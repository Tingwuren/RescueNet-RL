"""FastAPI application exposing training and simulation endpoints."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse

from configs.default_config import get_default_config
from data.resource_dataset import ResourceDataset
from server.schemas import (
    MahimahiSimulateRequest,
    MahimahiSimulateResponse,
    MahimahiTraceInfo,
    SceneImportRequest,
    SceneImportResponse,
    SimulationRequest,
    SimulationResponse,
    TrainRequest,
    TrainResponse,
    TrainingStatus,
)
from server.training_manager import TrainingManager
from server.mahimahi_manager import MahimahiManager
from server.ns3_replay_manager import Ns3ReplayManager
from services.evaluation import build_env, build_scene_preview, evaluate_policy, export_episode_scene, load_policy

app = FastAPI(title="RescueNet-RL API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

training_manager = TrainingManager()
mahimahi_manager = MahimahiManager()
ns3_replay_manager = Ns3ReplayManager()
default_config = get_default_config()
dataset_path = Path(default_config["multimodal_env"]["dataset_path"])
dataset = ResourceDataset(dataset_path)


def _classify_candidate_site(region_grid, row: int, col: int) -> str:
    label = region_grid.cell_label(row, col)
    if label and not label.startswith("cell-"):
        return "重点保障"

    row_norm = row / max(1, region_grid.rows - 1)
    col_norm = col / max(1, region_grid.cols - 1)
    is_edge = row_norm <= 0.16 or row_norm >= 0.84 or col_norm <= 0.16 or col_norm >= 0.84
    is_center = 0.35 <= row_norm <= 0.65 and 0.35 <= col_norm <= 0.65
    is_corridor = abs(row_norm - col_norm) <= 0.12 or abs((row_norm + col_norm) - 1.0) <= 0.12

    if is_center:
        return "核心覆盖"
    if is_corridor:
        return "中继转发"
    if is_edge:
        return "边缘补盲"
    return "机动接入"


def _build_candidate_site_preview(scenario_name: str) -> List[Dict[str, object]]:
    config = get_default_config()
    config["experiment"]["env_type"] = "multimodal"
    config["multimodal_env"]["scenario_name"] = scenario_name
    config["multimodal_env"]["seed"] = 42
    env = build_env(config, "multimodal")
    try:
        preview = []
        for site_index, coords in enumerate(env.candidate_locations):
            row = int(coords[0])
            col = int(coords[1])
            lat_min, lat_max, lon_min, lon_max = env.region_grid.cell_bounds(row, col)
            region_label = env.region_grid.cell_label(row, col)
            preview.append(
                {
                    "site_index": site_index,
                    "x": row,
                    "y": col,
                    "region_label": region_label,
                    "category": _classify_candidate_site(env.region_grid, row, col),
                    "lat_lon_bounds": {
                        "lat_min": lat_min,
                        "lat_max": lat_max,
                        "lon_min": lon_min,
                        "lon_max": lon_max,
                    },
                }
            )
        return preview
    finally:
        env.close()


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/train/latest-artifact")
def latest_training_artifact() -> Dict[str, object]:
    artifact_dir = Path(default_config["logging"]["artifact_dir"])
    meta_path = artifact_dir / "policy_meta.json"
    metrics_path = artifact_dir / "training_metrics.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No training artifact metadata found.")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    metrics = {}
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

    config = metrics.get("config", {}) if isinstance(metrics, dict) else {}
    experiment_cfg = config.get("experiment", {}) if isinstance(config, dict) else {}
    multimodal_cfg = config.get("multimodal_env", {}) if isinstance(config, dict) else {}
    updated_at = max(
        meta_path.stat().st_mtime if meta_path.exists() else 0.0,
        metrics_path.stat().st_mtime if metrics_path.exists() else 0.0,
    )
    return {
        "algorithm": meta.get("algorithm") or experiment_cfg.get("algorithm") or "ppo",
        "env_type": meta.get("env_type") or experiment_cfg.get("env_type") or "multimodal",
        "checkpoint_path": meta.get("policy_path"),
        "scenario_name": multimodal_cfg.get("scenario_name"),
        "reward_mode": multimodal_cfg.get("reward_mode"),
        "updated_at": updated_at,
    }


@app.get("/api/train/artifacts")
def list_training_artifacts() -> Dict[str, List[Dict[str, object]]]:
    artifact_dir = Path(default_config["logging"]["artifact_dir"])
    runs_dir = artifact_dir / "runs"
    artifacts: List[Dict[str, object]] = []
    if not runs_dir.exists():
        return {"artifacts": artifacts}

    for meta_path in runs_dir.glob("*/policy_meta.json"):
        metrics_path = meta_path.with_name("training_metrics.json")
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue

        metrics = {}
        if metrics_path.exists():
            try:
                with metrics_path.open("r", encoding="utf-8") as handle:
                    metrics = json.load(handle)
            except (OSError, json.JSONDecodeError):
                metrics = {}

        config = metrics.get("config", {}) if isinstance(metrics, dict) else {}
        experiment_cfg = config.get("experiment", {}) if isinstance(config, dict) else {}
        multimodal_cfg = config.get("multimodal_env", {}) if isinstance(config, dict) else {}
        policy_path = meta.get("policy_path")
        if not policy_path:
            continue

        updated_at = max(
            meta_path.stat().st_mtime if meta_path.exists() else 0.0,
            metrics_path.stat().st_mtime if metrics_path.exists() else 0.0,
        )
        artifacts.append(
            {
                "algorithm": meta.get("algorithm") or experiment_cfg.get("algorithm") or "ppo",
                "env_type": meta.get("env_type") or experiment_cfg.get("env_type") or "multimodal",
                "checkpoint_path": policy_path,
                "scenario_name": multimodal_cfg.get("scenario_name"),
                "reward_mode": multimodal_cfg.get("reward_mode"),
                "updated_at": updated_at,
                "run_dir": str(meta_path.parent),
            }
        )

    artifacts.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return {"artifacts": artifacts}


@app.get("/api/train/artifacts/detail")
def training_artifact_detail(run_dir: str) -> Dict[str, object]:
    artifact_dir = Path(default_config["logging"]["artifact_dir"]).resolve()
    runs_dir = (artifact_dir / "runs").resolve()
    requested_dir = Path(run_dir).resolve()

    try:
        requested_dir.relative_to(runs_dir)
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid training run directory.") from error

    meta_path = requested_dir / "policy_meta.json"
    metrics_path = requested_dir / "training_metrics.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Training artifact metadata not found.")

    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Failed to read policy metadata.") from error

    metrics = {}
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    episode_rewards = metrics.get("episode_rewards") if isinstance(metrics, dict) else []
    episode_coverages = metrics.get("episode_coverages") if isinstance(metrics, dict) else []
    episode_broadcasts = metrics.get("episode_broadcasts") if isinstance(metrics, dict) else []
    episode_timesteps = metrics.get("episode_timesteps") if isinstance(metrics, dict) else []
    eval_history = metrics.get("eval_history") if isinstance(metrics, dict) else []
    config = metrics.get("config", {}) if isinstance(metrics, dict) else {}
    experiment_cfg = config.get("experiment", {}) if isinstance(config, dict) else {}
    multimodal_cfg = config.get("multimodal_env", {}) if isinstance(config, dict) else {}
    algorithm_key = meta.get("algorithm") or experiment_cfg.get("algorithm") or "ppo"
    algorithm_cfg = config.get(str(algorithm_key).lower(), {}) if isinstance(config, dict) else {}
    train_cfg = config.get("train", {}) if isinstance(config, dict) else {}
    updated_at = max(
        meta_path.stat().st_mtime if meta_path.exists() else 0.0,
        metrics_path.stat().st_mtime if metrics_path.exists() else 0.0,
    )

    return {
        "algorithm": algorithm_key,
        "env_type": meta.get("env_type") or experiment_cfg.get("env_type") or "multimodal",
        "checkpoint_path": meta.get("policy_path"),
        "scenario_name": multimodal_cfg.get("scenario_name"),
        "reward_mode": multimodal_cfg.get("reward_mode"),
        "updated_at": updated_at,
        "run_dir": str(requested_dir),
        "episode_count": len(episode_rewards) if isinstance(episode_rewards, list) else 0,
        "total_timesteps": episode_timesteps[-1] if isinstance(episode_timesteps, list) and episode_timesteps else train_cfg.get("total_timesteps"),
        "last_reward": episode_rewards[-1] if isinstance(episode_rewards, list) and episode_rewards else None,
        "best_reward": max(episode_rewards) if isinstance(episode_rewards, list) and episode_rewards else None,
        "last_coverage": episode_coverages[-1] if isinstance(episode_coverages, list) and episode_coverages else None,
        "best_coverage": max(episode_coverages) if isinstance(episode_coverages, list) and episode_coverages else None,
        "last_broadcast": episode_broadcasts[-1] if isinstance(episode_broadcasts, list) and episode_broadcasts else None,
        "best_broadcast": max(episode_broadcasts) if isinstance(episode_broadcasts, list) and episode_broadcasts else None,
        "eval_history": eval_history if isinstance(eval_history, list) else [],
        "config": {
            "experiment": experiment_cfg if isinstance(experiment_cfg, dict) else {},
            "train": train_cfg if isinstance(train_cfg, dict) else {},
            "multimodal_env": multimodal_cfg if isinstance(multimodal_cfg, dict) else {},
            "algorithm": algorithm_cfg if isinstance(algorithm_cfg, dict) else {},
        },
    }


@app.get("/api/scenarios")
def list_scenarios() -> Dict[str, List[Dict[str, object]]]:
    scenarios = []
    for name in dataset.list_scenarios():
        record = dataset.get(name)
        candidate_site_preview = _build_candidate_site_preview(name)
        region_grid = record.region_grid.to_public_dict() if getattr(record, "region_grid", None) else None
        reward_profiles = [
            {
                "key": key,
                "label": profile.label,
                "description": profile.description,
                "coverage_weight": profile.coverage_weight,
                "bandwidth_weight": profile.bandwidth_weight,
                "throughput_weight": profile.throughput_weight,
                "broadcast_weight": profile.broadcast_weight,
                "device_cost_weight": profile.device_cost_weight,
                "bandwidth_cost_weight": profile.bandwidth_cost_weight,
            }
            for key, profile in sorted(record.reward_profiles.items())
        ]
        base_stations = [
            {
                "name": profile.name,
                "label": profile.label,
                "max_throughput": profile.max_throughput,
                "max_users": profile.max_users,
                "device_cost": profile.device_cost,
                "bandwidth_cost": profile.bandwidth_cost,
                "supported_modes": profile.supported_modes,
            }
            for profile in record.base_station_profiles.values()
        ]
        scenarios.append(
            {
                "name": record.name,
                "disaster_type": record.disaster_type,
                "grid_size": record.grid_size,
                "region_grid": region_grid,
                "num_users": record.num_users,
                "candidate_sites": record.candidate_sites,
                "max_steps": record.max_steps,
                "has_residual_network": record.has_residual_network,
                "reward_profiles": reward_profiles,
                "default_reward_profile": record.default_reward_profile,
                "base_stations": base_stations,
                "candidate_site_preview": candidate_site_preview,
            }
        )
    return {"scenarios": scenarios}


@app.post("/api/train", response_model=TrainResponse)
def start_training(request: TrainRequest) -> TrainResponse:
    run = training_manager.start_run(
        scenario_name=request.scenario_name,
        env_type=request.env_type,
        algorithm=request.algorithm,
        total_timesteps=request.total_timesteps,
        stochastic_eval=request.stochastic_eval,
        reward_mode=request.reward_mode,
        learning_rate=request.learning_rate,
        discount_factor=request.discount_factor,
        batch_size=request.batch_size,
        rollout_steps=request.rollout_steps,
        entropy_coef=request.entropy_coef,
        clip_range=request.clip_range,
        eval_interval=request.eval_interval,
    )
    return TrainResponse(run_id=run.run_id)


@app.get("/api/train/{run_id}", response_model=TrainingStatus)
def get_training_status(run_id: str) -> TrainingStatus:
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return TrainingStatus(
        run_id=run.run_id,
        status=run.status,
        scenario_name=run.scenario_name,
        env_type=run.env_type,
        algorithm=run.algorithm,
        reward_mode=run.reward_mode,
        started_at=run.started_at,
        updated_at=run.updated_at,
        error=run.error,
    )


@app.get("/api/train/{run_id}/stream")
async def stream_training_events(run_id: str):
    run = training_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if run.status in {"completed", "failed"} and run.events.empty():
                break
            try:
                event = await asyncio.to_thread(run.events.get, True, 0.5)
            except queue.Empty:
                continue
            payload = json.dumps(event)
            yield f"data: {payload}\n\n"
        yield f"data: {json.dumps({'type': 'end', 'status': run.status})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/simulate", response_model=SimulationResponse)
def simulate_strategy(request: SimulationRequest) -> SimulationResponse:
    config = get_default_config()
    config["experiment"]["env_type"] = request.env_type
    config["experiment"]["algorithm"] = request.algorithm
    if request.env_type == "multimodal":
        config["multimodal_env"]["scenario_name"] = request.scenario_name
        if request.reward_mode is not None:
            config["multimodal_env"]["reward_mode"] = request.reward_mode
    if request.eval_seed is not None:
        torch.manual_seed(request.eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.eval_seed)
        np.random.seed(request.eval_seed)

    checkpoint_path = Path(request.checkpoint_path)
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {checkpoint_path}")
    env = build_env(config, request.env_type)
    try:
        policy = load_policy(checkpoint_path, env, config, request.env_type, algorithm=request.algorithm)

        custom_state = [device.model_dump() for device in request.custom_devices]
        custom_base_stations = (
            [station.model_dump() for station in request.custom_base_stations]
            if request.custom_base_stations is not None
            else None
        )
        rewards, coverages, reports = evaluate_policy(
            env=env,
            policy=policy,
            episodes=request.episodes,
            deterministic=not request.stochastic_eval,
            render=False,
            custom_user_state=custom_state or None,
            custom_base_stations=custom_base_stations,
        )
        scene_export = None
        if reports:
            export_dir = Path(config["logging"]["artifact_dir"]) / "scene_exports"
            scene_export = export_episode_scene(reports[0], env, export_dir)

        return SimulationResponse(
            avg_reward=float(np.mean(rewards)),
            avg_final_coverage=float(np.mean(coverages)),
            reports=reports,
            scene_export=scene_export,
        )
    finally:
        env.close()


@app.post("/api/simulate/scene", response_model=SceneImportResponse)
def import_simulation_scene(request: SceneImportRequest) -> SceneImportResponse:
    config = get_default_config()
    config["experiment"]["env_type"] = request.env_type
    if request.env_type == "multimodal":
        config["multimodal_env"]["scenario_name"] = request.scenario_name

    env = build_env(config, request.env_type)
    custom_base_stations = (
        [station.model_dump() for station in request.custom_base_stations]
        if request.custom_base_stations is not None
        else None
    )

    try:
        preview = build_scene_preview(env, custom_base_stations=custom_base_stations)
        return SceneImportResponse(**preview)
    finally:
        env.close()


@app.post("/api/simulate/stream")
def stream_simulation(request: SimulationRequest):
    def encode_sse(event: Dict[str, object]) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def build_event(event_type: str, payload: Dict[str, object]) -> Dict[str, object]:
        return {
            "type": event_type,
            "payload": payload,
            "timestamp": time.time(),
        }

    event_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()

    def push_event(event_type: str, payload: Dict[str, object]) -> None:
        event_queue.put(build_event(event_type, payload))

    def run_simulation() -> None:
        env = None
        final_state = "failed"
        try:
            push_event("status", {"state": "initializing"})
            push_event("log", {"message": f"开始加载场景 {request.scenario_name}。"})
            config = get_default_config()
            config["experiment"]["env_type"] = request.env_type
            config["experiment"]["algorithm"] = request.algorithm
            if request.env_type == "multimodal":
                config["multimodal_env"]["scenario_name"] = request.scenario_name
                if request.reward_mode is not None:
                    config["multimodal_env"]["reward_mode"] = request.reward_mode
            if request.eval_seed is not None:
                torch.manual_seed(request.eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(request.eval_seed)
                np.random.seed(request.eval_seed)

            checkpoint_path = Path(request.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            env = build_env(config, request.env_type)
            custom_state = [device.model_dump() for device in request.custom_devices]
            custom_base_stations = (
                [station.model_dump() for station in request.custom_base_stations]
                if request.custom_base_stations is not None
                else None
            )

            push_event(
                "log",
                {
                    "message": f"加载模型 {checkpoint_path}，算法={request.algorithm}，episodes={request.episodes}，stochastic={request.stochastic_eval}，seed={request.eval_seed if request.eval_seed is not None else 'auto'}。"
                },
            )
            policy = load_policy(checkpoint_path, env, config, request.env_type, algorithm=request.algorithm)
            push_event("status", {"state": "running"})
            push_event("log", {"message": "测试开始，终端将持续输出策略执行过程。"})

            def push_progress(event: Dict[str, object]) -> None:
                message = str(event.get("message", "")).strip()
                if not message:
                    return
                push_event(
                    "log",
                    {
                        "message": message,
                        "event_type": str(event.get("type", "log")),
                    },
                )

            rewards, coverages, reports = evaluate_policy(
                env=env,
                policy=policy,
                episodes=request.episodes,
                deterministic=not request.stochastic_eval,
                render=False,
                custom_user_state=custom_state or None,
                custom_base_stations=custom_base_stations,
                progress_callback=push_progress,
            )

            scene_export = None
            if reports:
                export_dir = Path(config["logging"]["artifact_dir"]) / "scene_exports"
                scene_export = export_episode_scene(reports[0], env, export_dir)

            response = SimulationResponse(
                avg_reward=float(np.mean(rewards)),
                avg_final_coverage=float(np.mean(coverages)),
                reports=reports,
                scene_export=scene_export,
            )
            push_event("result", response.model_dump())
            push_event("status", {"state": "completed"})
            final_state = "completed"
        except Exception as exc:  # pylint: disable=broad-except
            push_event("error", {"message": str(exc)})
            push_event("status", {"state": "failed"})
        finally:
            if env is not None:
                env.close()
            push_event("end", {"state": final_state})

    def event_generator():
        worker = threading.Thread(target=run_simulation, daemon=True)
        worker.start()

        while True:
            try:
                event = event_queue.get(timeout=0.5)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            yield encode_sse(event)
            if event.get("type") == "end":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Mahimahi endpoints
# ---------------------------------------------------------------------------

@app.get("/api/mahimahi/status")
def mahimahi_status() -> Dict[str, object]:
    return {
        "mahimahi_available": mahimahi_manager.mahimahi_available,
        "traces_dir": str(mahimahi_manager.traces_dir),
    }


@app.get("/api/mahimahi/traces")
def list_mahimahi_traces() -> Dict[str, List[Dict[str, object]]]:
    return {"traces": mahimahi_manager.list_traces()}


@app.get("/api/mahimahi/traces/{trace_name}")
def get_trace_analysis(
    trace_name: str,
    duration_s: float = 60,
    window_ms: int = 500,
) -> Dict[str, object]:
    try:
        return mahimahi_manager.analyze_trace(trace_name, duration_s, window_ms)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_name}' not found")


@app.post("/api/mahimahi/simulate", response_model=MahimahiSimulateResponse)
def mahimahi_simulate(request: MahimahiSimulateRequest) -> MahimahiSimulateResponse:
    try:
        result = mahimahi_manager.simulate(
            trace_name=request.trace_name,
            duration_s=request.duration_s,
            rtt_ms=request.rtt_ms,
            buffer_packets=request.buffer_packets,
            window_ms=request.window_ms,
        )
        return MahimahiSimulateResponse(**result)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Trace '{request.trace_name}' not found")


@app.post("/api/mahimahi/simulate/stream")
def mahimahi_simulate_stream(request: MahimahiSimulateRequest):
    """SSE stream that pushes simulation progress and final results."""

    def encode_sse(event: Dict[str, object]) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    event_queue: "queue.Queue[Dict[str, object]]" = queue.Queue()

    def run() -> None:
        try:
            event_queue.put({"type": "status", "payload": {"state": "initializing"}, "timestamp": time.time()})
            event_queue.put({
                "type": "log",
                "payload": {"message": f"加载 trace: {request.trace_name}, RTT={request.rtt_ms}ms, 时长={request.duration_s}s"},
                "timestamp": time.time(),
            })

            result = mahimahi_manager.simulate(
                trace_name=request.trace_name,
                duration_s=request.duration_s,
                rtt_ms=request.rtt_ms,
                buffer_packets=request.buffer_packets,
                window_ms=request.window_ms,
            )

            n = len(result.get("capacity", []))
            chunk_size = max(1, n // 20)
            for i in range(0, n, chunk_size):
                chunk = {
                    "capacity": result["capacity"][i : i + chunk_size],
                    "throughput": result["throughput"][i : i + chunk_size],
                    "sending_rate": result["sending_rate"][i : i + chunk_size],
                }
                event_queue.put({"type": "data_chunk", "payload": chunk, "timestamp": time.time()})
                time.sleep(0.05)

            event_queue.put({"type": "result", "payload": {"stats": result["stats"]}, "timestamp": time.time()})
            event_queue.put({"type": "end", "payload": {"state": "completed"}, "timestamp": time.time()})
        except Exception as exc:
            event_queue.put({"type": "error", "payload": {"message": str(exc)}, "timestamp": time.time()})
            event_queue.put({"type": "end", "payload": {"state": "failed"}, "timestamp": time.time()})

    def generator():
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        while True:
            try:
                event = event_queue.get(timeout=0.5)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue
            yield encode_sse(event)
            if event.get("type") == "end":
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.on_event("startup")
async def startup_event() -> None:
    ns3_replay_manager.start()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    ns3_replay_manager.stop()


@app.get("/api/ns3/status")
def ns3_status() -> Dict[str, object]:
    return ns3_replay_manager.status()


@app.post("/api/ns3/run")
def ns3_run() -> Dict[str, object]:
    return ns3_replay_manager.start_simulation()


@app.post("/api/import")
def ns3_manual_import() -> Dict[str, object]:
    exp_id = ns3_replay_manager.import_trace()
    return {"success": exp_id is not None, "exp_id": exp_id}


@app.get("/api/experiments")
def list_ns3_experiments() -> List[Dict[str, object]]:
    return ns3_replay_manager.list_experiments()


@app.get("/api/exp/{exp_id}/charts")
def get_ns3_charts(exp_id: int) -> Dict[str, List[float]]:
    return ns3_replay_manager.get_charts(exp_id)


@app.get("/api/exp/{exp_id}/frame/{frame_idx}")
def get_ns3_frame(exp_id: int, frame_idx: int) -> Dict[str, object]:
    frame = ns3_replay_manager.get_frame(exp_id, frame_idx)
    if frame is None:
        raise HTTPException(status_code=404, detail="Frame not found.")
    return frame


def _resolve_frontend_dist() -> Path:
    configured = Path(os.environ.get("FRONTEND_DIST", "frontend/dist"))
    if configured.is_absolute():
        return configured
    project_root = Path(__file__).resolve().parents[1]
    return project_root / configured


def _mount_frontend_static() -> None:
    frontend_dist = _resolve_frontend_dist()
    if not frontend_dist.exists():
        return

    # Keep API routes unchanged, and serve built frontend for root/static files.
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


def _mount_ns3_native_static() -> None:
    if not ns3_replay_manager.ns3_root.exists():
        return
    app.mount("/ns3-native", StaticFiles(directory=ns3_replay_manager.ns3_root, html=True), name="ns3-native")


_mount_ns3_native_static()
_mount_frontend_static()
