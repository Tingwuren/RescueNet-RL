"""FastAPI application exposing training and simulation endpoints."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from configs.default_config import get_default_config
from data.resource_dataset import ResourceDataset
from server.schemas import (
    SceneImportRequest,
    SceneImportResponse,
    SimulationRequest,
    SimulationResponse,
    TrainRequest,
    TrainResponse,
    TrainingStatus,
)
from server.training_manager import TrainingManager
from services.evaluation import build_env, build_scene_preview, evaluate_policy, export_episode_scene, load_policy

app = FastAPI(title="RescueNet-RL API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

training_manager = TrainingManager()
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

    checkpoint_path = Path(request.checkpoint_path)
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

            checkpoint_path = Path(request.checkpoint_path)
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
                    "message": f"加载模型 {checkpoint_path}，算法={request.algorithm}，episodes={request.episodes}。"
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
