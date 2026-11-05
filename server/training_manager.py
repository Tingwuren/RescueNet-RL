"""Background training orchestration for the RescueNet-RL API."""

from __future__ import annotations

import queue
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from algos.ppo import PPOTrainer
from configs.default_config import apply_evaluation_protocol, apply_level4_algorithm_profile, get_default_config
from planning.broadcast_architecture import export_architecture


def _update_interval_from_step_interval(step_interval: int, rollout_steps: int) -> int:
    return max(1, (max(1, int(step_interval)) + max(1, int(rollout_steps)) - 1) // max(1, int(rollout_steps)))


def _run_timestamp(started_at: float) -> str:
    return datetime.fromtimestamp(float(started_at)).strftime("%Y%m%d_%H%M%S")


def _run_dir_name(run: "TrainingRun") -> str:
    return f"{_run_timestamp(run.started_at)}_{run.algorithm}_{run.scenario_name}_{run.run_id}"


@dataclass
class TrainingRun:
    run_id: str
    scenario_name: str
    env_type: str
    algorithm: str
    reward_mode: Optional[str] = None
    evaluation_protocol: Optional[str] = None
    status: str = "pending"
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    events: "queue.Queue[Dict[str, Any]]" = field(default_factory=queue.Queue)
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None


class TrainingManager:
    """Track and execute asynchronous PPO training jobs."""

    def __init__(self) -> None:
        self._runs: Dict[str, TrainingRun] = {}
        self._lock = threading.Lock()

    def start_run(
        self,
        *,
        scenario_name: str,
        env_type: str,
        algorithm: str,
        total_timesteps: Optional[int],
        stochastic_eval: bool,
        reward_mode: Optional[str],
        evaluation_protocol: Optional[str],
        learning_rate: Optional[float],
        discount_factor: Optional[float],
        batch_size: Optional[int],
        rollout_steps: Optional[int],
        entropy_coef: Optional[float],
        clip_range: Optional[float],
        eval_interval: Optional[int],
        custom_base_stations: Optional[List[Dict[str, Any]]] = None,
    ) -> TrainingRun:
        run_id = uuid.uuid4().hex
        run = TrainingRun(
            run_id=run_id,
            scenario_name=scenario_name,
            env_type=env_type,
            algorithm=algorithm,
            reward_mode=reward_mode,
            evaluation_protocol=evaluation_protocol,
        )
        with self._lock:
            self._runs[run_id] = run

        thread = threading.Thread(
            target=self._execute_training,
            args=(
                run,
                scenario_name,
                env_type,
                algorithm,
                total_timesteps,
                stochastic_eval,
                reward_mode,
                evaluation_protocol,
                learning_rate,
                discount_factor,
                batch_size,
                rollout_steps,
                entropy_coef,
                clip_range,
                eval_interval,
                custom_base_stations,
            ),
            daemon=True,
        )
        run.thread = thread
        thread.start()
        return run

    def get_run(self, run_id: str) -> Optional[TrainingRun]:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> Dict[str, TrainingRun]:
        with self._lock:
            return dict(self._runs)

    def _push_event(self, run: TrainingRun, event: Dict[str, Any]) -> None:
        event["timestamp"] = time.time()
        run.events.put(event)
        run.updated_at = event["timestamp"]

    def _publish_latest_artifacts(self, run_artifact_dir: Path, latest_artifact_dir: Path) -> None:
        latest_artifact_dir.mkdir(parents=True, exist_ok=True)
        for name in ("policy_meta.json", "training_metrics.json", "training_coverage_curve.png"):
            src = run_artifact_dir / name
            if not src.exists():
                continue
            shutil.copy2(src, latest_artifact_dir / name)

    def _write_run_metadata(
        self,
        *,
        run: TrainingRun,
        run_artifact_dir: Path,
        policy_path: Optional[Path] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta_path = run_artifact_dir / "policy_meta.json"
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}

        if policy_path is None:
            existing_policy = metadata.get("policy_path")
            policy_path = Path(existing_policy) if existing_policy else None

        metadata.update(
            {
                "run_id": run.run_id,
                "run_name": run_artifact_dir.name,
                "algorithm": run.algorithm,
                "env_type": run.env_type,
                "scenario_name": run.scenario_name,
                "reward_mode": run.reward_mode,
                "evaluation_protocol": run.evaluation_protocol,
                "artifact_dir": str(run_artifact_dir),
                "created_at": run.started_at,
                "created_at_iso": datetime.fromtimestamp(run.started_at).isoformat(timespec="seconds"),
            }
        )
        if policy_path is not None:
            metadata["policy_path"] = str(policy_path)
        if extra:
            metadata.update(extra)

        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _resolve_demo_policy_path(self, algorithm: str, scenario_name: str) -> Optional[Path]:
        policy_name = "dqn_policy.pt" if algorithm == "dqn" else f"{algorithm}_policy.pt"
        preferred = Path("artifacts/runs") / f"{algorithm}_{scenario_name}_demo_level4" / policy_name
        if preferred.exists():
            return preferred
        candidate_paths = [
            *Path("artifacts/runs").glob(f"{algorithm}_*/{policy_name}"),
            *Path("artifacts/runs").glob(f"*_{algorithm}_*/{policy_name}"),
        ]
        candidates = sorted(
            {path.resolve(): path for path in candidate_paths}.values(),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _write_demo_curve(self, eval_history: List[Dict[str, float]], output_path: Path) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            steps = [item["step"] for item in eval_history]
            coverage = [item["avg_coverage"] * 100 for item in eval_history]
            broadcast = [item["avg_broadcast"] * 100 for item in eval_history]
            plt.figure(figsize=(8, 4.5))
            plt.plot(steps, coverage, marker="o", label="coverage")
            plt.plot(steps, broadcast, marker="s", label="broadcast")
            plt.xlabel("step")
            plt.ylabel("percent")
            plt.ylim(40, 100)
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        except Exception:
            # The live monitor uses SSE metrics; the image is only an artifact convenience.
            pass

    def _execute_demo_training(
        self,
        run: TrainingRun,
        scenario_name: str,
        env_type: str,
        algorithm: str,
        total_timesteps: Optional[int],
        reward_mode: Optional[str],
        evaluation_protocol: Optional[str],
        learning_rate: Optional[float],
        discount_factor: Optional[float],
        batch_size: Optional[int],
        rollout_steps: Optional[int],
        eval_interval: Optional[int],
        custom_base_stations: Optional[List[Dict[str, Any]]],
    ) -> None:
        config = get_default_config()
        config["experiment"]["env_type"] = "multimodal" if algorithm == "hmarl" else env_type
        config["experiment"]["algorithm"] = algorithm
        config["train"]["total_timesteps"] = int(total_timesteps or 1000)
        config["train"]["rollout_steps"] = int(rollout_steps or 1024)
        config["train"]["eval_interval_steps"] = int(eval_interval or config["train"]["total_timesteps"])
        config["train"]["eval_interval"] = 1
        if learning_rate is not None:
            config.setdefault(algorithm, {})["learning_rate"] = learning_rate
        if discount_factor is not None:
            config.setdefault(algorithm, {})["gamma"] = discount_factor
        if batch_size is not None:
            config.setdefault(algorithm, {})["mini_batch_size"] = batch_size
        if config["experiment"]["env_type"] == "multimodal":
            config["multimodal_env"]["scenario_name"] = scenario_name
            if reward_mode is not None:
                config["multimodal_env"]["reward_mode"] = reward_mode
        apply_evaluation_protocol(config, evaluation_protocol)
        profile = apply_level4_algorithm_profile(config, algorithm)
        run.evaluation_protocol = config.get("evaluation", {}).get("protocol", "standard")

        latest_artifact_dir = Path(config["logging"]["artifact_dir"])
        run_artifact_dir = latest_artifact_dir / "runs" / _run_dir_name(run)
        run_artifact_dir.mkdir(parents=True, exist_ok=True)
        config["logging"]["artifact_dir"] = str(run_artifact_dir)

        policy_name = "dqn_policy.pt" if algorithm == "dqn" else f"{algorithm}_policy.pt"
        policy_path = run_artifact_dir / policy_name
        source_policy = self._resolve_demo_policy_path(algorithm, scenario_name)
        if source_policy and source_policy.exists():
            shutil.copy2(source_policy, policy_path)
        else:
            policy_path.write_bytes(b"demo-policy-placeholder\n")

        device_count = len(custom_base_stations or [])
        run.status = "running"
        self._push_event(run, {"type": "status", "payload": {"state": "running"}})
        self._push_event(
            run,
            {
                "type": "log",
                "payload": {
                    "message": "训练入口缺失，已切换快速演示训练流，用于展示设备同步、训练事件和曲线变化。"
                },
            },
        )
        if profile:
            self._push_event(
                run,
                {
                    "type": "log",
                    "payload": {
                        "message": f"已启用特别严重场景算法基准：{profile['name']} scenario_kind={profile['scenario_kind']}。"
                    },
                },
            )
        if custom_base_stations is not None:
            self._push_event(
                run,
                {
                    "type": "log",
                    "payload": {"message": f"已加载场景设备配置：{device_count} 个基站作为待部署资源进入训练。"},
                },
            )

        total_steps = int(config["train"]["total_timesteps"])
        step_points = [max(1, round(total_steps * ratio)) for ratio in (0.12, 0.25, 0.38, 0.52, 0.68, 0.84, 1.0)]
        coverages = [0.58, 0.66, 0.74, 0.82, 0.89, 0.94, 0.985]
        broadcasts = [0.52, 0.61, 0.70, 0.79, 0.86, 0.92, 0.976]
        rewards = [8.4, 10.7, 13.1, 15.8, 18.2, 20.5, 22.8]
        eval_history: List[Dict[str, float]] = []
        for index, step in enumerate(step_points, start=1):
            coverage = coverages[index - 1]
            broadcast = broadcasts[index - 1]
            reward = rewards[index - 1]
            eval_history.append(
                {
                    "step": float(step),
                    "avg_reward": reward,
                    "avg_coverage": coverage,
                    "avg_broadcast": broadcast,
                }
            )
            self._push_event(
                run,
                {
                    "type": "episode",
                    "payload": {
                        "episode": index,
                        "step": step,
                        "steps": step,
                        "total_timesteps": total_steps,
                        "reward": reward,
                        "coverage": coverage,
                        "broadcast": broadcast,
                        "hierarchy": {
                            "summary": {
                                "target_region_id": f"R{index}",
                                "l2_link_count": 3 + index,
                                "l3_deployed_devices": min(device_count or 98, 70 + index * 4),
                            }
                        },
                        "hierarchical_rewards": {"l3_final": reward / 10},
                    },
                },
            )
            time.sleep(0.85)

        metrics = {
            "episode_rewards": rewards,
            "episode_coverages": coverages,
            "episode_broadcasts": broadcasts,
            "episode_timesteps": step_points,
            "eval_history": eval_history,
            "config": config,
        }
        run.metrics = metrics
        (run_artifact_dir / "training_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_run_metadata(
            run=run,
            run_artifact_dir=run_artifact_dir,
            policy_path=policy_path,
            extra={
                "env_type": config["experiment"]["env_type"],
                "demo_fast_training": True,
                "demo_note": "Fast deterministic training stream for system demonstration when train.py is unavailable.",
            },
        )
        self._write_demo_curve(eval_history, run_artifact_dir / "training_coverage_curve.png")
        self._publish_latest_artifacts(run_artifact_dir, latest_artifact_dir)

        run.status = "completed"
        self._push_event(
            run,
            {
                "type": "log",
                "payload": {"message": "快速演示训练完成，训练曲线与策略权重已归档。"},
            },
        )
        self._push_event(run, {"type": "status", "payload": {"state": "completed", "step": total_steps}})

    def _execute_training(
        self,
        run: TrainingRun,
        scenario_name: str,
        env_type: str,
        algorithm: str,
        total_timesteps: Optional[int],
        stochastic_eval: bool,
        reward_mode: Optional[str],
        evaluation_protocol: Optional[str],
        learning_rate: Optional[float],
        discount_factor: Optional[float],
        batch_size: Optional[int],
        rollout_steps: Optional[int],
        entropy_coef: Optional[float],
        clip_range: Optional[float],
        eval_interval: Optional[int],
        custom_base_stations: Optional[List[Dict[str, Any]]],
    ) -> None:
        run.status = "initializing"
        self._push_event(run, {"type": "status", "payload": {"state": "initializing"}})
        try:
            try:
                from train import build_policy, make_env, plot_training_metrics
            except ModuleNotFoundError as exc:
                if exc.name != "train":
                    raise
                if os.getenv("RESCUENET_ALLOW_DEMO_TRAINING") != "1":
                    raise RuntimeError("真实训练入口 train.py 不可用，未启用快速演示训练。") from exc
                self._execute_demo_training(
                    run,
                    scenario_name,
                    env_type,
                    algorithm,
                    total_timesteps,
                    reward_mode,
                    evaluation_protocol,
                    learning_rate,
                    discount_factor,
                    batch_size,
                    rollout_steps,
                    eval_interval,
                    custom_base_stations,
                )
                return

            self._push_event(
                run,
                {
                    "type": "log",
                    "payload": {"message": "已加载真实训练入口，开始执行真实模型训练。"},
                },
            )
            config = get_default_config()
            config["experiment"]["env_type"] = env_type
            config["experiment"]["algorithm"] = algorithm
            if algorithm == "hmarl" and env_type != "multimodal":
                env_type = "multimodal"
                config["experiment"]["env_type"] = env_type
                run.env_type = env_type
            if env_type == "multimodal":
                config["multimodal_env"]["scenario_name"] = scenario_name
                if reward_mode is not None:
                    config["multimodal_env"]["reward_mode"] = reward_mode
            apply_evaluation_protocol(config, evaluation_protocol)
            profile = apply_level4_algorithm_profile(config, algorithm)
            run.evaluation_protocol = config.get("evaluation", {}).get("protocol", "standard")
            if total_timesteps:
                config["train"]["total_timesteps"] = total_timesteps
            if rollout_steps:
                config["train"]["rollout_steps"] = rollout_steps
            if eval_interval:
                eval_interval_steps = max(1, int(eval_interval))
                eval_interval_updates = _update_interval_from_step_interval(
                    eval_interval_steps,
                    int(config["train"].get("rollout_steps") or 1),
                )
                config["train"]["eval_interval_steps"] = eval_interval_steps
                config["train"]["eval_interval_updates"] = eval_interval_updates
                config["train"]["eval_interval"] = eval_interval_updates
            config["train"]["eval_deterministic"] = not stochastic_eval

            algo_cfg = config.get(algorithm, {})
            if learning_rate is not None:
                algo_cfg["learning_rate"] = learning_rate
            if discount_factor is not None:
                algo_cfg["gamma"] = discount_factor
            if batch_size is not None:
                if algorithm == "dqn":
                    algo_cfg["batch_size"] = batch_size
                else:
                    algo_cfg["mini_batch_size"] = batch_size
            if entropy_coef is not None and algorithm in {"ppo", "a3c", "mppo", "hmarl"}:
                algo_cfg["entropy_coef"] = entropy_coef
            if clip_range is not None and algorithm in {"ppo", "a3c", "mppo", "hmarl"}:
                algo_cfg["clip_coef"] = clip_range

            if profile:
                self._push_event(
                    run,
                    {
                        "type": "log",
                        "payload": {
                            "message": (
                                f"已启用特别严重场景算法基准：{profile['name']} "
                                f"algorithm={profile['algorithm']} scenario_kind={profile['scenario_kind']}。"
                            )
                        },
                    },
                )

            latest_artifact_dir = Path(config["logging"]["artifact_dir"])
            latest_artifact_dir.mkdir(parents=True, exist_ok=True)
            run_artifact_dir = latest_artifact_dir / "runs" / _run_dir_name(run)
            run_artifact_dir.mkdir(parents=True, exist_ok=True)
            config["logging"]["artifact_dir"] = str(run_artifact_dir)

            env = make_env(config, env_type)
            eval_env = make_env(config, env_type)
            if custom_base_stations is not None:
                for target_env in (env, eval_env):
                    if hasattr(target_env, "set_custom_base_stations"):
                        target_env.set_custom_base_stations(custom_base_stations)
                self._push_event(
                    run,
                    {
                        "type": "log",
                        "payload": {
                            "message": f"已加载场景设备配置：{len(custom_base_stations)} 个基站作为待部署资源进入训练。"
                        },
                    },
                )
            device = config["train"].get("device", "auto")
            policy = build_policy(env, config, env_type=env_type, device=device)

            from algos.dqn import DQNTrainer
            from algos.a3c import A3CTrainer
            from algos.mppo import MPPOTrainer
            from algos.hmarl import HMARLTrainer

            trainer_cls = {
                "ppo": PPOTrainer,
                "dqn": DQNTrainer,
                "a3c": A3CTrainer,
                "mppo": MPPOTrainer,
                "hmarl": HMARLTrainer,
            }.get(algorithm, PPOTrainer)

            trainer = trainer_cls(
                env=env,
                eval_env=eval_env,
                policy=policy,
                config=config,
                progress_callback=lambda event: self._push_event(run, event),
            )

            run.status = "running"
            self._push_event(run, {"type": "status", "payload": {"state": "running"}})

            metrics = trainer.train()
            run.metrics = metrics
            plot_training_metrics(metrics, run_artifact_dir / "training_coverage_curve.png", skip=1)

            if env_type == "multimodal":
                dataset_path = config["multimodal_env"]["dataset_path"]
                architecture_path = run_artifact_dir / f"broadcast_architecture_{scenario_name}.json"
                export_architecture(dataset_path, scenario_name, architecture_path)

            policy_name = "dqn_policy.pt" if algorithm == "dqn" else f"{algorithm}_policy.pt"
            self._write_run_metadata(
                run=run,
                run_artifact_dir=run_artifact_dir,
                policy_path=run_artifact_dir / policy_name,
            )
            self._publish_latest_artifacts(run_artifact_dir, latest_artifact_dir)

            run.status = "completed"
            self._push_event(
                run,
                {
                    "type": "status",
                    "payload": {"state": "completed", "step": trainer.global_step},
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            run.status = "failed"
            run.error = str(exc)
            self._push_event(run, {"type": "error", "payload": {"message": str(exc)}})
