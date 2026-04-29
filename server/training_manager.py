"""Background training orchestration for the RescueNet-RL API."""

from __future__ import annotations

import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from algos.ppo import PPOTrainer
from configs.default_config import get_default_config
from planning.broadcast_architecture import export_architecture
from train import build_policy, make_env, plot_training_metrics


@dataclass
class TrainingRun:
    run_id: str
    scenario_name: str
    env_type: str
    algorithm: str
    reward_mode: Optional[str] = None
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
        learning_rate: Optional[float],
        discount_factor: Optional[float],
        batch_size: Optional[int],
        rollout_steps: Optional[int],
        entropy_coef: Optional[float],
        clip_range: Optional[float],
        eval_interval: Optional[int],
    ) -> TrainingRun:
        run_id = uuid.uuid4().hex
        run = TrainingRun(
            run_id=run_id,
            scenario_name=scenario_name,
            env_type=env_type,
            algorithm=algorithm,
            reward_mode=reward_mode,
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
                learning_rate,
                discount_factor,
                batch_size,
                rollout_steps,
                entropy_coef,
                clip_range,
                eval_interval,
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

    def _execute_training(
        self,
        run: TrainingRun,
        scenario_name: str,
        env_type: str,
        algorithm: str,
        total_timesteps: Optional[int],
        stochastic_eval: bool,
        reward_mode: Optional[str],
        learning_rate: Optional[float],
        discount_factor: Optional[float],
        batch_size: Optional[int],
        rollout_steps: Optional[int],
        entropy_coef: Optional[float],
        clip_range: Optional[float],
        eval_interval: Optional[int],
    ) -> None:
        run.status = "initializing"
        self._push_event(run, {"type": "status", "payload": {"state": "initializing"}})
        try:
            config = get_default_config()
            config["experiment"]["env_type"] = env_type
            config["experiment"]["algorithm"] = algorithm
            if env_type == "multimodal":
                config["multimodal_env"]["scenario_name"] = scenario_name
                if reward_mode is not None:
                    config["multimodal_env"]["reward_mode"] = reward_mode
            if total_timesteps:
                config["train"]["total_timesteps"] = total_timesteps
            if rollout_steps:
                config["train"]["rollout_steps"] = rollout_steps
            if eval_interval:
                config["train"]["eval_interval"] = eval_interval
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
            if entropy_coef is not None and algorithm in {"ppo", "a3c", "mppo"}:
                algo_cfg["entropy_coef"] = entropy_coef
            if clip_range is not None and algorithm in {"ppo", "a3c", "mppo"}:
                algo_cfg["clip_coef"] = clip_range

            latest_artifact_dir = Path(config["logging"]["artifact_dir"])
            latest_artifact_dir.mkdir(parents=True, exist_ok=True)
            run_artifact_dir = latest_artifact_dir / "runs" / f"{algorithm}_{scenario_name}_{run.run_id}"
            run_artifact_dir.mkdir(parents=True, exist_ok=True)
            config["logging"]["artifact_dir"] = str(run_artifact_dir)

            env = make_env(config, env_type)
            eval_env = make_env(config, env_type)
            device = config["train"].get("device", "auto")
            policy = build_policy(env, config, env_type=env_type, device=device)

            from algos.dqn import DQNTrainer
            from algos.a3c import A3CTrainer
            from algos.mppo import MPPOTrainer

            trainer_cls = {
                "ppo": PPOTrainer,
                "dqn": DQNTrainer,
                "a3c": A3CTrainer,
                "mppo": MPPOTrainer,
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
            self._publish_latest_artifacts(run_artifact_dir, latest_artifact_dir)

            if env_type == "multimodal":
                dataset_path = config["multimodal_env"]["dataset_path"]
                architecture_path = run_artifact_dir / f"broadcast_architecture_{scenario_name}.json"
                export_architecture(dataset_path, scenario_name, architecture_path)

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
