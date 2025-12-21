"""Background training orchestration for the RescueNet-RL API."""

from __future__ import annotations

import queue
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
        total_timesteps: Optional[int],
        stochastic_eval: bool,
    ) -> TrainingRun:
        run_id = uuid.uuid4().hex
        run = TrainingRun(run_id=run_id, scenario_name=scenario_name, env_type=env_type)
        with self._lock:
            self._runs[run_id] = run

        thread = threading.Thread(
            target=self._execute_training,
            args=(run, scenario_name, env_type, total_timesteps, stochastic_eval),
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

    def _execute_training(
        self,
        run: TrainingRun,
        scenario_name: str,
        env_type: str,
        total_timesteps: Optional[int],
        stochastic_eval: bool,
    ) -> None:
        run.status = "initializing"
        self._push_event(run, {"type": "status", "payload": {"state": "initializing"}})
        try:
            config = get_default_config()
            config["experiment"]["env_type"] = env_type
            if env_type == "multimodal":
                config["multimodal_env"]["scenario_name"] = scenario_name
            if total_timesteps:
                config["train"]["total_timesteps"] = total_timesteps
            config["train"]["eval_deterministic"] = not stochastic_eval

            artifact_dir = Path(config["logging"]["artifact_dir"])
            artifact_dir.mkdir(parents=True, exist_ok=True)

            env = make_env(config, env_type)
            eval_env = make_env(config, env_type)
            device = config["train"].get("device", "auto")
            policy = build_policy(env, config["model"], env_type=env_type, device=device)

            trainer = PPOTrainer(
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
            plot_training_metrics(metrics, artifact_dir / "training_coverage_curve.png", skip=1)

            if env_type == "multimodal":
                dataset_path = config["multimodal_env"]["dataset_path"]
                architecture_path = artifact_dir / f"broadcast_architecture_{scenario_name}.json"
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
