"""HMARL trainer with metrics logging and per-update hierarchy I/O reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from algos.hmarl import HMARLTrainer


class RescuenetHMARLTrainer(HMARLTrainer):
    """Extends HMARLTrainer with metrics + L1/L2/L3 console reports after updates."""

    def __init__(
        self,
        *args,
        hierarchy_report_interval: int = 1,
        hierarchy_report_enabled: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.update_records: List[Dict[str, float]] = []
        self.eval_records: List[Dict[str, float]] = []
        self.hierarchy_report_interval = max(1, int(hierarchy_report_interval))
        self.hierarchy_report_enabled = bool(hierarchy_report_enabled)
        self._hierarchy_update_count = 0
        self._env_spec = None

    def _update_policy(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        info = super()._update_policy(batch)
        record = {
            "update": float(len(self.update_records) + 1),
            "step": float(self.global_step),
            **{key: float(value) for key, value in info.items()},
        }
        record["policy_loss_total"] = float(record.get("policy_loss", 0.0))
        self.update_records.append(record)
        self._hierarchy_update_count += 1
        self._maybe_print_hierarchy_report()
        return info

    def _maybe_print_hierarchy_report(self) -> None:
        if not self.hierarchy_report_enabled:
            return
        if self._hierarchy_update_count % self.hierarchy_report_interval != 0:
            return

        total_timesteps = int(self.train_cfg.get("total_timesteps", 1))
        rollout_steps = int(self.train_cfg.get("rollout_steps", 1))
        total_updates = max(1, (total_timesteps + rollout_steps - 1) // rollout_steps)
        progress = min(1.0, self._hierarchy_update_count / total_updates)

        mean_coverage = None
        if self.episode_coverages:
            tail = self.episode_coverages[-min(5, len(self.episode_coverages)) :]
            mean_coverage = float(np.mean(tail))
            progress = min(1.0, 0.65 * progress + 0.35 * mean_coverage)

        from rescuenet.hierarchy_report import print_hierarchy_report

        seed = int(self.train_cfg.get("seed", 0))
        disaster_label = "台风风暴潮"
        n_subregions = 5
        if self._env_spec is not None:
            disaster_label = getattr(self._env_spec, "l1_disaster_label", disaster_label)
            n_subregions = getattr(self._env_spec, "n_subregions", 5)
        print_hierarchy_report(
            progress=progress,
            phase="测试",
            update_idx=self._hierarchy_update_count,
            global_step=self.global_step,
            mean_coverage=mean_coverage,
            seed=seed + self._hierarchy_update_count,
            disaster_label=disaster_label,
            n_subregions=n_subregions,
            suppress_networking_handoff=True,
        )

    def evaluate(self, episodes: int = 5, deterministic: bool = True) -> Tuple[float, float, float]:
        eval_reward, eval_cov, eval_broadcast = super().evaluate(
            episodes=episodes, deterministic=deterministic
        )
        self.eval_records.append(
            {
                "episode": float(max(1, self.completed_episodes)),
                "step": float(self.global_step),
                "avg_reward": float(eval_reward),
                "avg_coverage": float(eval_cov),
                "avg_broadcast": float(eval_broadcast),
            }
        )
        return eval_reward, eval_cov, eval_broadcast

    def train(self) -> Dict[str, Any]:
        from rescuenet.hierarchy_report import get_env_spec, print_environment_banner

        alias = self.config.get("experiment", {}).get("scenario_alias")
        self._env_spec = get_env_spec(alias, self.config)
        print_environment_banner(self._env_spec, phase="训练")

        metrics = super().train()
        metrics["rescuenet_update_records"] = self.update_records
        metrics["rescuenet_eval_records"] = self.eval_records
        if self.completed_episodes > 0:
            metrics["avg_episode_steps"] = float(self.global_step / max(1, self.completed_episodes))
        return metrics

    def _save_artifacts(self, metrics: Dict[str, Any]) -> None:
        metrics_path = self.artifact_dir / "training_metrics.json"
        meta_path = self.artifact_dir / "policy_meta.json"
        weights_dir = self.artifact_dir / "weights"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        with meta_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "algorithm": self.algo_key,
                    "env_type": self.config.get("experiment", {}).get("env_type", "baseline"),
                    "weights_dir": str(weights_dir),
                    "layer_weights": {
                        "L1": str(weights_dir / "L1.pt"),
                        "L2": str(weights_dir / "L2.pt"),
                        "L3": str(weights_dir / "L3.pt"),
                    },
                    "evaluation_protocol": self.config.get("evaluation", {}).get("protocol", "standard"),
                    "config": self.config.get(self.algo_key, {}),
                },
                handle,
                indent=2,
            )
        print(f"Training metrics saved to {self.artifact_dir.resolve()}")
