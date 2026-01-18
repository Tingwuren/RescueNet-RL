"""MPPO trainer wrapper built on PPO with multi-head actor."""

from __future__ import annotations

from typing import Any, Dict, Optional

from algos.ppo import PPOTrainer


class MPPOTrainer(PPOTrainer):
    """Reuse PPO loop with MPPOPolicy; pulls hyperparameters from config['mppo'] if present."""

    def __init__(
        self,
        env,
        eval_env,
        policy,
        config: Dict[str, Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> None:
        config.setdefault("experiment", {})
        config["experiment"]["algorithm"] = "mppo"
        super().__init__(env=env, eval_env=eval_env, policy=policy, config=config, progress_callback=progress_callback)
