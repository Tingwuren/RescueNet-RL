"""N3C trainer wrapper (multi-head critic) built on PPO."""

from __future__ import annotations

from typing import Any, Dict, Optional

from algos.ppo import PPOTrainer


class N3CTrainer(PPOTrainer):
    """Reuse PPO loop with N3CPolicy; pulls hyperparameters from config['n3c'] if present."""

    def __init__(
        self,
        env,
        eval_env,
        policy,
        config: Dict[str, Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> None:
        config.setdefault("experiment", {})
        config["experiment"]["algorithm"] = "n3c"
        super().__init__(env=env, eval_env=eval_env, policy=policy, config=config, progress_callback=progress_callback)
