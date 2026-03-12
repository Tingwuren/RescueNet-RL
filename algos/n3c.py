"""A3C trainer wrapper (multi-head critic) built on PPO."""

from __future__ import annotations

from typing import Any, Dict, Optional

from algos.ppo import PPOTrainer


class A3CTrainer(PPOTrainer):
    """Reuse PPO loop with A3CPolicy; pulls hyperparameters from config['a3c'] if present."""

    def __init__(
        self,
        env,
        eval_env,
        policy,
        config: Dict[str, Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> None:
        config.setdefault("experiment", {})
        if "a3c" not in config and "n3c" in config:
            config["a3c"] = config["n3c"]
        config["experiment"]["algorithm"] = "a3c"
        super().__init__(env=env, eval_env=eval_env, policy=policy, config=config, progress_callback=progress_callback)


# Backward compatibility alias.
N3CTrainer = A3CTrainer
