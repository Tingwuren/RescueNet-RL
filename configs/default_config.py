"""Centralized configuration for RescueNet-RL experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "env": {
        "grid_size": 10,
        "num_users": 45,
        "candidate_sites": 25,
        "max_steps": 25,
        "initial_outage_fraction": 0.65,
        "coverage_radius": 2.5,
        "max_base_stations": 7,
        "coverage_reward": 1.0,
        "deployment_cost": 0.3,
        "invalid_action_penalty": 0.2,
        "seed": 42,
    },
    "model": {
        "hidden_sizes": [128, 128],
    },
    "ppo": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "update_epochs": 10,
        "mini_batch_size": 128,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "train": {
        "total_timesteps": 8000,
        "rollout_steps": 1024,
        "seed": 123,
        "log_interval": 5,
        "eval_interval": 4,
        "eval_episodes": 5,
        "device": "auto",
        "log_episodes": False,
    },
    "logging": {
        "artifact_dir": "artifacts",
    },
}


def get_default_config() -> Dict[str, Dict[str, Any]]:
    """Return a deepcopy of the default nested configuration dictionary."""
    return deepcopy(DEFAULT_CONFIG)
