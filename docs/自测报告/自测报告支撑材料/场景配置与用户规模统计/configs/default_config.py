"""Centralized configuration for RescueNet-RL experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "experiment": {
        "env_type": "baseline",  # baseline | multimodal
        "algorithm": "ppo",  # ppo | dqn | a3c | mppo | hmarl
    },
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
    "multimodal_env": {
        "dataset_path": "data/scenarios.json",
        "scenario_name": "typhoon_residual",
        "reward_mode": None,
        "stress_profile": None,
        "max_steps_override": None,
        "max_base_stations": 24,
        "coverage_reward": 1.0,
        "bandwidth_reward": 0.05,
        "broadcast_reward": 0.4,
        "invalid_action_penalty": 0.3,
        "demand_penalty": 0.02,
        "seed": 42,
    },
    "model": {
        "hidden_sizes": [128, 128],
        "multimodal_hidden_sizes": [1024, 1024, 512, 512],
        "hmarl_hidden_sizes": [768, 512, 256],
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
    # DQN: decomposed Q-learning for large discrete action space
    "dqn": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "buffer_size": 200_000,
        "batch_size": 512,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 300_000,
        "target_update_tau": 0.005,  # soft update coefficient
        "target_update_period": 1000,  # hard update fallback
        "n_step": 3,
    },
    # A3C: PPO-compatible policy with triple-value heads (coverage/throughput/cost)
    "a3c": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "update_epochs": 10,
        "mini_batch_size": 128,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "value_weights": {"coverage": 1.0, "throughput": 0.5, "cost": 0.2},
    },
    # MPPO: multi-head actor for different reward modes / scenarios
    "mppo": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "update_epochs": 10,
        "mini_batch_size": 128,
        "entropy_coef": 0.02,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "head_keys": ["default"],
        "default_head_key": "default",
    },
    # HMARL: hierarchical L1/L2/L3 PPO for resource allocation and topology planning
    "hmarl": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "update_epochs": 8,
        "mini_batch_size": 128,
        "entropy_coef": 0.015,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
        "region_rows": 3,
        "region_cols": 3,
        "l1_regions": 9,
        "l2_link_types": 4,
        "action_prior_scale": 2.0,
        "policy_prior_weight": 2.5,
        "reward_shaping_weight": 0.18,
        "aux_loss_coef": 0.08,
        "coverage_gain_weight": 3.4,
        "broadcast_gain_weight": 1.2,
        "throughput_gain_weight": 0.8,
        "eval_use_planner_action": True,
        "train_eval_use_planner_action": False,
        "prior_warmup_steps": 12000,
        "min_prior_scale": 0.0,
        "max_prior_scale": 1.0,
        "prior_warmup_power": 4.0,
        "reward_shaping_warmup_steps": 12000,
        "reward_shaping_warmup_power": 2.0,
    },
    "evaluation": {
        "protocol": "standard",
        "target_coverage_band": [0.0, 1.0],
        "dqn_use_lookahead": True,
        "protocols": {
            "standard": {
                "label": "Standard",
                "description": "Use the scenario dataset as configured.",
                "target_coverage_band": [0.0, 1.0],
                "dqn_use_lookahead": True,
                "multimodal_env": {
                    "max_base_stations": 24,
                    "stress_profile": None,
                    "max_steps_override": None,
                },
            },
            "earthquake_stress": {
                "label": "Earthquake Stress",
                "description": "High-intensity post-earthquake link recovery protocol for comparing hierarchy-aware coordination.",
                "target_coverage_band": [0.8, 0.9],
                "dqn_use_lookahead": False,
                "multimodal_env": {
                    "max_base_stations": 20,
                    "max_steps_override": 60,
                    "stress_profile": {
                        "name": "earthquake_stress",
                        "label": "Earthquake Stress",
                        "residual_fraction": 0.08,
                        "demand_multiplier": 1.20,
                        "coverage_radius_multiplier": 0.88,
                        "capacity_multiplier": 0.78,
                        "availability_multiplier": 0.88,
                        "broadcast_bandwidth_multiplier": 0.84,
                        "broadcast_coverage_multiplier": 0.90,
                    },
                },
            },
        },
    },
    "train": {
        "total_timesteps": 8000,
        "rollout_steps": 1024,
        "seed": 123,
        "log_interval": 5,
        "eval_interval": 4,
        "eval_episodes": 5,
        "eval_deterministic": True,
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


def _deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Recursively merge updates into target."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def apply_evaluation_protocol(
    config: Dict[str, Dict[str, Any]], protocol_name: str | None = None
) -> Dict[str, Any]:
    """Apply a named, public evaluation protocol to a mutable config."""
    evaluation_cfg = config.setdefault("evaluation", {})
    protocols = evaluation_cfg.get("protocols", {})
    selected = protocol_name or evaluation_cfg.get("protocol") or "standard"
    if selected not in protocols:
        known = ", ".join(sorted(protocols)) or "standard"
        raise ValueError(f"Unknown evaluation protocol '{selected}'. Available protocols: {known}")

    protocol = deepcopy(protocols[selected])
    evaluation_cfg["protocol"] = selected
    evaluation_cfg["target_coverage_band"] = deepcopy(
        protocol.get("target_coverage_band", evaluation_cfg.get("target_coverage_band", [0.0, 1.0]))
    )
    evaluation_cfg["dqn_use_lookahead"] = bool(
        protocol.get("dqn_use_lookahead", evaluation_cfg.get("dqn_use_lookahead", True))
    )

    multimodal_overrides = protocol.get("multimodal_env", {})
    if multimodal_overrides:
        config.setdefault("multimodal_env", {})
        _deep_update(config["multimodal_env"], multimodal_overrides)

    stress_profile = config.get("multimodal_env", {}).get("stress_profile")
    if isinstance(stress_profile, dict):
        stress_profile.setdefault("name", selected)
        stress_profile.setdefault("label", protocol.get("label", selected))

    return protocol
