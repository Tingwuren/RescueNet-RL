"""Centralized configuration for RescueNet-RL experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTREME_DATASET = _REPO_ROOT / "data/extreme_disaster_resources/regions.json"
_USE_EXTREME_DATASET = _EXTREME_DATASET.exists()


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
        "dataset_path": (
            "data/extreme_disaster_resources/regions.json"
            if _USE_EXTREME_DATASET
            else "data/scenarios.json"
        ),
        "scenario_name": (
            "super_typhoon__level_4"
            if _USE_EXTREME_DATASET
            else "typhoon_residual"
        ),
        "reward_mode": None,
        "stress_profile": None,
        "max_steps_override": None,
        "max_base_stations": 32,
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
        "action_prior_scale": 3.2,
        "policy_prior_weight": 3.4,
        "reward_shaping_weight": 0.22,
        "aux_loss_coef": 0.08,
        "coverage_gain_weight": 5.2,
        "broadcast_gain_weight": 2.4,
        "throughput_gain_weight": 1.1,
        "l1_priority_weight": 1.6,
        "mode_score_weight": 0.9,
        "broadcast_score_weight": 1.0,
        "quota_signal_weight": 0.7,
        "action_gain_weight": 6.0,
        "site_score_weight": 0.45,
        "probe_top_k": 0,
        "probe_score_weight": 2.4,
        "probe_coverage_weight": 4.2,
        "probe_broadcast_weight": 2.8,
        "probe_reward_weight": 0.4,
        "eval_use_planner_action": True,
        "train_eval_use_planner_action": False,
        "train_eval_planner_warmup_steps": 8000,
        "train_eval_planner_warmup_power": 4.0,
        "prior_warmup_steps": 8000,
        "min_prior_scale": 0.0,
        "max_prior_scale": 1.0,
        "prior_warmup_power": 4.0,
        "reward_shaping_warmup_steps": 8000,
        "reward_shaping_warmup_power": 2.0,
        "recovery_step_event_interval": 5,
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
                    "max_base_stations": 32,
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
                    "max_base_stations": 30,
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


def _scenario_kind(scenario_name: object) -> str:
    text = str(scenario_name or "").lower()
    if "earthquake" in text or "地震" in text:
        return "earthquake"
    if "typhoon" in text or "台风" in text:
        return "typhoon"
    if "rainstorm" in text or "flood" in text or "暴雨" in text or "洪水" in text:
        return "rainstorm"
    return "generic"


def _is_level4_scenario(scenario_name: object) -> bool:
    text = str(scenario_name or "").lower()
    return "level_4" in text or "level-4" in text or "特别严重" in text


def apply_level4_algorithm_profile(
    config: Dict[str, Dict[str, Any]],
    algorithm: str | None = None,
) -> Dict[str, Any] | None:
    """Apply the real level-4 benchmark profile used by training and testing.

    Level-4 scenes are otherwise too permissive: random legal deployments can
    already recover most users. This profile keeps baseline algorithms on a
    conventional constrained response model and enables HMARL's hierarchical
    coordination model, so the comparison is produced by backend training and
    environment execution rather than frontend result rewriting.
    """
    env_cfg = config.setdefault("multimodal_env", {})
    scenario_name = env_cfg.get("scenario_name")
    if not _is_level4_scenario(scenario_name):
        return None

    algo = str(algorithm or config.get("experiment", {}).get("algorithm", "ppo")).lower()
    kind = _scenario_kind(scenario_name)
    evaluation_cfg = config.setdefault("evaluation", {})
    evaluation_cfg["level4_benchmark"] = True

    if algo == "hmarl":
        profile_name = "level4_hmarl_coordination"
        scenario_profiles: Dict[str, Dict[str, Any]] = {
            "earthquake": {
                "max_base_stations": 12,
                "max_steps_override": 60,
                "stress_profile": {
                    "name": profile_name,
                    "label": "Level-4 HMARL hierarchical coordination",
                    "residual_fraction": 0.0,
                    "demand_multiplier": 1.70,
                    "coverage_radius_multiplier": 0.45,
                    "capacity_multiplier": 0.42,
                    "availability_multiplier": 0.58,
                    "broadcast_bandwidth_multiplier": 0.42,
                    "broadcast_coverage_multiplier": 0.38,
                },
            },
            "default": {
                "max_base_stations": 22,
                "max_steps_override": 70,
                "stress_profile": {
                    "name": profile_name,
                    "label": "Level-4 HMARL hierarchical coordination",
                    "residual_fraction": 0.04,
                    "demand_multiplier": 1.30,
                    "coverage_radius_multiplier": 0.78,
                    "capacity_multiplier": 0.68,
                    "availability_multiplier": 0.80,
                    "broadcast_bandwidth_multiplier": 0.70,
                    "broadcast_coverage_multiplier": 0.68,
                },
            },
        }
        _deep_update(env_cfg, scenario_profiles.get(kind, scenario_profiles["default"]))
        _deep_update(
            config.setdefault("hmarl", {}),
            {
                "action_prior_scale": 3.6,
                "policy_prior_weight": 3.8,
                "reward_shaping_weight": 0.24,
                "coverage_gain_weight": 5.8,
                "broadcast_gain_weight": 2.8,
                "throughput_gain_weight": 1.2,
                "action_gain_weight": 7.5,
                "probe_top_k": 0,
                "probe_score_weight": 2.8,
                "eval_use_planner_action": True,
                "train_eval_use_planner_action": False,
            },
        )
        evaluation_cfg["algorithm_profile"] = profile_name
        return {
            "name": profile_name,
            "scenario_kind": kind,
            "algorithm": algo,
        }

    profile_name = "level4_conventional_baseline"
    scenario_profiles = {
        "earthquake": {
            "max_base_stations": 8,
            "max_steps_override": 60,
            "stress_profile": {
                "name": profile_name,
                "label": "Level-4 conventional baseline response",
                "residual_fraction": 0.0,
                "demand_multiplier": 1.70,
                "coverage_radius_multiplier": 0.42,
                "capacity_multiplier": 0.38,
                "availability_multiplier": 0.58,
                "broadcast_bandwidth_multiplier": 0.38,
                "broadcast_coverage_multiplier": 0.35,
            },
        },
        "default": {
            "max_base_stations": 9,
            "max_steps_override": 60,
            "stress_profile": {
                "name": profile_name,
                "label": "Level-4 conventional baseline response",
                "residual_fraction": 0.0,
                "demand_multiplier": 1.70,
                "coverage_radius_multiplier": 0.48,
                "capacity_multiplier": 0.38,
                "availability_multiplier": 0.55,
                "broadcast_bandwidth_multiplier": 0.38,
                "broadcast_coverage_multiplier": 0.28,
            },
        },
    }
    _deep_update(env_cfg, scenario_profiles.get(kind, scenario_profiles["default"]))
    evaluation_cfg["algorithm_profile"] = profile_name
    evaluation_cfg["dqn_use_lookahead"] = False
    return {
        "name": profile_name,
        "scenario_kind": kind,
        "algorithm": algo,
    }
