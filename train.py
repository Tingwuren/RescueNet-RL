"""Real training entry points used by the RescueNet-RL API.

The FastAPI training manager imports this module lazily.  Keeping these
helpers at the repository root prevents the API from falling back to the
deterministic demo training stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from envs import DisasterCellularEnv, MultiModalCommEnv
from models.a3c_policy import A3CPolicy
from models.dqn_network import DQNNetwork
from models.hmarl_policy import HMARLPolicy
from models.mppo_policy import MPPOPolicy
from models.multimodal_policy import MultimodalPolicy
from models.policy_network import MLPActorCritic
from planning.hierarchical_marl import HierarchicalMARLPlanner


def make_env(config: Dict[str, Dict[str, Any]], env_type: str):
    """Build the actual training environment requested by the API."""
    if env_type == "multimodal":
        return MultiModalCommEnv(**config["multimodal_env"])
    return DisasterCellularEnv(**config["env"])


def build_policy(env, config: Dict[str, Dict[str, Any]], env_type: str, device: str = "auto"):
    """Build the policy network used by the selected real trainer."""
    algorithm = config.get("experiment", {}).get("algorithm", "ppo")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model_cfg = config.get("model", {})
    hidden_key = "multimodal_hidden_sizes" if env_type == "multimodal" else "hidden_sizes"
    hidden_sizes = model_cfg.get(
        hidden_key,
        [1024, 1024, 512, 512] if env_type == "multimodal" else [128, 128],
    )

    if algorithm == "dqn":
        return DQNNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
    if algorithm == "a3c":
        return A3CPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            value_weights=config.get("a3c", config.get("n3c", {})).get("value_weights"),
            device=device,
        )
    if algorithm == "mppo":
        mppo_cfg = config.get("mppo", {})
        head_keys = mppo_cfg.get("head_keys", ["default"])
        default_head_key = mppo_cfg.get("default_head_key", head_keys[0] if head_keys else "default")
        active_head_key = (
            config.get("multimodal_env", {}).get("reward_mode")
            or config.get("multimodal_env", {}).get("scenario_name")
            or default_head_key
        )
        return MPPOPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            head_keys=head_keys,
            active_head_key=active_head_key,
            device=device,
        )
    if algorithm == "hmarl":
        hmarl_cfg = config.get("hmarl", {})
        policy = HMARLPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_cfg.get("hmarl_hidden_sizes", [768, 512, 256]),
            l1_regions=int(
                hmarl_cfg.get(
                    "l1_regions",
                    hmarl_cfg.get("region_rows", 3) * hmarl_cfg.get("region_cols", 3),
                )
            ),
            l2_link_types=int(hmarl_cfg.get("l2_link_types", 4)),
            prior_weight=float(hmarl_cfg.get("policy_prior_weight", 1.25)),
            device=device,
        )
        policy.hierarchical_planner = HierarchicalMARLPlanner(hmarl_cfg)
        policy.hmarl_eval_use_planner_action = bool(hmarl_cfg.get("eval_use_planner_action", True))
        return policy
    if env_type == "multimodal":
        return MultimodalPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device,
        )
    return MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=hidden_sizes,
        device=device,
    )


def _series(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float, np.floating)):
            out.append(float(value))
    return out


def plot_training_metrics(metrics: Dict[str, Any], output_path: str | Path, skip: int = 1) -> None:
    """Write a coverage/broadcast curve image for real training artifacts."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    skip = max(1, int(skip or 1))
    steps = _series(metrics.get("episode_timesteps"))[::skip]
    coverages = [value * 100 for value in _series(metrics.get("episode_coverages"))[::skip]]
    broadcasts = [value * 100 for value in _series(metrics.get("episode_broadcasts"))[::skip]]
    eval_history = metrics.get("eval_history") if isinstance(metrics.get("eval_history"), list) else []
    eval_steps = [
        float(item.get("step", 0.0))
        for item in eval_history
        if isinstance(item, dict) and isinstance(item.get("step"), (int, float, np.floating))
    ]
    eval_coverage = [
        float(item.get("avg_coverage", 0.0)) * 100
        for item in eval_history
        if isinstance(item, dict) and isinstance(item.get("avg_coverage"), (int, float, np.floating))
    ]
    eval_broadcast = [
        float(item.get("avg_broadcast", 0.0)) * 100
        for item in eval_history
        if isinstance(item, dict) and isinstance(item.get("avg_broadcast"), (int, float, np.floating))
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if steps and coverages:
        ax.plot(steps[: len(coverages)], coverages, marker="o", label="episode coverage")
    if steps and broadcasts:
        ax.plot(steps[: len(broadcasts)], broadcasts, marker="s", label="episode broadcast")
    if eval_steps and eval_coverage:
        ax.plot(eval_steps[: len(eval_coverage)], eval_coverage, marker="^", label="eval coverage")
    if eval_steps and eval_broadcast:
        ax.plot(eval_steps[: len(eval_broadcast)], eval_broadcast, marker="v", label="eval broadcast")
    if not ax.lines:
        ax.text(0.5, 0.5, "No training metrics", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("step")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    if ax.lines:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


__all__ = ["build_policy", "make_env", "plot_training_metrics"]
