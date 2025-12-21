"""Evaluation and coverage visualization for trained PPO policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from configs.default_config import get_default_config
from services.evaluation import build_env, evaluate_policy, format_episode_report, load_policy


def parse_args() -> argparse.Namespace:
    """CLI parser for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy.")
    parser.add_argument("--checkpoint", type=str, default="artifacts/ppo_policy.pt", help="Path to policy checkpoint.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Print per-step render info.")
    parser.add_argument(
        "--env-type",
        type=str,
        choices=["baseline", "multimodal"],
        default=None,
        help="Select evaluation environment variant.",
    )
    parser.add_argument("--scenario-name", type=str, default=None, help="Scenario to load when env-type=multimodal.")
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Sample from the policy during evaluation (recommended for multimodal envs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config()
    if args.env_type:
        config["experiment"]["env_type"] = args.env_type
    env_type = config["experiment"].get("env_type", "baseline")
    if args.scenario_name:
        config["multimodal_env"]["scenario_name"] = args.scenario_name

    checkpoint_path = Path(args.checkpoint)
    artifact_dir = checkpoint_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(config, env_type)
    policy = load_policy(checkpoint_path, env, config, env_type)

    deterministic = not args.stochastic_eval
    rewards, coverages, reports = evaluate_policy(
        env,
        policy,
        args.episodes,
        deterministic=deterministic,
        render=args.render,
    )
    for report in reports:
        print(format_episode_report(report))
    print(
        f"Evaluation complete over {args.episodes} episodes | "
        f"avg_reward={np.mean(rewards):.2f} | avg_final_coverage={np.mean(coverages):.2%}"
    )


if __name__ == "__main__":
    main()
