#!/usr/bin/env python3
"""Validate HMARL checkpoint trained via rescuenet bridge (weights/L1|L2|L3.pt)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HMARL_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch

from ._scenarios import apply_multimodal_scenario
from .bootstrap import HMARL_ROOT, setup_repo_path
from .weights_io import merge_layer_weights, weights_dir_for_scenario


def default_weights_dir(scenario_alias: str) -> Path:
    return weights_dir_for_scenario(HMARL_ROOT / "checkpoints", scenario_alias)


def load_policy_from_weights(weights_dir: Path, env, config, env_type: str = "multimodal"):
    setup_repo_path(chdir=False)
    from train import build_policy  # noqa: E402

    device = config.get("train", {}).get("device", "auto")
    policy = build_policy(env, config, env_type=env_type, device=device)
    state_dict = merge_layer_weights(weights_dir)
    state_dict = {
        key: value.to(policy.device) if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }
    policy.load_state_dict(state_dict)
    policy.eval()

    from planning.hierarchical_marl import HierarchicalMARLPlanner

    hmarl_cfg = config.get("hmarl", {})
    policy.hierarchical_planner = HierarchicalMARLPlanner(hmarl_cfg)
    policy.hmarl_eval_use_planner_action = bool(hmarl_cfg.get("eval_use_planner_action", True))
    return policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HMARL layer weights (RescueNet-RL evaluator).")
    parser.add_argument("--scenario", default="super_typhoon", help="HMARL scenario alias.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Directory containing L1.pt, L2.pt, L3.pt (default: checkpoints/<scenario>/weights).",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--eval-protocol", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_repo_path(chdir=True)

    from configs.default_config import apply_evaluation_protocol, get_default_config  # noqa: E402
    from services.evaluation import build_env, evaluate_policy  # noqa: E402

    scenario_alias = args.scenario
    weights_dir = args.weights_dir or default_weights_dir(scenario_alias)
    for name in ("L1.pt", "L2.pt", "L3.pt"):
        if not (weights_dir / name).exists():
            raise SystemExit(
                f"Missing {weights_dir / name}\n"
                f"Train first: python rescuenet/train.py --scenario {scenario_alias}"
            )

    np.random.seed(args.seed)
    config = get_default_config()
    config["experiment"]["env_type"] = "multimodal"
    config["experiment"]["algorithm"] = "hmarl"
    _, scenario_name = apply_multimodal_scenario(config, scenario_alias)
    apply_evaluation_protocol(config, args.eval_protocol)

    env = build_env(config, "multimodal")
    try:
        policy = load_policy_from_weights(weights_dir, env, config, "multimodal")
        rewards, coverages, reports = evaluate_policy(env, policy, args.episodes, deterministic=True)
    finally:
        env.close()

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_coverage = float(np.mean(coverages)) if coverages else 0.0
    print(f"[rescuenet] weights: {weights_dir.resolve()}")
    print(f"[rescuenet] alias={scenario_alias} -> scenario_name={scenario_name} | episodes={args.episodes}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average final coverage: {avg_coverage:.2%}")
    print("Episode coverages: " + ", ".join(f"{float(v):.2%}" for v in coverages))

    if avg_coverage < args.min_coverage:
        raise SystemExit(
            f"Validation failed: avg coverage {avg_coverage:.2%} < required {args.min_coverage:.2%}"
        )
    print(f"Validation passed: {avg_coverage:.2%} >= {args.min_coverage:.2%}")


if __name__ == "__main__":
    main()
