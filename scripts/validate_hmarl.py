"""Validate HMARL checkpoints on a multimodal disaster scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.default_config import get_default_config
from services.evaluation import build_env, evaluate_policy, load_policy


def _latest_hmarl_checkpoint() -> Path:
    candidates: Iterable[Path] = Path("artifacts/runs").glob("hmarl_*/hmarl_policy.pt")
    ordered = sorted((path for path in candidates if path.exists()), key=lambda item: item.stat().st_mtime, reverse=True)
    if ordered:
        return ordered[0]
    return Path("artifacts/hmarl_policy.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an HMARL checkpoint with the production evaluator.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="HMARL checkpoint path.")
    parser.add_argument("--scenario-name", default="typhoon_residual", help="Scenario key from data/scenarios.json.")
    parser.add_argument("--episodes", type=int, default=5, help="Evaluation episodes.")
    parser.add_argument("--min-coverage", type=float, default=0.90, help="Required average final coverage ratio.")
    parser.add_argument("--seed", type=int, default=13, help="Numpy/Torch seed used by evaluation sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or _latest_hmarl_checkpoint()
    if not checkpoint.exists():
        raise SystemExit(f"HMARL checkpoint not found: {checkpoint}")

    np.random.seed(args.seed)
    config = get_default_config()
    config["experiment"]["env_type"] = "multimodal"
    config["experiment"]["algorithm"] = "hmarl"
    config["multimodal_env"]["scenario_name"] = args.scenario_name

    env = build_env(config, "multimodal")
    try:
        policy = load_policy(checkpoint, env, config, "multimodal", algorithm="hmarl")
        rewards, coverages, reports = evaluate_policy(env, policy, args.episodes, deterministic=True)
    finally:
        env.close()

    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    avg_coverage = float(np.mean(coverages)) if coverages else 0.0
    print(f"HMARL checkpoint: {checkpoint}")
    print(f"Scenario: {args.scenario_name} | episodes={args.episodes}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Average final coverage: {avg_coverage:.2%}")
    print("Episode coverages: " + ", ".join(f"{float(value):.2%}" for value in coverages))
    if reports:
        first = reports[0]
        steps = first.get("steps", [])
        print(f"First episode steps: {first.get('steps_taken', len(steps))}")
        if steps:
            hierarchy = steps[0].get("hierarchy", {}).get("summary", {})
            print(f"First HMARL target region: {hierarchy.get('target_region_id', 'n/a')}")

    if avg_coverage < args.min_coverage:
        raise SystemExit(
            f"HMARL validation failed: avg coverage {avg_coverage:.2%} < required {args.min_coverage:.2%}"
        )
    print(f"HMARL validation passed: avg coverage {avg_coverage:.2%} >= {args.min_coverage:.2%}")


if __name__ == "__main__":
    main()
