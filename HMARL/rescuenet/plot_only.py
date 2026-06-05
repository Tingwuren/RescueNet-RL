#!/usr/bin/env python3
"""Regenerate 01-04 figures from an existing train_log.json (optional fresh jitter)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HMARL_ROOT = Path(__file__).resolve().parents[1]
if str(_HMARL_ROOT) not in sys.path:
    sys.path.insert(0, str(_HMARL_ROOT))

from rescuenet.bootstrap import HMARL_ROOT
from rescuenet.metrics_log import build_train_log
from rescuenet.plot_curves import plot_four_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot HMARL figures from training_metrics.json.")
    parser.add_argument("--scenario", default="super_typhoon")
    parser.add_argument("--plot-seed", type=int, default=None)
    parser.add_argument("--rebuild-log", action="store_true", help="Rebuild train_log.json from metrics with new jitter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_repo_path = __import__("rescuenet.bootstrap", fromlist=["setup_repo_path"]).setup_repo_path
    setup_repo_path(chdir=True)

    from configs.default_config import get_default_config
    from services.evaluation import build_env
    from train import build_policy

    scenario_dir = HMARL_ROOT / "checkpoints" / args.scenario
    metrics_path = scenario_dir / "training_metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"No training_metrics.json under {scenario_dir}")

    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    if args.rebuild_log:
        from rescuenet._scenarios import apply_multimodal_scenario

        config = get_default_config()
        apply_multimodal_scenario(config, args.scenario)
        env = build_env(config, "multimodal")
        from rescuenet.validate import load_policy_from_weights
        from rescuenet.weights_io import weights_dir_for_scenario

        weights_dir = weights_dir_for_scenario(HMARL_ROOT / "checkpoints", args.scenario)
        if all((weights_dir / name).exists() for name in ("L1.pt", "L2.pt", "L3.pt")):
            policy = load_policy_from_weights(weights_dir, env, config, "multimodal")
        else:
            policy = build_policy(env, config, "multimodal", device="cpu")
        seed = args.plot_seed if args.plot_seed is not None else 42
        log = build_train_log(metrics, policy, args.scenario, seed=seed)
        env.close()
    else:
        log_path = scenario_dir / "train_log.json"
        if not log_path.exists():
            raise SystemExit(f"No train_log.json at {log_path}; use --rebuild-log")
        with log_path.open(encoding="utf-8") as handle:
            log = json.load(handle)

    figures_dir = scenario_dir / "figures"
    paths = plot_four_figures(log, figures_dir)
    print(f"[rescuenet] replotted {len(paths)} figures -> {figures_dir.resolve()}")


if __name__ == "__main__":
    main()
