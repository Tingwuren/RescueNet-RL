"""Train and test the real level-4 benchmark for all supported algorithms."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algos.a3c import A3CTrainer
from algos.dqn import DQNTrainer
from algos.hmarl import HMARLTrainer
from algos.mppo import MPPOTrainer
from algos.ppo import PPOTrainer
from configs.default_config import (
    apply_evaluation_protocol,
    apply_level4_algorithm_profile,
    get_default_config,
)
from planning.broadcast_architecture import export_architecture
from services.evaluation import build_env, evaluate_policy, load_policy
from train import build_policy, make_env, plot_training_metrics


SCENARIOS = [
    "extreme_rainstorm__level_4",
    "super_typhoon__level_4",
    "destructive_earthquake__level_4",
]
ALGORITHMS = ["ppo", "dqn", "a3c", "mppo", "hmarl"]
TRAINERS = {
    "ppo": PPOTrainer,
    "dqn": DQNTrainer,
    "a3c": A3CTrainer,
    "mppo": MPPOTrainer,
    "hmarl": HMARLTrainer,
}


def _configure_run(
    *,
    algorithm: str,
    scenario_name: str,
    timesteps: int,
    rollout_steps: int,
    eval_episodes: int,
    artifact_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    config = get_default_config()
    config["experiment"]["env_type"] = "multimodal"
    config["experiment"]["algorithm"] = algorithm
    config["multimodal_env"]["scenario_name"] = scenario_name
    apply_evaluation_protocol(config, "standard")
    apply_level4_algorithm_profile(config, algorithm)

    config["train"]["total_timesteps"] = timesteps
    config["train"]["rollout_steps"] = min(rollout_steps, timesteps)
    config["train"]["eval_episodes"] = eval_episodes
    config["train"]["eval_deterministic"] = True
    config["train"]["log_interval"] = max(1, rollout_steps if algorithm == "dqn" else 1)
    config["train"]["eval_interval"] = max(1, rollout_steps if algorithm == "dqn" else 1)
    config["logging"]["artifact_dir"] = str(artifact_dir)

    for key in ("ppo", "a3c", "mppo", "hmarl"):
        if key in config:
            config[key]["mini_batch_size"] = min(64, max(16, rollout_steps))
            config[key]["update_epochs"] = min(3, int(config[key].get("update_epochs", 3)))
    config["dqn"]["batch_size"] = min(64, max(16, rollout_steps))
    config["dqn"]["epsilon_decay_steps"] = max(1, timesteps)
    if algorithm == "dqn":
        config["model"]["multimodal_hidden_sizes"] = [256, 128]

    hmarl_warmup = max(1, timesteps)
    config["hmarl"]["train_eval_planner_warmup_steps"] = hmarl_warmup
    config["hmarl"]["train_eval_planner_warmup_power"] = 4.0
    config["hmarl"]["prior_warmup_steps"] = hmarl_warmup
    config["hmarl"]["reward_shaping_warmup_steps"] = hmarl_warmup
    return config


def _policy_path(artifact_dir: Path, algorithm: str) -> Path:
    return artifact_dir / ("dqn_policy.pt" if algorithm == "dqn" else f"{algorithm}_policy.pt")


def run_one(
    *,
    algorithm: str,
    scenario_name: str,
    output_root: Path,
    timesteps: int,
    rollout_steps: int,
    eval_episodes: int,
    test_episodes: int,
) -> Dict[str, Any]:
    run_dir = output_root / "runs" / f"{algorithm}_{scenario_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = _configure_run(
        algorithm=algorithm,
        scenario_name=scenario_name,
        timesteps=timesteps,
        rollout_steps=rollout_steps,
        eval_episodes=eval_episodes,
        artifact_dir=run_dir,
    )

    torch.manual_seed(int(config["train"].get("seed", 123)))
    np.random.seed(int(config["train"].get("seed", 123)))

    env = make_env(config, "multimodal")
    eval_env = make_env(config, "multimodal")
    policy = build_policy(env, config, env_type="multimodal", device=config["train"].get("device", "auto"))
    trainer = TRAINERS[algorithm](env=env, eval_env=eval_env, policy=policy, config=config)

    started = time.time()
    try:
        metrics = trainer.train()
    finally:
        env.close()
        eval_env.close()

    plot_training_metrics(metrics, run_dir / "training_coverage_curve.png", skip=1)
    export_architecture(config["multimodal_env"]["dataset_path"], scenario_name, run_dir / f"broadcast_architecture_{scenario_name}.json")

    test_env = build_env(config, "multimodal")
    try:
        loaded_policy = load_policy(_policy_path(run_dir, algorithm), test_env, config, "multimodal", algorithm=algorithm)
        rewards, coverages, reports = evaluate_policy(
            env=test_env,
            policy=loaded_policy,
            episodes=test_episodes,
            deterministic=True,
            dqn_use_lookahead=bool(config.get("evaluation", {}).get("dqn_use_lookahead", True)),
        )
    finally:
        test_env.close()

    final_states = [report.get("final_state", {}) for report in reports]
    broadcasts = [float(state.get("broadcast_ratio", 0.0)) for state in final_states]
    steps = [int(report.get("steps_taken", 0)) for report in reports]
    result = {
        "algorithm": algorithm,
        "scenario_name": scenario_name,
        "evaluation_protocol": config.get("evaluation", {}).get("protocol", "standard"),
        "algorithm_profile": config.get("evaluation", {}).get("algorithm_profile"),
        "level4_benchmark": bool(config.get("evaluation", {}).get("level4_benchmark", False)),
        "real_training": True,
        "demo_synthetic": False,
        "total_timesteps": timesteps,
        "test_episodes": test_episodes,
        "coverage_rate": float(np.mean(coverages)) if coverages else 0.0,
        "broadcast_rate": float(np.mean(broadcasts)) if broadcasts else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifact_dir": str(run_dir),
        "policy_path": str(_policy_path(run_dir, algorithm)),
    }
    (run_dir / "test_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="artifacts/real_level4_benchmark")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--dqn-timesteps", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--test-episodes", type=int, default=1)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for scenario_name in SCENARIOS:
        for algorithm in ALGORITHMS:
            timesteps = args.dqn_timesteps if algorithm == "dqn" else args.timesteps
            print(f"[real-level4] train/test algorithm={algorithm} scenario={scenario_name}", flush=True)
            row = run_one(
                algorithm=algorithm,
                scenario_name=scenario_name,
                output_root=output_root,
                timesteps=max(1, timesteps),
                rollout_steps=max(1, args.rollout_steps),
                eval_episodes=max(1, args.eval_episodes),
                test_episodes=max(1, args.test_episodes),
            )
            rows.append(row)
            print(
                "[real-level4] result "
                f"coverage={row['coverage_rate']:.2%} broadcast={row['broadcast_rate']:.2%} "
                f"profile={row['algorithm_profile']}",
                flush=True,
            )

    summary_json = output_root / "real_level4_training_test_summary.json"
    summary_csv = output_root / "real_level4_training_test_summary.csv"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"[real-level4] summary_json={summary_json}", flush=True)
    print(f"[real-level4] summary_csv={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
