"""Generate level-4 demo training/test artifacts for the training history UI.

The generated runs are explicit demo/synthetic fixtures. They are useful for UI
review and report walkthroughs where deterministic convergence curves are
needed, but they should not be treated as measured research results.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.default_config import apply_evaluation_protocol, get_default_config
from train import build_policy, make_env


SCENARIOS = [
    {
        "name": "extreme_rainstorm__level_4",
        "label": "暴雨",
        "disaster_type": "rainstorm",
        "protocol": "standard",
    },
    {
        "name": "super_typhoon__level_4",
        "label": "台风",
        "disaster_type": "typhoon",
        "protocol": "standard",
    },
    {
        "name": "destructive_earthquake__level_4",
        "label": "地震",
        "disaster_type": "earthquake",
        "protocol": "earthquake_stress",
    },
]

ALGORITHMS = [
    {"key": "ppo", "label": "PPO"},
    {"key": "dqn", "label": "DQN"},
    {"key": "a3c", "label": "A3C"},
    {"key": "mppo", "label": "MPPO"},
    {"key": "hmarl", "label": "HMARL"},
]

FINAL_TARGETS = {
    "extreme_rainstorm__level_4": {
        "ppo": (0.712, 0.682),
        "dqn": (0.643, 0.616),
        "a3c": (0.735, 0.704),
        "mppo": (0.785, 0.758),
        "hmarl": (0.986, 0.982),
    },
    "super_typhoon__level_4": {
        "ppo": (0.694, 0.661),
        "dqn": (0.628, 0.604),
        "a3c": (0.724, 0.701),
        "mppo": (0.772, 0.742),
        "hmarl": (0.982, 0.976),
    },
    "destructive_earthquake__level_4": {
        "ppo": (0.681, 0.644),
        "dqn": (0.612, 0.592),
        "a3c": (0.708, 0.676),
        "mppo": (0.758, 0.731),
        "hmarl": (0.971, 0.966),
    },
}


def stable_unit(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def convergence_curve(start: float, final: float, count: int, curvature: float) -> List[float]:
    values: List[float] = []
    max_progress = 1.0 - math.exp(-curvature)
    previous = start
    for index in range(count):
        t = index / max(1, count - 1)
        progress = (1.0 - math.exp(-curvature * t)) / max_progress
        value = start + (final - start) * progress
        value = max(previous, min(final, value))
        values.append(round(value, 6))
        previous = value
    values[-1] = round(final, 6)
    return values


def build_config(scenario_name: str, algorithm: str, protocol: str, artifact_dir: Path) -> Dict:
    config = get_default_config()
    config["experiment"]["env_type"] = "multimodal"
    config["experiment"]["algorithm"] = algorithm
    config["multimodal_env"]["scenario_name"] = scenario_name
    config["multimodal_env"]["reward_mode"] = "coverage_priority" if algorithm == "hmarl" else "coverage_balance"
    config["train"]["total_timesteps"] = 12000
    config["train"]["eval_interval"] = 1000
    config["train"]["eval_episodes"] = 5
    config["train"]["device"] = "cpu"
    config["logging"]["artifact_dir"] = str(artifact_dir)
    config["model"]["multimodal_hidden_sizes"] = [768, 256]
    config["model"]["hmarl_hidden_sizes"] = [256, 128]
    config["mppo"]["head_keys"] = ["default"]
    config["mppo"]["default_head_key"] = "default"
    apply_evaluation_protocol(config, protocol)
    return config


def write_policy_checkpoint(config: Dict, algorithm: str, output_path: Path) -> None:
    env = make_env(config, "multimodal")
    try:
        torch.manual_seed(7)
        policy = build_policy(env, config, env_type="multimodal", device="cpu")
        torch.save(policy.state_dict(), output_path)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def plot_curves(eval_history: List[Dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = [item["step"] for item in eval_history]
    coverage = [item["avg_coverage"] * 100.0 for item in eval_history]
    broadcast = [item["avg_broadcast"] * 100.0 for item in eval_history]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, coverage, marker="o", label="Coverage")
    plt.plot(steps, broadcast, marker="s", label="Broadcast")
    plt.xlabel("Environment Steps")
    plt.ylabel("Rate (%)")
    plt.ylim(0, 105)
    plt.title("Demo Level-4 Training Convergence")
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rows_to_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate() -> None:
    runs_dir = Path("artifacts/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    generated_at = time.time()
    summary_rows: List[Dict[str, object]] = []

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        for algorithm in ALGORITHMS:
            algo_key = algorithm["key"]
            run_id = f"demo_level4_{algo_key}_{scenario_name}"
            run_dir = runs_dir / f"{algo_key}_{scenario_name}_demo_level4"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True)

            final_coverage, final_broadcast = FINAL_TARGETS[scenario_name][algo_key]
            start_seed = stable_unit(scenario_name, algo_key, "start")
            coverage_start = 0.118 + start_seed * 0.052
            broadcast_start = 0.096 + start_seed * 0.046
            curvature = 4.0 if algo_key == "hmarl" else 2.45 + stable_unit(algo_key, scenario_name) * 0.55

            eval_steps = [float(step) for step in range(1000, 12001, 1000)]
            eval_coverages = convergence_curve(coverage_start, final_coverage, len(eval_steps), curvature)
            eval_broadcasts = convergence_curve(broadcast_start, final_broadcast, len(eval_steps), curvature * 0.92)
            eval_history = []
            for step, coverage, broadcast in zip(eval_steps, eval_coverages, eval_broadcasts):
                reward = 4.0 + coverage * 8.0 + broadcast * 5.0
                if algo_key == "hmarl":
                    reward += 1.25
                eval_history.append(
                    {
                        "step": step,
                        "avg_reward": round(reward, 6),
                        "avg_coverage": coverage,
                        "avg_broadcast": broadcast,
                    }
                )

            episode_count = 96
            episode_steps = [int((index + 1) * 125) for index in range(episode_count)]
            episode_coverages = convergence_curve(coverage_start * 0.82, final_coverage, episode_count, curvature * 1.08)
            episode_broadcasts = convergence_curve(broadcast_start * 0.84, final_broadcast, episode_count, curvature)
            episode_rewards = [
                round(3.5 + cov * 8.0 + br * 5.0 + (1.1 if algo_key == "hmarl" else 0.0), 6)
                for cov, br in zip(episode_coverages, episode_broadcasts)
            ]

            config = build_config(scenario_name, algo_key, str(scenario["protocol"]), run_dir)
            checkpoint_path = run_dir / f"{algo_key}_policy.pt"
            write_policy_checkpoint(config, algo_key, checkpoint_path)

            metrics = {
                "episode_rewards": episode_rewards,
                "episode_coverages": episode_coverages,
                "episode_broadcasts": episode_broadcasts,
                "episode_timesteps": episode_steps,
                "eval_history": eval_history,
                "config": config,
                "demo_synthetic": True,
                "demo_note": "Deterministic demo artifact for UI/report walkthrough; not a measured training run.",
            }
            write_json(run_dir / "training_metrics.json", metrics)
            write_json(
                run_dir / "policy_meta.json",
                {
                    "algorithm": algo_key,
                    "env_type": "multimodal",
                    "policy_path": str(checkpoint_path),
                    "evaluation_protocol": scenario["protocol"],
                    "demo_synthetic": True,
                    "demo_note": "Deterministic demo artifact for UI/report walkthrough; not a measured training run.",
                    "config": config.get(algo_key, {}),
                },
            )

            test_coverage = round(max(0.0, final_coverage - 0.004 + stable_unit(scenario_name, algo_key, "test-c") * 0.003), 6)
            test_broadcast = round(max(0.0, final_broadcast - 0.005 + stable_unit(scenario_name, algo_key, "test-b") * 0.003), 6)
            test_reward = round(4.5 + test_coverage * 8.5 + test_broadcast * 5.2 + (1.4 if algo_key == "hmarl" else 0.0), 6)
            test_results = {
                "scenario_name": scenario_name,
                "scenario_label": scenario["label"],
                "disaster_type": scenario["disaster_type"],
                "severity_level": "level_4",
                "algorithm": algo_key,
                "coverage_rate": test_coverage,
                "broadcast_rate": test_broadcast,
                "avg_reward": test_reward,
                "episodes": 5,
                "status": "completed",
                "demo_synthetic": True,
                "note": "Deterministic demo test summary; non-HMARL algorithms intentionally remain below 80%.",
            }
            write_json(run_dir / "test_results.json", test_results)
            write_json(
                run_dir / f"broadcast_architecture_{scenario_name}.json",
                {
                    "scenario": scenario_name,
                    "algorithm": algo_key,
                    "severity_level": "level_4",
                    "demo_synthetic": True,
                    "final_training_coverage": final_coverage,
                    "final_training_broadcast": final_broadcast,
                    "test_results": test_results,
                },
            )
            plot_curves(eval_history, run_dir / "training_coverage_curve.png")

            summary_rows.append(
                {
                    "scenario": scenario["label"],
                    "scenario_name": scenario_name,
                    "severity_level": "level_4",
                    "algorithm": algo_key.upper(),
                    "final_training_coverage": final_coverage,
                    "final_training_broadcast": final_broadcast,
                    "test_coverage": test_coverage,
                    "test_broadcast": test_broadcast,
                    "avg_reward": test_reward,
                    "run_dir": str(run_dir),
                }
            )

    summary = {
        "generated_at": generated_at,
        "demo_synthetic": True,
        "note": "Level-4 deterministic demo artifacts. HMARL is configured as the best performer; other algorithms are below 80% and varied.",
        "runs": summary_rows,
    }
    write_json(Path("artifacts/demo_level4_training_test_summary.json"), summary)
    rows_to_csv(Path("artifacts/demo_level4_training_test_summary.csv"), summary_rows)


if __name__ == "__main__":
    generate()
