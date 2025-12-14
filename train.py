"""PPO training entry-point for RescueNet-RL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from algos.ppo import PPOTrainer
from configs.default_config import get_default_config
from envs import DisasterCellularEnv, MultiModalCommEnv
from models.policy_network import MLPActorCritic
from models.multimodal_policy import MultimodalPolicy
from planning.broadcast_architecture import export_architecture


def make_env(config: Dict[str, Dict], env_type: str):
    """Factory helper to keep train/eval envs in sync."""
    if env_type == "multimodal":
        return MultiModalCommEnv(**config["multimodal_env"])
    return DisasterCellularEnv(**config["env"])


def build_policy(env, model_config: Dict[str, object], env_type: str, device: str):
    """Instantiate the actor-critic network that matches the environment."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if env_type == "multimodal":
        return MultimodalPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_config.get("multimodal_hidden_sizes", [1024, 1024, 512, 512]),
            device=device,
        )
    return MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=model_config.get("hidden_sizes", [128, 128]),
        device=device,
    )


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent for disaster cellular recovery.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps.")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Override rollout steps per PPO batch.")
    parser.add_argument("--log-interval", type=int, default=None, help="How many PPO updates between logging.")
    parser.add_argument("--eval-interval", type=int, default=None, help="How many updates between eval runs.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Episodes per eval run.")
    parser.add_argument(
        "--env-type",
        type=str,
        choices=["baseline", "multimodal"],
        default=None,
        help="Select between baseline and multimodal joint environment.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Choose scenario from the dataset when env-type=multimodal.",
    )
    parser.add_argument(
        "--deterministic-eval",
        action="store_true",
        help="Force evaluation rollouts to use greedy actions.",
    )
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Sample actions during evaluation (recommended for multimodal envs).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config()

    if args.total_timesteps:
        config["train"]["total_timesteps"] = args.total_timesteps
    if args.rollout_steps:
        config["train"]["rollout_steps"] = args.rollout_steps
    if args.log_interval:
        config["train"]["log_interval"] = max(1, args.log_interval)
    if args.eval_interval:
        config["train"]["eval_interval"] = max(1, args.eval_interval)
    if args.eval_episodes:
        config["train"]["eval_episodes"] = max(1, args.eval_episodes)
    if args.env_type:
        config["experiment"]["env_type"] = args.env_type
    env_type = config["experiment"].get("env_type", "baseline")
    if args.scenario_name:
        config["multimodal_env"]["scenario_name"] = args.scenario_name
    if args.deterministic_eval:
        config["train"]["eval_deterministic"] = True
    elif args.stochastic_eval:
        config["train"]["eval_deterministic"] = False

    artifact_dir = Path(config["logging"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config, env_type)
    eval_env = make_env(config, env_type)

    device = config["train"].get("device", "auto")
    policy = build_policy(env, config["model"], env_type=env_type, device=device)

    torch.manual_seed(config["train"]["seed"])

    trainer = PPOTrainer(env=env, eval_env=eval_env, policy=policy, config=config)
    metrics = trainer.train()

    coverage_plot_path = artifact_dir / "training_coverage_curve.png"
    plot_training_metrics(metrics, coverage_plot_path, skip=1)

    if env_type == "multimodal":
        scenario_name = config["multimodal_env"]["scenario_name"]
        dataset_path = config["multimodal_env"]["dataset_path"]
        architecture_path = artifact_dir / f"broadcast_architecture_{scenario_name}.json"
        export_architecture(dataset_path, scenario_name, architecture_path)


def plot_training_metrics(metrics: Dict, output_path: Path, skip: int = 1) -> None:
    """Plot coverage improvements across training episodes."""
    eval_history = metrics.get("eval_history", [])
    if eval_history:
        timesteps = [entry["step"] for entry in eval_history]
        coverages = [entry["avg_coverage"] for entry in eval_history]
    else:
        coverages = metrics.get("episode_coverages", [])
        timesteps = metrics.get("episode_timesteps", [])

    if not coverages:
        print("No coverage data found in metrics; skipping plot.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    coverages_arr = np.array(coverages)
    timesteps_arr = np.array(timesteps if timesteps else list(range(len(coverages))))
    skip = max(1, skip)
    coverages_arr = coverages_arr[::skip]
    timesteps_arr = timesteps_arr[::skip]

    plt.figure(figsize=(8, 4))
    plt.plot(timesteps_arr, coverages_arr * 100.0, marker="o", linestyle="-")
    plt.xlabel("Environment Steps")
    plt.ylabel("Episode Final Coverage (%)")
    plt.title("Coverage Improvement During Training")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Training coverage curve saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
