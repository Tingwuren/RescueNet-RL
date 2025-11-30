"""PPO training entry-point for RescueNet-RL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from algos.ppo import PPOTrainer
from configs.default_config import get_default_config
from envs import DisasterCellularEnv
from models.policy_network import MLPActorCritic


def make_env(env_config: Dict[str, float]) -> DisasterCellularEnv:
    """Factory helper to keep train/eval envs in sync."""
    return DisasterCellularEnv(**env_config)


def build_policy(env: DisasterCellularEnv, model_config: Dict[str, object], device: str) -> MLPActorCritic:
    """Instantiate the actor-critic network that matches the environment."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
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
    parser.add_argument("--artifact-dir", type=str, default=None, help="Output folder for checkpoints.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or auto.")
    parser.add_argument("--log-interval", type=int, default=None, help="How many PPO updates between logging.")
    parser.add_argument("--eval-interval", type=int, default=None, help="How many updates between eval runs.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Episodes per eval run.")
    parser.add_argument(
        "--log-episodes",
        action="store_true",
        help="Print episode-level summaries (including env.render output) during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config()

    if args.total_timesteps:
        config["train"]["total_timesteps"] = args.total_timesteps
    if args.rollout_steps:
        config["train"]["rollout_steps"] = args.rollout_steps
    if args.artifact_dir:
        config["logging"]["artifact_dir"] = args.artifact_dir
    if args.device:
        config["train"]["device"] = args.device
    if args.log_interval:
        config["train"]["log_interval"] = max(1, args.log_interval)
    if args.eval_interval:
        config["train"]["eval_interval"] = max(1, args.eval_interval)
    if args.eval_episodes:
        config["train"]["eval_episodes"] = max(1, args.eval_episodes)
    if args.log_episodes:
        config["train"]["log_episodes"] = True

    artifact_dir = Path(config["logging"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(config["env"])
    eval_env = make_env(config["env"])

    device = config["train"].get("device", "auto")
    policy = build_policy(env, config["model"], device=device)

    torch.manual_seed(config["train"]["seed"])

    trainer = PPOTrainer(env=env, eval_env=eval_env, policy=policy, config=config)
    trainer.train()


if __name__ == "__main__":
    main()
