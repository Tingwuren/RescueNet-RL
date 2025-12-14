"""Evaluation and coverage visualization for trained PPO policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from configs.default_config import get_default_config
from envs import DisasterCellularEnv, MultiModalCommEnv
from models.policy_network import MLPActorCritic
from models.multimodal_policy import MultimodalPolicy


def build_env(config: Dict[str, Dict], env_type: str):
    if env_type == "multimodal":
        return MultiModalCommEnv(**config["multimodal_env"])
    return DisasterCellularEnv(**config["env"])


def load_policy(checkpoint: Path, env, config: Dict[str, Dict], env_type: str):
    """Rebuild the policy network and load a saved checkpoint."""
    if env_type == "multimodal":
        policy = MultimodalPolicy(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_sizes=config["model"].get("multimodal_hidden_sizes", [1024, 1024, 512, 512]),
            device=config["train"].get("device", "auto"),
        )
    else:
        policy = MLPActorCritic(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_sizes=config["model"].get("hidden_sizes", [128, 128]),
            device=config["train"].get("device", "auto"),
        )
    state_dict = torch.load(checkpoint, map_location=policy.device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def evaluate_policy(
    env,
    policy,
    episodes: int,
    deterministic: bool = True,
    render: bool = False,
) -> Tuple[List[float], List[float]]:
    """Run deterministic rollouts and return per-episode rewards and coverage."""
    rewards: List[float] = []
    coverages: List[float] = []
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        final_cov = 0.0
        steps = 0
        while not done:
            action, _, _ = policy.act(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            final_cov = float(info.get("coverage_ratio", final_cov))
            steps += 1
            if render:
                print(
                    f"[Eval] Episode {episode} Step {steps} | reward={reward:+.2f} | coverage={final_cov:.2%}"
                )
                env.render()
        rewards.append(ep_reward)
        coverages.append(final_cov)
    return rewards, coverages


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
    rewards, coverages = evaluate_policy(
        env,
        policy,
        args.episodes,
        deterministic=deterministic,
        render=args.render,
    )
    print(
        f"Evaluation complete over {args.episodes} episodes | "
        f"avg_reward={np.mean(rewards):.2f} | avg_final_coverage={np.mean(coverages):.2%}"
    )


if __name__ == "__main__":
    main()
