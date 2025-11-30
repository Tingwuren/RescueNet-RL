"""Evaluation and coverage visualization for trained PPO policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from configs.default_config import get_default_config
from envs import DisasterCellularEnv
from models.policy_network import MLPActorCritic


def load_policy(checkpoint: Path, env: DisasterCellularEnv, config: Dict[str, Dict]) -> MLPActorCritic:
    """Rebuild the policy network and load a saved checkpoint."""
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
    env: DisasterCellularEnv,
    policy: MLPActorCritic,
    episodes: int,
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
            action, _, _ = policy.act(obs, deterministic=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_default_config()

    checkpoint_path = Path(args.checkpoint)
    artifact_dir = checkpoint_path.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)

    env = DisasterCellularEnv(**config["env"])
    policy = load_policy(checkpoint_path, env, config)

    rewards, coverages = evaluate_policy(env, policy, args.episodes, render=args.render)
    print(
        f"Evaluation complete over {args.episodes} episodes | "
        f"avg_reward={np.mean(rewards):.2f} | avg_final_coverage={np.mean(coverages):.2%}"
    )


if __name__ == "__main__":
    main()
