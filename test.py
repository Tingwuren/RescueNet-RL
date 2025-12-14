"""Utility script to report policy parameter counts for different environments."""

from __future__ import annotations

import argparse

from configs.default_config import get_default_config
from envs import DisasterCellularEnv, MultiModalCommEnv
from models.policy_network import MLPActorCritic
from models.multimodal_policy import MultimodalPolicy


def build_env(config, env_type: str):
    if env_type == "multimodal":
        return MultiModalCommEnv(**config["multimodal_env"])
    return DisasterCellularEnv(**config["env"])


def build_policy(env, model_config, env_type: str):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    if env_type == "multimodal":
        return MultimodalPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=model_config.get("multimodal_hidden_sizes", [1024, 1024, 512, 512]),
            device="cpu",
        )
    return MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=model_config.get("hidden_sizes", [128, 128]),
        device="cpu",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report parameter count of the configured RL policy.")
    parser.add_argument(
        "--env-type",
        type=str,
        choices=["baseline", "multimodal"],
        default=None,
        help="Select which environment/policy to instantiate.",
    )
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Override scenario when env-type=multimodal.",
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

    env = build_env(config, env_type)
    policy = build_policy(env, config["model"], env_type)
    param_count = sum(p.numel() for p in policy.parameters())
    print(f"Environment type: {env_type}")
    if env_type == "multimodal":
        print(f"Scenario: {config['multimodal_env']['scenario_name']}")
    print(f"Policy parameter count: {param_count:,}")


if __name__ == "__main__":
    main()
