"""Lightweight PPO trainer for the RescueNet-RL setting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from envs import DisasterCellularEnv
from models.policy_network import MLPActorCritic


class PPOTrainer:
    """Self-contained PPO training loop."""

    def __init__(
        self,
        env: DisasterCellularEnv,
        eval_env: DisasterCellularEnv,
        policy: MLPActorCritic,
        config: Dict[str, Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.config = config
        self.algo_key = config.get("experiment", {}).get("algorithm", "ppo")

        self.train_cfg = config["train"]
        self.ppo_cfg = config.get(self.algo_key, config["ppo"])
        self.log_cfg = config["logging"]

        self.device = policy.device
        self.optimizer = Adam(self.policy.parameters(), lr=self.ppo_cfg["learning_rate"])
        self.global_step = 0

        self.episode_rewards: list[float] = []
        self.episode_coverages: list[float] = []
        self.episode_broadcasts: list[float] = []
        self.episode_timesteps: list[int] = []
        self.eval_history: list[Dict[str, float]] = []

        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.completed_episodes = 0
        self.log_episodes = bool(self.train_cfg.get("log_episodes", False))

        self.artifact_dir = Path(self.log_cfg.get("artifact_dir", "artifacts"))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback

    def train(self) -> Dict[str, Any]:
        """Main PPO training loop."""
        total_timesteps: int = self.train_cfg["total_timesteps"]
        rollout_steps: int = self.train_cfg["rollout_steps"]
        log_interval: int = max(1, self.train_cfg["log_interval"])
        eval_interval: int = max(1, self.train_cfg["eval_interval"])
        eval_episodes: int = self.train_cfg["eval_episodes"]

        obs, _ = self.env.reset(seed=self.train_cfg.get("seed"))
        update_idx = 0

        while self.global_step < total_timesteps:
            steps_remaining = total_timesteps - self.global_step
            batch_steps = min(rollout_steps, steps_remaining)
            batch, obs = self._collect_rollout(obs, batch_steps)
            advantages, returns = self._compute_gae(
                batch["rewards"],
                batch["values"],
                batch["dones"],
                batch["last_value"],
            )
            batch["advantages"] = advantages
            batch["returns"] = returns
            loss_info = self._update_policy(batch)
            update_idx += 1

            if update_idx % log_interval == 0:
                mean_reward = np.mean(self.episode_rewards[-log_interval:]) if self.episode_rewards else 0.0
                mean_coverage = np.mean(self.episode_coverages[-log_interval:]) if self.episode_coverages else 0.0
                mean_broadcast = np.mean(self.episode_broadcasts[-log_interval:]) if self.episode_broadcasts else 0.0
                print(
                    f"[Update {update_idx}] step={self.global_step} | "
                    f"mean_reward={mean_reward:.2f} | mean_coverage={mean_coverage:.2%} | "
                    f"mean_broadcast={mean_broadcast:.2%} | "
                    f"loss_pi={loss_info['policy_loss']:.3f} | loss_v={loss_info['value_loss']:.3f}"
                )
                self._emit_progress(
                    "update",
                    {
                        "update": update_idx,
                        "step": self.global_step,
                        "mean_reward": float(mean_reward),
                        "mean_coverage": float(mean_coverage),
                        "mean_broadcast": float(mean_broadcast),
                        "loss_pi": float(loss_info["policy_loss"]),
                        "loss_v": float(loss_info["value_loss"]),
                        **{
                            key: float(value)
                            for key, value in loss_info.items()
                            if key not in {"policy_loss", "value_loss"}
                        },
                    },
                )

            if update_idx % eval_interval == 0:
                eval_reward, eval_cov, eval_broadcast = self.evaluate(
                    episodes=eval_episodes,
                    deterministic=self.train_cfg.get("eval_deterministic", True),
                )
                self.eval_history.append(
                    {
                        "step": float(self.global_step),
                        "avg_reward": float(eval_reward),
                        "avg_coverage": float(eval_cov),
                        "avg_broadcast": float(eval_broadcast),
                    }
                )
                print(
                    f"    Eval -> avg_reward={eval_reward:.2f} | "
                    f"avg_final_coverage={eval_cov:.2%} | avg_broadcast={eval_broadcast:.2%}"
                )
                self._emit_progress(
                    "evaluation",
                    {
                        "step": self.global_step,
                        "avg_reward": float(eval_reward),
                        "avg_coverage": float(eval_cov),
                        "avg_broadcast": float(eval_broadcast),
                    },
                )

        metrics = {
            "episode_rewards": self.episode_rewards,
            "episode_coverages": self.episode_coverages,
            "episode_broadcasts": self.episode_broadcasts,
            "episode_timesteps": self.episode_timesteps,
            "eval_history": self.eval_history,
            "config": self.config,
        }
        self._save_artifacts(metrics)
        self._emit_progress(
            "completed",
            {
                "step": self.global_step,
                "episodes": self.completed_episodes,
                "total_timesteps": int(self.train_cfg["total_timesteps"]),
            },
        )
        return metrics

    def _emit_progress(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        try:
            enriched_payload = dict(payload)
            enriched_payload.setdefault("step", int(self.global_step))
            enriched_payload.setdefault("global_step", int(self.global_step))
            enriched_payload.setdefault("total_timesteps", int(self.train_cfg["total_timesteps"]))
            enriched_payload.setdefault("algorithm", self.algo_key)
            self.progress_callback({"type": event_type, "payload": enriched_payload})
        except Exception as exc:
            print(f"[PPOTrainer] progress callback error: {exc}")

    def _episode_info_payload(self, info: Dict[str, Any]) -> Dict[str, Any]:
        coverage = float(info.get("coverage_ratio", 0.0))
        total_users = int(getattr(self.env, "num_users", 0) or 0)
        payload: Dict[str, Any] = {
            "total_users": total_users,
            "connected_users": int(round(coverage * total_users)) if total_users > 0 else 0,
            "remaining_budget": info.get("remaining_budget"),
            "avg_user_throughput": info.get("avg_user_throughput"),
            "recent_throughput": info.get("recent_throughput"),
            "device_cost": info.get("device_cost"),
            "bandwidth_cost": info.get("bandwidth_cost"),
            "reward_mode": info.get("reward_mode"),
            "reward_label": info.get("reward_label"),
            "evaluation_protocol": info.get("evaluation_protocol"),
            "has_residual_network": info.get("has_residual_network"),
        }
        reward_breakdown = info.get("reward_breakdown")
        if isinstance(reward_breakdown, dict):
            payload["reward_breakdown"] = {
                key: float(value)
                for key, value in reward_breakdown.items()
                if isinstance(value, (int, float))
            }
        return {key: value for key, value in payload.items() if value is not None}

    def _collect_rollout(
        self, start_obs: np.ndarray, steps: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        obs_list, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        obs = start_obs

        for _ in range(steps):
            action, log_prob, value = self.policy.act(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            obs_list.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)

            self.global_step += 1
            self.current_episode_return += reward
            self.current_episode_length += 1

            if done:
                coverage = float(info.get("coverage_ratio", 0.0))
                broadcast = float(info.get("broadcast_ratio", 0.0))
                self.episode_rewards.append(self.current_episode_return)
                self.episode_coverages.append(coverage)
                self.episode_broadcasts.append(broadcast)
                self.episode_timesteps.append(self.global_step)
                self.completed_episodes += 1
                if self.log_episodes:
                    reason = info.get("reason", "episode_end")
                    print(
                        f"[Episode {self.completed_episodes}] steps={self.current_episode_length} | "
                        f"reward={self.current_episode_return:.2f} | coverage={coverage:.2%} | "
                        f"broadcast={broadcast:.2%} | reason={reason}"
                    )
                    self.env.render()
                self._emit_progress(
                    "episode",
                    {
                        "episode": self.completed_episodes,
                        "steps": self.current_episode_length,
                        "reward": float(self.current_episode_return),
                        "coverage": coverage,
                        "broadcast": broadcast,
                        "reason": info.get("reason", "episode_end"),
                        **self._episode_info_payload(info),
                    },
                )
                self.current_episode_return = 0.0
                self.current_episode_length = 0
                next_obs, _ = self.env.reset()

            obs = next_obs

        last_value = self.policy.get_value(obs)
        batch = {
            "obs": np.asarray(obs_list, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "last_value": last_value,
        }
        return batch, obs

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        gamma = self.ppo_cfg["gamma"]
        gae_lambda = self.ppo_cfg["gae_lambda"]
        advantages = np.zeros_like(rewards)
        last_adv = 0.0

        values_ext = np.append(values, last_value)

        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + gamma * values_ext[step + 1] * mask - values_ext[step]
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            advantages[step] = last_adv

        returns = advantages + values
        return advantages, returns

    def _update_policy(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        clip_coef = self.ppo_cfg["clip_coef"]
        update_epochs = self.ppo_cfg["update_epochs"]
        mini_batch_size = self.ppo_cfg["mini_batch_size"]
        entropy_coef = self.ppo_cfg["entropy_coef"]
        value_coef = self.ppo_cfg["value_coef"]
        max_grad_norm = self.ppo_cfg["max_grad_norm"]

        num_samples = obs.size(0)
        idxs = np.arange(num_samples)

        policy_loss_val = 0.0
        value_loss_val = 0.0

        for _ in range(update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, num_samples, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = idxs[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                new_log_probs, entropy, values = self.policy.evaluate_actions(batch_obs, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                policy_loss_val = float(policy_loss.item())
                value_loss_val = float(value_loss.item())

        return {"policy_loss": policy_loss_val, "value_loss": value_loss_val}

    def evaluate(self, episodes: int = 5, deterministic: bool = True) -> Tuple[float, float, float]:
        """Roll out the current policy deterministically for reporting."""
        rewards = []
        coverages = []
        broadcasts = []
        for _ in range(episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            final_cov = 0.0
            final_broadcast = 0.0
            while not done:
                action, _, _ = self.policy.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
                total_reward += reward
                final_cov = float(info.get("coverage_ratio", final_cov))
                final_broadcast = float(info.get("broadcast_ratio", final_broadcast))
                if done:
                    break
            rewards.append(total_reward)
            coverages.append(final_cov)
            broadcasts.append(final_broadcast)
        return float(np.mean(rewards)), float(np.mean(coverages)), float(np.mean(broadcasts))

    def _save_artifacts(self, metrics: Dict[str, Any]) -> None:
        policy_path = self.artifact_dir / f"{self.algo_key}_policy.pt"
        metrics_path = self.artifact_dir / "training_metrics.json"
        meta_path = self.artifact_dir / "policy_meta.json"

        torch.save(self.policy.state_dict(), policy_path)
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "algorithm": self.algo_key,
                    "env_type": self.config.get("experiment", {}).get("env_type", "baseline"),
                    "policy_path": str(policy_path),
                    "evaluation_protocol": self.config.get("evaluation", {}).get("protocol", "standard"),
                    "algorithm_profile": self.config.get("evaluation", {}).get("algorithm_profile"),
                    "level4_benchmark": self.config.get("evaluation", {}).get("level4_benchmark", False),
                    "scenario_name": self.config.get("multimodal_env", {}).get("scenario_name"),
                    "config": self.config.get(self.algo_key, {}),
                },
                fp,
                indent=2,
            )
        print(f"Artifacts saved to {self.artifact_dir.resolve()}")
