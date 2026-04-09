"""Deep Q-Network (DQN) trainer for large discrete action spaces."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from models.dqn_network import DQNNetwork


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    n_steps: int = 1


class ReplayBuffer:
    """Simple replay buffer for off-policy Q-learning."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage: list[Transition] = []
        self.idx = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.idx] = transition
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int) -> Transition:
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        batch = [self.storage[i] for i in indices]
        obs = np.stack([t.obs for t in batch], axis=0)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_obs = np.stack([t.next_obs for t in batch], axis=0)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return Transition(obs=obs, action=actions, reward=rewards, next_obs=next_obs, done=dones)


class DQNTrainer:
    """Lightweight DQN trainer with epsilon-greedy exploration."""

    def __init__(
        self,
        env,
        eval_env,
        policy: DQNNetwork,
        config: Dict[str, Dict[str, Any]],
        progress_callback: Optional[callable] = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env
        self.policy = policy
        self.config = config
        self.algo_key = "dqn"

        self.train_cfg = config["train"]
        self.dqn_cfg = config.get("dqn", config.get("dqa", {}))
        self.log_cfg = config["logging"]

        self.device = policy.device
        self.optimizer = Adam(self.policy.parameters(), lr=self.dqn_cfg["learning_rate"])
        self.target_net = DQNNetwork(
            obs_dim=policy.obs_dim,
            action_dim=policy.action_dim,
            hidden_sizes=policy.hidden_sizes,
            device=policy.device,
        )
        self.policy.hard_update(self.target_net)

        self.replay_buffer = ReplayBuffer(self.dqn_cfg["buffer_size"])
        self.n_step = max(1, int(self.dqn_cfg.get("n_step", 1)))
        self.gamma = float(self.dqn_cfg["gamma"])
        self.nstep_buffer: Deque[Transition] = deque(maxlen=self.n_step)

        self.global_step = 0
        self.completed_episodes = 0
        self.episode_rewards: list[float] = []
        self.episode_coverages: list[float] = []
        self.episode_broadcasts: list[float] = []
        self.episode_timesteps: list[int] = []
        self.eval_history: list[Dict[str, float]] = []
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.log_episodes = bool(self.train_cfg.get("log_episodes", False))
        self.progress_callback = progress_callback
        self.total_timesteps = int(self.train_cfg["total_timesteps"])
        self.eval_deterministic = bool(self.train_cfg.get("eval_deterministic", True))

        self.artifact_dir = Path(self.log_cfg.get("artifact_dir", "artifacts"))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.epsilon_start = float(self.dqn_cfg["epsilon_start"])
        self.epsilon_end = float(self.dqn_cfg["epsilon_end"])
        configured_decay = int(self.dqn_cfg["epsilon_decay_steps"])
        self.epsilon_decay_steps = min(
            configured_decay,
            max(1, int(self.total_timesteps * 0.8)),
        )
        self.dqn_cfg["epsilon_decay_steps"] = self.epsilon_decay_steps
        self.target_update_tau = float(self.dqn_cfg.get("target_update_tau", 0.005))
        self.target_update_period = int(self.dqn_cfg.get("target_update_period", 1000))

    def _emit_progress(self, event_type: str, payload: Dict[str, Any]) -> None:
        if not self.progress_callback:
            return
        try:
            self.progress_callback({"type": event_type, "payload": payload})
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[DQNTrainer] progress callback error: {exc}")

    def _epsilon(self) -> float:
        fraction = min(1.0, self.global_step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def _current_action_mask(self, env) -> Optional[np.ndarray]:
        if hasattr(env, "get_action_mask"):
            return env.get_action_mask()
        return None

    def _action_mask_from_observation(self, obs_batch: np.ndarray) -> Optional[np.ndarray]:
        if hasattr(self.env, "action_mask_from_observation"):
            return self.env.action_mask_from_observation(obs_batch)
        return None

    def _append_nstep(self, transition: Transition) -> Optional[Transition]:
        self.nstep_buffer.append(transition)
        if len(self.nstep_buffer) < self.n_step and not transition.done:
            return None

        reward_sum = 0.0
        discount = 1.0
        next_obs = transition.next_obs
        done_flag = transition.done
        steps_used = 0
        for item in self.nstep_buffer:
            reward_sum += discount * item.reward
            discount *= self.gamma
            next_obs = item.next_obs
            done_flag = item.done
            steps_used += 1
            if item.done:
                break
        first = self.nstep_buffer.popleft()
        return Transition(
            obs=first.obs,
            action=first.action,
            reward=reward_sum,
            next_obs=next_obs,
            done=done_flag,
            n_steps=steps_used,
        )

    def _flush_nstep(self) -> None:
        while self.nstep_buffer:
            reward_sum = 0.0
            discount = 1.0
            next_obs = self.nstep_buffer[0].next_obs
            done_flag = self.nstep_buffer[0].done
            steps_used = 0
            for item in list(self.nstep_buffer):
                reward_sum += discount * item.reward
                discount *= self.gamma
                next_obs = item.next_obs
                done_flag = item.done
                steps_used += 1
                if item.done:
                    break
            first = self.nstep_buffer.popleft()
            self.replay_buffer.add(
                Transition(
                    obs=first.obs,
                    action=first.action,
                    reward=reward_sum,
                    next_obs=next_obs,
                    done=done_flag,
                    n_steps=steps_used,
                )
            )

    def _update_q_network(self) -> Dict[str, float]:
        batch = self.replay_buffer.sample(self.dqn_cfg["batch_size"])
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)
        n_steps = torch.as_tensor(batch.n_steps, dtype=torch.float32, device=self.device)

        q_values: Tensor = self.policy(obs)
        q_action = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_action_mask = self._action_mask_from_observation(batch.next_obs)
            next_policy_q = self.policy(next_obs)
            next_target_all = self.target_net(next_obs)
            if next_action_mask is not None:
                next_mask = torch.as_tensor(next_action_mask, dtype=torch.bool, device=self.device)
                masked_policy_q = next_policy_q.masked_fill(~next_mask, float("-inf"))
                next_actions = masked_policy_q.argmax(dim=1, keepdim=True)
                next_target_q = next_target_all.gather(1, next_actions).squeeze(-1)
                no_valid = ~next_mask.any(dim=1)
                next_target_q = next_target_q.masked_fill(no_valid, 0.0)
            else:
                next_actions = next_policy_q.argmax(dim=1, keepdim=True)
                next_target_q = next_target_all.gather(1, next_actions).squeeze(-1)
            discount = torch.pow(torch.full_like(n_steps, self.gamma), n_steps)
            target = rewards + (1.0 - dones) * discount * next_target_q

        loss = F.smooth_l1_loss(q_action, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()

        if self.global_step % self.target_update_period == 0:
            self.policy.hard_update(self.target_net)
        else:
            self.policy.soft_update(self.target_net, self.target_update_tau)

        return {"q_loss": float(loss.item())}

    def _log_episode(self, info: Dict[str, Any]) -> None:
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
        self._emit_progress(
            "episode",
            {
                "episode": self.completed_episodes,
                "steps": self.current_episode_length,
                "reward": float(self.current_episode_return),
                "coverage": coverage,
                "broadcast": broadcast,
                "reason": info.get("reason", "episode_end"),
            },
        )
        self.current_episode_return = 0.0
        self.current_episode_length = 0

    def train(self) -> Dict[str, Any]:
        total_timesteps: int = self.total_timesteps
        log_interval: int = max(1, self.train_cfg["log_interval"])
        eval_interval: int = max(1, self.train_cfg["eval_interval"])
        eval_episodes: int = self.train_cfg["eval_episodes"]

        obs, _ = self.env.reset(seed=self.train_cfg.get("seed"))
        last_info: Dict[str, Any] = {}
        q_loss_val = 0.0

        while self.global_step < total_timesteps:
            epsilon = self._epsilon()
            action, _ = self.policy.act(
                obs,
                epsilon=epsilon,
                action_mask=self._current_action_mask(self.env),
            )
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)

            t = Transition(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
            ready = self._append_nstep(t)
            if ready:
                self.replay_buffer.add(ready)

            obs = next_obs
            self.global_step += 1
            self.current_episode_return += reward
            self.current_episode_length += 1
            last_info = info

            if done:
                self._flush_nstep()
                self._log_episode(info)
                obs, _ = self.env.reset()

            if len(self.replay_buffer) >= self.dqn_cfg["batch_size"]:
                q_loss_val = self._update_q_network().get("q_loss", q_loss_val)

            if self.global_step % log_interval == 0:
                mean_reward = np.mean(self.episode_rewards[-log_interval:]) if self.episode_rewards else 0.0
                mean_coverage = np.mean(self.episode_coverages[-log_interval:]) if self.episode_coverages else 0.0
                print(
                    f"[DQN] step={self.global_step} | epsilon={epsilon:.3f} | "
                    f"mean_reward={mean_reward:.2f} | mean_coverage={mean_coverage:.2%} | q_loss={q_loss_val:.4f}"
                )
                self._emit_progress(
                    "update",
                    {
                        "step": self.global_step,
                        "epsilon": float(epsilon),
                        "mean_reward": float(mean_reward),
                        "mean_coverage": float(mean_coverage),
                        "q_loss": float(q_loss_val),
                    },
                )

            if self.global_step % eval_interval == 0:
                eval_reward, eval_cov, eval_broadcast = self.evaluate(
                    episodes=eval_episodes,
                    deterministic=self.eval_deterministic,
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

    def evaluate(self, episodes: int = 5, deterministic: bool = True) -> Tuple[float, float, float]:
        rewards = []
        coverages = []
        broadcasts = []
        eval_epsilon = 0.0 if deterministic else self.epsilon_end
        for _ in range(episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            final_cov = 0.0
            final_broadcast = 0.0
            while not done:
                action, _ = self.policy.act(
                    obs,
                    epsilon=eval_epsilon,
                    deterministic=deterministic,
                    action_mask=self._current_action_mask(self.eval_env),
                )
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
                ep_reward += reward
                final_cov = float(info.get("coverage_ratio", final_cov))
                final_broadcast = float(info.get("broadcast_ratio", final_broadcast))
            rewards.append(ep_reward)
            coverages.append(final_cov)
            broadcasts.append(final_broadcast)
        return float(np.mean(rewards)), float(np.mean(coverages)), float(np.mean(broadcasts))

    def _save_artifacts(self, metrics: Dict[str, Any]) -> None:
        policy_path = self.artifact_dir / "dqn_policy.pt"
        metrics_path = self.artifact_dir / "training_metrics.json"
        meta_path = self.artifact_dir / "policy_meta.json"

        torch.save(self.policy.state_dict(), policy_path)
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        with meta_path.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "algorithm": "dqn",
                    "env_type": self.config.get("experiment", {}).get("env_type", "baseline"),
                    "policy_path": str(policy_path),
                    "config": self.config.get("dqn", {}),
                },
                fp,
                indent=2,
            )
        print(f"Artifacts saved to {self.artifact_dir.resolve()}")


# Backward compatibility alias.
DQATrainer = DQNTrainer
