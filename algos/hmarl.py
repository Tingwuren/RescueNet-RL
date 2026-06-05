"""Hierarchical multi-agent PPO trainer."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from algos.ppo import PPOTrainer
from planning.hierarchical_marl import HierarchicalMARLPlanner


class HMARLTrainer(PPOTrainer):
    """PPO trainer with L1/L2 auxiliary supervision and L3 action priors."""

    def __init__(
        self,
        env,
        eval_env,
        policy,
        config: Dict[str, Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        config.setdefault("experiment", {})
        config["experiment"]["algorithm"] = "hmarl"
        config.setdefault("hmarl", {})
        super().__init__(env=env, eval_env=eval_env, policy=policy, config=config, progress_callback=progress_callback)
        self.hmarl_cfg = config.get("hmarl", {})
        self.planner = HierarchicalMARLPlanner(self.hmarl_cfg)
        self.eval_planner = HierarchicalMARLPlanner(self.hmarl_cfg)
        self.reward_shaping_weight = float(self.hmarl_cfg.get("reward_shaping_weight", 0.12))
        self.aux_loss_coef = float(self.hmarl_cfg.get("aux_loss_coef", 0.08))
        self.train_eval_use_planner_action = bool(self.hmarl_cfg.get("train_eval_use_planner_action", False))
        default_eval_warmup = int(self.train_cfg.get("total_timesteps", 8000) * 0.75)
        self.train_eval_planner_warmup_steps = max(
            0,
            int(self.hmarl_cfg.get("train_eval_planner_warmup_steps", default_eval_warmup)),
        )
        self.train_eval_planner_warmup_power = max(
            1.0,
            float(self.hmarl_cfg.get("train_eval_planner_warmup_power", 1.0)),
        )
        self.prior_warmup_steps = max(0, int(self.hmarl_cfg.get("prior_warmup_steps", 12000)))
        self.min_prior_scale = float(self.hmarl_cfg.get("min_prior_scale", 0.0))
        self.max_prior_scale = float(self.hmarl_cfg.get("max_prior_scale", 1.0))
        self.prior_warmup_power = max(1.0, float(self.hmarl_cfg.get("prior_warmup_power", 4.0)))
        self.reward_shaping_warmup_steps = max(0, int(self.hmarl_cfg.get("reward_shaping_warmup_steps", 12000)))
        self.reward_shaping_warmup_power = max(1.0, float(self.hmarl_cfg.get("reward_shaping_warmup_power", 2.0)))
        self.last_hierarchy_plan: Dict[str, Any] = {}
        self.step_loss_interval = int(self.train_cfg.get("step_loss_interval", 0) or 0)
        self.env_step_log_interval = int(self.train_cfg.get("env_step_log_interval", 0) or 0)
        self._opt_step = 0

    def _warmup_fraction(self, warmup_steps: int, power: float = 1.0) -> float:
        if warmup_steps <= 0:
            return 1.0
        progress = float(np.clip(self.global_step / max(1, warmup_steps), 0.0, 1.0))
        return progress ** max(1.0, power)

    def _current_prior_scale(self) -> float:
        progress = self._warmup_fraction(self.prior_warmup_steps, self.prior_warmup_power)
        return self.min_prior_scale + (self.max_prior_scale - self.min_prior_scale) * progress

    def _scale_action_prior(self, prior: np.ndarray) -> np.ndarray:
        return (prior * self._current_prior_scale()).astype(np.float32)

    def _current_reward_shaping_weight(self) -> float:
        return self.reward_shaping_weight * self._warmup_fraction(
            self.reward_shaping_warmup_steps,
            self.reward_shaping_warmup_power,
        )

    def _collect_rollout(
        self, start_obs: np.ndarray, steps: int
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        obs_list, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        action_priors, l1_targets, l2_targets = [], [], []
        obs = start_obs

        for _ in range(steps):
            prior, plan = self.planner.build_action_prior(self.env)
            scaled_prior = self._scale_action_prior(prior)
            action, log_prob, value = self.policy.act(obs, action_prior=scaled_prior)
            next_obs, env_reward, terminated, truncated, info = self.env.step(action)
            done = bool(terminated or truncated)
            hierarchy_reward = float(plan.get("rewards", {}).get("l3_final", 0.0))
            reward = float(env_reward) + self._current_reward_shaping_weight() * hierarchy_reward

            obs_list.append(obs)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            action_priors.append(scaled_prior.astype(np.float32))
            l1_targets.append(int(plan.get("l1_target", 0)))
            l2_targets.append(int(plan.get("l2_target", 0)))

            self.global_step += 1
            self.current_episode_return += reward
            self.current_episode_length += 1
            self.last_hierarchy_plan = plan

            if self.env_step_log_interval > 0 and (
                self.global_step % self.env_step_log_interval == 0
            ):
                coverage = float(info.get("coverage_ratio", 0.0))
                broadcast = float(info.get("broadcast_ratio", 0.0))
                print(
                    f"    [EnvStep {self.global_step}] reward={reward:.3f} | "
                    f"env_reward={float(env_reward):.3f} | "
                    f"coverage={coverage:.2%} | broadcast={broadcast:.2%} | "
                    f"ep_len={self.current_episode_length}",
                    flush=True,
                )

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
                        f"[HMARL Episode {self.completed_episodes}] steps={self.current_episode_length} | "
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
                        "hierarchy": plan.get("summary", {}),
                        "hierarchical_rewards": plan.get("rewards", {}),
                        "reason": info.get("reason", "episode_end"),
                        **self._episode_info_payload(info),
                    },
                )
                self.current_episode_return = 0.0
                self.current_episode_length = 0
                next_obs, _ = self.env.reset()

            obs = next_obs

        last_prior, _ = self.planner.build_action_prior(self.env)
        last_prior = self._scale_action_prior(last_prior)
        last_value = self.policy.get_value(obs)
        batch = {
            "obs": np.asarray(obs_list, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "last_value": last_value,
            "action_priors": np.asarray(action_priors, dtype=np.float32),
            "l1_targets": np.asarray(l1_targets, dtype=np.int64),
            "l2_targets": np.asarray(l2_targets, dtype=np.int64),
            "last_action_prior": last_prior.astype(np.float32),
        }
        return batch, obs

    def _update_policy(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        action_priors = torch.as_tensor(batch["action_priors"], dtype=torch.float32, device=self.device)
        l1_targets = torch.as_tensor(batch["l1_targets"], dtype=torch.int64, device=self.device)
        l2_targets = torch.as_tensor(batch["l2_targets"], dtype=torch.int64, device=self.device)

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
        aux_loss_val = 0.0

        for epoch_idx in range(update_epochs):
            np.random.shuffle(idxs)
            mb_idx = 0
            for start in range(0, num_samples, mini_batch_size):
                end = start + mini_batch_size
                batch_idx = idxs[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_priors = action_priors[batch_idx]

                new_log_probs, entropy, values, l1_logits, l2_logits = self.policy.evaluate_actions_with_prior(
                    batch_obs,
                    batch_actions,
                    action_prior=batch_priors,
                )
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()
                aux_loss = F.cross_entropy(l1_logits, l1_targets[batch_idx]) + F.cross_entropy(
                    l2_logits, l2_targets[batch_idx]
                )
                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy_loss
                    + self.aux_loss_coef * aux_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                self.optimizer.step()

                policy_loss_val = float(policy_loss.item())
                value_loss_val = float(value_loss.item())
                aux_loss_val = float(aux_loss.item())
                self._opt_step += 1
                if self.step_loss_interval > 0 and (self._opt_step % self.step_loss_interval == 0):
                    print(
                        f"    [OptStep {self._opt_step}] env_step={self.global_step} | "
                        f"epoch={epoch_idx + 1}/{update_epochs} mb={mb_idx + 1} | "
                        f"loss_pi={policy_loss_val:.3f} | loss_v={value_loss_val:.3f} | aux={aux_loss_val:.3f}",
                        flush=True,
                    )
                mb_idx += 1

        return {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "aux_loss": aux_loss_val,
        }

    def evaluate(self, episodes: int = 5, deterministic: bool = True) -> Tuple[float, float, float]:
        rewards = []
        coverages = []
        broadcasts = []
        planner_fraction = self._warmup_fraction(
            self.train_eval_planner_warmup_steps,
            self.train_eval_planner_warmup_power,
        )
        for _ in range(episodes):
            obs, _ = self.eval_env.reset()
            done = False
            total_reward = 0.0
            final_cov = 0.0
            final_broadcast = 0.0
            episode_step = 0
            max_eval_steps = max(1, int(getattr(self.eval_env, "max_steps", 1)))
            planner_step_limit = int(round(max_eval_steps * planner_fraction))
            while not done:
                prior, plan = self.eval_planner.build_action_prior(self.eval_env)
                scaled_prior = self._scale_action_prior(prior) * planner_fraction
                if self.train_eval_use_planner_action and episode_step < planner_step_limit:
                    action = int(plan.get("recommended_action", 0))
                else:
                    action, _, _ = self.policy.act(obs, deterministic=deterministic, action_prior=scaled_prior)
                obs, env_reward, terminated, truncated, info = self.eval_env.step(action)
                episode_step += 1
                hierarchy_reward = float(plan.get("rewards", {}).get("l3_final", 0.0))
                total_reward += float(env_reward) + self._current_reward_shaping_weight() * hierarchy_reward
                done = bool(terminated or truncated)
                final_cov = float(info.get("coverage_ratio", final_cov))
                final_broadcast = float(info.get("broadcast_ratio", final_broadcast))
            rewards.append(total_reward)
            coverages.append(final_cov)
            broadcasts.append(final_broadcast)
        return float(np.mean(rewards)), float(np.mean(coverages)), float(np.mean(broadcasts))
