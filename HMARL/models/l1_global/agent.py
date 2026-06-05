"""
L1 全局统筹智能体 — 标准强化学习 Agent 封装。

职责：全局灾情评估、区域优先级（隐含于配额分配）、设备总量调度；
输出 N×5 设备配额矩阵 Q，作为 L2/L3 硬约束上界。

用法：
    cfg = L1Config(n_regions=5)
    agent = L1GlobalAgent(cfg)
    obs = agent.build_observation(raw_state)
    action, log_prob, value, info = agent.act(obs, global_inventory=inv)
    quota = info["quota_matrix"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .actor import L1Actor
from .critic import L1Critic
from .l1_spaces import (
    L1Config,
    L1GlobalState,
    decode_action_to_quota,
    encode_observation,
    project_quota_to_inventory,
    quota_to_dict,
    state_from_dict,
)


class L1GlobalAgent:
    """L1 层完整 RL Agent：观测编码 + Actor/Critic + 动作解码 + 约束投影。"""

    def __init__(
        self,
        config: Optional[L1Config] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.cfg = config or L1Config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = L1Actor(self.cfg).to(self.device)
        self.critic = L1Critic(self.cfg).to(self.device)
        self._last_quota: Optional[np.ndarray] = None
        self._global_inventory: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ 观测
    def build_observation(
        self,
        raw_state: Union[Dict[str, Any], L1GlobalState],
    ) -> np.ndarray:
        if isinstance(raw_state, dict):
            state = state_from_dict(raw_state, self.cfg)
        else:
            state = raw_state
        return encode_observation(state, self.cfg)

    def get_obs_dim(self) -> int:
        return self.cfg.obs_dim

    def get_action_dim(self) -> int:
        return self.cfg.action_dim

    # ------------------------------------------------------------------ 交互
    def reset(self) -> None:
        self._last_quota = None
        self._global_inventory = None

    def act(
        self,
        observation: np.ndarray,
        global_inventory: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        """
        选择动作并解码为配额矩阵。

        Returns:
            action: 连续动作向量 (N*5,)
            log_prob: 标量
            value: 状态价值标量
            info: 含 quota_matrix、quota_dict、global_inventory
        """
        self._global_inventory = np.asarray(global_inventory, dtype=np.float32).reshape(5)

        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            value = self.critic(obs_t).item()
            if deterministic:
                action_t = self.actor.deterministic_action(obs_t)
                dist = self.actor.get_distribution(obs_t)
                log_prob = dist.log_prob(action_t).sum().item()
            else:
                action_t, log_prob_t = self.actor.sample(obs_t)
                log_prob = log_prob_t.item()

        action = action_t.squeeze(0).cpu().numpy()
        Q = decode_action_to_quota(action, self._global_inventory, self.cfg)
        Q = project_quota_to_inventory(Q, self._global_inventory)
        self._last_quota = Q

        info = {
            "quota_matrix": Q,
            "quota_dict": quota_to_dict(Q),
            "global_inventory": self._global_inventory.copy(),
            "n_regions": self.cfg.n_regions,
            "n_device_types": self.cfg.n_device_types,
        }
        return action, log_prob, value, info

    def evaluate(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO 更新用：log_prob, entropy, value（batch 维）。"""
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            act_t = act_t.unsqueeze(0)

        log_prob, entropy = self.actor.evaluate(obs_t, act_t)
        value = self.critic(obs_t)
        return log_prob, entropy, value

    # ------------------------------------------------------------------ 解码与约束
    def decode_action(
        self,
        action: np.ndarray,
        global_inventory: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        inv = global_inventory if global_inventory is not None else self._global_inventory
        if inv is None:
            raise ValueError("global_inventory 未设置，请先 act() 或传入 global_inventory")
        Q = decode_action_to_quota(action, inv, self.cfg)
        return project_quota_to_inventory(Q, inv)

    def get_quota_matrix(self) -> Optional[np.ndarray]:
        """最近一次 act() 产生的 N×5 配额矩阵。"""
        return self._last_quota

    def get_quota_for_region(self, region_id: int) -> Dict[str, int]:
        """返回某区域 5 类设备配额上限，供 L2/L3 约束检查。"""
        if self._last_quota is None:
            raise RuntimeError("尚未执行 act()，无配额矩阵")
        from .l1_spaces import DEVICE_NAMES

        row = self._last_quota[int(region_id)]
        return {DEVICE_NAMES[j]: int(row[j]) for j in range(self.cfg.n_device_types)}

    def check_l2_l3_feasible(
        self,
        region_id: int,
        deployed_counts: np.ndarray,
    ) -> Tuple[bool, str]:
        """
        检查 L2/L3 部署是否违反 L1 硬约束。

        deployed_counts: shape (5,) 该区域已部署/拟部署数量
        """
        cap = self.get_quota_for_region(region_id)
        from .l1_spaces import DEVICE_NAMES

        deployed = np.asarray(deployed_counts, dtype=np.int32).reshape(5)
        for j, name in enumerate(DEVICE_NAMES):
            if deployed[j] > cap[name]:
                return False, f"区域{region_id} {name}: 部署{deployed[j]} > 配额{cap[name]}"
        return True, "ok"

    # ------------------------------------------------------------------ 持久化
    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / "actor.pt")
        torch.save(self.critic.state_dict(), directory / "critic.pt")
        torch.save(
            {
                "n_regions": self.cfg.n_regions,
                "hidden_dims": self.cfg.hidden_dims,
                "log_std_init": self.cfg.log_std_init,
            },
            directory / "config.pt",
        )

    def load(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        self.actor.load_state_dict(
            torch.load(directory / "actor.pt", map_location=self.device, weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(directory / "critic.pt", map_location=self.device, weights_only=True)
        )

    def count_parameters(self) -> Dict[str, int]:
        actor_n = sum(p.numel() for p in self.actor.parameters())
        critic_n = sum(p.numel() for p in self.critic.parameters())
        return {
            "actor": actor_n,
            "critic": critic_n,
            "total": actor_n + critic_n,
        }

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
