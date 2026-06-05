"""
L2 区域调控智能体 — 标准强化学习 Agent（单区域）。

多区域时由 L2RegionalMARL 调度，本类负责单个区域的 act / evaluate / 动作解码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .actor import L2Actor
from .critic import L2Critic
from .l2_spaces import (
    L2Config,
    L2RegionState,
    compute_neighbor_message,
    decode_regional_action,
    encode_observation,
    region_state_from_dict,
)


class L2RegionalAgent:
    """单个区域的 L2 RL Agent。"""

    def __init__(
        self,
        region_id: int,
        config: Optional[L2Config] = None,
        device: Optional[Union[str, torch.device]] = None,
        shared_actor: Optional[L2Actor] = None,
        shared_critic: Optional[L2Critic] = None,
    ):
        self.region_id = int(region_id)
        self.cfg = config or L2Config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if shared_actor is not None and shared_critic is not None:
            self.actor = shared_actor
            self.critic = shared_critic
            self._shared = True
        else:
            self.actor = L2Actor(self.cfg).to(self.device)
            self.critic = L2Critic(self.cfg).to(self.device)
            self._shared = False

        self._last_migrations: List[Dict[str, Any]] = []
        self._last_links: List[Dict[str, Any]] = []
        self._state: Optional[L2RegionState] = None

    def build_observation(
        self,
        raw_state: Union[Dict[str, Any], L2RegionState],
        neighbor_messages: Dict[int, np.ndarray],
    ) -> np.ndarray:
        if isinstance(raw_state, dict):
            state = region_state_from_dict(raw_state, self.region_id, self.cfg)
        else:
            state = raw_state
        self._state = state
        return encode_observation(state, neighbor_messages, self.cfg)

    def get_communication_message(self, state: Optional[L2RegionState] = None) -> np.ndarray:
        """对外广播的邻居通信向量。"""
        st = state or self._state
        if st is None:
            raise ValueError("需要先 build_observation 或传入 state")
        return compute_neighbor_message(st)

    def reset(self) -> None:
        self._last_migrations = []
        self._last_links = []
        self._state = None

    def act(
        self,
        observation: np.ndarray,
        state: L2RegionState,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        self._state = state

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
        migrations, links = decode_regional_action(action, self.region_id, state, self.cfg)
        self._last_migrations = migrations
        self._last_links = links

        info = {
            "region_id": self.region_id,
            "migrations": migrations,
            "links": links,
        }
        return action, log_prob, value, info

    def evaluate(
        self,
        observation: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
            act_t = act_t.unsqueeze(0)
        log_prob, entropy = self.actor.evaluate(obs_t, act_t)
        value = self.critic(obs_t)
        return log_prob, entropy, value

    def decode_action(
        self,
        action: np.ndarray,
        state: L2RegionState,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        return decode_regional_action(action, self.region_id, state, self.cfg)

    def save(self, directory: Union[str, Path]) -> None:
        if self._shared:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / f"actor_r{self.region_id}.pt")
        torch.save(self.critic.state_dict(), directory / f"critic_r{self.region_id}.pt")

    def count_parameters(self) -> Dict[str, int]:
        actor_n = sum(p.numel() for p in self.actor.parameters())
        critic_n = sum(p.numel() for p in self.critic.parameters())
        return {"actor": actor_n, "critic": critic_n, "total": actor_n + critic_n}

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()
