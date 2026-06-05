"""
L3 用户配置智能体 — 标准 PPO RL Agent。

职责：在 L1 配额与 L2 迁移/链路约束下，输出 72 维部署动作并生成组网拓扑。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .actor import L3Actor
from .critic import L3Critic
from .l3_spaces import (
    L3Config,
    L3SubRegionState,
    L3UpperConstraints,
    check_constraints,
    decode_action,
    encode_observation,
    subregion_from_dict,
)
from .topology import build_topology_graph


class L3LocalAgent:
    """单子区域 L3 执行智能体。"""

    def __init__(
        self,
        subregion_id: int,
        region_id: int = 0,
        config: Optional[L3Config] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.subregion_id = int(subregion_id)
        self.region_id = int(region_id)
        self.cfg = config or L3Config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = L3Actor(self.cfg).to(self.device)
        self.critic = L3Critic(self.cfg).to(self.device)
        self._decoded: Optional[Dict[str, Any]] = None
        self._topology: Optional[Dict[str, Any]] = None
        self._constraints: Optional[L3UpperConstraints] = None

    def get_obs_dim(self) -> int:
        return self.cfg.obs_dim

    def get_action_dim(self) -> int:
        return self.cfg.action_dim

    def reset(self) -> None:
        self._decoded = None
        self._topology = None
        self._constraints = None

    def build_observation(
        self,
        raw_state: Union[Dict[str, Any], L3SubRegionState],
        constraints: L3UpperConstraints,
    ) -> np.ndarray:
        if isinstance(raw_state, dict):
            state = subregion_from_dict(raw_state, self.subregion_id, self.region_id)
        else:
            state = raw_state
        self._constraints = constraints
        return encode_observation(state, constraints, self.cfg)

    def act(
        self,
        observation: np.ndarray,
        constraints: L3UpperConstraints,
        state: Optional[L3SubRegionState] = None,
        l2_links: Optional[List[Dict[str, Any]]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        self._constraints = constraints

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
        decoded = decode_action(action, constraints, self.cfg)
        ok, msg = check_constraints(decoded, constraints)

        grid_centers = state.grid_centers if state is not None else None
        topology = build_topology_graph(
            self.subregion_id,
            self.region_id,
            decoded,
            grid_centers=grid_centers,
            l2_links=l2_links,
        )

        self._decoded = decoded
        self._topology = topology

        info = {
            "subregion_id": self.subregion_id,
            "region_id": self.region_id,
            "decoded": decoded,
            "topology": topology,
            "constraint_ok": ok,
            "constraint_msg": msg,
            "effective_quota": constraints.effective_quota().tolist(),
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

    def get_deployment_matrix(self) -> np.ndarray:
        if self._decoded is None:
            raise RuntimeError("尚未执行 act()")
        return self._decoded["deployment"]

    def get_topology(self) -> Dict[str, Any]:
        if self._topology is None:
            raise RuntimeError("尚未执行 act()")
        return self._topology

    def export_topology(self, path: Union[str, Path]) -> None:
        """导出组网拓扑 JSON，供现场部署或可视化。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_topology(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def constraints_from_upper_layers(
        l1_quota_row: np.ndarray,
        l2_transfer_in: Optional[np.ndarray] = None,
        l2_transfer_out: Optional[np.ndarray] = None,
        l2_link: Optional[Dict[str, Any]] = None,
    ) -> L3UpperConstraints:
        """从 L1 配额行与 L2 指令构造约束。"""
        z5 = np.zeros(5, dtype=np.float32)
        link = l2_link or {}
        return L3UpperConstraints(
            l1_quota=np.asarray(l1_quota_row, dtype=np.float32).reshape(5),
            l2_transfer_in=np.asarray(l2_transfer_in if l2_transfer_in is not None else z5).reshape(5),
            l2_transfer_out=np.asarray(l2_transfer_out if l2_transfer_out is not None else z5).reshape(5),
            link_active=float(link.get("active", 0.0)),
            link_type=int(link.get("link_type", 0)),
            link_peer_region=int(link.get("peer_region", -1)),
            link_deploy_grid=int(link.get("deploy_grid", 0)),
        )

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), directory / f"l3_actor_s{self.subregion_id}.pt")
        torch.save(self.critic.state_dict(), directory / f"l3_critic_s{self.subregion_id}.pt")

    def load(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        self.actor.load_state_dict(
            torch.load(directory / f"l3_actor_s{self.subregion_id}.pt", map_location=self.device, weights_only=True)
        )
        self.critic.load_state_dict(
            torch.load(directory / f"l3_critic_s{self.subregion_id}.pt", map_location=self.device, weights_only=True)
        )

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
