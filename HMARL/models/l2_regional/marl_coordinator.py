"""
L2 多智能体协同层：每区域一个 L2RegionalAgent，参数共享，有限邻居通信。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .actor import L2Actor
from .agent import L2RegionalAgent
from .critic import L2Critic
from .l2_spaces import (
    L2Config,
    L2RegionState,
    apply_migrations_to_quotas,
    compute_neighbor_message,
    merge_links,
    merge_migrations,
    region_state_from_dict,
)


class L2RegionalMARL:
    """
    N 个区域 L2 智能体协同调度。

    - 参数共享：所有区域共用同一 Actor/Critic（同质区域）
    - 通信：每步交换 neighbor_msg_dim 维摘要
    - 输出：全局迁移表、全局链路表、调剂后配额
    """

    def __init__(
        self,
        config: Optional[L2Config] = None,
        device: Optional[Union[str, torch.device]] = None,
        parameter_sharing: bool = True,
    ):
        self.cfg = config or L2Config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.parameter_sharing = parameter_sharing

        self.shared_actor = L2Actor(self.cfg).to(self.device)
        self.shared_critic = L2Critic(self.cfg).to(self.device)

        self.agents: List[L2RegionalAgent] = []
        for r in range(self.cfg.n_regions):
            if parameter_sharing:
                ag = L2RegionalAgent(
                    r, self.cfg, self.device, self.shared_actor, self.shared_critic
                )
            else:
                ag = L2RegionalAgent(r, self.cfg, self.device)
            self.agents.append(ag)

        self._last_migration_matrix: Optional[np.ndarray] = None
        self._last_link_matrix: Optional[np.ndarray] = None
        self._adjusted_quota: Optional[np.ndarray] = None

    def reset(self) -> None:
        for ag in self.agents:
            ag.reset()
        self._last_migration_matrix = None
        self._last_link_matrix = None
        self._adjusted_quota = None

    def _build_neighbor_graph(
        self, region_states: List[Union[Dict[str, Any], L2RegionState]]
    ) -> List[L2RegionState]:
        states: List[L2RegionState] = []
        for i, raw in enumerate(region_states):
            if isinstance(raw, dict):
                raw = dict(raw)
                if "neighbor_ids" not in raw:
                    # 默认链式邻居
                    nbs = []
                    if i > 0:
                        nbs.append(i - 1)
                    if i < len(region_states) - 1:
                        nbs.append(i + 1)
                    raw["neighbor_ids"] = nbs
                states.append(region_state_from_dict(raw, i, self.cfg))
            else:
                states.append(raw)
        return states

    def act_all(
        self,
        region_states: List[Union[Dict[str, Any], L2RegionState]],
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        """
        所有区域智能体同步决策一步。

        Args:
            region_states: 长度 N 的列表，每项为该区域状态字典

        Returns:
            含 migration_matrix, link_matrix, adjusted_quota, per_region_info
        """
        states = self._build_neighbor_graph(region_states)

        # 1) 交换通信摘要
        messages: Dict[int, np.ndarray] = {
            s.region_id: compute_neighbor_message(s) for s in states
        }

        all_migrations: List[Dict[str, Any]] = []
        all_links: List[Dict[str, Any]] = []
        per_region: List[Dict[str, Any]] = []

        l1_quotas = np.stack([s.l1_quota for s in states], axis=0)

        # 2) 各区域独立 act
        for ag, st in zip(self.agents, states):
            nbs = {nid: messages[nid] for nid in st.neighbor_ids if nid in messages}
            obs = ag.build_observation(st, nbs)
            action, log_prob, value, info = ag.act(obs, st, deterministic=deterministic)
            all_migrations.extend(info["migrations"])
            all_links.extend(info["links"])
            per_region.append(
                {
                    "region_id": st.region_id,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "migrations": info["migrations"],
                    "links": info["links"],
                }
            )

        mig_mat = merge_migrations(all_migrations)
        link_mat = merge_links(all_links)
        adjusted = apply_migrations_to_quotas(l1_quotas, all_migrations, self.cfg.n_regions)

        self._last_migration_matrix = mig_mat
        self._last_link_matrix = link_mat
        self._adjusted_quota = adjusted

        return {
            "migration_matrix": mig_mat,
            "link_matrix": link_mat,
            "adjusted_quota": adjusted,
            "l1_quota": l1_quotas,
            "per_region": per_region,
            "n_migrations": len(all_migrations),
            "n_links": len(all_links),
        }

    def get_adjusted_quota(self) -> Optional[np.ndarray]:
        """L1 配额经 L2 区域间调剂后的有效上限 (N, 5)。"""
        return self._adjusted_quota

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.shared_actor.state_dict(), directory / "l2_actor_shared.pt")
        torch.save(self.shared_critic.state_dict(), directory / "l2_critic_shared.pt")
        torch.save(
            {
                "n_regions": self.cfg.n_regions,
                "max_migrations": self.cfg.max_migrations,
                "max_links": self.cfg.max_links,
            },
            directory / "l2_config.pt",
        )

    def load(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        self.shared_actor.load_state_dict(
            torch.load(directory / "l2_actor_shared.pt", map_location=self.device, weights_only=True)
        )
        self.shared_critic.load_state_dict(
            torch.load(directory / "l2_critic_shared.pt", map_location=self.device, weights_only=True)
        )

    def count_parameters(self) -> Dict[str, int]:
        return self.agents[0].count_parameters()
