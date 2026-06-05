"""
多个子区域 L3 智能体并行执行（参数共享），汇总子区域拓扑为区域级组网图。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .actor import L3Actor
from .agent import L3LocalAgent
from .critic import L3Critic
from .l3_spaces import L3Config, L3UpperConstraints, subregion_from_dict


class L3SubRegionMARL:
    """同一区域内多个子区域 L3 智能体（共享 Actor/Critic）。"""

    def __init__(
        self,
        n_subregions: int,
        region_id: int = 0,
        config: Optional[L3Config] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.cfg = config or L3Config()
        self.n_subregions = n_subregions
        self.region_id = region_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.shared_actor = L3Actor(self.cfg).to(self.device)
        self.shared_critic = L3Critic(self.cfg).to(self.device)
        self.agents = [
            L3LocalAgent(i, region_id, self.cfg, self.device) for i in range(n_subregions)
        ]
        # 注入共享网络
        for ag in self.agents:
            ag.actor = self.shared_actor
            ag.critic = self.shared_critic

    def act_all(
        self,
        subregion_states: List[Dict[str, Any]],
        constraints: L3UpperConstraints,
        l2_links: Optional[List[Dict[str, Any]]] = None,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        per_sub: List[Dict[str, Any]] = []
        topologies: List[Dict[str, Any]] = []

        for ag, raw in zip(self.agents, subregion_states):
            st = subregion_from_dict(raw, ag.subregion_id, self.region_id)
            obs = ag.build_observation(st, constraints)
            action, log_prob, value, info = ag.act(
                obs, constraints, state=st, l2_links=l2_links, deterministic=deterministic
            )
            per_sub.append(
                {
                    "subregion_id": ag.subregion_id,
                    "log_prob": log_prob,
                    "value": value,
                    "constraint_ok": info["constraint_ok"],
                    "n_nodes": info["topology"]["n_nodes"],
                }
            )
            topologies.append(info["topology"])

        return {
            "region_id": self.region_id,
            "per_subregion": per_sub,
            "topologies": topologies,
            "merged_topology": _merge_topologies(topologies, self.region_id),
        }

    def count_parameters(self) -> Dict[str, int]:
        return self.agents[0].count_parameters()

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.shared_actor.state_dict(), directory / "l3_actor_shared.pt")
        torch.save(self.shared_critic.state_dict(), directory / "l3_critic_shared.pt")


def _merge_topologies(topologies: List[Dict[str, Any]], region_id: int) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    coverage: List[Dict[str, Any]] = []
    for topo in topologies:
        nodes.extend(topo.get("nodes", []))
        edges.extend(topo.get("edges", []))
        coverage.extend(topo.get("coverage", []))
    return {
        "region_id": region_id,
        "n_subregions": len(topologies),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "coverage": coverage,
    }
