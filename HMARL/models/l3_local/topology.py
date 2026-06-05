"""
根据 L3 部署动作与 L2 链路规划生成组网拓扑（可视化/导出用）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .l3_spaces import DEVICE_NAMES, N_DEVICES, N_GRIDS


def build_topology_graph(
    subregion_id: int,
    region_id: int,
    decoded_action: Dict[str, Any],
    grid_centers: Optional[np.ndarray] = None,
    l2_links: Optional[List[Dict[str, Any]]] = None,
    coverage_radius_km: float = 1.5,
) -> Dict[str, Any]:
    """
    生成组网拓扑 JSON 结构。

    nodes: 设备节点（网格位置、类型、功率/带宽参数）
    edges: 链路（含 L2 跨区链路与网格内逻辑连接）
    coverage: 覆盖圆
    """
    deployment = decoded_action["deployment"]
    work_params = decoded_action["work_params"]

    if grid_centers is None:
        # 默认 12 网格在子区域内 3×4 布局（归一化坐标）
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.1, 0.9, 3)
        centers = []
        for y in ys:
            for x in xs:
                centers.append([float(x), float(y)])
        grid_centers = np.asarray(centers[:N_GRIDS], dtype=np.float32)
    else:
        grid_centers = np.asarray(grid_centers, dtype=np.float32).reshape(N_GRIDS, 2)

    nodes: List[Dict[str, Any]] = []
    coverage: List[Dict[str, Any]] = []
    node_id = 0

    for j, dname in enumerate(DEVICE_NAMES):
        power_ratio = float(work_params[j, 0])
        bw_ratio = float(work_params[j, 1])
        for g in range(N_GRIDS):
            cnt = int(deployment[j, g])
            if cnt <= 0:
                continue
            pos = grid_centers[g].tolist()
            for _ in range(cnt):
                nodes.append(
                    {
                        "id": node_id,
                        "subregion_id": subregion_id,
                        "region_id": region_id,
                        "device_type": dname,
                        "device_type_id": j,
                        "grid_index": g,
                        "position": pos,
                        "power_ratio": power_ratio,
                        "bandwidth_ratio": bw_ratio,
                    }
                )
                coverage.append(
                    {
                        "node_id": node_id,
                        "center": pos,
                        "radius_km": coverage_radius_km * (0.5 + 0.5 * power_ratio),
                        "device_type": dname,
                    }
                )
                node_id += 1

    edges: List[Dict[str, Any]] = []
    # 同网格多设备互连
    by_grid: Dict[int, List[int]] = {}
    for n in nodes:
        by_grid.setdefault(n["grid_index"], []).append(n["id"])
    for g, ids in by_grid.items():
        for i in range(len(ids) - 1):
            edges.append({"source": ids[i], "target": ids[i + 1], "link_kind": "intra_grid"})

    if l2_links:
        for lk in l2_links:
            edges.append(
                {
                    "source": f"region_{lk.get('region_a')}",
                    "target": f"region_{lk.get('region_b')}",
                    "link_kind": "cross_region",
                    "link_type": lk.get("link_type_name", ""),
                    "deploy_position": lk.get("deploy_position", 0.0),
                }
            )

    return {
        "subregion_id": subregion_id,
        "region_id": region_id,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "coverage": coverage,
        "global_params": {
            "rescue_priority_weight": decoded_action.get("rescue_priority_weight"),
            "cross_region_reserve_ratio": decoded_action.get("cross_region_reserve_ratio"),
        },
    }
