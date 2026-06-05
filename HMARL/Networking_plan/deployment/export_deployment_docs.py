"""Generate deployment table documents from network_plan.json (per design doc §2.1).

Three-level structure matching HMARL hierarchy:
  01 全局设备分配清单    — L1 allocates devices to L2 管辖区
  02 区域资源调度表      — Each L2 管辖区 schedules its 子区域
  03 子区域设备部署明细表 — Each 子区域 has 12 grids
  04 设备点位与拓扑连接表 — Node coordinates plus intra/inter-region topology
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

DEVICE_LABELS = {
    "residual_bs": "残余基站",
    "emergency_bs": "应急基站",
    "portable_gateway": "便携广播网关",
    "mesh_relay": "Mesh中继",
    "comm_uav": "通信UAV",
    "satellite_terminal": "卫星终端",
    "relay_5g": "5G中继",
}

L2_LABELS_POOL = [
    "重灾核心管辖区",
    "次重灾管辖区",
    "一般受灾管辖区",
    "边缘恢复管辖区",
    "低优先级管辖区",
]

SUBREGION_LABELS_POOL = [
    "核心指挥镇", "重灾避难镇A", "重灾避难镇B", "沿海/河谷重灾镇", "核心周边镇",
    "次重灾镇A", "次重灾镇B", "次重灾镇C", "过渡镇A", "过渡镇B",
    "一般受灾镇A", "一般受灾镇B", "一般受灾镇C", "一般受灾镇D", "一般受灾镇E",
    "边缘恢复镇A", "边缘恢复镇B", "边缘恢复镇C", "低优先级镇A", "低优先级镇B",
]


def _plan_dims(plan: Dict[str, Any]):
    """Extract n_subregions / n_l2 / subs_per_l2 from plan metadata."""
    n_sub = int(plan.get("n_regions", plan.get("n_l3", 20)))
    n_l2 = int(plan.get("n_l2", max(1, n_sub // 5)))
    subs = max(1, n_sub // n_l2) if n_l2 > 0 else n_sub
    return n_sub, n_l2, subs


def _sub_label(rid: int) -> str:
    suffix = SUBREGION_LABELS_POOL[rid] if rid < len(SUBREGION_LABELS_POOL) else f"镇{rid:02d}"
    return f"子区域{rid:02d}-{suffix}"


def _l2_label(l2_id: int) -> str:
    suffix = L2_LABELS_POOL[l2_id] if l2_id < len(L2_LABELS_POOL) else f"管辖区{l2_id}"
    return f"L2-{l2_id} {suffix}"


def _type_label(node_type: str) -> str:
    return DEVICE_LABELS.get(node_type, node_type)


def _node_grid_label(node: Dict[str, Any]) -> str:
    if node.get("grid_label"):
        return str(node["grid_label"])
    if node.get("grid_index") is not None:
        return f"G{int(node['grid_index']):02d}"
    return "—"


def _node_coord_text(node: Dict[str, Any]) -> str:
    cell = node.get("grid_cell") or {}
    pos = node.get("position") or {}
    row = cell.get("row")
    col = cell.get("col")
    x = pos.get("x")
    y = pos.get("y")
    if row is not None and col is not None and x is not None and y is not None:
        return f"r{row},c{col} ({float(x):.2f},{float(y):.2f})"
    if x is not None and y is not None:
        return f"({float(x):.2f},{float(y):.2f})"
    return "—"


def _node_topology_role(node: Dict[str, Any]) -> str:
    return str(node.get("topology_role") or node.get("role") or "—")


def _node_ref(node: Dict[str, Any]) -> str:
    return f"N{int(node.get('id', 0)):03d}/{_sub_label(int(node.get('region_id', 0)))}/{_node_grid_label(node)}"


def _link_endpoint_text(link: Dict[str, Any], nodes_by_id: Dict[int, Dict[str, Any]], field: str) -> str:
    node_id = link.get(field)
    if node_id is None:
        return "—"
    node = nodes_by_id.get(int(node_id))
    if not node:
        return f"N{int(node_id):03d}"
    return _node_ref(node)


def _link_line(link: Dict[str, Any], nodes_by_id: Dict[int, Dict[str, Any]]) -> str:
    source = _link_endpoint_text(link, nodes_by_id, "source_node")
    target = _link_endpoint_text(link, nodes_by_id, "target_node")
    link_type = str(link.get("link_type", ""))
    layer = str(link.get("layer", ""))
    bandwidth = link.get("bandwidth_mbps")
    purpose = str(link.get("purpose", ""))
    note = str(link.get("note", ""))
    bw_text = f"{bandwidth}Mbps" if bandwidth is not None else "—"
    suffix = f" | {note}" if note else ""
    return f"    {source} --[{link_type}/{layer}/{bw_text}]--> {target}  ({purpose}){suffix}"


def _aggregate_by_region_type(nodes: List[Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for node in nodes:
        rid = int(node.get("region_id", 0))
        counts[rid][node.get("type", "unknown")] += 1
    return counts


def _global_totals(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    totals: Dict[str, int] = defaultdict(int)
    for node in nodes:
        totals[node.get("type", "unknown")] += 1
    return totals


# ---------------------------------------------------------------------------
# 01  全局设备分配清单
# ---------------------------------------------------------------------------
def render_global_device_allocation(plan: Dict[str, Any]) -> str:
    """《全局设备分配清单》— L1 全局统筹层向 L2 管辖区的配额分配。"""
    n_sub, n_l2, subs_per_l2 = _plan_dims(plan)

    nodes = plan.get("nodes", [])
    totals = _global_totals(nodes)
    by_region = _aggregate_by_region_type(nodes)
    types = sorted({n.get("type", "") for n in nodes})

    l2_agg: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rid, type_counts in by_region.items():
        l2_id = min(rid // subs_per_l2, n_l2 - 1)
        for t, cnt in type_counts.items():
            l2_agg[l2_id][t] += cnt

    lines = [
        "《全局设备分配清单》",
        "=" * 72,
        f"场景标识: {plan.get('scenario_id')} ({plan.get('scenario_name', '')})",
        f"组网模式: {plan.get('network_mode')} ({plan.get('network_mode_name', '')})",
        f"主回传链路: {plan.get('primary_backhaul', '')}",
        f"生成时间: {plan.get('generated_at', '')}",
        "",
        "[全局汇总]",
        f"  节点总数: {len(nodes)}",
        f"  残余复用: {plan.get('residual_nodes_reused', 0)}",
        f"  应急部署: {plan.get('emergency_nodes_deployed', 0)}",
        f"  L1 智能体: 1 | L2 智能体: {n_l2} | L3 智能体: {n_sub}",
        "",
        "设备类型 | 全局合计",
        "-" * 30,
    ]
    for t in types:
        lines.append(f"  {_type_label(t):12s} | {totals.get(t, 0):>6d}")
    lines.append("")
    lines.append(f"[L1 配额分配 → {n_l2} 个 L2 管辖区]")
    col_header = "L2 管辖区 | " + " | ".join(f"{_type_label(t)[:6]:>6s}" for t in types) + " | 小计"
    lines.append(col_header)
    lines.append("-" * len(col_header))
    for l2_id in range(n_l2):
        row = l2_agg.get(l2_id, {})
        vals = [str(row.get(t, 0)) for t in types]
        sub = sum(row.get(t, 0) for t in types)
        s_start = l2_id * subs_per_l2
        s_end = min(s_start + subs_per_l2, n_sub) - 1
        subs_tag = f"(子区域{s_start:02d}–{s_end:02d})"
        lines.append(
            f"  {_l2_label(l2_id)} {subs_tag} | "
            + " | ".join(f"{v:>6s}" for v in vals)
            + f" | {sub:>4d}"
        )
    lines.append("-" * len(col_header))
    grand = sum(totals.values())
    col_totals = [str(totals.get(t, 0)) for t in types]
    lines.append("  合计 | " + " | ".join(f"{v:>6s}" for v in col_totals) + f" | {grand:>4d}")

    lines.append("")
    lines.append("[部署优先级]")
    for i, step in enumerate(plan.get("deploy_priority") or []):
        lines.append(f"  {i + 1}. {step}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 02  区域资源调度表
# ---------------------------------------------------------------------------
def render_regional_resource_schedule(plan: Dict[str, Any]) -> str:
    """《区域资源调度表》— L2 管辖区 → 下辖子区域的资源调度。"""
    n_sub, n_l2, subs_per_l2 = _plan_dims(plan)

    regional = plan.get("regional_tasks", [])
    nodes = plan.get("nodes", [])
    by_region = _aggregate_by_region_type(nodes)

    comm_by_region: Dict[int, List[str]] = defaultdict(list)
    for node in nodes:
        rid = int(node.get("region_id", 0))
        cm = node.get("comm_mode", "")
        if cm and cm not in comm_by_region[rid]:
            comm_by_region[rid].append(cm)

    lines = [
        "《区域资源调度表》",
        "=" * 72,
        f"场景: {plan.get('scenario_id')} | 模式: {plan.get('network_mode')}",
        f"L2 管辖区数: {n_l2} | 每管辖区下辖子区域: {subs_per_l2} | 子区域总计: {n_sub}",
        "",
    ]

    for l2_id in range(n_l2):
        sub_start = l2_id * subs_per_l2
        sub_end = min(sub_start + subs_per_l2, n_sub)
        l2_nodes = sum(
            len([n for n in nodes if int(n.get("region_id", -1)) == r])
            for r in range(sub_start, sub_end)
        )
        lines.append(f"{'─' * 72}")
        lines.append(f"  {_l2_label(l2_id)}  (子区域 {sub_start:02d}–{sub_end - 1:02d}, 节点合计 {l2_nodes})")
        lines.append(f"{'─' * 72}")
        lines.append("  子区域 | 节点数 | 业务任务 | 主用制式 | 调度说明")
        lines.append("  " + "-" * 68)

        for rid in range(sub_start, sub_end):
            task = next((t for t in regional if int(t.get("region_id", -1)) == rid), None)
            n_count = task.get("node_count", 0) if task else 0
            tasks_str = ", ".join((task.get("business_tasks", []) if task else [])[:2])
            comm = "/".join(comm_by_region.get(rid, [])[:3])
            note = "L2跨区调剂后执行" if rid < sub_start + 2 else "按L1配额就地部署"
            lines.append(
                f"  {_sub_label(rid):20s} | {n_count:>4d} | {tasks_str[:22]:22s} | "
                f"{comm[:16]:16s} | {note}"
            )

        lines.append("")
        lines.append("  [管辖区设备分布]")
        for rid in range(sub_start, sub_end):
            row = by_region.get(rid, {})
            if not row:
                continue
            parts = ", ".join(f"{_type_label(k)}×{v}" for k, v in sorted(row.items()))
            lines.append(f"    {_sub_label(rid)}: {parts}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 03  子区域设备部署明细表
# ---------------------------------------------------------------------------
def render_subregion_deployment_detail(plan: Dict[str, Any]) -> str:
    """《子区域设备部署明细表》— L3 网格级节点部署明细。"""
    n_sub, n_l2, subs_per_l2 = _plan_dims(plan)

    nodes = plan.get("nodes", [])
    grids_per_sub = 12

    by_region: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_region[int(node.get("region_id", 0))].append(node)

    lines = [
        "《子区域设备部署明细表》",
        "=" * 72,
        f"场景: {plan.get('scenario_id')} | 模式: {plan.get('network_mode')}",
        f"子区域总数: {n_sub} | 每子区域网格: {grids_per_sub} | 总网格: {n_sub * grids_per_sub}",
        f"说明: 子网格 G00–G{grids_per_sub - 1:02d} 对应 L3 部署矩阵列 (5设备×{grids_per_sub}网格)。",
        "坐标: grid_cell 为 3×4 行列 (r0–r2,c0–c3); position 为子区域内归一化坐标 (x,y)∈[0,1]。",
        "",
    ]
    schema = plan.get("placement_schema") or {}
    if schema:
        lines.append(
            f"落点决策: seed={schema.get('placement_seed')} progress={schema.get('progress')} "
            f"mode={schema.get('network_mode')}"
        )
        lines.append("")

    for l2_id in range(n_l2):
        sub_start = l2_id * subs_per_l2
        sub_end = min(sub_start + subs_per_l2, n_sub)
        lines.append(f"{'═' * 72}")
        lines.append(f"  {_l2_label(l2_id)}  (子区域 {sub_start:02d}–{sub_end - 1:02d})")
        lines.append(f"{'═' * 72}")

        for rid in range(sub_start, sub_end):
            region_nodes = by_region.get(rid, [])
            lines.append(f"  ┌─ {_sub_label(rid)}  ({len(region_nodes)} 节点)")
            lines.append(
                f"  │ {'节点ID':>6s} | {'网格':>4s} | {'坐标':>18s} | {'设备类型':10s} | "
                f"{'来源':12s} | {'制式':12s} | {'状态':8s} | 角色"
            )
            lines.append(f"  │ {'-' * 86}")
            sorted_nodes = sorted(
                region_nodes,
                key=lambda n: (int(n.get("grid_index", 99)), int(n.get("id", 0))),
            )
            for idx, node in enumerate(sorted_nodes):
                lines.append(
                    f"  │ {node.get('id', idx):>6} | {_node_grid_label(node):>4s} | "
                    f"{_node_coord_text(node):>18s} | "
                    f"{_type_label(node.get('type', '')):10s} | {node.get('source', ''):12s} | "
                    f"{node.get('comm_mode', ''):12s} | {node.get('status', ''):8s} | "
                    f"{_node_topology_role(node)}"
                )
            lines.append(f"  └─ 小计: {len(region_nodes)} 节点")
            lines.append("")

    lines.append(f"[合计] 部署节点 {len(nodes)} 个, 覆盖 {n_sub} 子区域 × {grids_per_sub} 网格")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 04  设备点位与拓扑连接表
# ---------------------------------------------------------------------------
def render_device_point_topology(plan: Dict[str, Any]) -> str:
    """《设备点位与拓扑连接表》— Point placement plus layered topology."""
    n_sub, n_l2, subs_per_l2 = _plan_dims(plan)
    nodes = plan.get("nodes", [])
    topology = plan.get("topology") or {}
    summary = topology.get("summary") or {}
    nodes_by_id = {int(node.get("id", 0)): node for node in nodes}

    lines = [
        "《设备点位与拓扑连接表》",
        "=" * 72,
        f"场景: {plan.get('scenario_id')} | 模式: {plan.get('network_mode')} | 拓扑模式: {topology.get('pattern', plan.get('topology_pattern'))}",
        f"子区域: {n_sub} | L2管辖区: {n_l2} | 主回传: {plan.get('primary_backhaul', '')}",
        "",
        "[一、点位坐标约定]",
        "  每个子区域划分为 3×4 共12个网格，G00–G11 按行优先编号。",
        "  position 使用子区域内归一化坐标 (x,y)∈[0,1]，用于先确定设备点位，再生成拓扑连接。",
        "",
        "[二、拓扑生成原则]",
        f"  Hub选举: {topology.get('hub_selection', '—')}",
        "  区域内: 子区域 hub 连接残余基站、应急基站、Mesh中继、广播网关和UAV/卫星上行节点。",
        "  区域间: 每个L2管辖区选核心子区域 hub，其余子区域 hub 星型接入，再由L1骨干连接相邻L2核心 hub。",
        "",
        "[三、设备点位明细]",
        "  节点ID | 子区域 | 网格 | 坐标 | 设备类型 | 制式 | 拓扑角色",
        "  " + "-" * 76,
    ]

    for node in sorted(nodes, key=lambda n: (int(n.get("region_id", 0)), int(n.get("grid_index", 99)), int(n.get("id", 0)))):
        rid = int(node.get("region_id", 0))
        lines.append(
            f"  N{int(node.get('id', 0)):03d} | {_sub_label(rid):20s} | {_node_grid_label(node):>4s} | "
            f"{_node_coord_text(node):>18s} | {_type_label(str(node.get('type', ''))):10s} | "
            f"{str(node.get('comm_mode', '')):12s} | {_node_topology_role(node)}"
        )

    lines.extend(
        [
            "",
            "[四、区域Hub映射]",
        ]
    )
    hub_map = topology.get("hub_map") or {}
    for rid_text in sorted(hub_map, key=lambda x: int(x)):
        rid = int(rid_text)
        hub_id = int(hub_map[rid_text])
        hub = nodes_by_id.get(hub_id, {"id": hub_id, "region_id": rid})
        lines.append(f"  {_sub_label(rid)} -> {_node_ref(hub)} ({_node_topology_role(hub)})")

    lines.extend(["", "[五、区域内拓扑连接]"])
    intra = topology.get("intra_region") or []
    for rid in range(n_sub):
        region_links = [link for link in intra if int(link.get("source_region", -1)) == rid]
        if not region_links:
            continue
        lines.append(f"  {_sub_label(rid)}")
        for link in region_links:
            lines.append(_link_line(link, nodes_by_id))

    lines.extend(["", "[六、区域间拓扑连接]"])
    inter = topology.get("inter_region") or []
    for l2_id in range(n_l2):
        sub_start = l2_id * subs_per_l2
        sub_end = min(sub_start + subs_per_l2, n_sub)
        l2_links = [
            link
            for link in inter
            if sub_start <= int(link.get("source_region", -1)) < sub_end
            or sub_start <= int(link.get("target_region", -1)) < sub_end
        ]
        if not l2_links:
            continue
        lines.append(f"  {_l2_label(l2_id)} (子区域{sub_start:02d}–{sub_end - 1:02d})")
        for link in l2_links:
            lines.append(_link_line(link, nodes_by_id))

    lines.extend(["", "[七、L1骨干与场景补丁]"])
    backbone = topology.get("backbone") or []
    if backbone:
        lines.append("  L1骨干链路")
        for link in backbone:
            lines.append(_link_line(link, nodes_by_id))
    overlays = topology.get("scenario_overlays") or []
    if overlays:
        lines.append("  场景补丁链路")
        for link in overlays:
            lines.append(f"    {link.get('link_type')} ({link.get('purpose')}) | {link.get('note', '')}")

    lines.extend(
        [
            "",
            "[八、拓扑统计]",
            f"  区域内链路: {summary.get('intra_region_links', len(intra))}",
            f"  区域间链路: {summary.get('inter_region_links', len(inter))}",
            f"  L1骨干链路: {summary.get('backbone_links', len(backbone))}",
            f"  场景补丁链路: {summary.get('scenario_overlay_links', len(overlays))}",
            f"  链路合计: {summary.get('total_links', len(plan.get('links', [])))}",
        ]
    )
    return "\n".join(lines) + "\n"


def export_deployment_documents(plan: Dict[str, Any], out_dir: Path) -> Dict[str, Path]:
    """Write deployment tables next to network_plan.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "01_全局设备分配清单.txt": render_global_device_allocation,
        "02_区域资源调度表.txt": render_regional_resource_schedule,
        "03_子区域设备部署明细表.txt": render_subregion_deployment_detail,
        "04_设备点位与拓扑连接表.txt": render_device_point_topology,
    }
    written: Dict[str, Path] = {}
    for filename, renderer in mapping.items():
        path = out_dir / filename
        path.write_text(renderer(plan), encoding="utf-8")
        written[filename] = path
    return written
