"""Build point-aware topology links for generated network plans."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Dict, Iterable, List, Optional, Tuple


HUB_PRIORITY: Dict[str, int] = {
    "emergency_bs": 0,
    "residual_bs": 1,
    "satellite_terminal": 2,
    "comm_uav": 3,
    "mesh_relay": 4,
    "portable_gateway": 5,
}


def _node_id(node: Dict[str, Any]) -> int:
    return int(node.get("id", 0))


def _region_id(node: Dict[str, Any]) -> int:
    return int(node.get("region_id", 0))


def _grid_index(node: Dict[str, Any]) -> int:
    return int(node.get("grid_index", 99))


def _position(node: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pos = node.get("position") or {}
    x = pos.get("x")
    y = pos.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def _distance_norm(source: Dict[str, Any], target: Dict[str, Any]) -> Optional[float]:
    src = _position(source)
    tgt = _position(target)
    if src is None or tgt is None:
        return None
    return round(float(sqrt((src[0] - tgt[0]) ** 2 + (src[1] - tgt[1]) ** 2)), 4)


def _sort_nodes(nodes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        nodes,
        key=lambda n: (
            HUB_PRIORITY.get(str(n.get("type", "")), 99),
            _grid_index(n),
            _node_id(n),
        ),
    )


def _select_region_hub(region_nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not region_nodes:
        return None
    return _sort_nodes(region_nodes)[0]


def _default_topology_role(node: Dict[str, Any]) -> str:
    node_type = str(node.get("type", ""))
    if node_type == "portable_gateway":
        return "broadcast_root"
    if node_type == "mesh_relay":
        return "relay"
    if node_type in {"comm_uav", "satellite_terminal"}:
        return "uplink"
    if node_type == "emergency_bs":
        return "patch_access"
    return "access"


def _intra_link_type(node: Dict[str, Any], network_mode: str) -> Tuple[str, str, int]:
    node_type = str(node.get("type", ""))
    if node_type == "satellite_terminal":
        return "Satellite_Ka", "backhaul", 30
    if node_type == "comm_uav":
        return ("UAV_relay" if network_mode == "no_residual" else "Satellite_Ka"), "uplink", 30
    if node_type in {"portable_gateway", "mesh_relay"}:
        return "WiFi6_mesh", "access", 20
    if node_type == "residual_bs":
        return "5G_residual_link", "access", 50
    return "5G_700MHz", "access", 40


def _add_link(
    links: List[Dict[str, Any]],
    *,
    scope: str,
    layer: str,
    source_node: int,
    target_node: int,
    link_type: str,
    bandwidth_mbps: int,
    source_region: Optional[int] = None,
    target_region: Optional[int] = None,
    purpose: str,
    distance_norm: Optional[float] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    link: Dict[str, Any] = {
        "id": len(links),
        "scope": scope,
        "layer": layer,
        "source_node": int(source_node),
        "target_node": int(target_node),
        "link_type": link_type,
        "bandwidth_mbps": int(bandwidth_mbps),
        "purpose": purpose,
    }
    if source_region is not None:
        link["source_region"] = int(source_region)
    if target_region is not None:
        link["target_region"] = int(target_region)
    if distance_norm is not None:
        link["distance_norm"] = distance_norm
    if note:
        link["note"] = note
    links.append(link)
    return link


def _build_intra_region_links(
    by_region: Dict[int, List[Dict[str, Any]]],
    hub_by_region: Dict[int, Dict[str, Any]],
    *,
    network_mode: str,
) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []

    for rid in sorted(by_region):
        region_nodes = sorted(by_region[rid], key=lambda n: (_grid_index(n), _node_id(n)))
        hub = hub_by_region.get(rid)
        if hub is None:
            continue

        direct_pairs: set[Tuple[int, int]] = set()
        residuals = [n for n in region_nodes if n.get("type") == "residual_bs"]
        for source, target in zip(residuals, residuals[1:]):
            direct_pairs.add(tuple(sorted((_node_id(source), _node_id(target)))))
            _add_link(
                links,
                scope="intra_region",
                layer="access",
                source_node=_node_id(source),
                target_node=_node_id(target),
                source_region=rid,
                target_region=rid,
                link_type="5G_residual_link",
                bandwidth_mbps=50,
                purpose="residual_bs_chain",
                distance_norm=_distance_norm(source, target),
            )

        for node in region_nodes:
            if _node_id(node) == _node_id(hub):
                continue
            if tuple(sorted((_node_id(hub), _node_id(node)))) in direct_pairs:
                continue
            link_type, layer, bandwidth = _intra_link_type(node, network_mode)
            _add_link(
                links,
                scope="intra_region",
                layer=layer,
                source_node=_node_id(hub),
                target_node=_node_id(node),
                source_region=rid,
                target_region=rid,
                link_type=link_type,
                bandwidth_mbps=bandwidth,
                purpose="region_hub_access",
                distance_norm=_distance_norm(hub, node),
            )

        broadcast_roots = [n for n in region_nodes if n.get("type") == "portable_gateway"]
        access_targets = [
            n
            for n in region_nodes
            if n.get("type") in {"residual_bs", "emergency_bs", "mesh_relay"}
        ]
        for root in broadcast_roots:
            for target in access_targets[:3]:
                if _node_id(root) == _node_id(target):
                    continue
                _add_link(
                    links,
                    scope="intra_region",
                    layer="broadcast",
                    source_node=_node_id(root),
                    target_node=_node_id(target),
                    source_region=rid,
                    target_region=rid,
                    link_type="WiFi6",
                    bandwidth_mbps=10,
                    purpose="broadcast_fanout",
                    distance_norm=_distance_norm(root, target),
                )

    return links


def _build_inter_region_links(
    hub_by_region: Dict[int, Dict[str, Any]],
    *,
    n_regions: int,
    n_l2: int,
    network_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    links: List[Dict[str, Any]] = []
    subs_per_l2 = max(1, n_regions // max(1, n_l2))
    l2_core_hubs: Dict[int, Dict[str, Any]] = {}

    for l2_id in range(n_l2):
        sub_start = l2_id * subs_per_l2
        sub_end = min(sub_start + subs_per_l2, n_regions)
        region_hubs = [
            hub_by_region[rid]
            for rid in range(sub_start, sub_end)
            if rid in hub_by_region
        ]
        if not region_hubs:
            continue
        core = region_hubs[0]
        l2_core_hubs[l2_id] = core
        for hub in region_hubs[1:]:
            link_type = "Microwave_relay" if network_mode == "with_residual" else "UAV_relay"
            _add_link(
                links,
                scope="inter_region",
                layer="inter_region",
                source_node=_node_id(core),
                target_node=_node_id(hub),
                source_region=_region_id(core),
                target_region=_region_id(hub),
                link_type=link_type,
                bandwidth_mbps=25 if network_mode == "with_residual" else 15,
                purpose=f"L2-{l2_id}_hub_spoke",
                note=f"L2-{l2_id} core hub connects subordinate subregion hub",
            )

    return links, l2_core_hubs


def _build_backbone_links(
    l2_core_hubs: Dict[int, Dict[str, Any]],
    *,
    network_mode: str,
    primary_backhaul: str,
) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    ordered = [l2_core_hubs[i] for i in sorted(l2_core_hubs)]
    if len(ordered) < 2:
        return links

    link_type = primary_backhaul if network_mode == "with_residual" else "Satellite_Ka"
    bandwidth = 50 if network_mode == "with_residual" else 30
    for source, target in zip(ordered, ordered[1:]):
        _add_link(
            links,
            scope="backbone",
            layer="backhaul",
            source_node=_node_id(source),
            target_node=_node_id(target),
            source_region=_region_id(source),
            target_region=_region_id(target),
            link_type=link_type,
            bandwidth_mbps=bandwidth,
            purpose="L1_backbone_chain",
            note="L1 backbone connects adjacent L2 core hubs",
        )
    return links


def _scenario_overlays(scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
    disaster_type = scenario.get("disaster_type")
    if disaster_type == "rainstorm":
        return [
            {
                "scope": "scenario_overlay",
                "layer": "resilience",
                "link_type": "patch_fiber",
                "purpose": "rainstorm_fiber_breakage_patch",
                "note": "暴雨场景光缆冲毁补丁路由",
                "breakage_rate": scenario.get("link_breakage_rate", 0.35),
            }
        ]
    if disaster_type == "typhoon":
        return [
            {
                "scope": "scenario_overlay",
                "layer": "resilience",
                "link_type": "local_blackout_bridge",
                "purpose": "typhoon_blackout_bridge",
                "note": "台风场景局部全阻桥接",
                "blackout_zones": scenario.get("local_blackout_zones", 4),
            }
        ]
    return []


def build_topology(
    plan: Dict[str, Any],
    topology_template: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach topology roles to nodes and return layered topology links."""
    nodes: List[Dict[str, Any]] = plan.get("nodes", [])
    n_regions = int(plan.get("n_regions", plan.get("n_l3", 1)))
    n_l2 = int(plan.get("n_l2", max(1, n_regions // 5)))
    network_mode = str(plan.get("network_mode", topology_template.get("network_mode", "")))
    primary_backhaul = str(plan.get("primary_backhaul", "Satellite_Ka"))

    by_region: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        node["topology_role"] = _default_topology_role(node)
        node["is_region_hub"] = False
        by_region[_region_id(node)].append(node)

    hub_by_region: Dict[int, Dict[str, Any]] = {}
    for rid, region_nodes in by_region.items():
        hub = _select_region_hub(region_nodes)
        if hub is None:
            continue
        hub["topology_role"] = "region_hub"
        hub["is_region_hub"] = True
        hub_by_region[rid] = hub

    intra_links = _build_intra_region_links(
        by_region,
        hub_by_region,
        network_mode=network_mode,
    )
    inter_links, l2_core_hubs = _build_inter_region_links(
        hub_by_region,
        n_regions=n_regions,
        n_l2=n_l2,
        network_mode=network_mode,
    )
    backbone_links = _build_backbone_links(
        l2_core_hubs,
        network_mode=network_mode,
        primary_backhaul=primary_backhaul,
    )
    overlays = _scenario_overlays(scenario)

    flat_links: List[Dict[str, Any]] = []
    for link in intra_links + inter_links + backbone_links:
        link = dict(link)
        link["id"] = len(flat_links)
        flat_links.append(link)

    for overlay in overlays:
        overlay = dict(overlay)
        overlay["id"] = len(flat_links)
        flat_links.append(overlay)

    return {
        "pattern": topology_template.get("topology_pattern"),
        "description": topology_template.get("description"),
        "hub_selection": "emergency_bs > residual_bs > satellite_terminal > comm_uav > mesh_relay > portable_gateway",
        "hub_map": {str(rid): _node_id(hub) for rid, hub in sorted(hub_by_region.items())},
        "l2_core_hubs": {str(l2_id): _node_id(hub) for l2_id, hub in sorted(l2_core_hubs.items())},
        "intra_region": intra_links,
        "inter_region": inter_links,
        "backbone": backbone_links,
        "scenario_overlays": overlays,
        "links": flat_links,
        "summary": {
            "regions_with_hub": len(hub_by_region),
            "intra_region_links": len(intra_links),
            "inter_region_links": len(inter_links),
            "backbone_links": len(backbone_links),
            "scenario_overlay_links": len(overlays),
            "total_links": len(flat_links),
        },
    }
