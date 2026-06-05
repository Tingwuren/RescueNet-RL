"""Core network plan builder: architecture + scenario + mode -> network_plan.json."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .parse_rl_output import parse_rl_output
from .task_split import split_tasks
from .topology_builder import build_topology
from ._config import (
    get_comm_mode_ids,
    load_architecture,
    load_network_mode,
    load_phased_deploy,
    load_scenario,
)

SCENARIO_N_REGIONS: Dict[str, int] = {
    "super_typhoon": 20,
    "extreme_rainstorm": 10,
}
DEFAULT_N_REGIONS = 20

EMERGENCY_TYPES_RESIDUAL = [
    ("emergency_bs", "5G_700MHz"),
    ("portable_gateway", "WiFi6"),
    ("mesh_relay", "WiFi6"),
    ("comm_uav", "Satellite_Ka"),
]
EMERGENCY_TYPES_NO_RESIDUAL = [
    ("satellite_terminal", "Satellite_Ka"),
    ("comm_uav", "Satellite_Ka"),
    ("emergency_bs", "5G_700MHz"),
    ("portable_gateway", "WiFi6"),
    ("mesh_relay", "WiFi6"),
]


def _scenario_scale(scenario: Dict[str, Any]) -> float:
    """Derive node scale factor from outage / damage parameters."""
    outage = (
        scenario.get("base_station_outage_min", 0.3)
        + scenario.get("base_station_outage_max", 0.5)
    ) / 2
    return 0.8 + outage


def _emergency_type_keys(network_mode: str) -> List[str]:
    types = EMERGENCY_TYPES_NO_RESIDUAL if network_mode == "no_residual" else EMERGENCY_TYPES_RESIDUAL
    return [name for name, _ in types]


def _emergency_weights(
    deploy_rules: Dict[str, Any],
    network_mode: str,
    type_keys: List[str],
) -> Dict[str, float]:
    """Resolve per-type weights; prefer emergency_device_weights from YAML."""
    explicit = deploy_rules.get("emergency_device_weights")
    if isinstance(explicit, dict) and explicit:
        weights = {k: float(explicit.get(k, 0.0)) for k in type_keys}
    else:
        raw = deploy_rules.get("node_source_weights") or {}
        weights = {k: float(raw.get(k, 0.0)) for k in type_keys}
        if network_mode == "with_residual" and sum(weights.values()) <= 0:
            weights = {k: 1.0 / len(type_keys) for k in type_keys}
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        return {k: 1.0 / len(type_keys) for k in type_keys}
    return {k: max(0.0, weights[k]) / total for k in type_keys}


def _split_integer_by_weights(total: int, weights: Dict[str, float], keys: List[str]) -> Dict[str, int]:
    """Largest-remainder allocation so counts sum exactly to total."""
    if total <= 0:
        return {k: 0 for k in keys}
    import numpy as np

    w = np.array([weights.get(k, 0.0) for k in keys], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(keys), dtype=np.float64)
    w /= w.sum()
    raw = w * total
    counts = np.floor(raw).astype(int)
    remainder = int(total - int(counts.sum()))
    if remainder > 0:
        fractional = raw - counts
        order = np.argsort(-fractional)
        for i in range(remainder):
            counts[order[i % len(keys)]] += 1
    return {keys[i]: int(counts[i]) for i in range(len(keys))}


def _region_ids_for_devices(count: int, n_regions: int, *, salt: str) -> List[int]:
    """Spread devices across subregions (deterministic, type-dependent stride)."""
    if count <= 0:
        return []
    start = abs(hash(salt)) % max(1, n_regions)
    stride = 3 if n_regions % 3 else 1
    return [(start + i * stride) % n_regions for i in range(count)]


def _emergency_source_label(etype: str) -> str:
    if etype == "emergency_bs":
        return "emergency_emergency_bs"
    prefix = etype.split("_")[0]
    return f"emergency_{prefix}"


def _build_nodes(
    scenario: Dict[str, Any],
    mode_cfg: Dict[str, Any],
    deploy_rules: Dict[str, Any],
    network_mode: str,
    n_regions: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Generate node list with mixed device types per region."""
    scale = _scenario_scale(scenario)
    residual_base = int(deploy_rules.get("residual_node_count_base", 0) * scale)
    emergency_base = int(deploy_rules.get("emergency_node_count_base", 10) * scale)

    residual_base = int(residual_base * n_regions / 20)
    emergency_base = int(emergency_base * n_regions / 20)

    if network_mode == "no_residual":
        residual_base = 0
        emergency_base = max(emergency_base, n_regions * 4)

    nodes: List[Dict[str, Any]] = []
    node_id = 0

    if network_mode == "with_residual":
        for i in range(residual_base):
            nodes.append(
                {
                    "id": node_id,
                    "region_id": i % n_regions,
                    "type": "residual_bs",
                    "source": "residual_bs",
                    "comm_mode": "5G_700MHz",
                    "status": "active",
                    "role": "access",
                }
            )
            node_id += 1

    emergency_types = (
        EMERGENCY_TYPES_NO_RESIDUAL if network_mode == "no_residual" else EMERGENCY_TYPES_RESIDUAL
    )
    comm_by_type = {name: comm for name, comm in emergency_types}
    type_keys = [name for name, _ in emergency_types]
    weights = _emergency_weights(deploy_rules, network_mode, type_keys)
    type_counts = _split_integer_by_weights(emergency_base, weights, type_keys)
    scenario_tag = str(scenario.get("scenario_id", network_mode))

    for etype in type_keys:
        count = int(type_counts.get(etype, 0))
        region_ids = _region_ids_for_devices(count, n_regions, salt=f"{scenario_tag}:{etype}")
        for rid in region_ids:
            nodes.append(
                {
                    "id": node_id,
                    "region_id": rid,
                    "type": etype,
                    "source": _emergency_source_label(etype),
                    "comm_mode": comm_by_type[etype],
                    "status": "deployed",
                    "role": "patch" if network_mode == "with_residual" else "access",
                }
            )
            node_id += 1

    residual_count = sum(1 for n in nodes if n.get("source") == "residual_bs")
    emergency_count = len(nodes) - residual_count
    return nodes, residual_count, emergency_count


def _build_phases(phased: Dict[str, Any]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for phase in phased.get("phases", []):
        for step in phase.get("steps", []):
            entry: Dict[str, Any] = {
                "step": step.get("step"),
                "phase_id": phase.get("phase_id"),
                "phase_name": phase.get("phase_name"),
                "action": step.get("action"),
                "layer": step.get("layer"),
                "description": step.get("description"),
            }
            if step.get("output"):
                entry["output"] = step["output"]
            if step.get("core_rule"):
                entry["core_rule"] = step["core_rule"]
            flat.append(entry)
    return flat


def build_network_plan(
    scenario_id: str,
    network_mode: str,
    *,
    placement_seed: Optional[int] = None,
    progress: float = 0.92,
) -> Dict[str, Any]:
    """Build complete network plan for scenario x mode combination."""
    scenario = load_scenario(scenario_id)
    arch = load_architecture()
    mode_data = load_network_mode(network_mode)
    phased = load_phased_deploy()
    rl_info = parse_rl_output(scenario_id)

    n_regions = SCENARIO_N_REGIONS.get(scenario_id, DEFAULT_N_REGIONS)

    mode_config = mode_data["mode_config"]
    deploy_rules = mode_data["deploy_rules"]
    topology_template = mode_data["topology_template"]

    nodes, residual_count, emergency_count = _build_nodes(
        scenario, mode_config, deploy_rules, network_mode, n_regions
    )
    phases = _build_phases(phased)
    comm_modes_used = get_comm_mode_ids(arch)

    if network_mode == "with_residual":
        primary_backhaul = mode_config.get("primary_backhaul", "5G_residual_link")
        deploy_priority = mode_config.get("deploy_priority", [])
    else:
        primary_backhaul = mode_config.get("primary_backhaul", "Satellite_Ka")
        deploy_priority = mode_config.get("deploy_priority", [])

    business_tasks = arch["L3"].get("business_tasks", [])
    regional_tasks = split_tasks(n_regions, nodes, business_tasks)

    n_l2 = max(1, n_regions // 5)

    plan: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scenario_id": scenario_id,
        "scenario_name": scenario.get("scenario_name", scenario_id),
        "network_mode": network_mode,
        "network_mode_name": mode_config.get("mode_name", network_mode),
        "n_regions": n_regions,
        "n_l2": n_l2,
        "n_l3": n_regions,
        "architecture": {
            "L1": arch["L1"].get("layer_name", "L1"),
            "L2": arch["L2"].get("layer_name", "L2"),
            "L3": arch["L3"].get("layer_name", "L3"),
        },
        "scenario_params": {
            "disaster_type": scenario.get("disaster_type"),
            "residual_pattern": scenario.get("residual_pattern"),
            "base_station_outage": [
                scenario.get("base_station_outage_min"),
                scenario.get("base_station_outage_max"),
            ],
            "road_pass_rate": scenario.get("road_pass_rate"),
        },
        "primary_backhaul": primary_backhaul,
        "deploy_priority": deploy_priority,
        "residual_nodes_reused": residual_count,
        "emergency_nodes_deployed": emergency_count,
        "comm_modes_used": comm_modes_used,
        "phases": phases,
        "nodes": nodes,
        "links": [],
        "topology_pattern": topology_template.get("topology_pattern"),
        "topology": {},
        "regional_tasks": regional_tasks,
        "rl_enhancement": rl_info,
    }

    if scenario.get("link_breakage_rate") is not None:
        plan["scenario_params"]["link_breakage_rate"] = scenario["link_breakage_rate"]
    if scenario.get("local_blackout_zones") is not None:
        plan["scenario_params"]["local_blackout_zones"] = scenario["local_blackout_zones"]

    from .grid_placement import assign_grid_placements

    seed = placement_seed if placement_seed is not None else hash((scenario_id, network_mode)) % (2**31)
    assign_grid_placements(plan, placement_seed=int(seed), progress=float(progress))
    topology = build_topology(plan, topology_template, scenario)
    plan["topology"] = topology
    plan["links"] = topology.get("links", [])

    return plan
