"""L3 grid placement: assign each plan node to a subregion grid cell and coordinates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

N_GRIDS = 12
N_DEVICE_SLOTS = 5
GRID_ROWS = 3
GRID_COLS = 4

# plan_builder node type -> L3 deployment matrix row (emergency devices only)
EMERGENCY_TYPE_TO_SLOT: Dict[str, int] = {
    "emergency_bs": 0,
    "portable_gateway": 1,
    "relay_5g": 2,
    "mesh_relay": 3,
    "comm_uav": 4,
    "satellite_terminal": 4,
}

RESIDUAL_PREFERRED_GRIDS = (0, 1, 2)


def default_grid_centers() -> np.ndarray:
    """12 grid centers in subregion-normalized coordinates (3 rows × 4 columns)."""
    xs = np.linspace(0.1, 0.9, GRID_COLS)
    ys = np.linspace(0.1, 0.9, GRID_ROWS)
    centers: List[List[float]] = []
    for y in ys:
        for x in xs:
            centers.append([float(x), float(y)])
    return np.asarray(centers[:N_GRIDS], dtype=np.float32)


def grid_index_to_cell(grid_index: int) -> Dict[str, int]:
    g = int(grid_index) % N_GRIDS
    return {"row": g // GRID_COLS, "col": g % GRID_COLS}


def grid_index_to_label(grid_index: int) -> str:
    return f"G{int(grid_index) % N_GRIDS:02d}"


def _jitter_position(
    base: np.ndarray,
    grid_index: int,
    rng: np.random.Generator,
    *,
    spread: float = 0.04,
) -> Tuple[float, float]:
    """Small per-node offset so co-located devices do not share identical coordinates."""
    g = int(grid_index) % N_GRIDS
    angle = float(rng.uniform(0, 2 * np.pi))
    radius = float(rng.uniform(0, spread))
    x = float(np.clip(base[0] + radius * np.cos(angle), 0.02, 0.98))
    y = float(np.clip(base[1] + radius * np.sin(angle), 0.02, 0.98))
    return x, y


def _compute_emergency_deployment(
    quota_row: np.ndarray,
    progress: float,
    rng: np.random.Generator,
    *,
    network_mode: str,
    grid_offset: int = 0,
    grid_count: int = N_GRIDS,
) -> np.ndarray:
    """Allocate emergency devices across grids (L3-style, mirrors hierarchy_report._l3_process)."""
    n_slots = min(grid_count, N_GRIDS - grid_offset)
    deployment = np.zeros((N_DEVICE_SLOTS, n_slots), dtype=np.int32)
    p = float(np.clip(progress, 0.0, 1.0))
    focus = int(rng.integers(0, max(1, n_slots)))

    for j in range(N_DEVICE_SLOTS):
        available = int(quota_row[j]) if j < len(quota_row) else 0
        if available <= 0:
            continue
        if p < 0.35:
            placed = min(available, int(rng.integers(1, max(2, available + 1))))
            for _ in range(placed):
                g = int(rng.integers(0, n_slots))
                deployment[j, g] += 1
            continue
        user_dist = np.ones(n_slots) * 0.05
        user_dist[focus] += 0.35 + 0.4 * p
        if network_mode == "no_residual":
            # Wider spread when no residual anchor grids exist
            user_dist += rng.uniform(0.0, 0.08, size=n_slots)
        user_dist /= user_dist.sum()
        alloc = (user_dist * available).astype(int)
        remainder = available - int(alloc.sum())
        for k in range(int(remainder)):
            idx = int(np.argsort(-user_dist)[k % n_slots])
            alloc[idx] += 1
        deployment[j, :] = alloc

    full = np.zeros((N_DEVICE_SLOTS, N_GRIDS), dtype=np.int32)
    full[:, grid_offset : grid_offset + n_slots] = deployment
    return full


def _slots_from_deployment(deployment: np.ndarray) -> List[Tuple[int, int]]:
    """Flatten deployment matrix to (device_slot, grid_index) assignments."""
    slots: List[Tuple[int, int]] = []
    for j in range(deployment.shape[0]):
        for g in range(deployment.shape[1]):
            for _ in range(int(deployment[j, g])):
                slots.append((j, g))
    return slots


def _region_placement(
    region_nodes: List[Dict[str, Any]],
    *,
    network_mode: str,
    progress: float,
    rng: np.random.Generator,
) -> None:
    """Mutate nodes in place with grid_index, grid_cell, position."""
    centers = default_grid_centers()
    residuals = [n for n in region_nodes if n.get("type") == "residual_bs" or n.get("source") == "residual_bs"]
    emergencies = [n for n in region_nodes if n not in residuals]

    # Residual devices: anchor on G00–G02 when mode allows
    if network_mode == "with_residual":
        for i, node in enumerate(residuals):
            g = RESIDUAL_PREFERRED_GRIDS[i % len(RESIDUAL_PREFERRED_GRIDS)]
            _attach_grid(node, g, centers, rng)
    else:
        for node in residuals:
            g = int(rng.integers(0, N_GRIDS))
            _attach_grid(node, g, centers, rng)

    if not emergencies:
        return

    quota = np.zeros(N_DEVICE_SLOTS, dtype=np.int32)
    buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    unmapped: List[Dict[str, Any]] = []

    for node in emergencies:
        slot = EMERGENCY_TYPE_TO_SLOT.get(str(node.get("type", "")))
        if slot is None:
            unmapped.append(node)
            continue
        quota[slot] += 1
        buckets[slot].append(node)

    if network_mode == "with_residual":
        grid_offset = max(RESIDUAL_PREFERRED_GRIDS) + 1
        grid_count = N_GRIDS - grid_offset
    else:
        grid_offset = 0
        grid_count = N_GRIDS

    deployment = _compute_emergency_deployment(
        quota,
        progress,
        rng,
        network_mode=network_mode,
        grid_offset=grid_offset,
        grid_count=grid_count,
    )
    slot_pairs = _slots_from_deployment(deployment)

    assigned: set = set()
    for j, g in slot_pairs:
        pool = buckets.get(j) or []
        node = None
        while pool:
            candidate = pool.pop(0)
            if id(candidate) not in assigned:
                node = candidate
                break
        if node is None and unmapped:
            node = unmapped.pop(0)
        if node is None:
            continue
        assigned.add(id(node))
        _attach_grid(node, g, centers, rng)

    for node in emergencies:
        if id(node) not in assigned:
            g = int(rng.integers(grid_offset, N_GRIDS)) if grid_offset < N_GRIDS else 0
            _attach_grid(node, g, centers, rng)


def _attach_grid(
    node: Dict[str, Any],
    grid_index: int,
    centers: np.ndarray,
    rng: np.random.Generator,
) -> None:
    g = int(grid_index) % N_GRIDS
    base = centers[g]
    x, y = _jitter_position(base, g, rng)
    cell = grid_index_to_cell(g)
    node["grid_index"] = g
    node["grid_label"] = grid_index_to_label(g)
    node["grid_cell"] = cell
    node["position"] = {
        "x": round(x, 4),
        "y": round(y, 4),
        "coordinate_system": "subregion_normalized",
    }


def assign_grid_placements(
    plan: Dict[str, Any],
    *,
    placement_seed: int,
    progress: float = 0.92,
) -> Dict[str, Any]:
    """
    Assign each node in plan['nodes'] a grid_index and normalized (x,y) position.

    Coordinate convention (per subregion, ~10 km²):
      - grid_index: 0–11, row-major 3×4 (G00=row0,col0 … G11=row2,col3)
      - position.x, position.y: [0,1] within the subregion (1.0 ≈ 10 km edge)
    """
    nodes: List[Dict[str, Any]] = plan.get("nodes", [])
    network_mode = str(plan.get("network_mode", "with_residual"))
    rng = np.random.default_rng(int(placement_seed))

    by_region: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        by_region[int(node.get("region_id", 0))].append(node)

    for region_id in sorted(by_region):
        region_rng = np.random.default_rng(int(placement_seed) ^ (region_id * 0x9E3779B9))
        _region_placement(
            by_region[region_id],
            network_mode=network_mode,
            progress=float(progress),
            rng=region_rng,
        )

    plan["placement_schema"] = {
        "grid_layout": f"{GRID_ROWS}x{GRID_COLS}",
        "n_grids_per_subregion": N_GRIDS,
        "coordinate_system": "subregion_normalized",
        "grid_index_range": [0, N_GRIDS - 1],
        "position_unit": "fraction_of_subregion_edge",
        "subregion_extent_km": 10.0,
        "fields": ["grid_index", "grid_label", "grid_cell", "position"],
        "placement_seed": int(placement_seed),
        "progress": float(progress),
        "network_mode": network_mode,
    }
    return plan
