"""Hierarchical MARL planner for emergency communication resource allocation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEVICE_TYPES = [
    "emergency_5g_bs",
    "portable_broadcast_gateway",
    "cellular_relay",
    "mesh_relay",
    "communication_uav",
]

DEVICE_LABELS = {
    "emergency_5g_bs": "5G emergency base station",
    "portable_broadcast_gateway": "portable broadcast gateway",
    "cellular_relay": "5G relay",
    "mesh_relay": "Mesh relay",
    "communication_uav": "communication UAV",
}

LINK_TYPES = ["none", "satellite_backhaul", "microwave_relay", "uav_relay"]


@dataclass
class RegionStats:
    """Aggregated state used by L1/L2 agents."""

    region_id: int
    row_band: int
    col_band: int
    total_users: int
    high_priority_users: int
    disconnected_users: int
    broadcast_missing_users: int
    demand_sum: float
    mean_demand: float
    concentration: float
    severity: float
    terrain_complexity: float
    road_passability: float
    power_recovery: float
    rescue_progress: float
    residual_bandwidth: float
    resource_score: float
    neighbor_resource_state: float
    secondary_risk: float

    @property
    def high_priority_ratio(self) -> float:
        return self.high_priority_users / max(1, self.total_users)

    @property
    def disconnected_ratio(self) -> float:
        return self.disconnected_users / max(1, self.total_users)

    @property
    def broadcast_missing_ratio(self) -> float:
        return self.broadcast_missing_users / max(1, self.total_users)


class HierarchicalMARLPlanner:
    """Rule-guided CTDE planner matching the L1/L2/L3 design document.

    The neural HMARL policy learns the final discrete deployment action, while this
    planner supplies hierarchy-aware action priors, 32-dim L3 state features,
    L1 quotas, L2 migrations/links, and the 72-dim executable L3 vector used by
    backend reports.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.region_rows = int(cfg.get("region_rows", 3))
        self.region_cols = int(cfg.get("region_cols", 3))
        self.action_prior_scale = float(cfg.get("action_prior_scale", 1.0))
        self.coverage_gain_weight = float(cfg.get("coverage_gain_weight", 3.4))
        self.broadcast_gain_weight = float(cfg.get("broadcast_gain_weight", 1.2))
        self.throughput_gain_weight = float(cfg.get("throughput_gain_weight", 0.8))
        self.l1_priority_weight = float(cfg.get("l1_priority_weight", 1.8))
        self.mode_score_weight = float(cfg.get("mode_score_weight", 1.0))
        self.broadcast_score_weight = float(cfg.get("broadcast_score_weight", 0.7))
        self.quota_signal_weight = float(cfg.get("quota_signal_weight", 0.6))
        self.action_gain_weight = float(cfg.get("action_gain_weight", 2.5))
        self.site_score_weight = float(cfg.get("site_score_weight", 0.4))
        self.probe_top_k = max(0, int(cfg.get("probe_top_k", 0)))
        self.probe_score_weight = float(cfg.get("probe_score_weight", 0.0))
        self.probe_coverage_weight = float(cfg.get("probe_coverage_weight", 4.0))
        self.probe_broadcast_weight = float(cfg.get("probe_broadcast_weight", 2.0))
        self.probe_reward_weight = float(cfg.get("probe_reward_weight", 0.25))

    @property
    def region_count(self) -> int:
        return max(1, self.region_rows * self.region_cols)

    def build_action_prior(self, env) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return per-action prior scores and the full hierarchy plan."""
        plan = self.build_plan(env)
        scores = np.zeros(int(env.action_space.n), dtype=np.float32)
        valid_mask = env.get_action_mask() if hasattr(env, "get_action_mask") else None
        priority = np.asarray(plan["l1"]["priority_scores"], dtype=np.float32)
        if priority.size == 0:
            priority = np.ones(self.region_count, dtype=np.float32)
        priority = priority / max(1e-6, float(priority.max()))

        mode_scores = self._mode_scores(env)
        broadcast_scores = self._broadcast_scores(env)
        quotas = np.asarray(plan["l1"]["quotas"], dtype=np.float32)
        site_region_ids = plan["site_region_ids"]
        action_gain_scores = self._action_gain_scores(env)

        per_site_options = int(env.num_comm_modes * env.num_broadcast_modes)
        for action in range(int(env.action_space.n)):
            site_idx = action // per_site_options
            rem = action % per_site_options
            broadcast_idx = rem // int(env.num_comm_modes)
            comm_idx = rem % int(env.num_comm_modes)
            region_id = int(site_region_ids[site_idx]) if site_idx < len(site_region_ids) else 0
            device_idx = self._device_index_for_mode(env.communication_modes[comm_idx])
            quota_signal = 1.0 if quotas[region_id, device_idx] > 0 else 0.0
            scores[action] = (
                self.l1_priority_weight * priority[min(region_id, len(priority) - 1)]
                + self.mode_score_weight * mode_scores[comm_idx]
                + self.broadcast_score_weight * broadcast_scores[broadcast_idx]
                + self.quota_signal_weight * quota_signal
                + self.action_gain_weight * action_gain_scores[action]
                + self.site_score_weight * plan["site_scores"][site_idx]
            )

        if valid_mask is not None:
            valid_mask = np.asarray(valid_mask, dtype=bool)
            scores = np.where(valid_mask, scores, -1e9).astype(np.float32)

        finite = np.isfinite(scores) & (scores > -1e8)
        if finite.any() and self.probe_top_k > 0 and self.probe_score_weight > 0.0:
            scores = self._apply_probe_scores(env, scores, finite)

        finite = np.isfinite(scores) & (scores > -1e8)
        if finite.any():
            mean = float(scores[finite].mean())
            std = float(scores[finite].std() + 1e-6)
            scores[finite] = ((scores[finite] - mean) / std) * self.action_prior_scale
        plan["recommended_action"] = int(np.argmax(scores)) if finite.any() else 0
        return scores, plan

    def _apply_probe_scores(self, env, scores: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
        candidate_actions = np.flatnonzero(finite_mask)
        if candidate_actions.size == 0:
            return scores
        top_k = min(self.probe_top_k, int(candidate_actions.size))
        ranked = candidate_actions[np.argsort(scores[candidate_actions])[-top_k:]]
        probe_scores = np.zeros_like(scores, dtype=np.float32)

        for action in ranked.tolist():
            try:
                probe = deepcopy(env)
                _, reward, _, _, info = probe.step(int(action))
            except Exception:  # pragma: no cover - planner must remain usable for any env-like object
                continue
            probe_scores[int(action)] = (
                self.probe_coverage_weight * float(info.get("coverage_ratio", 0.0))
                + self.probe_broadcast_weight * float(info.get("broadcast_ratio", 0.0))
                + self.probe_reward_weight * float(reward)
            )

        active = probe_scores[ranked] > 0
        if np.any(active):
            active_scores = probe_scores[ranked][active]
            min_score = float(active_scores.min())
            max_score = float(active_scores.max())
            if max_score > min_score:
                probe_scores[ranked] = (probe_scores[ranked] - min_score) / (max_score - min_score)
            scores[ranked] = scores[ranked] + self.probe_score_weight * probe_scores[ranked]
        return scores.astype(np.float32)

    def build_plan(self, env) -> Dict[str, Any]:
        """Build L1 quotas, L2 coordination, and L3 executable plans."""
        regions = self._collect_region_stats(env)
        site_region_ids, site_scores = self._site_region_scores(env, regions)
        l1 = self._plan_l1(env, regions)
        l2 = self._plan_l2(env, regions, l1["quotas"])
        l3 = self._plan_l3(env, regions, l1["quotas"], l2, site_region_ids, site_scores)
        rewards = self._hierarchical_rewards(env, regions, l1, l2, l3)
        target_region = int(l1["target_region_id"])
        return {
            "algorithm": "hmarl",
            "region_shape": [self.region_rows, self.region_cols],
            "regions": [self._region_public_dict(region) for region in regions],
            "site_region_ids": site_region_ids,
            "site_scores": site_scores,
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "rewards": rewards,
            "l1_target": target_region,
            "l2_target": int(l2.get("dominant_link_type_index", 0)),
            "summary": self._summary(regions, l1, l2, l3, rewards),
        }

    def _collect_region_stats(self, env) -> List[RegionStats]:
        rows = int(getattr(env, "grid_rows", getattr(env, "grid_size", 1)))
        cols = int(getattr(env, "grid_cols", getattr(env, "grid_size", 1)))
        region_count = self.region_count
        positions = np.asarray(getattr(env, "user_positions", np.zeros((0, 2))), dtype=np.int32)
        demands = np.asarray(getattr(env, "user_demands", np.zeros((len(positions),))), dtype=np.float32)
        connected = np.asarray(getattr(env, "user_connected", np.zeros((len(positions),), dtype=bool)), dtype=bool)
        broadcast = np.asarray(getattr(env, "broadcast_served", np.zeros((len(positions),), dtype=bool)), dtype=bool)
        demand_threshold = float(np.quantile(demands, 0.75)) if demands.size else 0.0

        region_user_indices: List[List[int]] = [[] for _ in range(region_count)]
        for idx, pos in enumerate(positions):
            region_user_indices[self._region_id(int(pos[0]), int(pos[1]), rows, cols)].append(idx)

        severity = self._scenario_severity(env)
        terrain = self._terrain_complexity(env)
        progress = float(getattr(env, "current_step", 0)) / max(1.0, float(getattr(env, "max_steps", 1)))
        residual_bw = self._residual_bandwidth(env)
        secondary_risk = float(np.clip(severity * (1.0 - progress) + 0.12 * (1.0 - self._coverage(env)), 0.0, 1.0))

        regions: List[RegionStats] = []
        resource_scores = []
        for region_id, indices in enumerate(region_user_indices):
            row_band = region_id // self.region_cols
            col_band = region_id % self.region_cols
            region_demands = demands[indices] if indices else np.zeros((0,), dtype=np.float32)
            total_users = len(indices)
            high_priority = int(np.count_nonzero(region_demands >= demand_threshold)) if indices else 0
            disconnected = int(np.count_nonzero(~connected[indices])) if indices else 0
            broadcast_missing = int(np.count_nonzero(~broadcast[indices])) if indices else 0
            concentration = self._region_concentration(positions[indices], rows, cols) if indices else 0.0
            road = float(np.clip(1.0 - 0.45 * severity - 0.25 * terrain + 0.35 * progress, 0.05, 1.0))
            power = float(np.clip(0.2 + 0.6 * progress + 0.2 * self._coverage(env), 0.0, 1.0))
            resource_score = float(np.clip((residual_bw / 300.0) + self._coverage(env) * 0.5, 0.0, 1.0))
            resource_scores.append(resource_score)
            regions.append(
                RegionStats(
                    region_id=region_id,
                    row_band=row_band,
                    col_band=col_band,
                    total_users=total_users,
                    high_priority_users=high_priority,
                    disconnected_users=disconnected,
                    broadcast_missing_users=broadcast_missing,
                    demand_sum=float(region_demands.sum()) if indices else 0.0,
                    mean_demand=float(region_demands.mean()) if indices else 0.0,
                    concentration=concentration,
                    severity=severity,
                    terrain_complexity=terrain,
                    road_passability=road,
                    power_recovery=power,
                    rescue_progress=progress,
                    residual_bandwidth=residual_bw,
                    resource_score=resource_score,
                    neighbor_resource_state=0.0,
                    secondary_risk=secondary_risk,
                )
            )

        for region in regions:
            neighbors = self._neighbor_ids(region.region_id)
            if neighbors:
                region.neighbor_resource_state = float(np.mean([resource_scores[n] for n in neighbors]))
        return regions

    def _plan_l1(self, env, regions: List[RegionStats]) -> Dict[str, Any]:
        inventory = self._inventory(env)
        priorities = np.asarray([self._priority_score(region) for region in regions], dtype=np.float32)
        if not np.isfinite(priorities).all() or float(priorities.sum()) <= 0:
            priorities = np.ones(len(regions), dtype=np.float32)
        weights = priorities / float(priorities.sum())
        quotas = np.zeros((len(regions), len(DEVICE_TYPES)), dtype=np.int32)
        for device_idx, count in enumerate(inventory):
            quotas[:, device_idx] = self._allocate_integer(count, weights)

        target_region = int(np.argmax(priorities)) if len(priorities) else 0
        allocated = quotas.sum(axis=0)
        hard_constraint_ok = bool(np.all(allocated <= np.asarray(inventory, dtype=np.int32)))
        return {
            "inventory": inventory,
            "quotas": quotas.tolist(),
            "priority_scores": priorities.astype(float).round(4).tolist(),
            "target_region_id": target_region,
            "hard_constraint_ok": hard_constraint_ok,
            "device_types": DEVICE_TYPES,
        }

    def _plan_l2(self, env, regions: List[RegionStats], quotas: List[List[int]]) -> Dict[str, Any]:
        quota_arr = np.asarray(quotas, dtype=np.float32)
        capacity_weights = np.asarray([180.0, 80.0, 120.0, 90.0, 70.0], dtype=np.float32)
        demand = np.asarray([region.demand_sum for region in regions], dtype=np.float32)
        demand_norm = demand / max(1.0, float(demand.max()) if demand.size else 1.0)
        capacity = quota_arr @ capacity_weights
        capacity_norm = capacity / max(1.0, float(capacity.max()) if capacity.size else 1.0)
        gaps = np.clip(demand_norm - 0.72 * capacity_norm, -1.0, 1.0)
        surplus_ids = [int(idx) for idx in np.argsort(gaps) if gaps[idx] < -0.08]
        deficit_ids = [int(idx) for idx in np.argsort(-gaps) if gaps[idx] > 0.08]

        migrations: List[Dict[str, Any]] = []
        for target_id in deficit_ids[:3]:
            source_id = self._best_source_region(target_id, surplus_ids, regions)
            if source_id is None:
                continue
            move = self._migration_vector(quota_arr[source_id])
            if sum(move) <= 0:
                continue
            migrations.append(
                {
                    "source_region": int(source_id),
                    "target_region": int(target_id),
                    "devices": move,
                }
            )

        links: List[Dict[str, Any]] = []
        dominant_link_type_index = 0
        for target_id in deficit_ids[:4]:
            peer_id = self._best_link_peer(target_id, regions, gaps)
            if peer_id is None:
                continue
            link_type = self._link_type(regions[target_id])
            dominant_link_type_index = max(dominant_link_type_index, LINK_TYPES.index(link_type))
            links.append(
                {
                    "region_a": int(peer_id),
                    "region_b": int(target_id),
                    "link_type": link_type,
                    "deployment_position": self._region_center_cell(env, target_id),
                }
            )

        return {
            "resource_gaps": gaps.astype(float).round(4).tolist(),
            "migrations": migrations,
            "links": links,
            "dominant_link_type_index": int(dominant_link_type_index),
            "coordination_flag": True,
        }

    def _plan_l3(
        self,
        env,
        regions: List[RegionStats],
        quotas: List[List[int]],
        l2: Dict[str, Any],
        site_region_ids: List[int],
        site_scores: List[float],
    ) -> Dict[str, Any]:
        quotas_arr = np.asarray(quotas, dtype=np.int32)
        gaps = np.asarray(l2.get("resource_gaps", []), dtype=np.float32)
        if gaps.size:
            target_region = int(np.argmax(gaps))
        else:
            target_region = int(np.argmax([self._priority_score(region) for region in regions]))
        target_region = int(np.clip(target_region, 0, len(regions) - 1))
        target_quota = quotas_arr[target_region].copy()
        for migration in l2.get("migrations", []):
            devices = np.asarray(migration.get("devices", [0] * len(DEVICE_TYPES)), dtype=np.int32)
            if int(migration.get("target_region", -1)) == target_region:
                target_quota += devices
            if int(migration.get("source_region", -1)) == target_region:
                target_quota = np.maximum(0, target_quota - devices)
        if int(target_quota.sum()) <= 0 and int(quotas_arr.sum()) > 0:
            source_region = int(np.argmax(quotas_arr.sum(axis=1)))
            donor_device = int(np.argmax(quotas_arr[source_region]))
            target_quota[donor_device] = 1

        local_sites = [
            site_idx for site_idx, region_id in enumerate(site_region_ids) if int(region_id) == target_region
        ]
        if not local_sites:
            local_sites = list(range(min(12, len(site_region_ids))))
        local_sites = sorted(local_sites, key=lambda idx: site_scores[idx], reverse=True)[:12]
        action_matrix = np.zeros((12, len(DEVICE_TYPES)), dtype=np.float32)
        for device_idx, count in enumerate(target_quota.tolist()):
            for n in range(max(0, int(count))):
                slot = n % 12
                action_matrix[slot, device_idx] = min(5.0, action_matrix[slot, device_idx] + 1.0)

        region = regions[target_region]
        power = np.clip(0.75 + 0.55 * region.severity - 0.15 * region.terrain_complexity, 0.5, 1.5)
        bandwidth = self._bandwidth_ratios(target_quota)
        params = []
        for idx in range(len(DEVICE_TYPES)):
            params.extend([float(power), float(bandwidth[idx])])
        schedule = [
            float(np.clip(0.45 + 0.5 * region.high_priority_ratio + 0.2 * region.severity, 0.0, 1.0)),
            float(np.clip(max([abs(float(g)) for g in gaps], default=0.0) * 0.25, 0.0, 0.3)),
        ]
        action_72 = action_matrix.flatten().tolist() + params + schedule
        topology = self._topology(env, target_region, local_sites, action_matrix, params, l2)
        return {
            "target_region_id": target_region,
            "target_region_state_32": self._state_32(region).round(4).tolist(),
            "action_72": [round(float(v), 4) for v in action_72],
            "deployment_matrix": action_matrix.astype(int).tolist(),
            "device_params": [round(float(v), 4) for v in params],
            "schedule": [round(float(v), 4) for v in schedule],
            "topology": topology,
        }

    def _hierarchical_rewards(
        self,
        env,
        regions: List[RegionStats],
        l1: Dict[str, Any],
        l2: Dict[str, Any],
        l3: Dict[str, Any],
    ) -> Dict[str, float]:
        quotas = np.asarray(l1["quotas"], dtype=np.float32)
        priorities = np.asarray(l1["priority_scores"], dtype=np.float32)
        inventory = np.asarray(l1["inventory"], dtype=np.float32)
        target_mask = priorities >= np.quantile(priorities, 0.66) if priorities.size else np.zeros(0, dtype=bool)
        priority_reward = float(np.mean(quotas[target_mask].sum(axis=1) > 0)) if target_mask.any() else 1.0
        fairness = 1.0 - self._gini(quotas.sum(axis=1) / np.maximum(1.0, np.asarray([r.total_users for r in regions])))
        efficiency = float(np.clip(quotas.sum() / max(1.0, inventory.sum()), 0.0, 1.0))
        global_penalty = 0.0 if l1.get("hard_constraint_ok", False) else 1.0
        r_l1 = 0.5 * priority_reward + 0.3 * fairness + 0.2 * efficiency - global_penalty

        gaps = np.asarray(l2.get("resource_gaps", []), dtype=np.float32)
        balance = float(1.0 - np.clip(np.std(np.clip(gaps, 0.0, 1.0)), 0.0, 1.0)) if gaps.size else 1.0
        connectivity = float(np.clip(len(l2.get("links", [])) / max(1, int(np.count_nonzero(gaps > 0.08))), 0.0, 1.0)) if gaps.size else 1.0
        reuse = float(np.clip(self._residual_bandwidth(env) / 300.0, 0.0, 1.0))
        migrated = sum(sum(item.get("devices", [])) for item in l2.get("migrations", []))
        migration_penalty = float(migrated / max(1.0, inventory.sum()))
        r_l2 = 0.4 * balance + 0.4 * connectivity + 0.2 * reuse - 0.1 * migration_penalty
        r_l2_final = 0.7 * r_l2 + 0.3 * r_l1

        coverage = self._coverage(env)
        broadcast = self._broadcast(env)
        target_region = regions[int(l3["target_region_id"])]
        high_sat = float(np.clip(coverage + 0.35 * target_region.high_priority_ratio, 0.0, 1.0))
        stage = float(getattr(env, "current_step", 0)) / max(1.0, float(getattr(env, "max_steps", 1)))
        alpha = 0.3 + 0.2 * stage
        beta = 0.4 - 0.2 * stage
        gamma = 0.2 + 0.1 * stage
        deployed = float(np.asarray(l3["deployment_matrix"], dtype=np.float32).sum())
        cost = deployed / max(1.0, inventory.sum())
        constraint_penalty = 1.0 if target_region.road_passability < 0.2 and deployed > 0 else 0.0
        r_l3 = alpha * coverage + beta * broadcast + gamma * high_sat - 0.1 * cost - 10.0 * constraint_penalty
        r_l3_final = 0.8 * r_l3 + 0.2 * r_l2_final
        return {
            "l1": round(float(r_l1), 4),
            "l2": round(float(r_l2_final), 4),
            "l3": round(float(r_l3), 4),
            "l3_final": round(float(r_l3_final), 4),
        }

    def _action_gain_scores(self, env) -> np.ndarray:
        """Estimate immediate L3 deployment utility for every executable action."""
        action_count = int(getattr(env, "action_space").n)
        scores = np.zeros(action_count, dtype=np.float32)
        locations = np.asarray(getattr(env, "candidate_locations", np.zeros((0, 2))), dtype=np.float32)
        positions = np.asarray(getattr(env, "user_positions", np.zeros((0, 2))), dtype=np.float32)
        if locations.size == 0 or positions.size == 0:
            return scores

        connected = np.asarray(getattr(env, "user_connected", np.zeros((len(positions),), dtype=bool)), dtype=bool)
        broadcast_served = np.asarray(
            getattr(env, "broadcast_served", np.zeros((len(positions),), dtype=bool)), dtype=bool
        )
        demands = np.asarray(getattr(env, "user_demands", np.zeros((len(positions),))), dtype=np.float32)
        mode_snapshot, broadcast_snapshot = env._get_time_snapshot() if hasattr(env, "_get_time_snapshot") else ({}, {})
        communication_modes = list(getattr(env, "communication_modes", []))
        broadcast_modes = list(getattr(env, "broadcast_modes", []))
        num_comm_modes = max(1, int(getattr(env, "num_comm_modes", len(communication_modes) or 1)))
        num_broadcast_modes = max(1, int(getattr(env, "num_broadcast_modes", len(broadcast_modes) or 1)))
        per_site_options = num_comm_modes * num_broadcast_modes
        target_users = float(getattr(env, "target_users_per_station", max(1.0, len(positions))))
        max_throughput = float(max(1.0, getattr(env, "max_base_throughput", 1.0)))
        max_device_cost = float(max(1.0, getattr(env, "max_device_cost", 1.0)))
        rows = int(getattr(env, "grid_rows", getattr(env, "grid_size", 1)))
        cols = int(getattr(env, "grid_cols", getattr(env, "grid_size", 1)))
        grid_span = float(max(1, rows, cols))
        scenario = getattr(env, "scenario", None)

        for site_idx, location in enumerate(locations):
            distances = np.linalg.norm(positions - location, axis=1)
            broadcast_gains: List[float] = []
            for broadcast_name in broadcast_modes:
                _, broadcast_coverage = broadcast_snapshot.get(broadcast_name, (0.0, 0.0))
                reach = max(0.0, float(broadcast_coverage)) * (grid_span / 2.0)
                new_broadcast = int(np.count_nonzero((distances <= reach) & (~broadcast_served)))
                broadcast_gains.append(float(new_broadcast) / max(1.0, target_users))

            for comm_idx, mode_name in enumerate(communication_modes):
                profile = getattr(scenario, "mode_profiles", {}).get(mode_name, {}) if scenario else {}
                base_station = scenario.get_base_station_for_mode(mode_name) if scenario else None
                coverage_radius = float(profile.get("coverage_radius", 3.0))
                available_bw, availability = mode_snapshot.get(mode_name, (0.0, 0.0))
                dynamic_capacity = max(0.0, float(available_bw) * float(availability))
                throughput_cap = (
                    float(base_station.max_throughput)
                    if base_station is not None
                    else float(profile.get("max_bandwidth", dynamic_capacity))
                )
                capacity = min(dynamic_capacity, throughput_cap)
                max_users = int(base_station.max_users if base_station is not None else profile.get("max_users", len(positions)))
                if max_users <= 0:
                    max_users = len(positions)
                device_cost = float(base_station.device_cost if base_station is not None else profile.get("device_cost", 0.0))

                candidate_indices = np.flatnonzero((distances <= coverage_radius) & (~connected))
                newly_connected = 0
                served_demand = 0.0
                if capacity > 0.0 and candidate_indices.size:
                    if candidate_indices.size > max_users:
                        order = np.lexsort((-demands[candidate_indices], distances[candidate_indices]))
                        selected = candidate_indices[order[:max_users]]
                    else:
                        selected = candidate_indices
                    newly_connected = int(selected.size)
                    served_demand = min(capacity, float(demands[selected].sum()))

                coverage_gain = float(newly_connected) / max(1.0, target_users)
                throughput_gain = float(served_demand) / max_throughput
                cost_penalty = device_cost / max_device_cost
                for broadcast_idx in range(num_broadcast_modes):
                    action = site_idx * per_site_options + broadcast_idx * num_comm_modes + comm_idx
                    if action >= action_count:
                        continue
                    broadcast_gain = broadcast_gains[broadcast_idx] if broadcast_idx < len(broadcast_gains) else 0.0
                    scores[action] = (
                        self.coverage_gain_weight * coverage_gain
                        + self.broadcast_gain_weight * broadcast_gain
                        + self.throughput_gain_weight * throughput_gain
                        - 0.12 * cost_penalty
                    )

        finite = np.isfinite(scores)
        if finite.any():
            min_score = float(scores[finite].min())
            max_score = float(scores[finite].max())
            if max_score > min_score:
                scores[finite] = (scores[finite] - min_score) / (max_score - min_score)
            else:
                scores[finite] = 0.0
        return scores.astype(np.float32)

    def _site_region_scores(self, env, regions: List[RegionStats]) -> Tuple[List[int], List[float]]:
        rows = int(getattr(env, "grid_rows", getattr(env, "grid_size", 1)))
        cols = int(getattr(env, "grid_cols", getattr(env, "grid_size", 1)))
        locations = np.asarray(getattr(env, "candidate_locations", np.zeros((0, 2))), dtype=np.int32)
        priorities = np.asarray([self._priority_score(region) for region in regions], dtype=np.float32)
        if priorities.size:
            priorities = priorities / max(1e-6, float(priorities.max()))
        site_region_ids: List[int] = []
        site_scores: List[float] = []
        for row, col in locations:
            region_id = self._region_id(int(row), int(col), rows, cols)
            site_region_ids.append(region_id)
            label_bonus = 0.15 if self._is_labeled_cell(env, int(row), int(col)) else 0.0
            site_scores.append(float(np.clip(priorities[region_id] + label_bonus, 0.0, 1.2)))
        return site_region_ids, site_scores

    def _state_32(self, region: RegionStats) -> np.ndarray:
        users = np.array(
            [
                np.clip(region.total_users / 2000.0, 0.0, 1.0),
                region.high_priority_ratio,
                np.clip(region.mean_demand / 40.0, 0.0, 1.0),
                region.concentration,
                np.clip(region.high_priority_users / 500.0, 0.0, 1.0),
                np.clip(region.disconnected_users / 2000.0, 0.0, 1.0),
                np.clip((region.total_users - region.disconnected_users) / 2000.0, 0.0, 1.0),
                region.disconnected_ratio,
            ],
            dtype=np.float32,
        )
        resources = np.array(
            [
                region.resource_score,
                min(1.0, region.resource_score * 0.9),
                np.clip(region.residual_bandwidth / 300.0, 0.0, 1.0),
                min(1.0, region.resource_score * 0.7),
                min(1.0, region.resource_score * 0.45),
                min(1.0, region.resource_score * 0.55),
                np.clip(region.residual_bandwidth / 200.0, 0.0, 1.0),
                region.resource_score,
            ],
            dtype=np.float32,
        )
        devices = np.array(
            [
                np.clip(region.resource_score, 0.0, 1.0),
                np.clip(region.broadcast_missing_ratio, 0.0, 1.0),
                np.clip(region.neighbor_resource_state, 0.0, 1.0),
                np.clip(1.0 - region.terrain_complexity, 0.0, 1.0),
                np.clip(region.power_recovery, 0.0, 1.0),
                np.clip(1.0 - region.terrain_complexity * 0.4, 0.0, 1.0),
                np.clip(region.severity * 0.35 + region.terrain_complexity * 0.2, 0.0, 1.0),
                np.clip(1.0 - region.road_passability, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        env_features = np.array(
            [
                region.severity,
                region.terrain_complexity,
                region.road_passability,
                region.power_recovery,
                region.rescue_progress,
                region.neighbor_resource_state,
                region.secondary_risk,
                region.rescue_progress,
            ],
            dtype=np.float32,
        )
        return np.concatenate([users, resources, devices, env_features]).astype(np.float32)

    def _topology(
        self,
        env,
        target_region: int,
        local_sites: List[int],
        action_matrix: np.ndarray,
        params: List[float],
        l2: Dict[str, Any],
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        locations = np.asarray(getattr(env, "candidate_locations", np.zeros((0, 2))), dtype=np.int32)
        for slot_idx, site_idx in enumerate(local_sites[:12]):
            if site_idx >= len(locations):
                continue
            row, col = int(locations[site_idx][0]), int(locations[site_idx][1])
            for device_idx, count in enumerate(action_matrix[slot_idx].astype(int).tolist()):
                if count <= 0:
                    continue
                nodes.append(
                    {
                        "site_index": int(site_idx),
                        "grid": [row, col],
                        "device_type": DEVICE_TYPES[device_idx],
                        "label": DEVICE_LABELS[DEVICE_TYPES[device_idx]],
                        "count": int(count),
                        "power_ratio": float(params[device_idx * 2]),
                        "bandwidth_ratio": float(params[device_idx * 2 + 1]),
                    }
                )
        return {
            "target_region_id": int(target_region),
            "nodes": nodes,
            "links": list(l2.get("links", [])),
        }

    def _summary(
        self,
        regions: List[RegionStats],
        l1: Dict[str, Any],
        l2: Dict[str, Any],
        l3: Dict[str, Any],
        rewards: Dict[str, float],
    ) -> Dict[str, Any]:
        target = int(l3.get("target_region_id", l1.get("target_region_id", 0)))
        return {
            "target_region_id": target,
            "target_users": regions[target].total_users if 0 <= target < len(regions) else 0,
            "l1_allocated_devices": int(np.asarray(l1["quotas"]).sum()),
            "l2_migration_count": len(l2.get("migrations", [])),
            "l2_link_count": len(l2.get("links", [])),
            "l3_deployed_devices": int(np.asarray(l3["deployment_matrix"]).sum()),
            "hierarchical_reward": rewards.get("l3_final", 0.0),
        }

    def _region_public_dict(self, region: RegionStats) -> Dict[str, Any]:
        return {
            "region_id": region.region_id,
            "row_band": region.row_band,
            "col_band": region.col_band,
            "total_users": region.total_users,
            "high_priority_ratio": round(region.high_priority_ratio, 4),
            "demand_sum": round(region.demand_sum, 4),
            "severity": round(region.severity, 4),
            "road_passability": round(region.road_passability, 4),
            "resource_score": round(region.resource_score, 4),
        }

    def _inventory(self, env) -> List[int]:
        total = int(round(float(getattr(env, "remaining_budget", getattr(env, "max_base_stations", 1)))))
        total = max(1, total)
        weights = np.asarray([0.3, 0.22, 0.18, 0.16, 0.14], dtype=np.float32)
        return self._allocate_integer(total, weights).tolist()

    def _allocate_integer(self, total: int, weights: np.ndarray) -> np.ndarray:
        total = max(0, int(total))
        if total == 0:
            return np.zeros_like(weights, dtype=np.int32)
        weights = np.asarray(weights, dtype=np.float32)
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights, dtype=np.float32)
        weights = weights / float(weights.sum())
        raw = weights * total
        allocation = np.floor(raw).astype(np.int32)
        remainder = total - int(allocation.sum())
        if remainder > 0:
            order = np.argsort(-(raw - allocation))
            for idx in order[:remainder]:
                allocation[idx] += 1
        return allocation

    def _migration_vector(self, quota_row: np.ndarray) -> List[int]:
        move = np.zeros(len(DEVICE_TYPES), dtype=np.int32)
        if quota_row.sum() <= 1:
            return move.tolist()
        donor_order = np.argsort(-quota_row)
        for idx in donor_order[:2]:
            if quota_row[idx] > 1:
                move[idx] = 1
        return move.tolist()

    def _mode_scores(self, env) -> np.ndarray:
        mode_snapshot, _ = env._get_time_snapshot() if hasattr(env, "_get_time_snapshot") else ({}, {})
        scores = []
        for mode in getattr(env, "communication_modes", []):
            available, availability = mode_snapshot.get(mode, (0.0, 0.0))
            profile = getattr(env, "scenario", None).mode_profiles.get(mode, {}) if getattr(env, "scenario", None) else {}
            max_bw = float(profile.get("max_bandwidth", max(1.0, available)))
            scores.append(float(np.clip((available / max(1.0, max_bw)) * availability, 0.0, 1.0)))
        return np.asarray(scores or [1.0], dtype=np.float32)

    def _broadcast_scores(self, env) -> np.ndarray:
        _, broadcast_snapshot = env._get_time_snapshot() if hasattr(env, "_get_time_snapshot") else ({}, {})
        stage = float(getattr(env, "current_step", 0)) / max(1.0, float(getattr(env, "max_steps", 1)))
        scores = []
        for mode in getattr(env, "broadcast_modes", []):
            available, coverage = broadcast_snapshot.get(mode, (0.0, 0.0))
            scores.append(float(np.clip(0.65 * coverage + 0.35 * (available / 150.0) + 0.2 * (1.0 - stage), 0.0, 1.0)))
        return np.asarray(scores or [1.0], dtype=np.float32)

    def _bandwidth_ratios(self, quota: np.ndarray) -> np.ndarray:
        weights = quota.astype(np.float32) + np.asarray([1.2, 1.0, 0.85, 0.75, 0.7], dtype=np.float32)
        weights = weights / max(1e-6, float(weights.sum()))
        return weights.astype(np.float32)

    def _region_id(self, row: int, col: int, rows: int, cols: int) -> int:
        row_band = min(self.region_rows - 1, int(row * self.region_rows / max(1, rows)))
        col_band = min(self.region_cols - 1, int(col * self.region_cols / max(1, cols)))
        return int(row_band * self.region_cols + col_band)

    def _neighbor_ids(self, region_id: int) -> List[int]:
        row = region_id // self.region_cols
        col = region_id % self.region_cols
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.region_rows and 0 <= nc < self.region_cols:
                neighbors.append(nr * self.region_cols + nc)
        return neighbors

    def _best_source_region(self, target_id: int, surplus_ids: List[int], regions: List[RegionStats]) -> Optional[int]:
        if not surplus_ids:
            return None
        neighbors = set(self._neighbor_ids(target_id))
        ordered = sorted(
            surplus_ids,
            key=lambda idx: (
                0 if idx in neighbors else 1,
                abs(regions[idx].row_band - regions[target_id].row_band)
                + abs(regions[idx].col_band - regions[target_id].col_band),
            ),
        )
        return int(ordered[0]) if ordered else None

    def _best_link_peer(self, target_id: int, regions: List[RegionStats], gaps: np.ndarray) -> Optional[int]:
        candidates = [idx for idx in range(len(regions)) if idx != target_id]
        if not candidates:
            return None
        return int(
            min(
                candidates,
                key=lambda idx: (
                    max(0.0, float(gaps[idx])),
                    abs(regions[idx].row_band - regions[target_id].row_band)
                    + abs(regions[idx].col_band - regions[target_id].col_band),
                ),
            )
        )

    def _link_type(self, region: RegionStats) -> str:
        if region.terrain_complexity > 0.65 or region.road_passability < 0.35:
            return "satellite_backhaul"
        if region.severity > 0.72:
            return "uav_relay"
        return "microwave_relay"

    def _region_center_cell(self, env, region_id: int) -> List[int]:
        rows = int(getattr(env, "grid_rows", getattr(env, "grid_size", 1)))
        cols = int(getattr(env, "grid_cols", getattr(env, "grid_size", 1)))
        row_band = region_id // self.region_cols
        col_band = region_id % self.region_cols
        row = int((row_band + 0.5) * rows / max(1, self.region_rows))
        col = int((col_band + 0.5) * cols / max(1, self.region_cols))
        return [int(np.clip(row, 0, rows - 1)), int(np.clip(col, 0, cols - 1))]

    def _device_index_for_mode(self, mode: str) -> int:
        lower = str(mode).lower()
        if "wifi" in lower or "broadcast" in lower:
            return 1
        if "satellite" in lower:
            return 2
        if "mesh" in lower:
            return 3
        if "uav" in lower:
            return 4
        return 0 if "5g" in lower or "cell" in lower else 3

    def _scenario_severity(self, env) -> float:
        scenario = getattr(env, "scenario", None)
        disaster_type = getattr(scenario, "disaster_type", "") if scenario else ""
        base = {"earthquake": 0.82, "flood": 0.74, "typhoon": 0.68}.get(disaster_type, 0.65)
        outage = 1.0 - self._coverage(env)
        broadcast_gap = 1.0 - self._broadcast(env)
        no_residual = 0.15 if not getattr(scenario, "has_residual_network", False) else 0.0
        return float(np.clip(0.45 * base + 0.35 * outage + 0.2 * broadcast_gap + no_residual, 0.0, 1.0))

    def _terrain_complexity(self, env) -> float:
        scenario = getattr(env, "scenario", None)
        disaster_type = getattr(scenario, "disaster_type", "") if scenario else ""
        return {"earthquake": 0.82, "flood": 0.46, "typhoon": 0.38}.get(disaster_type, 0.5)

    def _residual_bandwidth(self, env) -> float:
        if not getattr(getattr(env, "scenario", None), "has_residual_network", False):
            return 0.0
        mode_snapshot, _ = env._get_time_snapshot() if hasattr(env, "_get_time_snapshot") else ({}, {})
        return float(sum(available * availability for available, availability in mode_snapshot.values()))

    def _coverage(self, env) -> float:
        return float(env._coverage_ratio()) if hasattr(env, "_coverage_ratio") else 0.0

    def _broadcast(self, env) -> float:
        return float(env._broadcast_ratio()) if hasattr(env, "_broadcast_ratio") else 0.0

    def _region_concentration(self, positions: np.ndarray, rows: int, cols: int) -> float:
        if len(positions) <= 1:
            return 1.0 if len(positions) else 0.0
        span = np.array([max(1, rows), max(1, cols)], dtype=np.float32)
        normalized = positions.astype(np.float32) / span
        spread = float(np.mean(np.std(normalized, axis=0)))
        return float(np.clip(1.0 - spread * 3.0, 0.0, 1.0))

    def _priority_score(self, region: RegionStats) -> float:
        return float(
            0.28 * region.severity
            + 0.24 * region.disconnected_ratio
            + 0.18 * region.high_priority_ratio
            + 0.14 * region.broadcast_missing_ratio
            + 0.1 * np.clip(region.mean_demand / 40.0, 0.0, 1.0)
            + 0.06 * (1.0 - region.road_passability)
        )

    def _is_labeled_cell(self, env, row: int, col: int) -> bool:
        grid = getattr(env, "region_grid", None)
        if not grid:
            return False
        label = grid.cell_label(row, col)
        return bool(label and not label.startswith("cell-"))

    def _gini(self, values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0 or float(arr.sum()) <= 0:
            return 0.0
        arr = np.sort(arr)
        n = arr.size
        index = np.arange(1, n + 1, dtype=np.float32)
        return float((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr) + 1e-6))
