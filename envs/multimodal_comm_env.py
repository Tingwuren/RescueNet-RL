"""Environment that couples multi-modal communication and broadcast resources.

Grid positions represent region-grid cells mapped to real-world bounds, and per-user observations
encode normalized row/column plus region-id semantics instead of free-form coordinates."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from data.resource_dataset import BaseStationProfile, RegionGrid, ResourceDataset, RewardProfile


class MultiModalCommEnv(gym.Env):
    """RL environment for joint communication/broadcast resource orchestration."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        dataset_path: str = "data/scenarios.json",
        scenario_name: str = "typhoon_residual",
        reward_mode: Optional[str] = None,
        max_base_stations: int = 10,
        coverage_reward: float = 1.0,
        bandwidth_reward: float = 0.05,
        broadcast_reward: float = 0.5,
        invalid_action_penalty: float = 0.3,
        demand_penalty: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dataset = ResourceDataset(dataset_path)
        self.scenario = self.dataset.get(scenario_name)
        self.reward_profile: RewardProfile = self.scenario.get_reward_profile(reward_mode)
        self.max_base_stations = max_base_stations
        self.coverage_reward = coverage_reward
        self.bandwidth_reward = bandwidth_reward
        self.broadcast_reward = broadcast_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.demand_penalty = demand_penalty
        self.reward_mode = self.reward_profile.key
        base_profiles = self.scenario.base_station_profiles or {}
        self.base_station_profiles = base_profiles
        self.max_base_throughput = max((profile.max_throughput for profile in base_profiles.values()), default=1.0)
        self.max_device_cost = max((profile.device_cost for profile in base_profiles.values()), default=1.0)
        self.max_bandwidth_cost_per_step = max(
            (profile.bandwidth_cost * profile.max_throughput for profile in base_profiles.values()),
            default=1.0,
        )
        self.max_total_demand = max(1.0, self.scenario.num_users * 40.0)
        self.target_users_per_station = max(1.0, self.scenario.num_users / max(1.0, self.max_base_stations))
        self.custom_base_station_specs: Optional[List[Dict[str, Any]]] = None
        self.residual_base_summary: List[Dict[str, Any]] = []

        self.np_random, _ = seeding.np_random(seed)
        self.region_grid: RegionGrid = self.scenario.region_grid
        self.grid_rows = self.region_grid.rows
        self.grid_cols = self.region_grid.cols
        self.grid_size = max(self.grid_rows, self.grid_cols, self.scenario.grid_size)
        self.num_users = self.scenario.num_users
        self.candidate_sites = self.scenario.candidate_sites
        self.max_steps = self.scenario.max_steps
        self.communication_modes = list(self.scenario.communication_modes)
        self.broadcast_modes = list(self.scenario.broadcast_modes)
        self.num_comm_modes = len(self.communication_modes)
        self.num_broadcast_modes = len(self.broadcast_modes)
        self.user_feature_dim = 6

        self.action_space = spaces.Discrete(
            self.candidate_sites * self.num_comm_modes * self.num_broadcast_modes
        )

        # Observation packs user state, deployment masks, mode/broadcast metrics and scalar context.
        obs_len = (
            self.num_users * self.user_feature_dim
            + self.candidate_sites * self.num_comm_modes
            + self.candidate_sites * self.num_broadcast_modes
            + self.num_comm_modes * 3
            + self.num_broadcast_modes * 3
            + 6
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_len,),
            dtype=np.float32,
        )

        self.seed_value = seed
        self.candidate_locations = self._generate_candidate_locations()
        self._init_state_containers()

    def _init_state_containers(self) -> None:
        self.user_positions = np.zeros((self.num_users, 2), dtype=np.int32)
        self.user_demands = np.zeros(self.num_users, dtype=np.float32)
        self.user_region_ids = np.zeros(self.num_users, dtype=np.int32)
        self.user_connected = np.zeros(self.num_users, dtype=bool)
        self.broadcast_served = np.zeros(self.num_users, dtype=bool)
        self.deployment_mask = np.zeros((self.candidate_sites, self.num_comm_modes), dtype=bool)
        self.broadcast_mask = np.zeros((self.candidate_sites, self.num_broadcast_modes), dtype=bool)
        self.mode_utilization = np.zeros(self.num_comm_modes, dtype=np.float32)
        self.broadcast_utilization = np.zeros(self.num_broadcast_modes, dtype=np.float32)
        self.remaining_budget = float(self.max_base_stations)
        self.current_step = 0
        self.current_time_idx = 0
        self.total_served_throughput = 0.0
        self.total_served_users = 0
        self._latest_throughput = 0.0
        self._latest_device_cost = 0.0
        self._latest_bandwidth_cost = 0.0
        self._latest_reward_breakdown: Dict[str, float] = {}
        self.residual_base_summary = []

    def _generate_candidate_locations(self) -> np.ndarray:
        coords: List[Tuple[int, int]] = []
        seen = set()
        while len(coords) < self.candidate_sites:
            candidate = (
                int(self.np_random.integers(0, self.grid_rows)),
                int(self.np_random.integers(0, self.grid_cols)),
            )
            if candidate in seen:
                continue
            seen.add(candidate)
            coords.append(candidate)
        return np.array(coords, dtype=np.int32)

    def _sample_users(self) -> None:
        clusters = self.scenario.user_clusters
        if not clusters:
            # Uniform fallback.
            rows = self.np_random.integers(0, self.grid_rows, size=(self.num_users,), dtype=np.int32)
            cols = self.np_random.integers(0, self.grid_cols, size=(self.num_users,), dtype=np.int32)
            self.user_positions = np.stack([rows, cols], axis=1)
            self.user_region_ids = np.array(
                [self.region_grid.cell_index(int(r), int(c)) for r, c in self.user_positions], dtype=np.int32
            )
            base_demand = 10.0
            self.user_demands = base_demand + self.np_random.normal(0, 2.0, size=self.num_users)
            self.user_demands = np.clip(self.user_demands, 2.0, 30.0).astype(np.float32)
            return

        weights = np.array([cluster["density"] for cluster in clusters], dtype=np.float32)
        weights = weights / weights.sum()
        for idx in range(self.num_users):
            choice = int(self.np_random.choice(len(clusters), p=weights))
            center = np.array(clusters[choice]["center"], dtype=np.float32)
            radius = float(clusters[choice].get("radius", 2.0))
            jitter = self.np_random.normal(0.0, radius * 0.4, size=2)
            point = center + jitter
            point = np.clip(point, [0.0, 0.0], [self.grid_rows - 1, self.grid_cols - 1])
            row, col = point.astype(np.int32)
            self.user_positions[idx] = (row, col)
            self.user_region_ids[idx] = self.region_grid.cell_index(int(row), int(col))
            demand = float(clusters[choice].get("demand_mbps", 10.0))
            demand += float(self.np_random.normal(0.0, demand * 0.15))
            self.user_demands[idx] = np.clip(demand, 2.0, 40.0)

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        per_site_options = self.num_comm_modes * self.num_broadcast_modes
        site_idx = action // per_site_options
        rem = action % per_site_options
        broadcast_idx = rem // self.num_comm_modes
        comm_idx = rem % self.num_comm_modes
        if site_idx >= self.candidate_sites:
            raise ValueError("Decoded site_idx exceeds candidate sites.")
        return site_idx, comm_idx, broadcast_idx

    def _get_time_snapshot(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        idx = min(self.current_time_idx, len(self.scenario.time_series) - 1)
        record = self.scenario.time_series[idx]
        mode_snapshot = {
            mode: (
                (record.mode_metrics.get(mode).available_bandwidth if record.mode_metrics.get(mode) else 0.0),
                (record.mode_metrics.get(mode).availability if record.mode_metrics.get(mode) else 0.0),
            )
            for mode in self.communication_modes
        }
        broadcast_snapshot = {
            mode: (
                (record.broadcast_metrics.get(mode).available_bandwidth if record.broadcast_metrics.get(mode) else 0.0),
                (record.broadcast_metrics.get(mode).coverage if record.broadcast_metrics.get(mode) else 0.0),
            )
            for mode in self.broadcast_modes
        }
        return mode_snapshot, broadcast_snapshot

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        del options
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        self._init_state_containers()
        self._sample_users()
        if self.custom_base_station_specs is not None:
            if self.custom_base_station_specs:
                self._apply_residual_base_stations(self.custom_base_station_specs)
        elif self.scenario.has_residual_network:
            residual_fraction = 0.25
            num_residual = int(self.num_users * residual_fraction)
            indices = self.np_random.choice(self.num_users, size=num_residual, replace=False)
            self.user_connected[indices] = True
            self.broadcast_served[indices] = True
        observation = self._get_observation()
        info = self._info_dict()
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space.")

        terminated = False
        truncated = False
        info: Dict[str, float] = {}

        if self.remaining_budget <= 0:
            self._latest_throughput = 0.0
            self._latest_device_cost = 0.0
            self._latest_bandwidth_cost = 0.0
            reward = -self.invalid_action_penalty
            self._latest_reward_breakdown = {
                "invalid_action_penalty": -self.invalid_action_penalty,
                "total": reward,
            }
            truncated = True
            info["reason"] = "budget_exhausted"
            observation = self._get_observation()
            info.update(self._info_dict())
            return observation, reward, terminated, truncated, info

        site_idx, comm_idx, broadcast_idx = self._decode_action(action)
        reward = 0.0
        if self.deployment_mask[site_idx, comm_idx]:
            reward = -self.invalid_action_penalty
            self._latest_throughput = 0.0
            self._latest_device_cost = 0.0
            self._latest_bandwidth_cost = 0.0
            self._latest_reward_breakdown = {
                "invalid_action_penalty": -self.invalid_action_penalty,
                "total": reward,
            }
        else:
            self.deployment_mask[site_idx, comm_idx] = True
            self.broadcast_mask[site_idx, broadcast_idx] = True
            self.remaining_budget -= 1.0
            mode_effect = self._deploy_comm(site_idx, comm_idx)
            broadcast_effect = self._activate_broadcast(site_idx, broadcast_idx)
            demand_gap = max(0.0, mode_effect["requested_demand"] - mode_effect["served_demand"])
            reward = self._compute_reward(mode_effect, broadcast_effect, demand_gap)

        self.current_step += 1
        self.current_time_idx += 1

        coverage_ratio = self._coverage_ratio()
        broadcast_ratio = self._broadcast_ratio()
        if coverage_ratio >= 0.999 and broadcast_ratio >= 0.9:
            terminated = True
            info["reason"] = "all_users_served"
            completion_bonus = (
                self.coverage_reward * self.reward_profile.coverage_weight
                + self.broadcast_reward * self.reward_profile.broadcast_weight
            )
            reward += completion_bonus
            self._latest_reward_breakdown["completion_bonus"] = float(completion_bonus)
            self._latest_reward_breakdown["total"] = float(reward)
        elif self.current_step >= self.max_steps:
            truncated = True
            info["reason"] = "max_steps"

        observation = self._get_observation()
        info.update(self._info_dict())
        return observation, reward, terminated, truncated, info

    def _deploy_comm(self, site_idx: int, comm_idx: int) -> Dict[str, float]:
        mode_name = self.communication_modes[comm_idx]
        location = self.candidate_locations[site_idx]
        profile = self.scenario.mode_profiles.get(mode_name, {})
        base_station: Optional[BaseStationProfile] = self.scenario.get_base_station_for_mode(mode_name)
        coverage_radius = float(profile.get("coverage_radius", 3.0))
        mode_snapshot, _ = self._get_time_snapshot()
        available_bw, availability = mode_snapshot[mode_name]
        dynamic_capacity = max(0.0, available_bw * availability)
        throughput_cap = base_station.max_throughput if base_station else float(profile.get("max_bandwidth", dynamic_capacity))
        capacity = min(dynamic_capacity, throughput_cap)
        max_users = int(base_station.max_users if base_station else profile.get("max_users", self.num_users))
        if max_users <= 0:
            max_users = self.num_users

        distances = np.linalg.norm(self.user_positions - location, axis=1)
        candidate_indices = np.where((distances <= coverage_radius) & (~self.user_connected))[0]
        served_mask = np.zeros(self.num_users, dtype=bool)
        requested = 0.0
        served = 0.0
        avg_throughput = 0.0
        newly_connected = 0

        if candidate_indices.size:
            if candidate_indices.size > max_users:
                selected = self.np_random.choice(candidate_indices, size=max_users, replace=False)
            else:
                selected = candidate_indices
            demands = self.user_demands[selected]
            requested = float(demands.sum())
            allocations = np.zeros_like(demands)
            if requested > 0 and capacity > 0:
                served = min(capacity, requested)
                fraction = served / requested if requested > 0 else 0.0
                allocations = demands * fraction
                avg_throughput = float(allocations.mean()) if allocations.size else 0.0
            served_mask[selected] = allocations > 0.0
            if served_mask.any():
                prev_connected = self.user_connected.copy()
                self.user_connected |= served_mask
                newly_connected = int(np.logical_and(~prev_connected, served_mask).sum())

        per_device_cost = base_station.device_cost if base_station else float(profile.get("device_cost", 0.0))
        bandwidth_cost = (base_station.bandwidth_cost if base_station else float(profile.get("bandwidth_cost", 0.0))) * served
        utilization_base = throughput_cap if throughput_cap > 0 else float(profile.get("max_bandwidth", capacity) + 1e-6)
        self.mode_utilization[comm_idx] = min(1.0, self.mode_utilization[comm_idx] + (served / (utilization_base + 1e-6)))
        self.total_served_throughput += served
        self.total_served_users += newly_connected
        self._latest_throughput = served
        self._latest_device_cost = per_device_cost
        self._latest_bandwidth_cost = bandwidth_cost

        return {
            "newly_connected": float(newly_connected) / max(1, self.num_users),
            "newly_connected_users": float(newly_connected),
            "requested_demand": requested,
            "served_demand": served,
            "avg_throughput": avg_throughput,
            "device_cost": per_device_cost,
            "bandwidth_cost": bandwidth_cost,
        }

    def _compute_reward(self, mode_effect: Dict[str, float], broadcast_effect: float, demand_gap: float) -> float:
        weights = self.reward_profile
        newly_connected_users = mode_effect.get("newly_connected_users", 0.0)
        new_broadcast_users = broadcast_effect * self.num_users
        coverage_progress = newly_connected_users / self.target_users_per_station
        broadcast_progress = new_broadcast_users / self.target_users_per_station
        served_norm = mode_effect.get("served_demand", 0.0) / max(1.0, self.max_base_throughput)
        throughput_norm = mode_effect.get("avg_throughput", 0.0) / max(1.0, self.max_base_throughput)
        coverage_term = self.coverage_reward * weights.coverage_weight * coverage_progress
        broadcast_term = self.broadcast_reward * weights.broadcast_weight * broadcast_progress
        device_cost_term = weights.device_cost_weight * (
            mode_effect.get("device_cost", 0.0) / max(1.0, self.max_device_cost)
        )
        bw_cost_term = weights.bandwidth_cost_weight * (
            mode_effect.get("bandwidth_cost", 0.0) / max(1.0, self.max_bandwidth_cost_per_step)
        )
        requested_demand = mode_effect.get("requested_demand", 0.0)
        demand_penalty = self.demand_penalty * (demand_gap / max(1.0, requested_demand))
        bandwidth_term = self.bandwidth_reward * weights.bandwidth_weight * served_norm
        throughput_term = weights.throughput_weight * throughput_norm
        reward = coverage_term + bandwidth_term + throughput_term + broadcast_term - device_cost_term - bw_cost_term - demand_penalty
        self._latest_reward_breakdown = {
            "coverage_term": float(coverage_term),
            "broadcast_term": float(broadcast_term),
            "bandwidth_term": float(bandwidth_term),
            "throughput_term": float(throughput_term),
            "device_cost_term": float(-device_cost_term),
            "bandwidth_cost_term": float(-bw_cost_term),
            "demand_penalty": float(-demand_penalty),
            "total": float(reward),
        }
        return reward

    def _activate_broadcast(self, site_idx: int, broadcast_idx: int) -> float:
        broadcast_name = self.broadcast_modes[broadcast_idx]
        _, broadcast_snapshot = self._get_time_snapshot()
        available_bw, coverage_ratio = broadcast_snapshot[broadcast_name]
        location = self.candidate_locations[site_idx]
        grid_span = max(self.grid_rows, self.grid_cols)
        reach = coverage_ratio * (grid_span / 2.0)
        distances = np.linalg.norm(self.user_positions - location, axis=1)
        coverage_mask = (distances <= reach) & (~self.broadcast_served)
        new_served = int(coverage_mask.sum())
        if new_served > 0:
            self.broadcast_served[coverage_mask] = True
        utilization = min(1.0, available_bw / (self.scenario.broadcast_profiles.get(broadcast_name, {}).get("max_bandwidth", available_bw) + 1e-6))
        self.broadcast_utilization[broadcast_idx] = max(self.broadcast_utilization[broadcast_idx], utilization)
        return float(new_served) / max(1, self.num_users)

    def _apply_residual_base_stations(self, specs: List[Dict[str, Any]]) -> None:
        summaries: List[Dict[str, Any]] = []
        for spec in specs:
            base_key = spec.get("base_station")
            mode_name = spec.get("mode")
            if not base_key or not mode_name:
                continue
            base_profile = self.base_station_profiles.get(base_key)
            if not base_profile:
                continue
            mode_profile = self.scenario.mode_profiles.get(mode_name, {})
            coverage_radius = float(mode_profile.get("coverage_radius", 3.0))
            location = np.array([spec["x"], spec["y"]], dtype=np.float32)
            distances = np.linalg.norm(self.user_positions - location, axis=1)
            coverage_mask = distances <= coverage_radius
            covered_users = int(np.count_nonzero(coverage_mask))
            if covered_users > 0:
                self.user_connected[coverage_mask] = True
            summaries.append(
                {
                    "base_station": base_profile.name,
                    "label": base_profile.label,
                    "mode": mode_name,
                    "x": spec["x"],
                    "y": spec["y"],
                    "connected_users": covered_users,
                    "coverage_radius": coverage_radius,
                }
            )
        self.residual_base_summary = summaries

    def _sanitize_base_station_spec(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not entry:
            return None
        base_key = str(entry.get("base_station", "")).strip()
        if not base_key or base_key not in self.base_station_profiles:
            return None
        profile = self.base_station_profiles[base_key]
        supported_modes = list(profile.supported_modes)
        mode_name = entry.get("mode")
        if mode_name not in supported_modes:
            mode_name = supported_modes[0] if supported_modes else None
        if not mode_name:
            return None
        x = int(np.clip(entry.get("x", 0), 0, self.grid_rows - 1))
        y = int(np.clip(entry.get("y", 0), 0, self.grid_cols - 1))
        return {
            "base_station": base_key,
            "mode": mode_name,
            "x": x,
            "y": y,
        }

    def set_custom_base_stations(self, base_stations: Optional[List[Dict[str, Any]]]) -> None:
        """Configure custom residual base-station deployments."""
        if base_stations is None:
            self.custom_base_station_specs = None
            return
        sanitized: List[Dict[str, Any]] = []
        for entry in base_stations:
            spec = self._sanitize_base_station_spec(entry)
            if spec:
                sanitized.append(spec)
        self.custom_base_station_specs = sanitized

    def _coverage_ratio(self) -> float:
        return float(self.user_connected.mean()) if self.user_connected.size else 0.0

    def _broadcast_ratio(self) -> float:
        return float(self.broadcast_served.mean()) if self.broadcast_served.size else 0.0

    def _info_dict(self) -> Dict[str, Any]:
        return {
            "coverage_ratio": self._coverage_ratio(),
            "broadcast_ratio": self._broadcast_ratio(),
            "remaining_budget": float(self.remaining_budget),
            "avg_user_throughput": self._average_user_throughput(),
            "recent_throughput": float(self._latest_throughput),
            "reward_mode": self.reward_profile.key,
            "reward_label": self.reward_profile.label,
            "device_cost": float(self._latest_device_cost),
            "bandwidth_cost": float(self._latest_bandwidth_cost),
            "reward_breakdown": dict(self._latest_reward_breakdown),
            "residual_base_stations": list(self.residual_base_summary),
        }

    def _average_user_throughput(self) -> float:
        if self.total_served_users <= 0:
            return 0.0
        return float(self.total_served_throughput / max(1, self.total_served_users))

    def _get_observation(self) -> np.ndarray:
        mode_snapshot, broadcast_snapshot = self._get_time_snapshot()
        row_max = max(1, self.grid_rows - 1)
        col_max = max(1, self.grid_cols - 1)
        region_max = max(1, self.region_grid.cell_count - 1)
        user_features = np.zeros((self.num_users, self.user_feature_dim), dtype=np.float32)
        user_features[:, 0] = np.clip(self.user_positions[:, 0] / row_max, 0.0, 1.0)
        user_features[:, 1] = np.clip(self.user_positions[:, 1] / col_max, 0.0, 1.0)
        user_features[:, 2] = np.clip(self.user_region_ids / region_max, 0.0, 1.0)
        user_features[:, 3] = np.clip(self.user_demands / 40.0, 0.0, 1.0)
        user_features[:, 4] = self.user_connected.astype(np.float32)
        user_features[:, 5] = self.broadcast_served.astype(np.float32)

        deploy_state = self.deployment_mask.astype(np.float32).flatten()
        broadcast_state = self.broadcast_mask.astype(np.float32).flatten()

        mode_features = []
        for idx, mode in enumerate(self.communication_modes):
            max_bw = float(self.scenario.mode_profiles.get(mode, {}).get("max_bandwidth", 1.0))
            available_bw, availability = mode_snapshot[mode]
            utilization = self.mode_utilization[idx]
            mode_features.extend(
                [
                    np.clip(available_bw / max(1.0, max_bw), 0.0, 1.0),
                    np.clip(availability, 0.0, 1.0),
                    np.clip(utilization, 0.0, 1.0),
                ]
            )

        broadcast_features = []
        for idx, mode in enumerate(self.broadcast_modes):
            max_bw = float(self.scenario.broadcast_profiles.get(mode, {}).get("max_bandwidth", 1.0))
            available_bw, coverage = broadcast_snapshot[mode]
            utilization = self.broadcast_utilization[idx]
            broadcast_features.extend(
                [
                    np.clip(available_bw / max(1.0, max_bw), 0.0, 1.0),
                    np.clip(coverage, 0.0, 1.0),
                    np.clip(utilization, 0.0, 1.0),
                ]
            )

        avg_throughput_norm = np.clip(self._average_user_throughput() / max(1.0, self.max_base_throughput), 0.0, 1.0)
        recent_throughput_norm = np.clip(self._latest_throughput / max(1.0, self.max_base_throughput), 0.0, 1.0)
        scalars = np.array(
            [
                float(self.scenario.has_residual_network),
                self.remaining_budget / max(1.0, self.max_base_stations),
                self.current_step / max(1.0, self.max_steps),
                self._coverage_ratio(),
                avg_throughput_norm,
                recent_throughput_norm,
            ],
            dtype=np.float32,
        )

        obs = np.concatenate(
            [
                user_features.flatten(),
                deploy_state,
                broadcast_state,
                np.array(mode_features, dtype=np.float32),
                np.array(broadcast_features, dtype=np.float32),
                scalars,
            ],
            dtype=np.float32,
        )
        return obs

    def apply_custom_user_state(self, users: List[Dict[str, float]]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Override sampled users with externally provided positions/demands."""
        if not users:
            observation = self._get_observation()
            info = self._info_dict()
            return observation, info

        self.total_served_throughput = 0.0
        self.total_served_users = 0
        self._latest_throughput = 0.0
        self._latest_device_cost = 0.0
        self._latest_bandwidth_cost = 0.0
        limit = min(len(users), self.num_users)
        self.user_connected[:] = False
        self.broadcast_served[:] = False

        for idx in range(limit):
            entry = users[idx]
            x = int(np.clip(entry.get("x", self.user_positions[idx, 0]), 0, self.grid_rows - 1))
            y = int(np.clip(entry.get("y", self.user_positions[idx, 1]), 0, self.grid_cols - 1))
            demand = float(entry.get("demand", self.user_demands[idx]))
            connected = bool(entry.get("connected", False))
            broadcast = bool(entry.get("broadcast_served", False))
            self.user_positions[idx] = (x, y)
            self.user_region_ids[idx] = self.region_grid.cell_index(int(x), int(y))
            self.user_demands[idx] = np.clip(demand, 0.5, 100.0)
            self.user_connected[idx] = connected
            self.broadcast_served[idx] = broadcast

        if self.custom_base_station_specs is not None:
            self.residual_base_summary = []
            if self.custom_base_station_specs:
                self._apply_residual_base_stations(self.custom_base_station_specs)

        observation = self._get_observation()
        info = self._info_dict()
        return observation, info

    def render(self) -> None:
        print(
            f"[MultiModalRender] step={self.current_step} coverage={self._coverage_ratio():.2%} "
            f"broadcast={self._broadcast_ratio():.2%} remaining_budget={self.remaining_budget:.1f}"
        )

    def close(self) -> None:
        return
