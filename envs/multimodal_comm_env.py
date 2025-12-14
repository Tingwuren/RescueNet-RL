"""Environment that couples multi-modal communication and broadcast resources."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from data.resource_dataset import ResourceDataset


class MultiModalCommEnv(gym.Env):
    """RL environment for joint communication/broadcast resource orchestration."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        dataset_path: str = "data/scenarios.json",
        scenario_name: str = "typhoon_residual",
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
        self.max_base_stations = max_base_stations
        self.coverage_reward = coverage_reward
        self.bandwidth_reward = bandwidth_reward
        self.broadcast_reward = broadcast_reward
        self.invalid_action_penalty = invalid_action_penalty
        self.demand_penalty = demand_penalty

        self.np_random, _ = seeding.np_random(seed)
        self.grid_size = self.scenario.grid_size
        self.num_users = self.scenario.num_users
        self.candidate_sites = self.scenario.candidate_sites
        self.max_steps = self.scenario.max_steps
        self.communication_modes = list(self.scenario.communication_modes)
        self.broadcast_modes = list(self.scenario.broadcast_modes)
        self.num_comm_modes = len(self.communication_modes)
        self.num_broadcast_modes = len(self.broadcast_modes)

        self.action_space = spaces.Discrete(
            self.candidate_sites * self.num_comm_modes * self.num_broadcast_modes
        )

        # Observation packs user state, deployment masks, mode/broadcast metrics and scalar context.
        obs_len = (
            self.num_users * 5
            + self.candidate_sites * self.num_comm_modes
            + self.candidate_sites * self.num_broadcast_modes
            + self.num_comm_modes * 3
            + self.num_broadcast_modes * 3
            + 4
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
        self.user_connected = np.zeros(self.num_users, dtype=bool)
        self.broadcast_served = np.zeros(self.num_users, dtype=bool)
        self.deployment_mask = np.zeros((self.candidate_sites, self.num_comm_modes), dtype=bool)
        self.broadcast_mask = np.zeros((self.candidate_sites, self.num_broadcast_modes), dtype=bool)
        self.mode_utilization = np.zeros(self.num_comm_modes, dtype=np.float32)
        self.broadcast_utilization = np.zeros(self.num_broadcast_modes, dtype=np.float32)
        self.remaining_budget = float(self.max_base_stations)
        self.current_step = 0
        self.current_time_idx = 0

    def _generate_candidate_locations(self) -> np.ndarray:
        coords: List[Tuple[int, int]] = []
        seen = set()
        while len(coords) < self.candidate_sites:
            candidate = (
                int(self.np_random.integers(0, self.grid_size)),
                int(self.np_random.integers(0, self.grid_size)),
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
            self.user_positions = self.np_random.integers(
                0, self.grid_size, size=(self.num_users, 2), dtype=np.int32
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
            point = np.clip(point, [0.0, 0.0], [self.grid_size - 1, self.grid_size - 1])
            self.user_positions[idx] = point.astype(np.int32)
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
        if self.scenario.has_residual_network:
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
            reward = -self.invalid_action_penalty
            truncated = True
            info["reason"] = "budget_exhausted"
            observation = self._get_observation()
            info.update(self._info_dict())
            return observation, reward, terminated, truncated, info

        site_idx, comm_idx, broadcast_idx = self._decode_action(action)
        reward = 0.0
        if self.deployment_mask[site_idx, comm_idx]:
            reward = -self.invalid_action_penalty
        else:
            self.deployment_mask[site_idx, comm_idx] = True
            self.broadcast_mask[site_idx, broadcast_idx] = True
            self.remaining_budget -= 1.0
            mode_effect = self._deploy_comm(site_idx, comm_idx)
            broadcast_effect = self._activate_broadcast(site_idx, broadcast_idx)
            demand_gap = max(0.0, mode_effect["requested_demand"] - mode_effect["served_demand"])
            reward = (
                self.coverage_reward * mode_effect["newly_connected"]
                + self.bandwidth_reward * mode_effect["served_demand"]
                + self.broadcast_reward * broadcast_effect
                - self.demand_penalty * demand_gap
            )

        self.current_step += 1
        self.current_time_idx += 1

        coverage_ratio = self._coverage_ratio()
        broadcast_ratio = self._broadcast_ratio()
        if coverage_ratio >= 0.999 and broadcast_ratio >= 0.9:
            terminated = True
            info["reason"] = "all_users_served"
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
        coverage_radius = float(profile.get("coverage_radius", 3.0))
        mode_snapshot, _ = self._get_time_snapshot()
        available_bw, availability = mode_snapshot[mode_name]
        capacity = max(0.0, available_bw * availability)

        distances = np.linalg.norm(self.user_positions - location, axis=1)
        coverage_mask = (distances <= coverage_radius) & (~self.user_connected)
        newly_connected = int(coverage_mask.sum())
        demanded = float(self.user_demands[coverage_mask].sum()) if newly_connected else 0.0
        served = min(capacity, demanded)
        if newly_connected:
            served_ratio = served / demanded if demanded > 0 else 0.0
            satisfied_mask = coverage_mask & (self.np_random.random(self.num_users) < served_ratio)
            self.user_connected[satisfied_mask] = True
        self.mode_utilization[comm_idx] = min(1.0, self.mode_utilization[comm_idx] + (served / (profile.get("max_bandwidth", capacity) + 1e-6)))
        return {
            "newly_connected": float(newly_connected) / max(1, self.num_users),
            "requested_demand": demanded,
            "served_demand": served,
        }

    def _activate_broadcast(self, site_idx: int, broadcast_idx: int) -> float:
        broadcast_name = self.broadcast_modes[broadcast_idx]
        _, broadcast_snapshot = self._get_time_snapshot()
        available_bw, coverage_ratio = broadcast_snapshot[broadcast_name]
        location = self.candidate_locations[site_idx]
        reach = coverage_ratio * (self.grid_size / 2.0)
        distances = np.linalg.norm(self.user_positions - location, axis=1)
        coverage_mask = (distances <= reach) & (~self.broadcast_served)
        new_served = int(coverage_mask.sum())
        if new_served > 0:
            self.broadcast_served[coverage_mask] = True
        utilization = min(1.0, available_bw / (self.scenario.broadcast_profiles.get(broadcast_name, {}).get("max_bandwidth", available_bw) + 1e-6))
        self.broadcast_utilization[broadcast_idx] = max(self.broadcast_utilization[broadcast_idx], utilization)
        return float(new_served) / max(1, self.num_users)

    def _coverage_ratio(self) -> float:
        return float(self.user_connected.mean()) if self.user_connected.size else 0.0

    def _broadcast_ratio(self) -> float:
        return float(self.broadcast_served.mean()) if self.broadcast_served.size else 0.0

    def _info_dict(self) -> Dict[str, float]:
        return {
            "coverage_ratio": self._coverage_ratio(),
            "broadcast_ratio": self._broadcast_ratio(),
            "remaining_budget": float(self.remaining_budget),
        }

    def _get_observation(self) -> np.ndarray:
        mode_snapshot, broadcast_snapshot = self._get_time_snapshot()
        grid_max = max(1, self.grid_size - 1)
        user_features = np.zeros((self.num_users, 5), dtype=np.float32)
        user_features[:, 0:2] = self.user_positions / grid_max
        user_features[:, 2] = np.clip(self.user_demands / 40.0, 0.0, 1.0)
        user_features[:, 3] = self.user_connected.astype(np.float32)
        user_features[:, 4] = self.broadcast_served.astype(np.float32)

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

        scalars = np.array(
            [
                float(self.scenario.has_residual_network),
                self.remaining_budget / max(1.0, self.max_base_stations),
                self.current_step / max(1.0, self.max_steps),
                self._coverage_ratio(),
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

    def render(self) -> None:
        print(
            f"[MultiModalRender] step={self.current_step} coverage={self._coverage_ratio():.2%} "
            f"broadcast={self._broadcast_ratio():.2%} remaining_budget={self.remaining_budget:.1f}"
        )

    def close(self) -> None:
        return
