"""Core Gymnasium environment for the disaster recovery cellular deployment task.

The environment models a simplified disaster region represented by a 2D grid. A
single centralized agent sequentially deploys identical mobile base stations to
restore service for disconnected users. The modeling assumptions are intentionally
minimal so that the environment can serve as a lightweight baseline for policy
experimentation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class DisasterCellularEnv(gym.Env):
    """Disaster-area mobile base station deployment environment.

    Observation (flattened vector):
        - Each user contributes (x_norm, y_norm, coverage_flag) => 3 * num_users values
        - Deployment mask for every candidate site (1 if a mobile base station is present)
        - Remaining budget fraction and normalized step index

    Action:
        - Discrete index selecting one of the pre-defined candidate deployment sites.

    Reward:
        reward_t = coverage_reward * N_newly_covered_users - deployment_cost
        Invalid actions (e.g., redeploying at the same site or exceeding the budget) incur
        a small negative penalty.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = 10,
        num_users: int = 35,
        candidate_sites: int = 20,
        max_steps: int = 20,
        initial_outage_fraction: float = 0.6,
        coverage_radius: float = 2.5,
        max_base_stations: int = 6,
        coverage_reward: float = 1.0,
        deployment_cost: float = 0.3,
        invalid_action_penalty: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if candidate_sites <= 0:
            raise ValueError("candidate_sites must be positive.")
        if candidate_sites > grid_size * grid_size:
            raise ValueError("candidate_sites cannot exceed number of grid cells.")
        if not (0.0 < initial_outage_fraction < 1.0):
            raise ValueError("initial_outage_fraction must be in (0, 1).")

        self.grid_size = grid_size
        self.num_users = num_users
        self.candidate_sites = candidate_sites
        self.max_steps = max_steps
        self.initial_outage_fraction = initial_outage_fraction
        self.coverage_radius = coverage_radius
        self.max_base_stations = max_base_stations
        self.coverage_reward = coverage_reward
        self.deployment_cost = deployment_cost
        self.invalid_action_penalty = invalid_action_penalty

        self.np_random, _ = seeding.np_random(seed)
        self.candidate_locations = self._generate_candidate_locations()

        obs_len = self.num_users * 3 + self.candidate_sites + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_len,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.candidate_sites)

        self.user_positions: np.ndarray
        self.user_connected: np.ndarray
        self.deployed_mask: np.ndarray
        self.current_step: int = 0
        self.remaining_budget: float = float(self.max_base_stations)

    def _generate_candidate_locations(self) -> np.ndarray:
        """Sample unique candidate grid coordinates."""
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

    def _generate_user_positions(self) -> np.ndarray:
        """Generate user coordinates on the grid."""
        return self.np_random.integers(
            0, self.grid_size, size=(self.num_users, 2), dtype=np.int32
        )

    def _get_observation(self) -> np.ndarray:
        """Assemble the flattened observation vector."""
        user_features = np.zeros((self.num_users, 3), dtype=np.float32)
        grid_max = max(1, self.grid_size - 1)
        user_features[:, 0] = self.user_positions[:, 0] / grid_max
        user_features[:, 1] = self.user_positions[:, 1] / grid_max
        user_features[:, 2] = self.user_connected.astype(np.float32)

        deployed = self.deployed_mask.astype(np.float32)
        budget_frac = np.array([self.remaining_budget / max(1, self.max_base_stations)], dtype=np.float32)
        step_frac = np.array([self.current_step / max(1, self.max_steps)], dtype=np.float32)

        obs = np.concatenate(
            [user_features.flatten(), deployed, budget_frac, step_frac],
            dtype=np.float32,
        )
        return obs

    def _deploy_and_cover(self, action: int) -> int:
        """Deploy a station at the chosen candidate and update user coverage."""
        location = self.candidate_locations[action]
        deltas = self.user_positions - location
        distances = np.linalg.norm(deltas, axis=1)
        newly_covered_mask = (distances <= self.coverage_radius) & (~self.user_connected)
        newly_covered = int(newly_covered_mask.sum())
        self.user_connected[newly_covered_mask] = True
        return newly_covered

    def _coverage_ratio(self) -> float:
        """Return the fraction of users that currently have service."""
        return float(self.user_connected.mean()) if self.user_connected.size else 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Reset the environment to a fresh disaster scenario."""
        del options  # unused
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        self.current_step = 0
        self.remaining_budget = float(self.max_base_stations)
        self.deployed_mask = np.zeros(self.candidate_sites, dtype=bool)

        self.user_positions = self._generate_user_positions()
        outage_mask = self.np_random.random(self.num_users) < self.initial_outage_fraction
        self.user_connected = ~outage_mask
        if not outage_mask.any():
            # Ensure at least one user starts without coverage.
            outage_idx = int(self.np_random.integers(0, self.num_users))
            self.user_connected[outage_idx] = False

        observation = self._get_observation()
        info = {"coverage_ratio": self._coverage_ratio()}
        return observation, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        """Apply a deployment action."""
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
            info["coverage_ratio"] = self._coverage_ratio()
            return observation, reward, terminated, truncated, info

        reward: float
        newly_covered = 0
        if self.deployed_mask[action]:
            reward = -self.invalid_action_penalty
        else:
            self.deployed_mask[action] = True
            self.remaining_budget -= 1.0
            newly_covered = self._deploy_and_cover(action)
            reward = self.coverage_reward * newly_covered - self.deployment_cost

        self.current_step += 1

        coverage_ratio = self._coverage_ratio()
        info["coverage_ratio"] = coverage_ratio
        info["newly_covered"] = float(newly_covered)
        info["remaining_budget"] = float(self.remaining_budget)

        if coverage_ratio >= 0.999:
            terminated = True
            info["reason"] = "all_users_covered"
        elif self.current_step >= self.max_steps:
            truncated = True
            info["reason"] = "max_steps"
        elif self.remaining_budget <= 0:
            truncated = True
            info["reason"] = "budget_exhausted"

        observation = self._get_observation()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Render a textual summary of the deployment state."""
        coverage_ratio = self._coverage_ratio()
        deployed_count = int(self.deployed_mask.sum())
        print(
            f"[Render] Step: {self.current_step} | Coverage: {coverage_ratio:.2%} | "
            f"Deployed stations: {deployed_count}/{self.max_base_stations} | "
            f"Remaining budget: {self.remaining_budget:.1f}"
        )

    def close(self) -> None:
        """Cleanup hook (not used for this simple environment)."""
        return
