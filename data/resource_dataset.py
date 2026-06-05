"""Scenario-aware communication resource dataset utilities."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


EXTREME_SCENARIO_USER_COUNTS: Dict[str, int] = {
    "extreme_rainstorm": 3500,
    "super_typhoon": 3200,
    "destructive_earthquake": 3900,
}


@dataclass
class ModeSnapshot:
    """Stores per-mode availability metadata for a specific timestamp."""

    available_bandwidth: float
    availability: float


@dataclass
class BroadcastSnapshot:
    """Broadcast network snapshot for a timestamp."""

    available_bandwidth: float
    coverage: float


@dataclass
class TimeStepRecord:
    """Aggregated resource metrics for one timestamp."""

    time: int
    mode_metrics: Dict[str, ModeSnapshot]
    broadcast_metrics: Dict[str, BroadcastSnapshot]


@dataclass
class DisasterScenario:
    """Parsed disaster scenario ready for environment consumption."""

    name: str
    disaster_type: str
    grid_size: int
    region_grid: "RegionGrid"
    num_users: int
    candidate_sites: int
    max_steps: int
    has_residual_network: bool
    communication_modes: List[str]
    broadcast_modes: List[str]
    mode_profiles: Dict[str, Dict[str, Any]]
    broadcast_profiles: Dict[str, Dict[str, Any]]
    user_clusters: List[Dict[str, float]]
    time_series: List[TimeStepRecord]
    base_station_profiles: Dict[str, "BaseStationProfile"]
    reward_profiles: Dict[str, "RewardProfile"]
    default_reward_profile: Optional[str] = None
    source_scenario: Optional[str] = None
    severity_level: Optional[str] = None
    severity_label: Optional[str] = None
    display_name: Optional[str] = None
    candidate_locations: Optional[List[Tuple[int, int]]] = None

    def get_reward_profile(self, key: Optional[str] = None) -> "RewardProfile":
        """Return reward profile by key or fallback to scenario default."""
        if key and key in self.reward_profiles:
            return self.reward_profiles[key]
        if self.default_reward_profile and self.default_reward_profile in self.reward_profiles:
            return self.reward_profiles[self.default_reward_profile]
        if not self.reward_profiles:
            raise ValueError(f"Scenario {self.name} does not define reward profiles.")
        return next(iter(self.reward_profiles.values()))

    def get_base_station_for_mode(self, mode: str) -> Optional["BaseStationProfile"]:
        """Resolve the base-station profile that backs a communication mode."""
        profile = self.mode_profiles.get(mode)
        if not profile:
            return None
        key = profile.get("base_station")
        if key and key in self.base_station_profiles:
            return self.base_station_profiles[key]
        # Fall back to first profile that lists the mode under supported modes.
        for entry in self.base_station_profiles.values():
            if mode in entry.supported_modes:
                return entry
        return None


@dataclass
class BaseStationProfile:
    """Describes deployable base-station hardware limits and costs."""

    name: str
    label: str
    max_throughput: float
    max_users: int
    device_cost: float
    bandwidth_cost: float
    supported_modes: List[str]


@dataclass
class RewardProfile:
    """Scenario-specific reward configuration weights."""

    key: str
    label: str
    description: str
    coverage_weight: float
    bandwidth_weight: float
    throughput_weight: float
    broadcast_weight: float
    device_cost_weight: float
    bandwidth_cost_weight: float


class ResourceDataset:
    """Loads and validates the multi-mode disaster communication dataset."""

    MIN_COMM_MODES = 4
    _USER_COUNT_CACHE: Dict[Tuple[str, str, str], int] = {}
    _TOPOLOGY_CACHE: Dict[Tuple[str, str, str, int, int], Tuple[List[List[int]], List[Dict[str, float]]]] = {}

    def __init__(self, dataset_path: Union[str, Path]) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")
        raw = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        scenario_entries = self._normalise_scenarios(raw)
        self._scenarios = {
            entry["name"]: self._parse_scenario(entry) for entry in scenario_entries
        }
        if not self._scenarios:
            raise ValueError("Dataset does not contain any scenarios.")

    def list_scenarios(self) -> List[str]:
        return sorted(self._scenarios.keys())

    def get(self, name: str) -> DisasterScenario:
        if name not in self._scenarios:
            raise KeyError(f"Scenario '{name}' not found in dataset.")
        return self._scenarios[name]

    def _parse_scenario(self, data: Dict) -> DisasterScenario:
        comm_modes = data.get("communication_modes", [])
        if len(comm_modes) < self.MIN_COMM_MODES:
            raise ValueError(
                f"Scenario {data.get('name')} must define at least {self.MIN_COMM_MODES} communication modes"
            )
        broadcast_modes = data.get("broadcast_modes", [])
        time_series = [self._parse_time_step(step) for step in data.get("time_series", [])]
        if not time_series:
            raise ValueError(f"Scenario {data.get('name')} must include time-series metrics")
        base_stations = self._parse_base_stations(data.get("base_stations", {}))
        reward_profiles = self._parse_reward_profiles(data.get("reward_profiles", {}), data.get("name"))
        default_reward = data.get("default_reward_profile")
        if not default_reward or default_reward not in reward_profiles:
            default_reward = next(iter(reward_profiles.keys()))
        region_grid = self._parse_region_grid(data.get("region_grid"), int(data["grid_size"]), data.get("name"))
        return DisasterScenario(
            name=data["name"],
            disaster_type=data["disaster_type"],
            grid_size=int(data["grid_size"]),
            region_grid=region_grid,
            num_users=int(data["num_users"]),
            candidate_sites=int(data["candidate_sites"]),
            max_steps=int(data["max_steps"]),
            has_residual_network=bool(data.get("has_residual_network", False)),
            communication_modes=list(comm_modes),
            broadcast_modes=list(broadcast_modes),
            mode_profiles=data.get("mode_profiles", {}),
            broadcast_profiles=data.get("broadcast_profiles", {}),
            user_clusters=data.get("user_clusters", []),
            time_series=time_series,
            base_station_profiles=base_stations,
            reward_profiles=reward_profiles,
            default_reward_profile=default_reward,
            source_scenario=data.get("source_scenario"),
            severity_level=data.get("severity_level"),
            severity_label=data.get("severity_label"),
            display_name=data.get("display_name") or data.get("label"),
            candidate_locations=[
                (int(item[0]), int(item[1]))
                for item in data.get("candidate_locations", [])
                if isinstance(item, (list, tuple)) and len(item) >= 2
            ] or None,
        )

    def _parse_time_step(self, step: Dict) -> TimeStepRecord:
        mode_metrics = {
            mode: ModeSnapshot(
                available_bandwidth=float(values["available_bandwidth"]),
                availability=float(values["availability"]),
            )
            for mode, values in step.get("mode_metrics", {}).items()
        }
        broadcast_metrics = {
            mode: BroadcastSnapshot(
                available_bandwidth=float(values["available_bandwidth"]),
                coverage=float(values["coverage"]),
            )
            for mode, values in step.get("broadcast_metrics", {}).items()
        }
        if not mode_metrics:
            raise ValueError("Time step is missing mode metrics")
        if not broadcast_metrics:
            raise ValueError("Time step is missing broadcast metrics")
        return TimeStepRecord(
            time=int(step["time"]),
            mode_metrics=mode_metrics,
            broadcast_metrics=broadcast_metrics,
        )

    def _parse_base_stations(self, table: Dict[str, Dict[str, Any]]) -> Dict[str, BaseStationProfile]:
        profiles: Dict[str, BaseStationProfile] = {}
        for name, entry in table.items():
            profiles[name] = BaseStationProfile(
                name=name,
                label=str(entry.get("label", name)),
                max_throughput=float(entry.get("max_throughput", entry.get("max_bandwidth", 0.0))),
                max_users=int(entry.get("max_users", 50)),
                device_cost=float(entry.get("device_cost", 1.0)),
                bandwidth_cost=float(entry.get("bandwidth_cost", 0.0)),
                supported_modes=list(entry.get("supported_modes", [])),
            )
        return profiles

    def _parse_reward_profiles(
        self, table: Dict[str, Dict[str, Any]], scenario_name: Optional[str]
    ) -> Dict[str, RewardProfile]:
        profiles: Dict[str, RewardProfile] = {}
        if not table:
            raise ValueError(f"Scenario {scenario_name} must define at least one reward profile.")
        for key, entry in table.items():
            profiles[key] = RewardProfile(
                key=key,
                label=str(entry.get("label", key)),
                description=str(entry.get("description", "")),
                coverage_weight=float(entry.get("coverage_weight", 1.0)),
                bandwidth_weight=float(entry.get("bandwidth_weight", 0.0)),
                throughput_weight=float(entry.get("throughput_weight", 0.0)),
                broadcast_weight=float(entry.get("broadcast_weight", 0.0)),
                device_cost_weight=float(entry.get("device_cost_weight", 0.0)),
                bandwidth_cost_weight=float(entry.get("bandwidth_cost_weight", 0.0)),
            )
        return profiles

    def _parse_region_grid(self, raw: Optional[Dict[str, Any]], grid_size: int, scenario_name: Optional[str]) -> "RegionGrid":
        """Map grid cells to real-world latitude/longitude bounds and human labels."""
        rows = int(raw.get("rows", grid_size)) if raw else grid_size
        cols = int(raw.get("cols", grid_size)) if raw else grid_size
        bounds = self._effective_geo_bounds(raw or {})
        lat_min = float(bounds.get("lat_min", 0.0))
        lat_max = float(bounds.get("lat_max", float(rows)))
        lon_min = float(bounds.get("lon_min", 0.0))
        lon_max = float(bounds.get("lon_max", float(cols)))
        labels = {str(key): str(value) for key, value in (raw.get("cell_labels", {}) if raw else {}).items()}
        name = str(raw.get("name", f"{scenario_name}_grid")) if raw else f"{scenario_name}_grid"
        return RegionGrid(
            name=name,
            rows=rows,
            cols=cols,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            cell_labels=labels,
        )

    def _normalise_scenarios(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Support both the legacy flat scenario file and the imported disaster regions file."""
        entries = raw.get("scenarios", [])
        normalised: List[Dict[str, Any]] = []
        for entry in entries:
            if "severity_levels" not in entry:
                normalised.append(entry)
                continue
            normalised.extend(self._expand_extreme_region_entry(entry))
        return normalised

    def _expand_extreme_region_entry(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand one imported disaster region into one trainable scenario per severity level."""
        base_name = str(entry["name"])
        grid = dict(entry.get("region_grid", {}))
        configured_grid_size = int(entry.get("grid_size") or 0)
        rows = int(grid.get("rows") or configured_grid_size or 24)
        cols = int(grid.get("cols") or configured_grid_size or rows)
        grid_size = max(rows, cols)
        grid["rows"] = rows
        grid["cols"] = cols
        grid["geo_bounds"] = self._effective_geo_bounds(grid)
        communication_modes = list(entry.get("communication_modes", []))
        output: List[Dict[str, Any]] = []

        for severity_key, severity_data in entry.get("severity_levels", {}).items():
            scenario_name = f"{base_name}__{severity_key}"
            mode_profiles = self._build_mode_profiles(
                severity_data.get("mode_profiles", {}),
                grid,
                communication_modes,
            )
            candidate_locations, user_clusters = self._load_deployment_topology(
                base_name,
                severity_key,
                rows,
                cols,
            )
            authoritative_users = self._authoritative_user_count(base_name)
            if authoritative_users > 0:
                num_users = authoritative_users
            else:
                num_users = self._infer_user_count(base_name, severity_key)
                if num_users <= 0:
                    num_users = self._estimate_user_count(severity_data.get("mode_profiles", {}), base_name)
                num_users = max(num_users, self._regional_display_user_floor(base_name, rows, cols))

            time_series = [
                self._build_time_step_with_broadcast(step, mode_profiles)
                for step in severity_data.get("time_series", [])
            ]
            if not time_series:
                time_series = [
                    self._build_time_step_with_broadcast(
                        {"time": 0, "mode_metrics": {
                            mode: {
                                "available_bandwidth": profile.get("max_bandwidth", 0.0),
                                "availability": profile.get("average_success_rate", 0.5),
                            }
                            for mode, profile in mode_profiles.items()
                        }},
                        mode_profiles,
                    )
                ]

            output.append(
                {
                    "name": scenario_name,
                    "display_name": f"{entry.get('label', base_name)} / {severity_data.get('label', severity_key)}",
                    "source_scenario": base_name,
                    "severity_level": severity_key,
                    "severity_label": severity_data.get("label", severity_key),
                    "disaster_type": entry.get("disaster_type", base_name),
                    "grid_size": grid_size,
                    "region_grid": grid,
                    "num_users": num_users,
                    "candidate_sites": max(1, len(candidate_locations)),
                    "candidate_locations": candidate_locations,
                    "max_steps": max(1, len(time_series)),
                    "has_residual_network": bool(entry.get("has_residual_network", False)),
                    "communication_modes": communication_modes,
                    "broadcast_modes": self._default_broadcast_modes(),
                    "mode_profiles": mode_profiles,
                    "broadcast_profiles": self._build_broadcast_profiles(mode_profiles),
                    "user_clusters": user_clusters or self._default_user_clusters(grid_size),
                    "time_series": time_series,
                    "base_stations": self._build_base_stations(mode_profiles),
                    "reward_profiles": self._default_reward_profiles(),
                    "default_reward_profile": self._default_reward_key(severity_key),
                }
            )
        return output

    def _build_mode_profiles(
        self,
        raw_profiles: Dict[str, Dict[str, Any]],
        grid: Dict[str, Any],
        communication_modes: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        cell_side_km = math.sqrt(float(grid.get("grid_cell_area_km2") or 1.0))
        profiles: Dict[str, Dict[str, Any]] = {}
        for mode in communication_modes:
            raw = dict(raw_profiles.get(mode, {}))
            radius_km = float(raw.get("coverage_radius", 1.0))
            radius_grid = max(1.0, radius_km / max(0.1, cell_side_km))
            raw["source_coverage_radius_km"] = round(radius_km, 4)
            raw["coverage_radius"] = round(radius_grid, 4)
            raw["max_bandwidth"] = float(raw.get("max_bandwidth", raw.get("available_bandwidth", 0.0)))
            raw["base_station"] = self._base_station_key(mode)
            profiles[mode] = raw
        return profiles

    def _build_time_step_with_broadcast(
        self,
        step: Dict[str, Any],
        mode_profiles: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        mode_metrics = step.get("mode_metrics", {})
        satellite = mode_metrics.get("Satellite_Ka", {})
        cellular = mode_metrics.get("5G_700MHz", {})
        shortwave = mode_metrics.get("Shortwave_HF", {})
        wifi = mode_metrics.get("WiFi6", {})

        def bandwidth(metric: Dict[str, Any], scale: float, floor: float = 0.0) -> float:
            return max(floor, float(metric.get("available_bandwidth", 0.0)) * scale)

        def coverage(metric: Dict[str, Any], scale: float, fallback: float) -> float:
            return float(max(0.0, min(0.98, float(metric.get("availability", fallback)) * scale)))

        broadcast_metrics = {
            "satellite_broadcast": {
                "available_bandwidth": round(bandwidth(satellite, 0.75, 1.0), 4),
                "coverage": round(coverage(satellite, 0.95, 0.85), 4),
            },
            "cell_broadcast": {
                "available_bandwidth": round(bandwidth(cellular, 0.18, 0.5), 4),
                "coverage": round(coverage(cellular, 0.85, 0.45), 4),
            },
            "emergency_radio": {
                "available_bandwidth": round(max(bandwidth(shortwave, 10.0), bandwidth(wifi, 0.05), 0.25), 4),
                "coverage": round(coverage(shortwave, 0.9, 0.65), 4),
            },
        }
        return {
            "time": int(step.get("time", 0)),
            "mode_metrics": mode_metrics,
            "broadcast_metrics": broadcast_metrics,
        }

    def _build_broadcast_profiles(self, mode_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        satellite_bw = float(mode_profiles.get("Satellite_Ka", {}).get("max_bandwidth", 40.0))
        cellular_bw = float(mode_profiles.get("5G_700MHz", {}).get("max_bandwidth", 60.0))
        return {
            "satellite_broadcast": {"max_bandwidth": max(1.0, satellite_bw * 0.75), "latency_ms": 240},
            "cell_broadcast": {"max_bandwidth": max(1.0, cellular_bw * 0.18), "latency_ms": 90},
            "emergency_radio": {"max_bandwidth": 8.0, "latency_ms": 60},
        }

    def _build_base_stations(self, mode_profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        stations: Dict[str, Dict[str, Any]] = {}
        for mode, profile in mode_profiles.items():
            key = self._base_station_key(mode)
            max_bandwidth = float(profile.get("max_bandwidth", 0.0))
            if mode == "Satellite_Ka":
                label, max_users, device_cost, bandwidth_cost = "Ka 卫星应急终端", 140, 1.85, 0.026
            elif mode == "WiFi6":
                label, max_users, device_cost, bandwidth_cost = "WiFi6 Mesh 应急节点", 90, 0.72, 0.017
            elif mode == "Shortwave_HF":
                label, max_users, device_cost, bandwidth_cost = "短波保底通信台", 220, 0.55, 0.012
            else:
                label, max_users, device_cost, bandwidth_cost = "5G 700MHz应急基站", 180, 1.05, 0.02
            stations[key] = {
                "label": label,
                "max_throughput": max(1.0, max_bandwidth),
                "max_users": max_users,
                "device_cost": device_cost,
                "bandwidth_cost": bandwidth_cost,
                "supported_modes": [mode],
            }
        return stations

    def _load_deployment_topology(
        self,
        scenario: str,
        severity: str,
        grid_rows: int,
        grid_cols: int,
    ) -> Tuple[List[List[int]], List[Dict[str, float]]]:
        scenario_dir = self.dataset_path.parent / scenario / severity
        if not scenario_dir.exists():
            return [], []
        cache_key = (str(self.dataset_path.parent.resolve()), scenario, severity, int(grid_rows), int(grid_cols))
        cached = self._TOPOLOGY_CACHE.get(cache_key)
        if cached is not None:
            locations, clusters = cached
            return [list(item) for item in locations], [dict(item) for item in clusters]

        cell_users: Dict[Tuple[int, int], float] = {}
        cell_status_score: Dict[Tuple[int, int], float] = {}
        cell_demand: Dict[Tuple[int, int], float] = {}
        for path in self._deployment_sample_paths(scenario_dir):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    grid = sample.get("grid_position", {})
                    row = int(max(0, min(grid_rows - 1, grid.get("row", 0))))
                    col = int(max(0, min(grid_cols - 1, grid.get("col", 0))))
                    key = (row, col)
                    cell_users[key] = cell_users.get(key, 0.0) + self._sample_user_signal(sample)
                    cell_demand[key] = cell_demand.get(key, 0.0) + self._sample_demand_signal(sample)
                    status = str(sample.get("operational_status", "active"))
                    cell_status_score[key] = cell_status_score.get(key, 0.0) + {
                        "offline": 3.0,
                        "degraded": 2.0,
                        "active": 1.0,
                    }.get(status, 1.0)

        ordered = sorted(
            cell_users,
            key=lambda item: (cell_users[item], cell_status_score.get(item, 0.0)),
            reverse=True,
        )
        locations = [[row, col] for row, col in ordered]

        positive = [(key, count) for key, count in cell_users.items() if count > 0]
        positive.sort(key=lambda item: item[1], reverse=True)
        top = positive[: min(24, len(positive))]
        total = sum(count for _, count in top) or 1.0
        cluster_radius = max(1.2, min(4.5, max(grid_rows, grid_cols) * 0.16))
        clusters = [
            {
                "center": [float(key[0]), float(key[1])],
                "radius": cluster_radius,
                "density": max(0.05, count / total),
                "demand_mbps": 8.0 + min(32.0, cell_demand.get(key, count) / 12.0),
            }
            for key, count in top
        ]
        self._TOPOLOGY_CACHE[cache_key] = ([list(item) for item in locations], [dict(item) for item in clusters])
        return locations, clusters

    def _deployment_sample_paths(self, scenario_dir: Path) -> List[Path]:
        paths: List[Path] = []
        for station_dir in sorted(item for item in scenario_dir.glob("*/*") if item.is_dir()):
            path = station_dir / "deployment_samples.jsonl"
            if not path.exists():
                path = station_dir / "cell_info.jsonl"
            if path.exists():
                paths.append(path)
        return paths

    @staticmethod
    def _sample_user_signal(sample: Dict[str, Any]) -> float:
        cell_users = float(sample.get("cell_user_count") or 0.0)
        connection_attempts = sample.get("connection_attempt_count")
        if connection_attempts is None:
            stats = sample.get("connection_statistics", {})
            if isinstance(stats, dict):
                connection_attempts = stats.get("attempt_count")
        attempt_signal = float(connection_attempts or 0.0) / 100.0
        return max(cell_users, attempt_signal, 1.0)

    @staticmethod
    def _sample_demand_signal(sample: Dict[str, Any]) -> float:
        downlink = sample.get("downlink_bandwidth_mbps", {})
        if isinstance(downlink, dict):
            return float(downlink.get("avg") or 0.0)
        return 0.0

    def _infer_user_count(self, scenario: str, severity: str) -> int:
        scenario_dir = self.dataset_path.parent / scenario / severity
        candidates = sorted(scenario_dir.glob("*/*/business_users.jsonl"))
        if not candidates:
            return 0
        cache_key = (str(self.dataset_path.parent.resolve()), scenario, severity)
        cached = self._USER_COUNT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # The business user bundles can be multiple GB across all scenarios.  Startup only
        # needs a scenario-level count, so infer it from each file's last user id instead
        # of scanning every JSONL row before the API begins listening.
        max_user_index = -1
        for path in candidates:
            try:
                last_line = self._last_nonempty_line(path)
                user_id = self._extract_jsonl_string_field(last_line, "user_id") or ""
            except (OSError, UnicodeDecodeError):
                continue
            suffix = user_id.rsplit("_", 1)[-1]
            if suffix.isdigit():
                max_user_index = max(max_user_index, int(suffix))

        count = max_user_index + 1 if max_user_index >= 0 else 0
        if count <= 0:
            count = self._infer_user_count_from_station_summaries(scenario_dir)
        if count > 0:
            self._USER_COUNT_CACHE[cache_key] = count
        return count

    @staticmethod
    def _infer_user_count_from_station_summaries(scenario_dir: Path) -> int:
        total = 0
        for path in sorted(scenario_dir.glob("*/*/resource_profile.json")):
            try:
                profile = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            total += int(profile.get("cell_user_count") or 0)
        return total

    @staticmethod
    def _last_nonempty_line(path: Path) -> str:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            block_size = 8192
            buffer = b""
            pos = end
            while pos > 0:
                read_size = min(block_size, pos)
                pos -= read_size
                handle.seek(pos)
                buffer = handle.read(read_size) + buffer
                lines = [line for line in buffer.splitlines() if line.strip()]
                if len(lines) >= 2 or pos == 0:
                    return lines[-1].decode("utf-8")
        return ""

    @staticmethod
    def _estimate_user_count(raw_profiles: Dict[str, Dict[str, Any]], scenario: str) -> int:
        profile_users = sum(
            int(profile.get("physical_station_count", profile.get("deployment_sample_count", 0))) * 10
            for profile in raw_profiles.values()
        )
        scenario_floor = {
            "extreme_rainstorm": 1500,
            "super_typhoon": 1500,
            "destructive_earthquake": 1200,
        }.get(scenario, 1000)
        return max(scenario_floor, profile_users)

    @staticmethod
    def _authoritative_user_count(scenario: str) -> int:
        return EXTREME_SCENARIO_USER_COUNTS.get(scenario, 0)

    @staticmethod
    def _regional_display_user_floor(scenario: str, rows: int, cols: int) -> int:
        if scenario not in {"destructive_earthquake", "extreme_rainstorm", "super_typhoon"}:
            return 0
        regional_scale = int(round(max(1, rows) * max(1, cols) * 30))
        return max(3000, min(4000, regional_scale))

    @staticmethod
    def _default_broadcast_modes() -> List[str]:
        return ["satellite_broadcast", "cell_broadcast", "emergency_radio"]

    @staticmethod
    def _default_user_clusters(grid_size: int) -> List[Dict[str, float]]:
        return [
            {"center": [grid_size * 0.30, grid_size * 0.35], "radius": 4.0, "density": 0.35, "demand_mbps": 12.0},
            {"center": [grid_size * 0.62, grid_size * 0.70], "radius": 5.0, "density": 0.40, "demand_mbps": 18.0},
            {"center": [grid_size * 0.50, grid_size * 0.20], "radius": 3.5, "density": 0.25, "demand_mbps": 10.0},
        ]

    @staticmethod
    def _default_reward_profiles() -> Dict[str, Dict[str, Any]]:
        return {
            "coverage_balance": {
                "label": "覆盖与广播均衡",
                "description": "兼顾用户恢复、广播覆盖和设备成本。",
                "coverage_weight": 1.10,
                "bandwidth_weight": 0.05,
                "throughput_weight": 0.04,
                "broadcast_weight": 0.45,
                "device_cost_weight": 0.24,
                "bandwidth_cost_weight": 0.16,
            },
            "coverage_priority": {
                "label": "覆盖优先",
                "description": "重灾条件下优先恢复离线用户覆盖。",
                "coverage_weight": 1.45,
                "bandwidth_weight": 0.035,
                "throughput_weight": 0.045,
                "broadcast_weight": 0.48,
                "device_cost_weight": 0.18,
                "bandwidth_cost_weight": 0.12,
            },
            "bandwidth_priority": {
                "label": "带宽优先",
                "description": "优先选择高吞吐链路保障视频和遥测业务。",
                "coverage_weight": 0.92,
                "bandwidth_weight": 0.10,
                "throughput_weight": 0.075,
                "broadcast_weight": 0.36,
                "device_cost_weight": 0.22,
                "bandwidth_cost_weight": 0.17,
            },
        }

    @staticmethod
    def _default_reward_key(severity: str) -> str:
        return "coverage_priority" if severity in {"level_3", "level_4", "level_3_severe", "level_4_extreme"} else "coverage_balance"

    @staticmethod
    def _extract_jsonl_string_field(line: str, field: str) -> Optional[str]:
        marker = f'"{field}"'
        start = line.find(marker)
        if start < 0:
            return None
        colon = line.find(":", start + len(marker))
        if colon < 0:
            return None
        first_quote = line.find('"', colon + 1)
        if first_quote < 0:
            return None
        second_quote = line.find('"', first_quote + 1)
        if second_quote < 0:
            return None
        return line[first_quote + 1:second_quote]

    @staticmethod
    def _base_station_key(mode: str) -> str:
        return {
            "5G_700MHz": "emergency_5g_700mhz_cell",
            "Satellite_Ka": "ka_satellite_terminal",
            "WiFi6": "wifi6_mesh_node",
            "Shortwave_HF": "shortwave_hf_station",
        }.get(mode, mode.lower().replace(" ", "_"))

    @staticmethod
    def _effective_geo_bounds(raw: Dict[str, Any]) -> Dict[str, float]:
        bounds = raw.get("geo_bounds", {}) if raw else {}
        if (
            bounds
            and bounds.get("lat_min") != bounds.get("lat_max")
            and bounds.get("lon_min") != bounds.get("lon_max")
        ):
            return bounds

        points = raw.get("geo_points") or []
        if points:
            anchor_lat = sum(float(point["lat"]) for point in points) / len(points)
            anchor_lon = sum(float(point["lon"]) for point in points) / len(points)
        else:
            anchor_lat = float(bounds.get("lat_min", 0.0)) if bounds else 0.0
            anchor_lon = float(bounds.get("lon_min", 0.0)) if bounds else 0.0
        area_km2 = float(raw.get("coverage_area_km2", 100.0))
        half_side_km = math.sqrt(max(area_km2, 1.0)) / 2.0
        km_per_deg_lat = 111.0
        km_per_deg_lon = max(1e-6, 111.0 * math.cos(math.radians(anchor_lat)))
        return {
            "lat_min": anchor_lat - half_side_km / km_per_deg_lat,
            "lat_max": anchor_lat + half_side_km / km_per_deg_lat,
            "lon_min": anchor_lon - half_side_km / km_per_deg_lon,
            "lon_max": anchor_lon + half_side_km / km_per_deg_lon,
        }


def load_dataset(dataset_path: Union[str, Path]) -> ResourceDataset:
    """Helper to load the dataset from disk."""
    return ResourceDataset(dataset_path)


@dataclass
class RegionGrid:
    """Semantic mapping from grid cells to geo bounds and labels."""

    name: str
    rows: int
    cols: int
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    cell_labels: Dict[str, str]

    @property
    def cell_count(self) -> int:
        return self.rows * self.cols

    def cell_index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def normalize_cell_index(self, row: int, col: int) -> float:
        return self.cell_index(row, col) / max(1, self.cell_count - 1)

    def normalize_row(self, row: int) -> float:
        return row / max(1, self.rows - 1)

    def normalize_col(self, col: int) -> float:
        return col / max(1, self.cols - 1)

    def cell_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        lat_step = (self.lat_max - self.lat_min) / max(1, self.rows)
        lon_step = (self.lon_max - self.lon_min) / max(1, self.cols)
        lat0 = self.lat_min + row * lat_step
        lat1 = lat0 + lat_step
        lon0 = self.lon_min + col * lon_step
        lon1 = lon0 + lon_step
        return lat0, lat1, lon0, lon1

    def cell_label(self, row: int, col: int) -> str:
        return self.cell_labels.get(f"{row},{col}", f"cell-{row}-{col}")

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "geo_bounds": {
                "lat_min": self.lat_min,
                "lat_max": self.lat_max,
                "lon_min": self.lon_min,
                "lon_max": self.lon_max,
            },
            "cell_labels": dict(self.cell_labels),
        }
