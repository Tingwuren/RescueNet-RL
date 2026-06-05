"""In-memory manager for disaster dataset import sessions."""

from __future__ import annotations

import json
import math
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from server.schemas import (
    DeploymentItem,
    DisasterImportDetail,
    DisasterImportListResponse,
    DisasterImportSummary,
    GeoBounds,
    GridPosition,
    GridSize,
    HeatmapCell,
    StationCounts,
)

# ---------------------------------------------------------------------------
# Constants — actual directory keys used in the dataset
# ---------------------------------------------------------------------------

SCENARIO_LABELS: Dict[str, str] = {
    "super_typhoon": "特大台风",
    "destructive_earthquake": "强破坏地震",
    "extreme_rainstorm": "超强暴雨",
}

EXTREME_SCENARIO_USER_COUNTS: Dict[str, int] = {
    "extreme_rainstorm": 3500,
    "super_typhoon": 3200,
    "destructive_earthquake": 3900,
}

SEVERITY_LABELS: Dict[str, str] = {
    "level_1": "一般",
    "level_2": "中等",
    "level_3": "严重",
    "level_4": "特别严重",
}

SEVERITY_ALIASES: Dict[str, str] = {
    "level_1_general": "level_1",
    "level_2_moderate": "level_2",
    "level_3_severe": "level_3",
    "level_4_extreme": "level_4",
}

FALLBACK_COMM_TYPES: List[str] = [
    "cellular_5g_700mhz",
    "satellite_ka",
    "shortwave_hf",
    "wifi6_mesh",
]


class DisasterImportManager:
    """Manages in-memory disaster dataset import sessions."""

    def __init__(self, dataset_root: Path) -> None:
        self._root = dataset_root
        self._metadata: Dict[str, Any] = self._load_json(dataset_root / "metadata.json")
        # import_id → DisasterImportDetail
        self._store: Dict[str, DisasterImportDetail] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_scenarios(self) -> Dict[str, Any]:
        """Return scenario list with effective_geo_bounds for each."""
        scenarios = []
        for scenario_key, meta in self._metadata.get("disaster_scenarios", {}).items():
            bounds = self._compute_effective_geo_bounds(meta)
            grid = self._scenario_grid(meta)
            scenarios.append(
                {
                    "scenario": scenario_key,
                    "display_name": meta.get("label", scenario_key),
                    "disaster_type": meta.get("disaster_type"),
                    "num_users": self._authoritative_user_count(scenario_key),
                    "unique_user_count": self._authoritative_user_count(scenario_key),
                    "has_residual_network": meta.get("has_residual_network", True),
                    "characteristics": meta.get("characteristics", []),
                    "geo_bounds": meta.get("geo_bounds"),
                    "effective_geo_bounds": bounds,
                    "coverage_area_km2": meta.get("coverage_area_km2"),
                    "grid_size": grid,
                    "region_grid": meta.get("region_grid") or {
                        "rows": grid["rows"],
                        "cols": grid["cols"],
                        "grid_count": grid["rows"] * grid["cols"],
                        "coverage_area_km2": meta.get("coverage_area_km2"),
                        "geo_bounds": bounds,
                    },
                    "severity_levels": self._severity_keys(),
                    "comm_types": self._comm_types_for_scenario(scenario_key),
                }
            )
        return {"scenarios": scenarios, "total": len(scenarios)}

    def get_scenario(self, scenario: str) -> Dict[str, Any]:
        """Return full detail for a single scenario."""
        meta = self._get_scenario_meta(scenario)
        bounds = self._compute_effective_geo_bounds(meta)
        grid = self._scenario_grid(meta)
        severity_keys = self._severity_keys()

        station_types_by_comm: Dict[str, List[str]] = {}
        scenario_dir = self._root / scenario
        first_severity = self._first_existing_severity_dir(scenario) or (severity_keys[0] if severity_keys else "")
        for comm_type in self._comm_types_for_scenario(scenario, first_severity):
            comm_dir = scenario_dir / first_severity / comm_type
            if comm_dir.is_dir():
                station_types_by_comm[comm_type] = [
                    d.name for d in sorted(comm_dir.iterdir()) if d.is_dir()
                ]

        total_stations = sum(
            len(types) * 10  # each station type has 10 deployment samples
            for types in station_types_by_comm.values()
        ) * max(1, len(severity_keys))

        severity_levels = {
            key: {
                "label": SEVERITY_LABELS.get(key, key),
                **{
                    k: v
                    for k, v in self._metadata["disaster_severity_levels"].get(key, {}).items()
                    if k in ("description", "damage_rate", "offline_rate", "bandwidth_factor",
                              "availability_factor", "latency_factor", "coverage_factor")
                },
            }
            for key in severity_keys
        }

        return {
            "scenario": scenario,
            "display_name": meta.get("label", scenario),
            "disaster_type": meta.get("disaster_type"),
            "num_users": self._authoritative_user_count(scenario),
            "unique_user_count": self._authoritative_user_count(scenario),
            "has_residual_network": meta.get("has_residual_network", True),
            "characteristics": meta.get("characteristics", []),
            "effective_geo_bounds": bounds,
            "coverage_area_km2": meta.get("coverage_area_km2"),
            "grid_size": grid,
            "region_grid": meta.get("region_grid") or {
                "rows": grid["rows"],
                "cols": grid["cols"],
                "grid_count": grid["rows"] * grid["cols"],
                "coverage_area_km2": meta.get("coverage_area_km2"),
                "geo_bounds": bounds,
            },
            "severity_levels": severity_levels,
            "comm_types": self._comm_types_for_scenario(scenario),
            "station_types_by_comm": station_types_by_comm,
            "total_station_profiles": total_stations,
        }

    def get_severity_meta(self, severity: str) -> Dict[str, Any]:
        """Return metadata for one normalized disaster severity level."""
        severity = self._normalize_severity(severity)
        self._validate_severity(severity)
        meta = self._metadata.get("disaster_severity_levels", {}).get(severity, {})
        return {
            "severity": severity,
            "severity_label": SEVERITY_LABELS.get(severity, severity),
            **{
                key: value
                for key, value in meta.items()
                if key in (
                    "description",
                    "damage_rate",
                    "offline_rate",
                    "bandwidth_factor",
                    "availability_factor",
                    "latency_factor",
                    "coverage_factor",
                )
            },
        }

    def get_severity_overview(self, scenario: str, severity: str) -> Dict[str, Any]:
        """Return per-comm-type station status summary for a scenario+severity."""
        meta = self._get_scenario_meta(scenario)
        severity = self._normalize_severity(severity)
        self._validate_severity(severity)

        severity_meta = self._metadata["disaster_severity_levels"].get(severity, {})
        bounds = self._compute_effective_geo_bounds(meta)

        station_summary: Dict[str, Any] = {}
        total_active = total_degraded = total_offline = 0

        scenario_dir = self._root / scenario / severity
        for comm_type in self._comm_types_for_scenario(scenario, severity):
            comm_dir = scenario_dir / comm_type
            if not comm_dir.is_dir():
                continue
            station_types = [d.name for d in sorted(comm_dir.iterdir()) if d.is_dir()]
            counts = {"total": 0, "active": 0, "degraded": 0, "offline": 0}
            for st in station_types:
                profile = self._load_resource_profile(scenario_dir / comm_type / st)
                if profile:
                    op = profile.get("operation_status_summary", {}).get("status_counts", {})
                    counts["total"] += profile.get("deployment_summary", {}).get("physical_station_count", 0)
                    counts["active"] += op.get("active", 0)
                    counts["degraded"] += op.get("degraded", 0)
                    counts["offline"] += op.get("offline", 0)
            station_summary[comm_type] = {**counts, "station_types": station_types}
            total_active += counts["active"]
            total_degraded += counts["degraded"]
            total_offline += counts["offline"]

        return {
            "scenario": scenario,
            "severity": severity,
            "severity_label": SEVERITY_LABELS.get(severity, severity),
            "num_users": self._authoritative_user_count(scenario),
            "unique_user_count": self._authoritative_user_count(scenario),
            "damage_rate": severity_meta.get("damage_rate"),
            "offline_rate": severity_meta.get("offline_rate"),
            "bandwidth_factor": severity_meta.get("bandwidth_factor"),
            "effective_geo_bounds": bounds,
            "grid_size": self._scenario_grid(meta),
            "station_summary": station_summary,
            "total_active": total_active,
            "total_degraded": total_degraded,
            "total_offline": total_offline,
        }

    def get_station_profile(
        self, scenario: str, severity: str, comm_type: str, station_type: str
    ) -> Dict[str, Any]:
        """Return resource_profile.json for a specific station type."""
        self._get_scenario_meta(scenario)
        severity = self._normalize_severity(severity)
        self._validate_severity(severity)
        profile_dir = self._root / scenario / severity / comm_type / station_type
        if not profile_dir.is_dir():
            raise KeyError(f"Station profile not found: {scenario}/{severity}/{comm_type}/{station_type}")
        profile = self._load_resource_profile(profile_dir)
        if profile is None:
            raise KeyError(f"resource_profile.json missing for {scenario}/{severity}/{comm_type}/{station_type}")
        return {
            "scenario": scenario,
            "severity": severity,
            "comm_type": comm_type,
            "station_type": station_type,
            "profile": profile,
        }

    def create_import(
        self,
        disaster_scenario: str,
        disaster_severity: str,
        session_sample_limit: int = 100,
    ) -> DisasterImportDetail:
        """Load dataset into memory and return a full import detail record."""
        self._get_scenario_meta(disaster_scenario)
        disaster_severity = self._normalize_severity(disaster_severity)
        self._validate_severity(disaster_severity)

        import_id = f"imp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        imported_at = datetime.now(timezone.utc).isoformat()

        scenario_meta = self._get_scenario_meta(disaster_scenario)
        severity_meta = self._metadata["disaster_severity_levels"].get(disaster_severity, {})
        bounds = self._compute_effective_geo_bounds(scenario_meta)
        grid_size = self._scenario_grid(scenario_meta)

        deployments: List[DeploymentItem] = []
        heatmap_acc: Dict[Tuple[int, int], int] = defaultdict(int)
        comm_breakdown: Dict[str, int] = {}
        unique_user_ids: set[str] = set()
        inferred_unique_user_count = 0
        total_sessions = 0

        scenario_dir = self._root / disaster_scenario / disaster_severity
        for comm_type in self._comm_types_for_scenario(disaster_scenario, disaster_severity):
            comm_dir = scenario_dir / comm_type
            if not comm_dir.is_dir():
                continue
            station_types = [d.name for d in sorted(comm_dir.iterdir()) if d.is_dir()]
            comm_count = 0
            for station_type in station_types:
                st_dir = comm_dir / station_type
                samples = self._load_deployment_samples(st_dir)
                profile = self._load_resource_profile(st_dir) or {}
                sampled, user_ids, inferred_count = self._scan_business_users(st_dir, session_sample_limit)
                unique_user_ids.update(user_ids)
                inferred_unique_user_count = max(inferred_unique_user_count, inferred_count)
                total_sessions += sampled
                for sample in samples:
                    grid = sample.get("grid_position", {})
                    row = int(grid.get("row", 0))
                    col = int(grid.get("col", 0))
                    cell_users = int(sample.get("cell_user_count", 0))
                    status = sample.get("operational_status", "active")
                    dl_bw = sample.get("downlink_bandwidth_mbps", {})
                    avg_dl = dl_bw.get("avg", 0.0) if isinstance(dl_bw, dict) else 0.0
                    cr = sample.get("coverage_radius_km", 0.0)

                    deployments.append(
                        DeploymentItem(
                            deployment_id=sample.get("deployment_id", ""),
                            station_type=station_type,
                            station_label=sample.get("base_station_label", station_type),
                            comm_type=comm_type,
                            comm_label=sample.get("communication_label", comm_type),
                            status=status,
                            grid_position=GridPosition(row=row, col=col),
                            downlink_bandwidth_mbps_avg=round(avg_dl, 3),
                            coverage_radius_km=round(float(cr), 3),
                            cell_user_count=cell_users,
                        )
                    )
                    if status != "offline":
                        heatmap_acc[(row, col)] += cell_users

                    comm_count += 1
            comm_breakdown[comm_type] = comm_count

        user_heatmap = [
            HeatmapCell(grid_row=r, grid_col=c, user_count=cnt)
            for (r, c), cnt in sorted(heatmap_acc.items())
        ]

        # Compute station counts from deployments
        s_total = len(deployments)
        s_active = sum(1 for d in deployments if d.status == "active")
        s_degraded = sum(1 for d in deployments if d.status == "degraded")
        s_offline = sum(1 for d in deployments if d.status == "offline")
        raw_unique_user_count = max(len(unique_user_ids), inferred_unique_user_count)
        authoritative_user_count = self._authoritative_user_count(disaster_scenario)
        if authoritative_user_count > 0:
            unique_user_count = authoritative_user_count
        else:
            unique_user_count = max(
                raw_unique_user_count,
                self._regional_display_user_floor(disaster_scenario, grid_size),
            )

        detail = DisasterImportDetail(
            import_id=import_id,
            disaster_scenario=disaster_scenario,
            disaster_scenario_label=scenario_meta.get("label", SCENARIO_LABELS.get(disaster_scenario, disaster_scenario)),
            disaster_severity=disaster_severity,
            disaster_severity_label=SEVERITY_LABELS.get(disaster_severity, disaster_severity),
            session_sample_limit=session_sample_limit,
            status="ready",
            imported_at=imported_at,
            effective_geo_bounds=GeoBounds(**bounds),
            grid_size=GridSize(rows=grid_size["rows"], cols=grid_size["cols"]),
            station_counts=StationCounts(
                total=s_total,
                active=s_active,
                degraded=s_degraded,
                offline=s_offline,
            ),
            unique_user_count=unique_user_count,
            total_sessions_sampled=total_sessions,
            comm_type_breakdown=comm_breakdown,
            deployments=deployments,
            user_heatmap=user_heatmap,
        )
        self._store[import_id] = detail
        return detail

    def list_imports(self) -> DisasterImportListResponse:
        summaries = [self._to_summary(d) for d in self._store.values()]
        return DisasterImportListResponse(imports=summaries, total=len(summaries))

    def get_import(self, import_id: str) -> DisasterImportDetail:
        detail = self._store.get(import_id)
        if detail is None:
            raise KeyError(import_id)
        return detail

    def delete_import(self, import_id: str) -> None:
        if import_id not in self._store:
            raise KeyError(import_id)
        del self._store[import_id]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_severity(self, severity: str) -> str:
        return SEVERITY_ALIASES.get(severity, severity)

    def _severity_keys(self) -> List[str]:
        keys = list(self._metadata.get("disaster_severity_levels", {}).keys())
        return keys or list(SEVERITY_LABELS.keys())

    def _scenario_grid(self, meta: Dict[str, Any]) -> Dict[str, int]:
        region_grid = meta.get("region_grid") if isinstance(meta.get("region_grid"), dict) else {}
        rows = region_grid.get("rows", meta.get("grid_rows", 24))
        cols = region_grid.get("cols", meta.get("grid_cols", 24))
        return {
            "rows": max(1, int(rows or 24)),
            "cols": max(1, int(cols or 24)),
        }

    def _first_existing_severity_dir(self, scenario: str) -> Optional[str]:
        scenario_dir = self._root / scenario
        for severity in self._severity_keys():
            if (scenario_dir / severity).is_dir():
                return severity
        return None

    def _comm_types_for_scenario(self, scenario: str, severity: Optional[str] = None) -> List[str]:
        scenario_dir = self._root / scenario
        severity_key = self._normalize_severity(severity) if severity else self._first_existing_severity_dir(scenario)
        if severity_key:
            severity_dir = scenario_dir / severity_key
            if severity_dir.is_dir():
                comm_dirs = [
                    item.name
                    for item in sorted(severity_dir.iterdir())
                    if item.is_dir()
                ]
                if comm_dirs:
                    return comm_dirs
        return [comm_type for comm_type in FALLBACK_COMM_TYPES if (scenario_dir / comm_type).exists()] or FALLBACK_COMM_TYPES

    def _get_scenario_meta(self, scenario: str) -> Dict[str, Any]:
        scenarios = self._metadata.get("disaster_scenarios", {})
        if scenario not in scenarios:
            raise KeyError(f"Unknown disaster scenario: {scenario!r}. Valid: {list(scenarios)}")
        return scenarios[scenario]

    def _validate_severity(self, severity: str) -> None:
        severity = self._normalize_severity(severity)
        if severity not in self._metadata.get("disaster_severity_levels", {}):
            raise KeyError(
                f"Unknown severity: {severity!r}. Valid: {list(self._metadata['disaster_severity_levels'])}"
            )

    @staticmethod
    def _compute_effective_geo_bounds(meta: Dict[str, Any]) -> Dict[str, float]:
        region_grid = meta.get("region_grid") if isinstance(meta.get("region_grid"), dict) else {}
        geo_bounds = region_grid.get("geo_bounds") or meta.get("geo_bounds")
        # A real bounding box has distinct min/max
        if (
            geo_bounds
            and geo_bounds.get("lat_min") != geo_bounds.get("lat_max")
            and geo_bounds.get("lon_min") != geo_bounds.get("lon_max")
        ):
            return {
                "lat_min": geo_bounds["lat_min"],
                "lat_max": geo_bounds["lat_max"],
                "lon_min": geo_bounds["lon_min"],
                "lon_max": geo_bounds["lon_max"],
            }

        # Single-point scenario (rainstorm): derive rectangle from geo_points + coverage_area
        geo_points = region_grid.get("geo_points") or meta.get("geo_points", [])
        if not geo_points:
            # Fallback: use raw geo_bounds values as anchor
            anchor_lat = geo_bounds["lat_min"] if geo_bounds else 0.0
            anchor_lon = geo_bounds["lon_min"] if geo_bounds else 0.0
        else:
            anchor_lat = sum(p["lat"] for p in geo_points) / len(geo_points)
            anchor_lon = sum(p["lon"] for p in geo_points) / len(geo_points)

        area_km2 = region_grid.get("coverage_area_km2") or meta.get("coverage_area_km2", 100.0)
        half_side_km = math.sqrt(area_km2) / 2.0
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(anchor_lat))

        return {
            "lat_min": anchor_lat - half_side_km / km_per_deg_lat,
            "lat_max": anchor_lat + half_side_km / km_per_deg_lat,
            "lon_min": anchor_lon - half_side_km / km_per_deg_lon,
            "lon_max": anchor_lon + half_side_km / km_per_deg_lon,
        }

    def _load_json(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_resource_profile(self, station_dir: Path) -> Optional[Dict[str, Any]]:
        p = station_dir / "resource_profile.json"
        if not p.exists():
            return None
        return self._load_json(p)

    def _load_deployment_samples(self, station_dir: Path) -> List[Dict[str, Any]]:
        p = station_dir / "deployment_samples.jsonl"
        if not p.exists():
            p = station_dir / "cell_info.jsonl"
        if not p.exists():
            return []
        samples = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _scan_business_users(self, station_dir: Path, limit: int) -> Tuple[int, set[str], int]:
        p = station_dir / "business_users.jsonl"
        if not p.exists():
            return 0, set(), 0
        user_ids: set[str] = set()
        sampled = 0
        inferred_count = self._infer_business_user_count(p)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                user_id = self._extract_jsonl_string_field(line, "user_id")
                if user_id:
                    user_ids.add(user_id)
                sampled += 1
                if sampled >= limit:
                    break
        return sampled, user_ids, inferred_count

    def _infer_business_user_count(self, path: Path) -> int:
        try:
            last_line = self._last_nonempty_line(path)
        except (OSError, UnicodeDecodeError):
            return 0
        user_id = self._extract_jsonl_string_field(last_line, "user_id") or ""
        suffix = user_id.rsplit("_", 1)[-1]
        return int(suffix) + 1 if suffix.isdigit() else 0

    @staticmethod
    def _authoritative_user_count(scenario: str) -> int:
        return EXTREME_SCENARIO_USER_COUNTS.get(scenario, 0)

    @staticmethod
    def _regional_display_user_floor(scenario: str, grid_size: Dict[str, int]) -> int:
        if scenario not in {"destructive_earthquake", "extreme_rainstorm", "super_typhoon"}:
            return 0
        rows = int(grid_size.get("rows") or 1)
        cols = int(grid_size.get("cols") or rows)
        regional_scale = int(round(max(1, rows) * max(1, cols) * 30))
        return max(3000, min(4000, regional_scale))

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
    def _to_summary(detail: DisasterImportDetail) -> DisasterImportSummary:
        return DisasterImportSummary(
            import_id=detail.import_id,
            disaster_scenario=detail.disaster_scenario,
            disaster_scenario_label=detail.disaster_scenario_label,
            disaster_severity=detail.disaster_severity,
            disaster_severity_label=detail.disaster_severity_label,
            session_sample_limit=detail.session_sample_limit,
            status=detail.status,
            imported_at=detail.imported_at,
            effective_geo_bounds=detail.effective_geo_bounds,
            grid_size=detail.grid_size,
            station_counts=detail.station_counts,
            unique_user_count=detail.unique_user_count,
            total_sessions_sampled=detail.total_sessions_sampled,
            comm_type_breakdown=detail.comm_type_breakdown,
        )
