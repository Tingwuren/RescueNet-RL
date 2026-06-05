"""SQLite persistence for per-scenario grid devices and device configs."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(value: Any, fallback: Any) -> Any:
    if value in (None, ""):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


class ScenarioDeviceManager:
    """Manage the scene -> grid -> device relational model.

    Tables:
    - scenario_registry: one row per training/testing scenario.
    - scenario_grid_cells: materialized grid cells for each scenario.
    - scenario_devices: base-station and user-device instances in grid cells.
    - scenario_device_type_configs: scene-scoped type-level parameter overrides.
    - scenario_device_events: audit trail for CRUD operations.
    """

    DEVICE_KIND_BASE_STATION = "base_station"
    DEVICE_KIND_USER = "user_device"

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_registry (
                    scenario_name TEXT PRIMARY KEY,
                    display_name TEXT,
                    disaster_type TEXT,
                    source_scenario TEXT,
                    severity_level TEXT,
                    grid_rows INTEGER NOT NULL,
                    grid_cols INTEGER NOT NULL,
                    grid_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_grid_cells (
                    cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_name TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    col_index INTEGER NOT NULL,
                    cell_label TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (scenario_name, row_index, col_index),
                    FOREIGN KEY (scenario_name)
                        REFERENCES scenario_registry(scenario_name)
                        ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_device_type_configs (
                    scenario_name TEXT NOT NULL,
                    device_kind TEXT NOT NULL,
                    type_key TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (scenario_name, device_kind, type_key),
                    FOREIGN KEY (scenario_name)
                        REFERENCES scenario_registry(scenario_name)
                        ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_devices (
                    device_uid TEXT PRIMARY KEY,
                    scenario_name TEXT NOT NULL,
                    device_kind TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    col_index INTEGER NOT NULL,
                    type_key TEXT NOT NULL,
                    mode TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    device_name TEXT,
                    device_category TEXT,
                    source_deployment_id TEXT,
                    source_station_type TEXT,
                    source_station_label TEXT,
                    cell_user_count INTEGER,
                    coverage_radius REAL,
                    coverage_radius_km REAL,
                    max_throughput REAL,
                    max_users INTEGER,
                    downlink_bandwidth_mbps REAL,
                    uplink_bandwidth_mbps REAL,
                    tx_power_watt REAL,
                    battery_duration_h REAL,
                    demand_mbps REAL,
                    connected INTEGER,
                    broadcast_served INTEGER,
                    notes TEXT,
                    override_fields_json TEXT NOT NULL DEFAULT '[]',
                    extra_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (scenario_name)
                        REFERENCES scenario_registry(scenario_name)
                        ON DELETE CASCADE,
                    FOREIGN KEY (scenario_name, row_index, col_index)
                        REFERENCES scenario_grid_cells(scenario_name, row_index, col_index)
                        ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scenario_devices_grid
                ON scenario_devices (
                    scenario_name, row_index, col_index, device_kind, type_key, mode, status
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_scenario_devices_kind
                ON scenario_devices (scenario_name, device_kind, status)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scenario_device_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario_name TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    device_uid TEXT,
                    device_kind TEXT,
                    type_key TEXT,
                    row_index INTEGER,
                    col_index INTEGER,
                    device_count INTEGER NOT NULL DEFAULT 0,
                    active_count INTEGER NOT NULL DEFAULT 0,
                    degraded_count INTEGER NOT NULL DEFAULT 0,
                    offline_count INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (scenario_name)
                        REFERENCES scenario_registry(scenario_name)
                        ON DELETE CASCADE
                )
                """
            )

    def ensure_scenario(
        self,
        scenario_name: str,
        *,
        display_name: Optional[str],
        disaster_type: Optional[str],
        source_scenario: Optional[str],
        severity_level: Optional[str],
        grid: Dict[str, Any],
        default_base_stations: List[Dict[str, Any]],
        legacy_entry: Optional[Dict[str, Any]] = None,
    ) -> None:
        rows = int(grid.get("rows") or 0)
        cols = int(grid.get("cols") or 0)
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Invalid grid for scenario {scenario_name}: rows={rows}, cols={cols}")

        with self._lock, self._conn:
            exists = self._conn.execute(
                "SELECT 1 FROM scenario_registry WHERE scenario_name = ?",
                (scenario_name,),
            ).fetchone()
            now = _utc_now()
            if exists:
                self._conn.execute(
                    """
                    UPDATE scenario_registry
                    SET display_name = ?, disaster_type = ?, source_scenario = ?, severity_level = ?,
                        grid_rows = ?, grid_cols = ?, grid_json = ?, updated_at = ?
                    WHERE scenario_name = ?
                    """,
                    (
                        display_name,
                        disaster_type,
                        source_scenario,
                        severity_level,
                        rows,
                        cols,
                        _json_dump(grid),
                        now,
                        scenario_name,
                    ),
                )
                self._ensure_grid_cells_locked(scenario_name, rows, cols)
                return

            self._conn.execute(
                """
                INSERT INTO scenario_registry (
                    scenario_name, display_name, disaster_type, source_scenario, severity_level,
                    grid_rows, grid_cols, grid_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario_name,
                    display_name,
                    disaster_type,
                    source_scenario,
                    severity_level,
                    rows,
                    cols,
                    _json_dump(grid),
                    now,
                    now,
                ),
            )
            self._ensure_grid_cells_locked(scenario_name, rows, cols)

            seed_specs = default_base_stations
            type_overrides: Dict[str, Any] = {}
            history: List[Dict[str, Any]] = []
            if isinstance(legacy_entry, dict):
                if isinstance(legacy_entry.get("base_stations"), list):
                    seed_specs = legacy_entry["base_stations"]
                if isinstance(legacy_entry.get("type_overrides"), dict):
                    type_overrides = legacy_entry["type_overrides"]
                if isinstance(legacy_entry.get("history"), list):
                    history = legacy_entry["history"]

            self._replace_base_stations_locked(scenario_name, seed_specs)
            self._replace_type_configs_locked(scenario_name, type_overrides)
            if history:
                for item in history[-200:]:
                    self._append_event_locked(
                        scenario_name,
                        str(item.get("operation") or "legacy_import"),
                        payload=item,
                        created_at=item.get("timestamp"),
                    )

    def has_scenario(self, scenario_name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM scenario_registry WHERE scenario_name = ?",
            (scenario_name,),
        ).fetchone()
        return row is not None

    def replace_state(
        self,
        scenario_name: str,
        base_stations: List[Dict[str, Any]],
        *,
        type_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        operation: str = "update",
    ) -> Dict[str, Any]:
        with self._lock, self._conn:
            previous_overrides = self.get_type_overrides(scenario_name)
            self._replace_base_stations_locked(scenario_name, base_stations)
            if type_overrides is not None:
                self._replace_type_configs_locked(scenario_name, type_overrides)
            else:
                self._replace_type_configs_locked(scenario_name, previous_overrides)
            counts = self._status_counts_locked(scenario_name)
            self._append_event_locked(
                scenario_name,
                operation,
                device_count=counts["total"],
                active_count=counts["active"],
                degraded_count=counts["degraded"],
                offline_count=counts["offline"],
            )
            return self.get_state(scenario_name)

    def reset_scenario(
        self,
        scenario_name: str,
        default_base_stations: List[Dict[str, Any]],
        *,
        operation: str = "reset",
    ) -> Dict[str, Any]:
        with self._lock, self._conn:
            self._replace_base_stations_locked(scenario_name, default_base_stations)
            self._conn.execute(
                """
                DELETE FROM scenario_device_type_configs
                WHERE scenario_name = ? AND device_kind = ?
                """,
                (scenario_name, self.DEVICE_KIND_BASE_STATION),
            )
            counts = self._status_counts_locked(scenario_name)
            self._append_event_locked(
                scenario_name,
                operation,
                device_count=counts["total"],
                active_count=counts["active"],
                degraded_count=counts["degraded"],
                offline_count=counts["offline"],
            )
            return self.get_state(scenario_name)

    def get_state(self, scenario_name: str) -> Dict[str, Any]:
        registry = self._conn.execute(
            "SELECT * FROM scenario_registry WHERE scenario_name = ?",
            (scenario_name,),
        ).fetchone()
        if not registry:
            return {}
        devices = [
            self._row_to_device_spec(row)
            for row in self._conn.execute(
                """
                SELECT * FROM scenario_devices
                WHERE scenario_name = ? AND device_kind = ?
                ORDER BY row_index, col_index, type_key, mode, device_uid
                """,
                (scenario_name, self.DEVICE_KIND_BASE_STATION),
            ).fetchall()
        ]
        history = [
            self._row_to_event(row)
            for row in self._conn.execute(
                """
                SELECT * FROM scenario_device_events
                WHERE scenario_name = ?
                ORDER BY event_id ASC
                """,
                (scenario_name,),
            ).fetchall()
        ]
        return {
            "base_stations": devices,
            "type_overrides": self.get_type_overrides(scenario_name),
            "history": history,
            "updated_at": self._timestamp_from_iso(registry["updated_at"]),
        }

    def get_type_overrides(self, scenario_name: str) -> Dict[str, Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT type_key, config_json
            FROM scenario_device_type_configs
            WHERE scenario_name = ? AND device_kind = ?
            """,
            (scenario_name, self.DEVICE_KIND_BASE_STATION),
        ).fetchall()
        return {str(row["type_key"]): _json_load(row["config_json"], {}) for row in rows}

    def schema_overview(self) -> Dict[str, Any]:
        rows = self._conn.execute(
            """
            SELECT name, sql
            FROM sqlite_master
            WHERE type = 'table' AND name LIKE 'scenario_%'
            ORDER BY name
            """
        ).fetchall()
        return {str(row["name"]): str(row["sql"]) for row in rows}

    def _ensure_grid_cells_locked(self, scenario_name: str, rows: int, cols: int) -> None:
        now = _utc_now()
        for row_index in range(rows):
            for col_index in range(cols):
                self._conn.execute(
                    """
                    INSERT INTO scenario_grid_cells (
                        scenario_name, row_index, col_index, cell_label, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, NULL, '{}', ?, ?)
                    ON CONFLICT(scenario_name, row_index, col_index)
                    DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (scenario_name, row_index, col_index, now, now),
                )

    def _replace_base_stations_locked(self, scenario_name: str, base_stations: List[Dict[str, Any]]) -> None:
        self._conn.execute(
            """
            DELETE FROM scenario_devices
            WHERE scenario_name = ? AND device_kind = ?
            """,
            (scenario_name, self.DEVICE_KIND_BASE_STATION),
        )
        now = _utc_now()
        for spec in base_stations:
            self._insert_device_locked(
                scenario_name,
                {
                    **spec,
                    "device_kind": self.DEVICE_KIND_BASE_STATION,
                    "type_key": spec.get("base_station"),
                    "row_index": spec.get("x", 0),
                    "col_index": spec.get("y", 0),
                },
                now=now,
            )
        self._conn.execute(
            "UPDATE scenario_registry SET updated_at = ? WHERE scenario_name = ?",
            (now, scenario_name),
        )

    def _replace_type_configs_locked(self, scenario_name: str, type_overrides: Dict[str, Dict[str, Any]]) -> None:
        now = _utc_now()
        self._conn.execute(
            """
            DELETE FROM scenario_device_type_configs
            WHERE scenario_name = ? AND device_kind = ?
            """,
            (scenario_name, self.DEVICE_KIND_BASE_STATION),
        )
        for type_key, config in (type_overrides or {}).items():
            cleaned = config if isinstance(config, dict) else {}
            if not cleaned:
                continue
            self._conn.execute(
                """
                INSERT INTO scenario_device_type_configs (
                    scenario_name, device_kind, type_key, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario_name,
                    self.DEVICE_KIND_BASE_STATION,
                    str(type_key),
                    _json_dump(cleaned),
                    now,
                    now,
                ),
            )

    def _insert_device_locked(self, scenario_name: str, device: Dict[str, Any], *, now: str) -> None:
        device_uid = str(device.get("device_uid") or "")
        if not device_uid:
            raise ValueError("scenario_devices.device_uid is required")
        row_index = int(device.get("row_index", device.get("x", 0)) or 0)
        col_index = int(device.get("col_index", device.get("y", 0)) or 0)
        device_kind = str(device.get("device_kind") or self.DEVICE_KIND_BASE_STATION)
        type_key = str(device.get("type_key") or device.get("base_station") or device.get("device_type") or "custom")
        known_keys = {
            "device_uid",
            "device_kind",
            "type_key",
            "base_station",
            "x",
            "y",
            "row_index",
            "col_index",
            "mode",
            "status",
            "device_name",
            "device_category",
            "deployment_id",
            "station_type",
            "station_label",
            "cell_user_count",
            "coverage_radius",
            "coverage_radius_km",
            "max_throughput",
            "max_users",
            "downlink_bandwidth_mbps",
            "uplink_bandwidth_mbps",
            "tx_power_watt",
            "battery_duration_h",
            "demand_mbps",
            "connected",
            "broadcast_served",
            "notes",
            "override_fields",
        }
        extra = {key: value for key, value in device.items() if key not in known_keys and value is not None}
        self._conn.execute(
            """
            INSERT INTO scenario_devices (
                device_uid, scenario_name, device_kind, row_index, col_index, type_key, mode, status,
                device_name, device_category, source_deployment_id, source_station_type,
                source_station_label, cell_user_count, coverage_radius, coverage_radius_km,
                max_throughput, max_users, downlink_bandwidth_mbps, uplink_bandwidth_mbps,
                tx_power_watt, battery_duration_h, demand_mbps, connected, broadcast_served,
                notes, override_fields_json, extra_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                device_uid,
                scenario_name,
                device_kind,
                row_index,
                col_index,
                type_key,
                device.get("mode"),
                device.get("status") or "active",
                device.get("device_name"),
                device.get("device_category"),
                device.get("deployment_id"),
                device.get("station_type"),
                device.get("station_label"),
                device.get("cell_user_count"),
                device.get("coverage_radius"),
                device.get("coverage_radius_km"),
                device.get("max_throughput"),
                device.get("max_users"),
                device.get("downlink_bandwidth_mbps"),
                device.get("uplink_bandwidth_mbps"),
                device.get("tx_power_watt"),
                device.get("battery_duration_h"),
                device.get("demand_mbps"),
                self._bool_to_int(device.get("connected")),
                self._bool_to_int(device.get("broadcast_served")),
                device.get("notes"),
                _json_dump(device.get("override_fields") or []),
                _json_dump(extra),
                now,
                now,
            ),
        )

    def _status_counts_locked(self, scenario_name: str) -> Dict[str, int]:
        rows = self._conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM scenario_devices
            WHERE scenario_name = ? AND device_kind = ?
            GROUP BY status
            """,
            (scenario_name, self.DEVICE_KIND_BASE_STATION),
        ).fetchall()
        counts = {"total": 0, "active": 0, "degraded": 0, "offline": 0}
        for row in rows:
            status = str(row["status"] or "unknown")
            count = int(row["count"] or 0)
            counts["total"] += count
            if status in counts:
                counts[status] = count
        return counts

    def _append_event_locked(
        self,
        scenario_name: str,
        operation: str,
        *,
        device_uid: Optional[str] = None,
        device_kind: Optional[str] = None,
        type_key: Optional[str] = None,
        row_index: Optional[int] = None,
        col_index: Optional[int] = None,
        device_count: Optional[int] = None,
        active_count: Optional[int] = None,
        degraded_count: Optional[int] = None,
        offline_count: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
        created_at: Optional[Any] = None,
    ) -> None:
        if device_count is None:
            counts = self._status_counts_locked(scenario_name)
            device_count = counts["total"]
            active_count = counts["active"]
            degraded_count = counts["degraded"]
            offline_count = counts["offline"]
        timestamp = self._iso_from_timestamp(created_at)
        self._conn.execute(
            """
            INSERT INTO scenario_device_events (
                scenario_name, operation, device_uid, device_kind, type_key, row_index, col_index,
                device_count, active_count, degraded_count, offline_count, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_name,
                operation,
                device_uid,
                device_kind,
                type_key,
                row_index,
                col_index,
                int(device_count or 0),
                int(active_count or 0),
                int(degraded_count or 0),
                int(offline_count or 0),
                _json_dump(payload or {}),
                timestamp,
            ),
        )

    def _row_to_device_spec(self, row: sqlite3.Row) -> Dict[str, Any]:
        extra = _json_load(row["extra_json"], {})
        spec: Dict[str, Any] = {
            **(extra if isinstance(extra, dict) else {}),
            "device_uid": row["device_uid"],
            "base_station": row["type_key"],
            "mode": row["mode"],
            "x": int(row["row_index"]),
            "y": int(row["col_index"]),
            "status": row["status"],
        }
        field_map = {
            "deployment_id": "source_deployment_id",
            "device_name": "device_name",
            "device_category": "device_category",
            "station_type": "source_station_type",
            "station_label": "source_station_label",
            "cell_user_count": "cell_user_count",
            "coverage_radius": "coverage_radius",
            "coverage_radius_km": "coverage_radius_km",
            "max_throughput": "max_throughput",
            "max_users": "max_users",
            "downlink_bandwidth_mbps": "downlink_bandwidth_mbps",
            "uplink_bandwidth_mbps": "uplink_bandwidth_mbps",
            "tx_power_watt": "tx_power_watt",
            "battery_duration_h": "battery_duration_h",
            "demand_mbps": "demand_mbps",
            "notes": "notes",
        }
        for public_key, column in field_map.items():
            value = row[column]
            if value is not None:
                spec[public_key] = value
        override_fields = _json_load(row["override_fields_json"], [])
        if override_fields:
            spec["override_fields"] = override_fields
        if row["connected"] is not None:
            spec["connected"] = bool(row["connected"])
        if row["broadcast_served"] is not None:
            spec["broadcast_served"] = bool(row["broadcast_served"])
        return spec

    def _row_to_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        payload = _json_load(row["payload_json"], {})
        event = payload if isinstance(payload, dict) else {}
        event.setdefault("timestamp", self._timestamp_from_iso(row["created_at"]))
        event.setdefault("operation", row["operation"])
        event.setdefault("device_count", int(row["device_count"] or 0))
        event.setdefault("active_count", int(row["active_count"] or 0))
        event.setdefault("degraded_count", int(row["degraded_count"] or 0))
        event.setdefault("offline_count", int(row["offline_count"] or 0))
        return event

    @staticmethod
    def _bool_to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        return 1 if bool(value) else 0

    @staticmethod
    def _timestamp_from_iso(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            text = str(value)
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            return datetime.fromisoformat(text).timestamp()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _iso_from_timestamp(value: Any) -> str:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if isinstance(value, str) and value:
            return value
        return _utc_now()
