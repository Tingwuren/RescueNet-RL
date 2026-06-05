"""SQLite-backed CRUD manager for dedicated emergency communication devices."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.schemas import DedicatedDevice, DedicatedDeviceCreate, DedicatedDeviceListResponse, DedicatedDeviceUpdate


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_device_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = uuid.uuid4().hex[:4].upper()
    return f"DED-{ts}-{suffix}"


class DeviceManager:
    """Persist dedicated devices in a local SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dedicated_devices (
                    device_id TEXT PRIMARY KEY,
                    device_name TEXT NOT NULL,
                    device_type TEXT NOT NULL DEFAULT 'custom',
                    device_category TEXT NOT NULL DEFAULT '其他',
                    is_dedicated INTEGER NOT NULL DEFAULT 1,
                    coverage_radius_km REAL NOT NULL DEFAULT 1.0,
                    downlink_bandwidth_mbps REAL NOT NULL DEFAULT 10.0,
                    uplink_bandwidth_mbps REAL NOT NULL DEFAULT 5.0,
                    max_users INTEGER NOT NULL DEFAULT 50,
                    tx_power_watt REAL,
                    battery_duration_h REAL,
                    supported_modes TEXT NOT NULL DEFAULT '[]',
                    image_url TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    deploy_position TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    bound_scenario TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dedicated_device_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    device_name TEXT NOT NULL,
                    device_type TEXT NOT NULL,
                    device_category TEXT NOT NULL,
                    result_status TEXT NOT NULL
                )
                """
            )

    def create(self, payload: DedicatedDeviceCreate) -> DedicatedDevice:
        data = payload.model_dump()
        now = _utc_now()
        for _ in range(8):
            device_id = generate_device_id()
            try:
                with self._lock, self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO dedicated_devices (
                            device_id, device_name, device_type, device_category, is_dedicated,
                            coverage_radius_km, downlink_bandwidth_mbps, uplink_bandwidth_mbps,
                            max_users, tx_power_watt, battery_duration_h, supported_modes,
                            image_url, status, deploy_position, created_at, updated_at, bound_scenario
                        ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?, ?, ?)
                        """,
                        (
                            device_id,
                            data["device_name"],
                            data["device_type"],
                            data["device_category"],
                            data["coverage_radius_km"],
                            data["downlink_bandwidth_mbps"],
                            data["uplink_bandwidth_mbps"],
                            data["max_users"],
                            data.get("tx_power_watt"),
                            data.get("battery_duration_h"),
                            json.dumps(data.get("supported_modes") or [], ensure_ascii=False),
                            data.get("image_url"),
                            now,
                            now,
                            data.get("bound_scenario"),
                        ),
                    )
                    device = self.get(device_id)
                    if device is None:
                        raise RuntimeError(f"Created device disappeared: {device_id}")
                    self._append_log("create", device)
                    return device
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("Failed to generate a unique dedicated device ID")

    def list(self, status: Optional[str] = None, device_type: Optional[str] = None) -> DedicatedDeviceListResponse:
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if device_type:
            clauses.append("device_type = ?")
            params.append(device_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM dedicated_devices {where} ORDER BY created_at DESC, device_id DESC",
            params,
        ).fetchall()
        devices = [self._row_to_device(row) for row in rows]
        active_count = int(
            self._conn.execute("SELECT COUNT(*) FROM dedicated_devices WHERE status = 'active'").fetchone()[0]
        )
        return DedicatedDeviceListResponse(devices=devices, total=len(devices), active_count=active_count)

    def get(self, device_id: str) -> Optional[DedicatedDevice]:
        row = self._conn.execute("SELECT * FROM dedicated_devices WHERE device_id = ?", (device_id,)).fetchone()
        return self._row_to_device(row) if row else None

    def update(self, device_id: str, payload: DedicatedDeviceUpdate) -> Optional[DedicatedDevice]:
        updates = payload.model_dump(exclude_unset=True)
        if not updates:
            return self.get(device_id)
        updates["updated_at"] = _utc_now()

        columns = []
        params: List[Any] = []
        for key, value in updates.items():
            columns.append(f"{key} = ?")
            if key in {"supported_modes", "deploy_position"}:
                params.append(json.dumps(value, ensure_ascii=False) if value is not None else None)
            else:
                params.append(value)
        params.append(device_id)

        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"UPDATE dedicated_devices SET {', '.join(columns)} WHERE device_id = ?",
                params,
            )
            if cursor.rowcount <= 0:
                return None
            device = self.get(device_id)
            if device:
                self._append_log("update", device)
            return device

    def update_status(self, device_id: str, status: str) -> Optional[DedicatedDevice]:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE dedicated_devices SET status = ?, updated_at = ? WHERE device_id = ?",
                (status, _utc_now(), device_id),
            )
            if cursor.rowcount <= 0:
                return None
            device = self.get(device_id)
            if device:
                self._append_log("enable" if status == "active" else "disable", device)
            return device

    def delete(self, device_id: str) -> bool:
        device = self.get(device_id)
        if device is None:
            return False
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM dedicated_devices WHERE device_id = ?", (device_id,))
            if cursor.rowcount <= 0:
                return False
            self._append_log("delete", device, result_status="deleted")
            return True

    def _append_log(self, operation: str, device: DedicatedDevice, result_status: Optional[str] = None) -> None:
        self._conn.execute(
            """
            INSERT INTO dedicated_device_logs (
                timestamp, operation, device_id, device_name, device_type, device_category, result_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                operation,
                device.device_id,
                device.device_name,
                device.device_type,
                device.device_category,
                result_status or device.status,
            ),
        )

    def _row_to_device(self, row: sqlite3.Row) -> DedicatedDevice:
        def parse_json(value: Any, fallback: Any) -> Any:
            if value in (None, ""):
                return fallback
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return fallback

        return DedicatedDevice(
            device_id=str(row["device_id"]),
            device_name=str(row["device_name"]),
            device_type=row["device_type"],
            device_category=str(row["device_category"]),
            is_dedicated=bool(row["is_dedicated"]),
            coverage_radius_km=float(row["coverage_radius_km"]),
            downlink_bandwidth_mbps=float(row["downlink_bandwidth_mbps"]),
            uplink_bandwidth_mbps=float(row["uplink_bandwidth_mbps"]),
            max_users=int(row["max_users"]),
            tx_power_watt=row["tx_power_watt"],
            battery_duration_h=row["battery_duration_h"],
            supported_modes=parse_json(row["supported_modes"], []),
            image_url=row["image_url"],
            status=row["status"],
            deploy_position=parse_json(row["deploy_position"], None),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            bound_scenario=row["bound_scenario"],
        )
