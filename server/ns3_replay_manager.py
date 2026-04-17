"""ns-3 replay integration for the main RescueNet API."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class Ns3ReplayManager:
    """Owns ns-3 replay data import, experiment queries, and simulation launch."""

    def __init__(self, ns3_root: str | Path = "ns-3.46.1") -> None:
        project_root = Path(__file__).resolve().parents[1]
        candidate = Path(ns3_root)
        self.ns3_root = candidate if candidate.is_absolute() else project_root / candidate
        self.db_file = self.ns3_root / "simulation_history.db"
        self.trace_file = self.ns3_root / "trace.json"
        self.ns3_script = self.ns3_root / "ns3"
        self.native_index = self.ns3_root / "index.html"
        self.log_file = self.ns3_root / "ns3_backend.log"

        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_running = False
        self._watcher_stop = threading.Event()

        self._process_lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._started_at: Optional[float] = None
        self._last_finished_at: Optional[float] = None
        self._last_exit_code: Optional[int] = None
        self._last_error: Optional[str] = None

        self.init_db()

    def init_db(self) -> None:
        self.ns3_root.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS experiments
               (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, date TEXT,
                duration REAL, total_nodes INTEGER, disaster_time REAL, frames INTEGER)"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS frame_data
               (id INTEGER PRIMARY KEY AUTOINCREMENT, exp_id INTEGER, frame_idx INTEGER,
                time REAL, tp REAL, loss REAL, disaster INTEGER, data_json TEXT)"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS node_stats
               (id INTEGER PRIMARY KEY AUTOINCREMENT, exp_id INTEGER, node_type INTEGER, total_count INTEGER)"""
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_exp ON frame_data(exp_id)")
        conn.commit()
        conn.close()

    def start(self) -> None:
        self.import_trace()
        if self._watcher_running:
            return
        self._watcher_running = True
        self._watcher_stop.clear()
        self._watcher_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()

        if self.experiment_count() == 0 and not self.trace_file.exists():
            self.start_simulation()

    def stop(self) -> None:
        self._watcher_running = False
        self._watcher_stop.set()

    def _watch_loop(self) -> None:
        while not self._watcher_stop.wait(2.0):
            if self.trace_file.exists():
                time.sleep(1.0)
                self.import_trace()

    def _get_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def experiment_count(self) -> int:
        conn = self._get_db()
        try:
            row = conn.execute("SELECT COUNT(*) AS count FROM experiments").fetchone()
            return int(row["count"] if row else 0)
        finally:
            conn.close()

    def import_trace(self) -> Optional[int]:
        if not self.trace_file.exists():
            return None

        try:
            content = self.trace_file.read_text(encoding="utf-8").strip()
            if content.endswith(","):
                content = content[:-1]
            if not content.startswith("["):
                content = "[" + content
            if not content.endswith("]"):
                content = content + "]"

            frames = json.loads(content)
            if not frames:
                return None

            first = frames[0]
            node_count = len(first.get("nodes", []))
            disaster_time = 0.0
            for frame in frames:
                if frame.get("disaster", 0) == 1:
                    disaster_time = float(frame.get("time", 0))
                    break
            duration = float(frames[-1].get("time", 0)) if frames else 0.0

            conn = self._get_db()
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO experiments (name, date, duration, total_nodes, disaster_time, frames)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    f"演练_{int(time.time())}",
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    duration,
                    node_count,
                    disaster_time,
                    len(frames),
                ),
            )
            exp_id = int(cur.lastrowid)

            for idx, frame in enumerate(frames):
                data_json = json.dumps(
                    {"nodes": frame.get("nodes", []), "links": frame.get("links", [])},
                    ensure_ascii=False,
                )
                cur.execute(
                    """INSERT INTO frame_data (exp_id, frame_idx, time, tp, loss, disaster, data_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        exp_id,
                        idx,
                        float(frame.get("time", 0)),
                        float(frame.get("tp", 0)),
                        float(frame.get("loss", 0)),
                        int(frame.get("disaster", 0)),
                        data_json,
                    ),
                )

            conn.commit()
            conn.close()
            self.trace_file.unlink(missing_ok=True)
            return exp_id
        except Exception as exc:
            self._last_error = f"trace import failed: {exc}"
            return None

    def list_experiments(self) -> List[Dict[str, Any]]:
        conn = self._get_db()
        try:
            rows = conn.execute("SELECT * FROM experiments ORDER BY id DESC").fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_charts(self, exp_id: int) -> Dict[str, List[float]]:
        conn = self._get_db()
        try:
            rows = conn.execute(
                """SELECT time, tp, loss FROM frame_data
                   WHERE exp_id = ? ORDER BY frame_idx""",
                (exp_id,),
            ).fetchall()
            return {
                "times": [float(row["time"]) for row in rows],
                "tps": [float(row["tp"]) for row in rows],
                "losses": [float(row["loss"]) for row in rows],
            }
        finally:
            conn.close()

    def get_frame(self, exp_id: int, frame_idx: int) -> Optional[Dict[str, Any]]:
        conn = self._get_db()
        try:
            row = conn.execute(
                "SELECT * FROM frame_data WHERE exp_id = ? AND frame_idx = ?",
                (exp_id, frame_idx),
            ).fetchone()
            if not row:
                return None
            data = json.loads(row["data_json"])
            return {
                "time": float(row["time"]),
                "tp": float(row["tp"]),
                "loss": float(row["loss"]),
                "disaster": int(row["disaster"]),
                "nodes": data.get("nodes", []),
                "links": data.get("links", []),
            }
        finally:
            conn.close()

    def start_simulation(self, force: bool = False) -> Dict[str, Any]:
        with self._process_lock:
            if self._process and self._process.poll() is None and not force:
                return self._status_unlocked()
            if not self.ns3_script.exists():
                self._last_error = f"ns3 launcher not found: {self.ns3_script}"
                return self._status_unlocked()
            if not os.access(self.ns3_script, os.X_OK):
                self.ns3_script.chmod(self.ns3_script.stat().st_mode | 0o111)

            self._last_error = None
            self._started_at = time.time()
            self._last_finished_at = None
            self._last_exit_code = None
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            command = (
                "export NS3_ALLOW_ROOT=1; "
                "./ns3 configure && "
                "./ns3 build && "
                "./ns3 run scratch/disaster-pro"
            )
            log_handle = open(self.log_file, "w", encoding="utf-8")
            self._process = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=self.ns3_root,
                env={**os.environ, "NS3_ALLOW_ROOT": "1"},
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            threading.Thread(target=self._wait_for_process, daemon=True).start()
            return self._status_unlocked()

    def _wait_for_process(self) -> None:
        proc: Optional[subprocess.Popen[str]]
        with self._process_lock:
            proc = self._process
        if proc is None:
            return
        exit_code = proc.wait()
        with self._process_lock:
            self._last_exit_code = exit_code
            self._last_finished_at = time.time()
            if exit_code != 0:
                self._last_error = f"ns-3 exited with code {exit_code}"
            self._process = None

    def status(self) -> Dict[str, Any]:
        with self._process_lock:
            return self._status_unlocked()

    def _status_unlocked(self) -> Dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        pid = self._process.pid if running and self._process else None
        return {
            "available": self.ns3_script.exists() and self.native_index.exists(),
            "running": running,
            "pid": pid,
            "started_at": self._started_at,
            "last_finished_at": self._last_finished_at,
            "last_exit_code": self._last_exit_code,
            "last_error": self._last_error,
            "db_exists": self.db_file.exists(),
            "trace_exists": self.trace_file.exists(),
            "experiment_count": self.experiment_count(),
            "native_path": "/ns3-native/index.html",
            "log_path": str(self.log_file),
        }
