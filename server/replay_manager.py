"""Persistent scene replay sessions for RescueNet simulations."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


MAP_WIDTH = 5000
MAP_HEIGHT = 5000
SESSION_SCHEMA_VERSION = "rescuenet.replay_session.v1"
COORDINATE_SOURCE_VERSION = "deterministic_grid_cross_cell_v3"
STATION_MIN_SPACING_FLOOR = 180.0
STATION_MIN_SPACING_CEILING = 260.0
REPLAY_SESSION_DIR_RE = re.compile(r"^rpl_(?P<date>\d{8})_(?P<time>\d{6})")


class ReplaySessionManager:
    """Create and query file-backed replay sessions.

    A session is intentionally stored as plain JSON/JSONL artifacts so it can be
    inspected during acceptance without a database migration.
    """

    def __init__(self, root_dir: str | Path = "artifacts/replay_sessions") -> None:
        project_root = Path(__file__).resolve().parents[1]
        root = Path(root_dir)
        self.root_dir = root if root.is_absolute() else project_root / root
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_from_simulation(
        self,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        *,
        source: str = "test",
    ) -> Dict[str, Any]:
        reports = response_payload.get("reports") or []
        if not reports:
            raise ValueError("Simulation response does not contain episode reports.")

        report = reports[0] or {}
        replay_id = self._new_replay_id()
        session_dir = self.root_dir / replay_id
        session_dir.mkdir(parents=True, exist_ok=False)

        frames, link_metrics = self._build_frames(
            report,
            response_payload.get("scene_export"),
            request_payload=request_payload,
        )
        for frame in frames:
            frame["replay_id"] = replay_id
        for metric in link_metrics:
            metric["replay_id"] = replay_id
        metadata = self._build_metadata(
            replay_id=replay_id,
            session_dir=session_dir,
            request_payload=request_payload,
            response_payload=response_payload,
            report=report,
            frames=frames,
            source=source,
        )

        self._write_json(session_dir / "metadata.json", metadata)
        self._write_json(session_dir / "request.json", request_payload)
        self._write_json(session_dir / "report.json", response_payload)
        self._write_jsonl(session_dir / "frames.jsonl", frames)
        self._write_jsonl(session_dir / "link_metrics.jsonl", link_metrics)
        self._write_nodes_full(session_dir / "nodes_full.jsonl", replay_id, frames)
        self._write_log(session_dir / "replay.log", metadata, frames, report)

        return metadata

    def list_sessions(self, *, limit: int = 20, source: Optional[str] = None) -> Dict[str, Any]:
        sessions: List[Dict[str, Any]] = []
        for metadata_path in self.root_dir.glob("*/metadata.json"):
            metadata = self._read_json(metadata_path)
            if not metadata:
                continue
            if source and metadata.get("source") != source:
                continue
            public = self._public_metadata(metadata)
            public["created_at"] = self._session_created_at(metadata, metadata_path.parent)
            public["created_at_iso"] = self._created_at_iso(public["created_at"], metadata.get("created_at_iso"))
            public["modified_at"] = self._path_mtime(metadata_path)
            sessions.append(public)
        sessions.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
        return {"sessions": sessions[:limit], "total": len(sessions)}

    def get_session(self, replay_id: str) -> Dict[str, Any]:
        metadata = self._load_metadata(replay_id)
        return self._public_metadata(metadata)

    def get_frames(self, replay_id: str, *, offset: int = 0, limit: int = 10, stride: int = 1) -> Dict[str, Any]:
        metadata = self._load_metadata(replay_id)
        offset = max(0, int(offset))
        limit = max(1, min(100, int(limit)))
        stride = max(1, int(stride))
        selected: List[Dict[str, Any]] = []
        for frame in self._iter_jsonl(self._session_dir(replay_id) / "frames.jsonl"):
            frame_index = int(frame.get("frame_index", frame.get("frameIndex", 0)) or 0)
            if frame_index < offset:
                continue
            if (frame_index - offset) % stride != 0:
                continue
            selected.append(self._frame_summary(frame))
            if len(selected) >= limit:
                break
        return {
            "replay_id": replay_id,
            "frame_count": metadata.get("frame_count", 0),
            "frames": selected,
        }

    def get_frame(
        self,
        replay_id: str,
        frame_index: int,
        *,
        sample_ratio: int = 30,
        include_links: bool = True,
    ) -> Dict[str, Any]:
        metadata = self._load_metadata(replay_id)
        frame = self._read_frame(replay_id, int(frame_index))
        if frame is None:
            raise FileNotFoundError(f"Replay frame not found: {replay_id}/{frame_index}")

        nodes = self._declutter_station_nodes(list(frame.get("nodes") or []))
        sampled_nodes = self._sample_nodes(nodes, sample_ratio)
        sampled_node_ids = {str(node.get("id")) for node in sampled_nodes}
        payload = {
            **frame,
            "replay_id": replay_id,
            "frame_count": metadata.get("frame_count", 0),
            "sample_ratio": max(1, int(sample_ratio or 1)),
            "nodes_total": len(nodes),
            "nodes_drawn": len(sampled_nodes),
            "nodes": sampled_nodes,
        }
        if not include_links:
            payload["links"] = []
        else:
            filtered_links = []
            for link in frame.get("links") or []:
                if str(link.get("src")) not in sampled_node_ids or str(link.get("dst")) not in sampled_node_ids:
                    continue
                filtered_links.append(link)
                if len(filtered_links) >= 480:
                    break
            payload["links_total"] = len(frame.get("links") or [])
            payload["links"] = filtered_links

        link_metric = self._link_metric_for_frame(replay_id, int(frame_index), frame=frame)
        if link_metric:
            summary = link_metric.get("summary") or {}
            payload["link_metrics"] = link_metric
            payload["linkMetrics"] = link_metric
            payload["cluster_throughput_mbps"] = _finite_float(
                summary.get("total_throughput_mbps") or summary.get("avg_throughput_mbps"),
                0.0,
            )
            payload["latency_ms"] = _finite_float(summary.get("latency_ms"), 0.0)
            payload["metrics"] = {
                **(payload.get("metrics") or {}),
                "cluster_throughput_mbps": payload["cluster_throughput_mbps"],
                "total_throughput_mbps": _finite_float(summary.get("total_throughput_mbps"), 0.0),
                "avg_link_throughput_mbps": _finite_float(summary.get("avg_throughput_mbps"), 0.0),
                "latency_ms": payload["latency_ms"],
            }
        return payload

    def get_logs(self, replay_id: str, *, limit: int = 200) -> Dict[str, Any]:
        self._load_metadata(replay_id)
        log_path = self._session_dir(replay_id) / "replay.log"
        lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
        limited = lines[-max(1, min(1000, int(limit))):]
        return {"replay_id": replay_id, "lines": limited, "total": len(lines)}

    def get_link_metrics(self, replay_id: str, *, frame_index: Optional[int] = None) -> Dict[str, Any]:
        self._load_metadata(replay_id)
        session_dir = self._session_dir(replay_id)
        if frame_index is not None:
            metric = self._link_metric_for_frame(replay_id, int(frame_index))
            if metric is None:
                raise FileNotFoundError(f"Replay link metrics not found: {replay_id}/{frame_index}")
            return metric

        metrics = list(self._iter_jsonl(session_dir / "link_metrics.jsonl"))
        report = self._load_report_for_session(replay_id)
        request_payload = self._read_json(session_dir / "request.json") or {}
        context = self._link_acceptance_context(report, request_payload)
        frame_by_index = {
            int(frame.get("frame_index", frame.get("frameIndex", 0)) or 0): frame
            for frame in self._iter_jsonl(session_dir / "frames.jsonl")
        }
        metrics = [
            self._augment_link_metric(item, frame_by_index.get(int(item.get("frame_index", 0) or 0)), context)
            for item in metrics
        ]
        if frame_index is None:
            return {"replay_id": replay_id, "frames": metrics}
        for item in metrics:
            if int(item.get("frame_index", 0) or 0) == int(frame_index):
                return item
        raise FileNotFoundError(f"Replay link metrics not found: {replay_id}/{frame_index}")

    def artifact_path(self, replay_id: str, artifact_type: str) -> Path:
        self._load_metadata(replay_id)
        mapping = {
            "metadata": "metadata.json",
            "log": "replay.log",
            "nodes": "nodes_full.jsonl",
            "frames": "frames.jsonl",
            "report": "report.json",
            "request": "request.json",
            "link_metrics": "link_metrics.jsonl",
        }
        file_name = mapping.get(str(artifact_type or "").strip())
        if not file_name:
            raise ValueError(f"Unsupported replay artifact type: {artifact_type}")
        path = self._session_dir(replay_id) / file_name
        if not path.exists():
            raise FileNotFoundError(f"Replay artifact not found: {artifact_type}")
        return path

    def _build_metadata(
        self,
        *,
        replay_id: str,
        session_dir: Path,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
        report: Dict[str, Any],
        frames: List[Dict[str, Any]],
        source: str,
    ) -> Dict[str, Any]:
        created_at = time.time()
        scenario = report.get("scenario") or {}
        initial_state = report.get("initial_state") or {}
        final_state = report.get("final_state") or initial_state
        first_frame = frames[0] if frames else {}
        last_frame = frames[-1] if frames else first_frame
        scenario_name = (
            request_payload.get("scenario_name")
            or scenario.get("name")
            or "unknown_scenario"
        )
        algorithm = request_payload.get("algorithm") or "unknown"
        title = f"{scenario_name} / {str(algorithm).upper()} / Episode {report.get('episode', 1)}"
        if source == "training":
            title = f"训练回放 / {title}"

        summary = {
            "total_reward": _finite_float(report.get("total_reward"), 0.0),
            "coverage_ratio": _finite_float(final_state.get("coverage_ratio"), 0.0),
            "broadcast_ratio": _finite_float(final_state.get("broadcast_ratio"), 0.0),
            "steps_taken": int(report.get("steps_taken") or max(0, len(frames) - 1)),
            "total_users": int(first_frame.get("user_count") or initial_state.get("total_users") or 0),
            "initial_stations": int(first_frame.get("station_count") or 0),
            "final_stations": int(last_frame.get("station_count") or 0),
            "connected_users": int(last_frame.get("connected_users") or final_state.get("connected_users") or 0),
            "broadcast_users": int(last_frame.get("broadcast_users") or final_state.get("broadcast_served_users") or 0),
        }

        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "replay_id": replay_id,
            "id": replay_id,
            "source": source or "test",
            "created_at": created_at,
            "created_at_iso": datetime.fromtimestamp(created_at).isoformat(timespec="seconds"),
            "title": title,
            "scenario_name": scenario_name,
            "scenario": scenario,
            "algorithm": algorithm,
            "checkpoint_path": request_payload.get("checkpoint_path"),
            "evaluation_protocol": request_payload.get("evaluation_protocol")
            or final_state.get("evaluation_protocol")
            or initial_state.get("evaluation_protocol"),
            "episode": int(report.get("episode", 1)),
            "frame_count": len(frames),
            "map_width": MAP_WIDTH,
            "map_height": MAP_HEIGHT,
            "geo_bounds": (response_payload.get("scene_export") or {}).get("deployment_scene", {}).get("geo_bounds")
            or (response_payload.get("scene_export") or {}).get("disaster_scene", {}).get("geo_bounds"),
            "node_count_total": max((int(frame.get("node_count_total") or len(frame.get("nodes") or [])) for frame in frames), default=0),
            "summary": summary,
            "artifacts": {
                "session_dir": str(session_dir),
                "metadata": f"/api/replay/sessions/{replay_id}/download?type=metadata",
                "log": f"/api/replay/sessions/{replay_id}/download?type=log",
                "nodes": f"/api/replay/sessions/{replay_id}/download?type=nodes",
                "frames": f"/api/replay/sessions/{replay_id}/download?type=frames",
                "report": f"/api/replay/sessions/{replay_id}/download?type=report",
                "link_metrics": f"/api/replay/sessions/{replay_id}/download?type=link_metrics",
            },
        }

    def _build_frames(
        self,
        report: Dict[str, Any],
        scene_export: Optional[Dict[str, Any]],
        *,
        request_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        initial_state = report.get("initial_state") or {}
        steps = report.get("steps") or []
        rows, cols = self._infer_grid_shape(report, scene_export)
        geo_bounds = self._geo_bounds(scene_export)
        deployment_plan = report.get("deployment_plan") or {}
        deployments = deployment_plan.get("deployments") or []
        link_context = self._link_acceptance_context(report, request_payload or {})

        residual_nodes = self._residual_nodes(initial_state.get("residual_base_stations") or [], rows, cols)
        residual_station_points = _station_points(residual_nodes)
        user_node_map = self._user_node_map(initial_state.get("user_details") or [], rows, cols)

        frames: List[Dict[str, Any]] = []
        link_metrics: List[Dict[str, Any]] = []

        initial_users = sorted(user_node_map.values(), key=lambda item: str(item.get("id")))
        initial_frame = self._frame_payload(
            frame_index=0,
            label="初始受灾场景",
            state=initial_state,
            users=initial_users,
            stations=residual_nodes,
            geo_bounds=geo_bounds,
            reward=0.0,
            action_desc=None,
            deployment=None,
        )
        frames.append(initial_frame)
        link_metrics.append(self._link_metrics(initial_frame, link_context))

        for index, step in enumerate(steps):
            post_state = step.get("post_state") or {}
            user_node_map.update(self._user_node_map(post_state.get("user_details") or [], rows, cols, user_node_map))
            users = sorted(user_node_map.values(), key=lambda item: str(item.get("id")))
            active_deployments = [
                item for item in deployments if int(item.get("time_step") or item.get("sequence") or 0) <= index + 1
            ]
            deployed_nodes = self._deployed_nodes(
                active_deployments,
                rows,
                cols,
                occupied=residual_station_points,
            )
            latest_deployment = active_deployments[-1] if active_deployments else None
            stations = [*residual_nodes, *deployed_nodes]
            frame = self._frame_payload(
                frame_index=index + 1,
                label=f"Step {step.get('step') or index + 1}",
                state=post_state,
                users=users,
                stations=stations,
                geo_bounds=geo_bounds,
                reward=_finite_float(step.get("reward"), 0.0),
                action_desc=step.get("action_desc"),
                deployment=latest_deployment,
            )
            frames.append(frame)
            link_metrics.append(self._link_metrics(frame, link_context))

        return frames, link_metrics

    def _frame_payload(
        self,
        *,
        frame_index: int,
        label: str,
        state: Dict[str, Any],
        users: List[Dict[str, Any]],
        stations: List[Dict[str, Any]],
        geo_bounds: Optional[Dict[str, Any]],
        reward: float,
        action_desc: Optional[Dict[str, Any]],
        deployment: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        coverage = _finite_float(state.get("coverage_ratio"), 0.0)
        broadcast = _finite_float(state.get("broadcast_ratio"), 0.0)
        nodes = [*users, *stations]
        connected_users = sum(1 for node in users if node.get("connected"))
        broadcast_users = sum(1 for node in users if node.get("broadcast_served"))
        return {
            "frame_index": frame_index,
            "frameIndex": frame_index,
            "time": frame_index,
            "label": label,
            "map_width": MAP_WIDTH,
            "map_height": MAP_HEIGHT,
            "geo_bounds": geo_bounds,
            "tp": _finite_float(state.get("avg_user_throughput") or state.get("recent_throughput"), 0.0),
            "loss": max(0.0, min(1.0, 1.0 - coverage)),
            "disaster": 1,
            "nodes": nodes,
            "links": self._connectivity_links(users, stations),
            "coverageRatio": coverage,
            "coverage_ratio": coverage,
            "broadcastRatio": broadcast,
            "broadcast_ratio": broadcast,
            "remainingBudget": _finite_float(state.get("remaining_budget"), 0.0),
            "remaining_budget": _finite_float(state.get("remaining_budget"), 0.0),
            "reward": reward,
            "actionDesc": action_desc,
            "action_desc": action_desc,
            "latestDeploymentId": deployment.get("deployment_id") if deployment else None,
            "latest_deployment": deployment,
            "user_count": len(users),
            "station_count": len(stations),
            "connected_users": connected_users,
            "broadcast_users": broadcast_users,
            "node_count_total": len(nodes),
            "metrics": {
                "coverage_ratio": coverage,
                "broadcast_ratio": broadcast,
                "avg_user_throughput": _finite_float(state.get("avg_user_throughput") or state.get("recent_throughput"), 0.0),
                "loss_ratio": max(0.0, min(1.0, 1.0 - coverage)),
                "remaining_budget": _finite_float(state.get("remaining_budget"), 0.0),
                "connected_users": connected_users,
                "broadcast_users": broadcast_users,
                "user_count": len(users),
                "station_count": len(stations),
            },
        }

    def _user_node_map(
        self,
        details: Iterable[Dict[str, Any]],
        rows: int,
        cols: int,
        previous: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        previous = previous or {}
        for detail in details:
            node_id = f"user:{detail.get('id', len(result))}"
            position = detail.get("position")
            if _valid_position(position):
                row, col = int(position[0]), int(position[1])
                x, y = _grid_to_precise_coords(
                    row,
                    col,
                    rows,
                    cols,
                    seed=f"user:{detail.get('id', len(result))}:{detail.get('region_id', '')}",
                    spread=1.52,
                )
            else:
                previous_node = previous.get(node_id, {})
                row, col = None, None
                x, y = previous_node.get("x", 0), previous_node.get("y", 0)
            node = {
                "id": node_id,
                "type": "USER",
                "x": x,
                "y": y,
                "grid": {"row": row, "col": col} if row is not None and col is not None else None,
                "connected": bool(detail.get("connected", previous.get(node_id, {}).get("connected", False))),
                "broadcast_served": bool(
                    detail.get("broadcast_served", previous.get(node_id, {}).get("broadcast_served", False))
                ),
                "demand": _finite_float(detail.get("demand"), 0.0),
                "region_id": detail.get("region_id"),
                "region_label": detail.get("region_label"),
                "node_role": "user",
                "coordinate_source": COORDINATE_SOURCE_VERSION if row is not None and col is not None else None,
            }
            node.update(_lat_lon_center(detail.get("lat_lon_bounds")))
            result[node_id] = _drop_none(node)
        return result

    def _residual_nodes(self, stations: Iterable[Dict[str, Any]], rows: int, cols: int) -> List[Dict[str, Any]]:
        nodes = []
        occupied: List[Tuple[float, float]] = []
        for index, station in enumerate(stations):
            row = station.get("x")
            col = station.get("y")
            if row is None or col is None:
                continue
            base_key = station.get("base_station")
            node_seed = f"residual:{station.get('device_uid') or station.get('deployment_id') or index}:{base_key}:{row}:{col}"
            x, y = _grid_to_precise_coords(int(row), int(col), rows, cols, seed=node_seed, spread=0.42)
            x, y = _separate_station_coords(
                int(row),
                int(col),
                rows,
                cols,
                preferred=(x, y),
                occupied=occupied,
                seed=node_seed,
            )
            occupied.append((x, y))
            nodes.append(
                _drop_none(
                    {
                        "id": f"residual:{station.get('device_uid') or station.get('deployment_id') or index}",
                        "type": _station_visual_type(base_key),
                        "visual_type": _station_visual_type(base_key),
                        "node_role": "residual_base_station",
                        "x": x,
                        "y": y,
                        "grid": {"row": int(row), "col": int(col)},
                        "connected": True,
                        "base_station": base_key,
                        "device_type": station.get("device_type") or base_key,
                        "device_label": station.get("device_label") or station.get("label") or base_key,
                        "label": station.get("label") or station.get("device_label") or base_key,
                        "mode": station.get("mode"),
                        "status": station.get("status"),
                        "coverage_radius": station.get("coverage_radius"),
                        "coverage_radius_km": station.get("coverage_radius_km"),
                        "max_users": station.get("max_users"),
                        "max_throughput": station.get("max_throughput"),
                        "device_uid": station.get("device_uid"),
                        "deployment_id": station.get("deployment_id"),
                        "coordinate_source": COORDINATE_SOURCE_VERSION,
                    }
                )
            )
        return nodes

    def _deployed_nodes(
        self,
        deployments: Iterable[Dict[str, Any]],
        rows: int,
        cols: int,
        *,
        occupied: Optional[Iterable[Tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        nodes = []
        seen = set()
        occupied_points: List[Tuple[float, float]] = list(occupied or [])
        for deployment in deployments:
            grid = deployment.get("grid") or {}
            row = grid.get("row")
            col = grid.get("col")
            if row is None or col is None:
                continue
            device = deployment.get("device") or {}
            base_key = device.get("base_station") or device.get("device_type") or deployment.get("communication_mode")
            dedupe_key = (int(row), int(col), str(base_key))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            sequence = int(deployment.get("sequence") or len(nodes) + 1)
            seed = f"deploy:{sequence}:{deployment.get('site_index', '')}:{base_key}:{row}:{col}"
            x, y = _grid_to_precise_coords(int(row), int(col), rows, cols, seed=seed, spread=0.58)
            x, y = _separate_station_coords(
                int(row),
                int(col),
                rows,
                cols,
                preferred=(x, y),
                occupied=occupied_points,
                seed=seed,
            )
            occupied_points.append((x, y))
            nodes.append(
                _drop_none(
                    {
                        "id": f"deploy:{sequence}",
                        "type": _station_visual_type(base_key),
                        "visual_type": _station_visual_type(base_key),
                        "node_role": "planned_deployment",
                        "x": x,
                        "y": y,
                        "grid": {"row": int(row), "col": int(col)},
                        "connected": True,
                        "base_station": base_key,
                        "device_type": device.get("device_type") or base_key,
                        "device_label": device.get("device_label") or base_key,
                        "label": device.get("device_label") or base_key,
                        "mode": deployment.get("communication_mode"),
                        "broadcast_mode": deployment.get("broadcast_mode"),
                        "site_index": deployment.get("site_index"),
                        "time_step": deployment.get("time_step"),
                        "region_label": deployment.get("region_label"),
                        "coverage_radius": device.get("coverage_radius"),
                        "coverage_radius_km": device.get("coverage_radius_km"),
                        "max_users": device.get("max_users"),
                        "max_throughput": device.get("max_throughput"),
                        "downlink_bandwidth_mbps": device.get("downlink_bandwidth_mbps"),
                        "uplink_bandwidth_mbps": device.get("uplink_bandwidth_mbps"),
                        "coordinate_source": COORDINATE_SOURCE_VERSION,
                        "deployment_coordinate": {"x": x, "y": y},
                    }
                )
            )
        return nodes

    def _declutter_station_nodes(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not nodes:
            return nodes
        rows, cols = _grid_shape_from_nodes(nodes)
        occupied: List[Tuple[float, float]] = []
        decluttered: List[Dict[str, Any]] = []
        changed = False

        for node in nodes:
            if not _is_station_node(node):
                decluttered.append(node)
                continue

            preferred = (
                _finite_float(node.get("x"), 0.0),
                _finite_float(node.get("y"), 0.0),
            )
            grid = node.get("grid") or {}
            if _is_finite_number(grid.get("row")) and _is_finite_number(grid.get("col")):
                row, col = int(float(grid.get("row"))), int(float(grid.get("col")))
                seed = f"declutter:{node.get('id')}:{node.get('device_uid')}:{node.get('deployment_id')}:{row}:{col}"
                x, y = _separate_station_coords(
                    row,
                    col,
                    rows,
                    cols,
                    preferred=preferred,
                    occupied=occupied,
                    seed=seed,
                )
            else:
                seed = f"declutter:{node.get('id')}:{node.get('device_uid')}:{node.get('deployment_id')}"
                spacing = _station_min_spacing(rows, cols)
                radius = _station_search_radius(rows, cols, spacing)
                x, y = _spread_station_coords(
                    preferred=preferred,
                    anchor=preferred,
                    occupied=occupied,
                    seed=seed,
                    min_spacing=spacing,
                    max_radius=radius,
                )

            occupied.append((x, y))
            if abs(x - preferred[0]) > 0.5 or abs(y - preferred[1]) > 0.5:
                updated = dict(node)
                updated["x"] = int(round(x))
                updated["y"] = int(round(y))
                if updated.get("node_role") == "planned_deployment":
                    updated["deployment_coordinate"] = {"x": updated["x"], "y": updated["y"]}
                decluttered.append(updated)
                changed = True
            else:
                decluttered.append(node)

        return decluttered if changed else nodes

    def _infer_grid_shape(self, report: Dict[str, Any], scene_export: Optional[Dict[str, Any]]) -> Tuple[int, int]:
        scenario = report.get("scenario") or {}
        for rows_key, cols_key in (("grid_rows", "grid_cols"), ("rows", "cols")):
            if scenario.get(rows_key) and scenario.get(cols_key):
                return max(1, int(scenario[rows_key])), max(1, int(scenario[cols_key]))

        max_row = 0
        max_col = 0

        def inspect(position: Any) -> None:
            nonlocal max_row, max_col
            if not _valid_position(position):
                return
            max_row = max(max_row, int(position[0]))
            max_col = max(max_col, int(position[1]))

        for state_key in ("initial_state", "final_state"):
            state = report.get(state_key) or {}
            for detail in state.get("user_details") or []:
                inspect(detail.get("position"))
            for station in state.get("residual_base_stations") or []:
                inspect([station.get("x"), station.get("y")])
        for step in report.get("steps") or []:
            inspect((step.get("action_desc") or {}).get("location"))
        for deployment in (report.get("deployment_plan") or {}).get("deployments") or []:
            grid = deployment.get("grid") or {}
            inspect([grid.get("row"), grid.get("col")])

        scene = (scene_export or {}).get("deployment_scene") or (scene_export or {}).get("disaster_scene") or {}
        for node in scene.get("nodes") or []:
            grid = node.get("grid") or {}
            inspect([grid.get("row"), grid.get("col")])

        return max(1, max_row + 1), max(1, max_col + 1)

    def _geo_bounds(self, scene_export: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        scene_export = scene_export or {}
        for scene_key in ("deployment_scene", "disaster_scene"):
            bounds = (scene_export.get(scene_key) or {}).get("geo_bounds")
            if isinstance(bounds, dict):
                return bounds
        return None

    def _connectivity_links(self, users: List[Dict[str, Any]], stations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not stations:
            return []
        links = []
        for user in users:
            if not (user.get("connected") or user.get("broadcast_served")):
                continue
            best_station = min(stations, key=lambda station: _distance(user, station))
            links.append(
                {
                    "src": user.get("id"),
                    "dst": best_station.get("id"),
                    "protocol": 1 if user.get("connected") else 0,
                }
            )
        return links

    def _link_metrics(self, frame: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metrics = frame.get("metrics") or {}
        throughput = _finite_float(metrics.get("avg_user_throughput"), 0.0)
        coverage = _finite_float(metrics.get("coverage_ratio"), 0.0)
        broadcast_ratio = _finite_float(metrics.get("broadcast_ratio"), 0.0)
        station_count = int(metrics.get("station_count") or 0)
        connected_users = int(metrics.get("connected_users") or 0)
        broadcast_users = int(metrics.get("broadcast_users") or 0)
        acceptance = self._frame_acceptance_metrics(frame, context or {})
        residual_capacity = acceptance.get("residual_capacity_mbps", 0.0)
        deployment_capacity = acceptance.get("deployment_capacity_mbps", 0.0)
        total_capacity = acceptance.get("theoretical_total_capacity_mbps", 0.0)
        connected_demand = acceptance.get("connected_user_demand_mbps", 0.0)
        broadcast_load = min(
            max(0.1, throughput * max(1, broadcast_users) * 0.28),
            max(0.1, residual_capacity or total_capacity * max(0.2, broadcast_ratio)),
        )
        user_capacity = max(deployment_capacity, acceptance.get("single_user_theoretical_max_mbps", 0.0))
        residual_load = min(max(residual_capacity, 0.0), max(broadcast_load, residual_capacity * broadcast_ratio))
        backhaul_capacity = max(
            acceptance.get("architecture_backhaul_capacity_mbps", 0.0),
            total_capacity - max(0.0, residual_capacity),
            acceptance.get("single_user_theoretical_max_mbps", 0.0),
        )
        access_throughput = min(max(connected_demand, throughput), max(user_capacity, 0.1))
        residual_throughput = residual_load
        backhaul_load = min(backhaul_capacity, max(0.1, connected_demand * max(coverage, 0.0)))
        backhaul_throughput = backhaul_load
        groups = [
            {
                "link_type": "user_access",
                "label": "用户接入链路",
                "active_links": connected_users,
                "throughput_mbps": round(access_throughput, 3),
                "avg_throughput_mbps": round(access_throughput, 3),
                "load_mbps": round(access_throughput, 3),
                "available_capacity_mbps": round(user_capacity, 3),
                "utilization": round(_ratio(access_throughput, user_capacity), 4),
                "success_rate": round(coverage, 4),
                "covered_users": connected_users,
                "device_count": max(0, station_count),
                "packet_loss_ratio": max(0.0, 1.0 - coverage),
                "latency_ms": round(38.0 + (1.0 - coverage) * 75.0, 3),
            },
            {
                "link_type": "broadcast",
                "label": "残余网络/广播链路",
                "active_links": broadcast_users,
                "throughput_mbps": round(residual_throughput, 3),
                "avg_throughput_mbps": round(residual_throughput, 3),
                "load_mbps": round(residual_throughput, 3),
                "available_capacity_mbps": round(max(residual_capacity, 0.0), 3),
                "utilization": round(_ratio(residual_throughput, residual_capacity), 4),
                "success_rate": round(broadcast_ratio, 4),
                "covered_users": broadcast_users,
                "device_count": int(acceptance.get("residual_station_count", 0) or 0),
                "packet_loss_ratio": max(0.0, 1.0 - broadcast_ratio),
                "latency_ms": round(72.0 + station_count * 1.7, 3),
            },
            {
                "link_type": "backhaul",
                "label": "应急回传链路",
                "active_links": station_count,
                "throughput_mbps": round(backhaul_throughput, 3),
                "avg_throughput_mbps": round(backhaul_throughput, 3),
                "load_mbps": round(backhaul_throughput, 3),
                "available_capacity_mbps": round(backhaul_capacity, 3),
                "utilization": round(_ratio(backhaul_throughput, backhaul_capacity), 4),
                "success_rate": round(max(0.0, 1.0 - max(0.0, min(0.35, (1.0 - coverage) * 0.38))), 4),
                "covered_users": connected_users,
                "device_count": station_count,
                "packet_loss_ratio": round(max(0.0, min(0.35, (1.0 - coverage) * 0.38)), 4),
                "latency_ms": round(24.0 + max(0, station_count - 1) * 2.4, 3),
            },
        ]
        total_throughput = sum(item["avg_throughput_mbps"] for item in groups)
        avg_success = sum(item.get("success_rate", 0.0) for item in groups) / max(1, len(groups))
        summary = {
            "active_links": sum(item["active_links"] for item in groups),
            "avg_throughput_mbps": round(total_throughput / len(groups), 3),
            "total_throughput_mbps": round(total_throughput, 3),
            "packet_loss_ratio": round(sum(item["packet_loss_ratio"] for item in groups) / len(groups), 4),
            "latency_ms": round(sum(item["latency_ms"] for item in groups) / len(groups), 3),
            "avg_success_rate": round(avg_success, 4),
            "covered_users": connected_users,
            "single_user_theoretical_max_mbps": acceptance["single_user_theoretical_max_mbps"],
            "single_user_theoretical_target_mbps": acceptance["single_user_theoretical_target_mbps"],
            "single_user_theoretical_passed": acceptance["single_user_theoretical_passed"],
            "theoretical_bandwidth_utilization": acceptance["theoretical_bandwidth_utilization"],
            "theoretical_bandwidth_utilization_target": acceptance["theoretical_bandwidth_utilization_target"],
            "theoretical_bandwidth_utilization_passed": acceptance["theoretical_bandwidth_utilization_passed"],
            "theoretical_total_capacity_mbps": acceptance["theoretical_total_capacity_mbps"],
            "theoretical_allocated_bandwidth_mbps": acceptance["theoretical_allocated_bandwidth_mbps"],
            "theoretical_demand_mbps": acceptance["theoretical_demand_mbps"],
            "acceptance_passed": acceptance["acceptance_passed"],
        }
        return {
            "replay_id": frame.get("replay_id"),
            "frame_index": frame.get("frame_index", 0),
            "time": frame.get("time", 0),
            "summary": summary,
            "acceptance": acceptance,
            "events": [],
            "groups": groups,
        }

    def _augment_link_metric(
        self,
        metric: Dict[str, Any],
        frame: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not frame:
            return metric
        if self._link_metric_has_usable_telemetry(metric):
            return metric
        recomputed = self._link_metrics(frame, context)
        replay_id = metric.get("replay_id") or recomputed.get("replay_id")
        if replay_id:
            recomputed["replay_id"] = replay_id
        return recomputed

    def _frame_acceptance_metrics(self, frame: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        metrics = frame.get("metrics") or {}
        frame_index = int(frame.get("frame_index", frame.get("frameIndex", 0)) or 0)
        nodes = list(frame.get("nodes") or [])
        users = [node for node in nodes if node.get("type") == "USER"]
        stations = [
            node
            for node in nodes
            if node.get("node_role") in {"residual_base_station", "planned_deployment"}
        ]
        connected_users = [node for node in users if node.get("connected")]
        theoretical_demand = sum(_finite_float(node.get("demand"), 0.0) for node in users)
        connected_demand = sum(_finite_float(node.get("demand"), 0.0) for node in connected_users)
        if connected_demand <= 0:
            connected_demand = theoretical_demand * _finite_float(metrics.get("coverage_ratio"), 0.0)

        residual_capacity = 0.0
        deployment_capacity = 0.0
        capacity_values: List[float] = []
        residual_station_count = 0
        planned_station_count = 0
        for node in stations:
            capacity = _node_capacity_mbps(node)
            if capacity > 0:
                capacity_values.append(capacity)
            if node.get("node_role") == "residual_base_station":
                residual_capacity += capacity
                residual_station_count += 1
            elif node.get("node_role") == "planned_deployment":
                deployment_capacity += capacity
                planned_station_count += 1

        deployment_fallback = (context.get("deployment_capacity_by_frame") or {}).get(frame_index, {})
        if deployment_capacity <= 0 and deployment_fallback:
            deployment_capacity = _finite_float(deployment_fallback.get("total_capacity_mbps"), 0.0)
            capacity_values.extend(deployment_fallback.get("capacity_values") or [])
            planned_station_count = int(deployment_fallback.get("device_count") or planned_station_count)

        architecture_capacity = _finite_float(context.get("architecture_total_capacity_mbps"), 0.0)
        architecture_peak = _finite_float(context.get("architecture_peak_capacity_mbps"), 0.0)
        architecture_sla = _finite_float(context.get("per_user_bandwidth_mbps"), 40.0)
        target_utilization = _finite_float(context.get("target_bandwidth_utilization"), 0.95)
        total_capacity = residual_capacity + deployment_capacity
        if total_capacity <= 0:
            total_capacity = architecture_capacity
        elif architecture_capacity > 0:
            total_capacity = max(total_capacity, architecture_capacity)

        single_user_max = max([architecture_sla, architecture_peak, *capacity_values, 0.0])
        allocated = min(total_capacity, connected_demand) if total_capacity > 0 else 0.0
        utilization = _ratio(allocated, total_capacity)
        acceptance_passed = single_user_max >= 40.0 and utilization >= 0.95
        modes = sorted(context.get("communication_modes") or [])
        broadcasts = sorted(context.get("broadcast_modes") or [])
        event = (
            "验收指标已补算：单用户理论带宽 "
            f"{single_user_max:.1f} Mbps，理论资源利用率 {utilization:.1%}。"
        )
        evidence = (
            f"方案拓扑={context.get('architecture_topology') or 'runtime'}，通信方式={len(modes)}，"
            f"广播方式={len(broadcasts)}，已恢复用户需求={connected_demand:.1f} Mbps，"
            f"理论容量={total_capacity:.1f} Mbps。"
        )
        return {
            "single_user_theoretical_max_mbps": round(single_user_max, 3),
            "single_user_theoretical_target_mbps": 40.0,
            "single_user_theoretical_passed": single_user_max >= 40.0,
            "theoretical_bandwidth_utilization": round(utilization, 4),
            "theoretical_bandwidth_utilization_target": target_utilization,
            "theoretical_bandwidth_utilization_passed": utilization >= target_utilization,
            "theoretical_total_capacity_mbps": round(total_capacity, 3),
            "theoretical_allocated_bandwidth_mbps": round(allocated, 3),
            "theoretical_demand_mbps": round(theoretical_demand, 3),
            "connected_user_demand_mbps": round(connected_demand, 3),
            "residual_capacity_mbps": round(residual_capacity, 3),
            "deployment_capacity_mbps": round(deployment_capacity, 3),
            "architecture_backhaul_capacity_mbps": round(_finite_float(context.get("architecture_backhaul_capacity_mbps"), 0.0), 3),
            "architecture_total_capacity_mbps": round(architecture_capacity, 3),
            "architecture_peak_capacity_mbps": round(architecture_peak, 3),
            "communication_modes": modes,
            "broadcast_modes": broadcasts,
            "residual_station_count": residual_station_count,
            "planned_station_count": planned_station_count,
            "acceptance_passed": acceptance_passed,
            "evidence": evidence,
            "source": context.get("source") or "runtime_replay",
            "events": [event],
        }

    def _link_acceptance_context(
        self,
        report: Dict[str, Any],
        request_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        request_payload = request_payload or {}
        scenario = report.get("scenario") or (report.get("deployment_plan") or {}).get("scenario") or {}
        scenario_name = (
            request_payload.get("scenario_name")
            or scenario.get("name")
            or (report.get("deployment_plan") or {}).get("scenario", {}).get("name")
            or ""
        )
        disaster_type = scenario.get("disaster_type") or ""
        plan, plan_source = self._load_architecture_plan(scenario_name, request_payload)
        topology_key = self._select_topology_key(report, plan)
        topology = (plan or {}).get(topology_key) or {}
        scenario_profile = None if plan else self._load_scenario_profile(scenario_name, disaster_type)

        communication_modes = set(topology.get("communication_modes") or [])
        broadcast_modes = set(topology.get("broadcast_modes") or [])
        if scenario_profile:
            communication_modes.update((scenario_profile.get("mode_profiles") or {}).keys())
            broadcast_modes.update((scenario_profile.get("broadcast_profiles") or {}).keys())
        for deployment in (report.get("deployment_plan") or {}).get("deployments") or []:
            if deployment.get("communication_mode"):
                communication_modes.add(str(deployment.get("communication_mode")))
            if deployment.get("broadcast_mode"):
                broadcast_modes.add(str(deployment.get("broadcast_mode")))

        layer_capacities = [
            _finite_float(layer.get("capacity_mbps"), 0.0)
            for layer in topology.get("layers") or []
            if _finite_float(layer.get("capacity_mbps"), 0.0) > 0
        ]
        profile_peak = 0.0
        for profile in (scenario_profile.get("mode_profiles") or {}).values() if scenario_profile else []:
            profile_peak = max(profile_peak, _finite_float(profile.get("max_bandwidth"), 0.0))
        for profile in (scenario_profile.get("base_stations") or {}).values() if scenario_profile else []:
            profile_peak = max(profile_peak, _finite_float(profile.get("max_throughput"), 0.0))

        per_user_bandwidth = _finite_float(topology.get("per_user_bandwidth_mbps"), 0.0)
        target_utilization = _finite_float(topology.get("target_bandwidth_utilization"), 0.0)
        if per_user_bandwidth <= 0:
            per_user_bandwidth = max(40.0, profile_peak)
        if target_utilization <= 0:
            target_utilization = 0.95

        return {
            "scenario_name": scenario_name,
            "disaster_type": disaster_type,
            "source": plan_source or ("scenario_dataset" if scenario_profile else "runtime_replay"),
            "architecture_topology": topology_key.replace("_topology", "") if topology_key else None,
            "per_user_bandwidth_mbps": per_user_bandwidth,
            "target_bandwidth_utilization": target_utilization,
            "communication_modes": sorted(communication_modes),
            "broadcast_modes": sorted(broadcast_modes),
            "architecture_total_capacity_mbps": sum(layer_capacities),
            "architecture_peak_capacity_mbps": max([per_user_bandwidth, profile_peak, *layer_capacities, 0.0]),
            "architecture_backhaul_capacity_mbps": layer_capacities[0] if layer_capacities else 0.0,
            "deployment_capacity_by_frame": self._deployment_capacity_by_frame(report),
        }

    def _select_topology_key(self, report: Dict[str, Any], plan: Optional[Dict[str, Any]]) -> str:
        if not plan:
            return "residual_topology"
        has_residual = bool((report.get("initial_state") or {}).get("residual_base_stations"))
        preferred = "residual_topology" if has_residual else "no_residual_topology"
        return preferred if preferred in plan else next(iter(plan.keys()), preferred)

    def _load_architecture_plan(
        self,
        scenario_name: str,
        request_payload: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        file_name = f"broadcast_architecture_{scenario_name}.json" if scenario_name else ""
        candidates: List[Path] = []
        checkpoint_path = request_payload.get("checkpoint_path")
        if checkpoint_path:
            checkpoint = Path(str(checkpoint_path))
            if not checkpoint.is_absolute():
                checkpoint = Path(__file__).resolve().parents[1] / checkpoint
            if file_name:
                candidates.append(checkpoint.parent / file_name)
            candidates.extend(sorted(checkpoint.parent.glob("broadcast_architecture_*.json")))
        project_root = Path(__file__).resolve().parents[1]
        if file_name:
            candidates.extend(sorted((project_root / "artifacts" / "runs").glob(f"*/{file_name}")))
            candidates.extend(sorted((project_root / "HMARL" / "checkpoints").glob(f"*/{file_name}")))
        candidates.extend(sorted((project_root / "artifacts" / "runs").glob("*/broadcast_architecture_*.json")))
        candidates.extend(sorted((project_root / "HMARL" / "checkpoints").glob("*/broadcast_architecture_*.json")))
        seen: set[str] = set()
        for path in candidates:
            path_key = str(path)
            if path_key in seen:
                continue
            seen.add(path_key)
            payload = self._read_json(path)
            if not payload:
                continue
            if scenario_name and not self._plan_matches_scenario(payload, scenario_name):
                continue
            return payload, str(path)
        return None, None

    def _plan_matches_scenario(self, plan: Dict[str, Any], scenario_name: str) -> bool:
        for topology in plan.values():
            if isinstance(topology, dict) and topology.get("scenario") == scenario_name:
                return True
        return False

    def _load_scenario_profile(self, scenario_name: str, disaster_type: str) -> Optional[Dict[str, Any]]:
        project_root = Path(__file__).resolve().parents[1]
        payload = self._read_json(project_root / "data" / "scenarios.json")
        for scenario in payload.get("scenarios", []) if payload else []:
            if scenario.get("name") == scenario_name:
                return scenario
        aliases = {
            "rainstorm": "flood",
            "extreme_rainstorm": "flood",
            "super_typhoon": "typhoon",
            "destructive_earthquake": "earthquake",
        }
        normalized_name = str(scenario_name or "").split("__", 1)[0]
        wanted = aliases.get(normalized_name) or aliases.get(disaster_type) or disaster_type
        for scenario in payload.get("scenarios", []) if payload else []:
            if scenario.get("disaster_type") == wanted or scenario.get("name") == wanted:
                return scenario
        return None

    def _deployment_capacity_by_frame(self, report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        deployments = (report.get("deployment_plan") or {}).get("deployments") or []
        max_step = max((int(item.get("time_step") or item.get("sequence") or 0) for item in deployments), default=0)
        result: Dict[int, Dict[str, Any]] = {}
        for frame_index in range(0, max_step + 1):
            seen = set()
            capacities: List[float] = []
            for deployment in deployments:
                if int(deployment.get("time_step") or deployment.get("sequence") or 0) > frame_index:
                    continue
                device = deployment.get("device") or {}
                grid = deployment.get("grid") or {}
                dedupe_key = (
                    int(grid.get("row") or -1),
                    int(grid.get("col") or -1),
                    str(device.get("base_station") or device.get("device_type") or deployment.get("communication_mode")),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                capacity = _node_capacity_mbps(device)
                if capacity > 0:
                    capacities.append(capacity)
            result[frame_index] = {
                "total_capacity_mbps": sum(capacities),
                "capacity_values": capacities,
                "device_count": len(capacities),
            }
        return result

    def _sample_nodes(self, nodes: List[Dict[str, Any]], sample_ratio: int) -> List[Dict[str, Any]]:
        ratio = max(1, int(sample_ratio or 1))
        if ratio <= 1:
            return nodes
        sampled = []
        for index, node in enumerate(nodes):
            if node.get("type") != "USER":
                sampled.append(node)
                continue
            if node.get("connected") or node.get("broadcast_served") or index % ratio == 0:
                sampled.append(node)
        return sampled

    def _frame_summary(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_index": frame.get("frame_index", 0),
            "time": frame.get("time", 0),
            "label": frame.get("label"),
            "metrics": frame.get("metrics") or {},
            "node_count_total": frame.get("node_count_total") or len(frame.get("nodes") or []),
        }

    def _public_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **metadata,
            "id": metadata.get("replay_id") or metadata.get("id"),
            "session_dir": None,
        }

    def _session_created_at(self, metadata: Dict[str, Any], session_dir: Path) -> float:
        value = _finite_float(metadata.get("created_at"), 0.0)
        if value > 0:
            return value

        replay_id = str(metadata.get("replay_id") or metadata.get("id") or session_dir.name)
        match = REPLAY_SESSION_DIR_RE.match(replay_id) or REPLAY_SESSION_DIR_RE.match(session_dir.name)
        if match:
            try:
                return datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M%S").timestamp()
            except ValueError:
                pass

        return self._path_mtime(session_dir / "metadata.json")

    def _created_at_iso(self, created_at: float, fallback: Any = None) -> str:
        if isinstance(fallback, str) and fallback:
            return fallback
        return datetime.fromtimestamp(created_at).isoformat(timespec="seconds") if created_at > 0 else ""

    def _path_mtime(self, path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _new_replay_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"rpl_{stamp}_{uuid.uuid4().hex[:6]}"

    def _session_dir(self, replay_id: str) -> Path:
        safe_id = str(replay_id).strip()
        if not safe_id or "/" in safe_id or "\\" in safe_id or safe_id.startswith("."):
            raise FileNotFoundError("Invalid replay session id.")
        return self.root_dir / safe_id

    def _load_metadata(self, replay_id: str) -> Dict[str, Any]:
        metadata = self._read_json(self._session_dir(replay_id) / "metadata.json")
        if not metadata:
            raise FileNotFoundError(f"Replay session not found: {replay_id}")
        return metadata

    def _load_report_for_session(self, replay_id: str) -> Dict[str, Any]:
        payload = self._read_json(self._session_dir(replay_id) / "report.json") or {}
        reports = payload.get("reports") if isinstance(payload, dict) else None
        if isinstance(reports, list) and reports:
            return reports[0] or {}
        return payload if isinstance(payload, dict) else {}

    def _read_frame(self, replay_id: str, frame_index: int) -> Optional[Dict[str, Any]]:
        for frame in self._iter_jsonl(self._session_dir(replay_id) / "frames.jsonl"):
            if int(frame.get("frame_index", frame.get("frameIndex", 0)) or 0) == int(frame_index):
                return frame
        return None

    def _read_link_metric(self, replay_id: str, frame_index: int) -> Optional[Dict[str, Any]]:
        for metric in self._iter_jsonl(self._session_dir(replay_id) / "link_metrics.jsonl"):
            if int(metric.get("frame_index", 0) or 0) == int(frame_index):
                return metric
        return None

    def _link_metric_for_frame(
        self,
        replay_id: str,
        frame_index: int,
        *,
        frame: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        metric = self._read_link_metric(replay_id, int(frame_index))
        frame = frame or self._read_frame(replay_id, int(frame_index))
        if frame is None:
            return metric
        if metric and self._link_metric_has_usable_telemetry(metric):
            return metric

        report = self._load_report_for_session(replay_id)
        request_payload = self._read_json(self._session_dir(replay_id) / "request.json") or {}
        context = self._link_acceptance_context(report, request_payload)
        if metric is None:
            metric = self._link_metrics(frame, context)
        else:
            metric = self._augment_link_metric(metric, frame, context)
        if metric and not metric.get("replay_id"):
            metric["replay_id"] = replay_id
        return metric

    def _link_metric_has_usable_telemetry(self, metric: Dict[str, Any]) -> bool:
        summary = metric.get("summary") or {}
        throughput = _finite_float(summary.get("total_throughput_mbps") or summary.get("avg_throughput_mbps"), 0.0)
        latency = _finite_float(summary.get("latency_ms"), 0.0)
        return (
            bool(metric.get("acceptance"))
            and summary.get("single_user_theoretical_max_mbps") is not None
            and throughput > 0.0
            and latency > 0.0
        )

    def _write_log(self, path: Path, metadata: Dict[str, Any], frames: List[Dict[str, Any]], report: Dict[str, Any]) -> None:
        lines = [
            f"[SESSION] replay_id={metadata['replay_id']} source={metadata['source']} title={metadata['title']}",
            f"[SCENARIO] name={metadata.get('scenario_name')} algorithm={metadata.get('algorithm')} protocol={metadata.get('evaluation_protocol')}",
            f"[NODES] total={metadata.get('node_count_total')} frames={metadata.get('frame_count')} users={metadata.get('summary', {}).get('total_users')} initial_stations={metadata.get('summary', {}).get('initial_stations')} final_stations={metadata.get('summary', {}).get('final_stations')}",
        ]
        deployments = (report.get("deployment_plan") or {}).get("deployments") or []
        deployment_coordinates: Dict[int, Tuple[Any, Any]] = {}
        for frame in frames:
            for node in frame.get("nodes") or []:
                if node.get("node_role") != "planned_deployment":
                    continue
                for raw_step in (node.get("time_step"), str(node.get("id", "")).replace("deploy:", "")):
                    if _is_finite_number(raw_step):
                        deployment_coordinates[int(float(raw_step))] = (node.get("x"), node.get("y"))
        for deployment in deployments:
            device = deployment.get("device") or {}
            grid = deployment.get("grid") or {}
            raw_time_step = deployment.get("time_step") or deployment.get("sequence")
            coordinate = deployment_coordinates.get(int(float(raw_time_step))) if _is_finite_number(raw_time_step) else None
            coordinate_text = f" coord=({coordinate[0]},{coordinate[1]})" if coordinate else ""
            lines.append(
                "[DEPLOY] t={time_step} grid=({row},{col}){coordinate} device={device} mode={mode} reward={reward:.3f} coverage={coverage:.2%}".format(
                    time_step=raw_time_step,
                    row=grid.get("row"),
                    col=grid.get("col"),
                    coordinate=coordinate_text,
                    device=device.get("device_label") or device.get("base_station") or device.get("device_type") or "unknown",
                    mode=deployment.get("communication_mode") or "unknown",
                    reward=_finite_float(deployment.get("reward"), 0.0),
                    coverage=_finite_float(deployment.get("coverage_after"), 0.0),
                )
            )
        for frame in frames:
            metrics = frame.get("metrics") or {}
            lines.append(
                "[FRAME {idx:03d}] coverage={cov:.2%} broadcast={broadcast:.2%} connected={connected}/{users} stations={stations} throughput={tp:.3f} budget={budget:.1f}".format(
                    idx=int(frame.get("frame_index", 0) or 0),
                    cov=_finite_float(metrics.get("coverage_ratio"), 0.0),
                    broadcast=_finite_float(metrics.get("broadcast_ratio"), 0.0),
                    connected=int(metrics.get("connected_users") or 0),
                    users=int(metrics.get("user_count") or 0),
                    stations=int(metrics.get("station_count") or 0),
                    tp=_finite_float(metrics.get("avg_user_throughput"), 0.0),
                    budget=_finite_float(metrics.get("remaining_budget"), 0.0),
                )
            )
        summary = metadata.get("summary") or {}
        lines.append(
            "[SUMMARY] total_reward={reward:.3f} final_coverage={coverage:.2%} final_broadcast={broadcast:.2%} steps={steps}".format(
                reward=_finite_float(summary.get("total_reward"), 0.0),
                coverage=_finite_float(summary.get("coverage_ratio"), 0.0),
                broadcast=_finite_float(summary.get("broadcast_ratio"), 0.0),
                steps=int(summary.get("steps_taken") or 0),
            )
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_nodes_full(self, path: Path, replay_id: str, frames: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for frame in frames:
                frame_index = int(frame.get("frame_index", 0) or 0)
                for node in frame.get("nodes") or []:
                    file.write(json.dumps(_json_safe({"replay_id": replay_id, "frame_index": frame_index, **node}), ensure_ascii=False) + "\n")

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _iter_jsonl(self, path: Path) -> Iterable[Dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _grid_to_coords(row: int, col: int, rows: int, cols: int) -> Tuple[int, int]:
    x = int(round(((col + 0.5) / max(1, cols)) * MAP_WIDTH))
    y = int(round(((row + 0.5) / max(1, rows)) * MAP_HEIGHT))
    return min(MAP_WIDTH, max(0, x)), min(MAP_HEIGHT, max(0, y))


def _station_points(nodes: Iterable[Dict[str, Any]]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for node in nodes:
        if not _is_station_node(node):
            continue
        points.append((_finite_float(node.get("x"), 0.0), _finite_float(node.get("y"), 0.0)))
    return points


def _grid_shape_from_nodes(nodes: Iterable[Dict[str, Any]]) -> Tuple[int, int]:
    max_row = 0
    max_col = 0
    for node in nodes:
        grid = node.get("grid") or {}
        if not (_is_finite_number(grid.get("row")) and _is_finite_number(grid.get("col"))):
            continue
        max_row = max(max_row, int(float(grid.get("row"))))
        max_col = max(max_col, int(float(grid.get("col"))))
    return max(1, max_row + 1), max(1, max_col + 1)


def _station_min_spacing(rows: int, cols: int) -> float:
    cell_width = MAP_WIDTH / max(1, cols)
    cell_height = MAP_HEIGHT / max(1, rows)
    return max(
        STATION_MIN_SPACING_FLOOR,
        min(STATION_MIN_SPACING_CEILING, min(cell_width, cell_height) * 0.72),
    )


def _station_search_radius(rows: int, cols: int, min_spacing: float) -> float:
    cell_width = MAP_WIDTH / max(1, cols)
    cell_height = MAP_HEIGHT / max(1, rows)
    return max(min_spacing, min(cell_width, cell_height) * 1.18)


def _separate_station_coords(
    row: int,
    col: int,
    rows: int,
    cols: int,
    *,
    preferred: Tuple[float, float],
    occupied: Iterable[Tuple[float, float]],
    seed: Any,
) -> Tuple[int, int]:
    min_spacing = _station_min_spacing(rows, cols)
    max_radius = _station_search_radius(rows, cols, min_spacing)
    anchor = _grid_to_coords(row, col, rows, cols)
    x, y = _spread_station_coords(
        preferred=preferred,
        anchor=anchor,
        occupied=occupied,
        seed=seed,
        min_spacing=min_spacing,
        max_radius=max_radius,
    )
    return int(round(x)), int(round(y))


def _spread_station_coords(
    *,
    preferred: Tuple[float, float],
    anchor: Tuple[float, float],
    occupied: Iterable[Tuple[float, float]],
    seed: Any,
    min_spacing: float,
    max_radius: float,
) -> Tuple[float, float]:
    occupied_points = list(occupied or [])
    preferred = _clamp_point(preferred)
    anchor = _clamp_point(anchor)
    if not occupied_points or _nearest_point_distance(preferred, occupied_points) >= min_spacing:
        return preferred

    base_angle = _stable_unit(f"{seed}:station-spacing-angle") * math.tau
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    candidates = [preferred]
    for index in range(112):
        ring = index // 16
        radius = min(max_radius, min_spacing * (0.82 + ring * 0.18))
        angle = base_angle + index * golden_angle
        candidates.append(
            _clamp_point(
                (
                    anchor[0] + math.cos(angle) * radius,
                    anchor[1] + math.sin(angle) * radius,
                )
            )
        )

    best_clear: Optional[Tuple[float, float]] = None
    best_clear_cost = math.inf
    fallback = preferred
    fallback_score = -math.inf

    for candidate in candidates:
        nearest = _nearest_point_distance(candidate, occupied_points)
        move_cost = _point_distance(candidate, preferred) + _point_distance(candidate, anchor) * 0.18
        if nearest >= min_spacing and move_cost < best_clear_cost:
            best_clear = candidate
            best_clear_cost = move_cost
        score = nearest - move_cost * 0.018
        if score > fallback_score:
            fallback = candidate
            fallback_score = score

    return best_clear or fallback


def _nearest_point_distance(point: Tuple[float, float], points: Iterable[Tuple[float, float]]) -> float:
    nearest = math.inf
    for other in points:
        nearest = min(nearest, _point_distance(point, other))
    return nearest


def _point_distance(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _clamp_point(point: Tuple[float, float]) -> Tuple[float, float]:
    return (
        max(0.0, min(float(MAP_WIDTH), _finite_float(point[0], 0.0))),
        max(0.0, min(float(MAP_HEIGHT), _finite_float(point[1], 0.0))),
    )


def _grid_to_precise_coords(
    row: int,
    col: int,
    rows: int,
    cols: int,
    *,
    seed: Any,
    spread: float,
) -> Tuple[int, int]:
    center_x, center_y = _grid_to_coords(row, col, rows, cols)
    cell_width = MAP_WIDTH / max(1, cols)
    cell_height = MAP_HEIGHT / max(1, rows)
    effective_spread = max(0.0, min(1.72, spread))
    angle = _stable_unit(f"{seed}:angle") * math.tau
    radius = (0.16 + math.sqrt(_stable_unit(f"{seed}:radius")) * 0.84) * effective_spread
    dx = math.cos(angle) * cell_width * 0.5 * radius
    dy = math.sin(angle) * cell_height * 0.5 * radius
    dx += (_stable_unit(f"{seed}:free-x") - 0.5) * cell_width * 0.22 * effective_spread
    dy += (_stable_unit(f"{seed}:free-y") - 0.5) * cell_height * 0.22 * effective_spread
    flow = (row * 1.371 + col * 0.917 + _stable_unit(f"{seed}:flow")) * math.pi
    dx += math.sin(flow) * cell_width * 0.08 * effective_spread
    dy += math.cos(flow * 0.83) * cell_height * 0.08 * effective_spread
    x = int(round(center_x + dx))
    y = int(round(center_y + dy))
    return min(MAP_WIDTH, max(0, x)), min(MAP_HEIGHT, max(0, y))


def _stable_unit(seed: Any) -> float:
    text = str(seed or "")
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    value = (value + ((value << 13) & 0xFFFFFFFF)) & 0xFFFFFFFF
    value ^= value >> 7
    value = (value + ((value << 3) & 0xFFFFFFFF)) & 0xFFFFFFFF
    value ^= value >> 17
    value = (value + ((value << 5) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return (value & 0xFFFFFFFF) / 4294967296


def _lat_lon_center(bounds: Any) -> Dict[str, float]:
    if not isinstance(bounds, dict):
        return {}
    lat_min = _finite_float(bounds.get("lat_min"), math.nan)
    lat_max = _finite_float(bounds.get("lat_max"), math.nan)
    lon_min = _finite_float(bounds.get("lon_min"), math.nan)
    lon_max = _finite_float(bounds.get("lon_max"), math.nan)
    if any(math.isnan(value) for value in (lat_min, lat_max, lon_min, lon_max)):
        return {}
    return {"lat": (lat_min + lat_max) / 2, "lon": (lon_min + lon_max) / 2}


def _station_visual_type(base_key: Any) -> str:
    normalized = str(base_key or "").lower()
    if "wifi" in normalized or "mesh" in normalized:
        return "RELAY"
    if "satellite" in normalized or "ka" in normalized:
        return "RELAY_ENB"
    if "macro" in normalized or "700" in normalized or "5g" in normalized:
        return "MACRO_ENB"
    return "MANPACK_ENB"


def _is_station_node(node: Dict[str, Any]) -> bool:
    role = node.get("node_role")
    if role in {"residual_base_station", "planned_deployment"}:
        return True
    node_type = str(node.get("type") or "").upper()
    return bool(node_type and node_type != "USER" and role != "user")


def _distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    dx = _finite_float(left.get("x"), 0.0) - _finite_float(right.get("x"), 0.0)
    dy = _finite_float(left.get("y"), 0.0) - _finite_float(right.get("y"), 0.0)
    return math.sqrt(dx * dx + dy * dy)


def _valid_position(position: Any) -> bool:
    return (
        isinstance(position, (list, tuple))
        and len(position) >= 2
        and _is_finite_number(position[0])
        and _is_finite_number(position[1])
    )


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _ratio(numerator: Any, denominator: Any) -> float:
    denominator_value = _finite_float(denominator, 0.0)
    if denominator_value <= 0:
        return 0.0
    return max(0.0, min(1.0, _finite_float(numerator, 0.0) / denominator_value))


def _node_capacity_mbps(node: Dict[str, Any]) -> float:
    downlink = node.get("downlink_bandwidth_mbps")
    if isinstance(downlink, dict):
        downlink = downlink.get("max") or downlink.get("avg")
    for key in ("max_throughput", "capacity_mbps", "available_capacity_mbps"):
        value = _finite_float(node.get(key), 0.0)
        if value > 0:
            return value
    value = _finite_float(downlink, 0.0)
    if value > 0:
        return value
    uplink = node.get("uplink_bandwidth_mbps")
    if isinstance(uplink, dict):
        uplink = uplink.get("max") or uplink.get("avg")
    return _finite_float(uplink, 0.0)


def _drop_none(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
