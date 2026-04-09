"""Mahimahi trace 解析与容量分析模块。

提供:
- Trace 文件解析（mahimahi 格式：每行一个毫秒时间戳，代表 1500 字节包的传输机会）
- 链路容量时间序列计算
- Trace 文件列表与元信息管理
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Dict, List

PACKET_SIZE_BYTES = 1500
BITS_PER_PACKET = PACKET_SIZE_BYTES * 8


class TraceAnalyzer:
    """解析 mahimahi trace 文件并计算容量时间序列。"""

    def __init__(self, trace_path: str | Path):
        self.trace_path = Path(trace_path)
        self.timestamps: List[int] = self._parse()

    def _parse(self) -> List[int]:
        ts: List[int] = []
        with open(self.trace_path) as fh:
            for line in fh:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    ts.append(int(stripped))
        return sorted(ts)

    @property
    def period_ms(self) -> int:
        return self.timestamps[-1] if self.timestamps else 0

    @property
    def total_packets(self) -> int:
        return len(self.timestamps)

    @property
    def avg_throughput_mbps(self) -> float:
        if not self.timestamps or self.period_ms == 0:
            return 0.0
        return (self.total_packets * BITS_PER_PACKET) / (self.period_ms / 1000) / 1e6

    def _packets_in_window(self, start_ms: int, end_ms: int) -> int:
        """统计 [start_ms, end_ms) 内的包传输机会数（考虑 trace 循环）。"""
        period = self.period_ms
        if period <= 0 or not self.timestamps:
            return 0

        first_loop = start_ms // period
        last_loop = (end_ms - 1) // period

        if first_loop == last_loop:
            lo = start_ms - first_loop * period
            hi = end_ms - first_loop * period
            return self._count_range(lo, hi)

        count = self._count_range(start_ms - first_loop * period, period)
        full_loops = last_loop - first_loop - 1
        if full_loops > 0:
            count += full_loops * len(self.timestamps)
        count += self._count_range(0, end_ms - last_loop * period)
        return count

    def _count_range(self, lo: int, hi: int) -> int:
        return bisect.bisect_left(self.timestamps, hi) - bisect.bisect_left(self.timestamps, lo)

    def capacity_series(self, duration_s: float = 60.0, window_ms: int = 500) -> List[Dict[str, float]]:
        """返回每个时间窗口的链路容量 (Mbps)。"""
        total_ms = int(duration_s * 1000)
        results: List[Dict[str, float]] = []
        for start in range(0, total_ms, window_ms):
            end = min(start + window_ms, total_ms)
            n = self._packets_in_window(start, end)
            cap = (n * BITS_PER_PACKET) / ((end - start) / 1000) / 1e6
            results.append({"time_s": round(start / 1000, 3), "value": round(cap, 3)})
        return results


TRACE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "emergency-command": {"label": "应急指挥中心链路 (10Mbps)"},
    "damaged-station": {"label": "震后受损基站链路"},
    "mobile-patrol": {"label": "灾区巡查车载链路"},
    "flood-emergency": {"label": "洪灾应急通信链路"},
    "temp-relay": {"label": "临时中继站链路"},
}


class MahimahiManager:
    """Trace 文件管理与容量分析。"""

    def __init__(self, traces_dir: str = "data/traces"):
        project_root = Path(__file__).resolve().parents[1]
        candidate = Path(traces_dir)
        self.traces_dir = candidate if candidate.is_absolute() else project_root / candidate
        self._traces_cache: List[Dict[str, Any]] = []
        self._analyzers: Dict[str, TraceAnalyzer] = {}
        self._load_all()

    def _load_all(self) -> None:
        """启动时一次性解析所有 trace 文件并缓存。"""
        if not self.traces_dir.exists():
            return
        for f in sorted(self.traces_dir.iterdir()):
            if f.is_file() and f.suffix == ".trace":
                try:
                    a = TraceAnalyzer(f)
                    self._analyzers[f.stem] = a
                    desc = TRACE_DESCRIPTIONS.get(f.stem, {})
                    self._traces_cache.append({
                        "name": f.stem,
                        "filename": f.name,
                        "label": desc.get("label", f.stem),
                        "period_ms": a.period_ms,
                        "total_packets": a.total_packets,
                        "avg_throughput_mbps": round(a.avg_throughput_mbps, 2),
                    })
                except Exception:
                    continue

    def list_traces(self) -> List[Dict[str, Any]]:
        return self._traces_cache

    def _get_analyzer(self, trace_name: str) -> TraceAnalyzer:
        if trace_name in self._analyzers:
            return self._analyzers[trace_name]
        raise FileNotFoundError(f"Trace '{trace_name}' not found")

    def analyze_trace(self, trace_name: str, duration_s: float = 60, window_ms: int = 500) -> Dict[str, Any]:
        a = self._get_analyzer(trace_name)
        return {
            "name": trace_name,
            "period_ms": a.period_ms,
            "total_packets": a.total_packets,
            "avg_throughput_mbps": round(a.avg_throughput_mbps, 2),
            "capacity": a.capacity_series(duration_s, window_ms),
        }

    def simulate(self, trace_name: str, duration_s: float = 60, window_ms: int = 500, **_kwargs) -> Dict[str, Any]:
        """返回 trace 的容量时间序列，供前端回放展示。"""
        a = self._get_analyzer(trace_name)
        return {
            "trace_name": trace_name,
            "duration_s": duration_s,
            "window_ms": window_ms,
            "capacity": a.capacity_series(duration_s, window_ms),
        }
