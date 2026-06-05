#!/usr/bin/env python3
"""Concurrent HTTP load probe for the RescueNet-RL service.

The script uses only the Python standard library so it can run on the
same machine as the project without installing a benchmark package.

Examples:
  python3 scripts/stress_test_concurrency.py --users 1200
  python3 scripts/stress_test_concurrency.py --users 1200 --preset api-smoke
  python3 scripts/stress_test_concurrency.py --users 1200 --duration 60 --output reports/concurrency_1200.json
  python3 scripts/stress_test_concurrency.py --users 1200 --endpoint "GET /api/health" --min-open-connections 1200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import ssl
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit


HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

PRESETS: Dict[str, Tuple[str, ...]] = {
    "health": ("GET /api/health",),
    "api-smoke": (
        "GET /api/health",
        "GET /api/scenarios",
        "GET /api/devices",
        "GET /api/replay/sessions?limit=5",
        "GET /api/mahimahi/status",
        "GET /api/ns3/status",
    ),
}


@dataclass(frozen=True)
class Target:
    scheme: str
    host: str
    port: int
    host_header: str
    base_path: str
    ssl_context: Optional[ssl.SSLContext]


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"


class Metrics:
    def __init__(self) -> None:
        self.total = 0
        self.success = 0
        self.failure = 0
        self.bytes_read = 0
        self.latencies_ms: List[float] = []
        self.status_counts: Counter[str] = Counter()
        self.error_counts: Counter[str] = Counter()
        self.endpoint_counts: Counter[str] = Counter()
        self.endpoint_success: Counter[str] = Counter()
        self.connection_attempts = 0
        self.connection_failures = 0
        self.open_connections = 0
        self.max_open_connections = 0
        self.active_requests = 0
        self.max_active_requests = 0
        self.virtual_users_started = 0
        self.virtual_users_finished = 0

    def mark_user_started(self) -> None:
        self.virtual_users_started += 1

    def mark_user_finished(self) -> None:
        self.virtual_users_finished += 1

    def request_started(self) -> None:
        self.active_requests += 1
        if self.active_requests > self.max_active_requests:
            self.max_active_requests = self.active_requests

    def request_finished(
        self,
        endpoint: Endpoint,
        latency_ms: float,
        status_code: Optional[int],
        error: Optional[str],
        bytes_read: int,
    ) -> None:
        self.active_requests -= 1
        self.total += 1
        self.latencies_ms.append(latency_ms)
        self.bytes_read += bytes_read
        self.endpoint_counts[endpoint.key] += 1

        if status_code is not None:
            self.status_counts[str(status_code)] += 1

        ok = status_code is not None and 200 <= status_code < 400 and error is None
        if ok:
            self.success += 1
            self.endpoint_success[endpoint.key] += 1
        else:
            self.failure += 1
            self.error_counts[error or f"HTTP {status_code}"] += 1

    def connection_opened(self) -> None:
        self.open_connections += 1
        if self.open_connections > self.max_open_connections:
            self.max_open_connections = self.open_connections

    def connection_closed(self) -> None:
        if self.open_connections > 0:
            self.open_connections -= 1


def parse_target(base_url: str) -> Target:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("--base-url must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("--base-url must include a host")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    default_port = 443 if parsed.scheme == "https" else 80
    host_header = parsed.hostname if port == default_port else f"{parsed.hostname}:{port}"
    ssl_context = ssl.create_default_context() if parsed.scheme == "https" else None
    return Target(
        scheme=parsed.scheme,
        host=parsed.hostname,
        port=port,
        host_header=host_header,
        base_path=parsed.path.rstrip("/"),
        ssl_context=ssl_context,
    )


def parse_endpoint(spec: str, default_method: str = "GET") -> Endpoint:
    raw = spec.strip()
    if not raw:
        raise ValueError("endpoint cannot be empty")

    method = default_method.upper()
    path = raw

    parts = raw.split(maxsplit=1)
    if len(parts) == 2 and parts[0].upper() in HTTP_METHODS:
        method = parts[0].upper()
        path = parts[1].strip()
    elif ":" in raw:
        candidate_method, candidate_path = raw.split(":", 1)
        if candidate_method.upper() in HTTP_METHODS:
            method = candidate_method.upper()
            path = candidate_path.strip()

    parsed = urlsplit(path)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
    elif not path.startswith("/"):
        path = f"/{path}"

    return Endpoint(method=method, path=path)


def resolve_path(target: Target, endpoint_path: str) -> str:
    if not target.base_path:
        return endpoint_path
    if endpoint_path == target.base_path or endpoint_path.startswith(f"{target.base_path}/"):
        return endpoint_path
    return f"{target.base_path}/{endpoint_path.lstrip('/')}"


def parse_headers(header_args: Sequence[str]) -> List[Tuple[str, str]]:
    headers: List[Tuple[str, str]] = []
    for raw in header_args:
        if ":" not in raw:
            raise ValueError(f"invalid header {raw!r}; expected 'Name: Value'")
        name, value = raw.split(":", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"invalid header {raw!r}; header name is empty")
        headers.append((name, value))
    return headers


def load_body(args: argparse.Namespace) -> bytes:
    if args.body_file and args.body:
        raise ValueError("use either --body or --body-file, not both")
    if args.body_file:
        return Path(args.body_file).read_bytes()
    if args.body:
        return args.body.encode("utf-8")
    return b""


def raise_fd_limit(target_users: int) -> Optional[str]:
    try:
        import resource
    except ImportError:
        return None

    needed = target_users + 256
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= needed:
        return None
    if hard < needed:
        return (
            f"Warning: open-file limit is {soft}/{hard}, lower than the recommended "
            f"{needed}. The test may fail before reaching {target_users} connections."
        )
    resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))
    return f"Raised open-file soft limit from {soft} to {needed}."


async def open_connection(target: Target, timeout_s: float, metrics: Metrics):
    metrics.connection_attempts += 1
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                target.host,
                target.port,
                ssl=target.ssl_context,
                server_hostname=target.host if target.ssl_context else None,
            ),
            timeout=timeout_s,
        )
    except Exception:
        metrics.connection_failures += 1
        raise
    metrics.connection_opened()
    return reader, writer


async def close_connection(writer: Optional[asyncio.StreamWriter], metrics: Metrics) -> None:
    if writer is None:
        return
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), timeout=1.0)
    except Exception:
        pass
    metrics.connection_closed()


def build_request(
    target: Target,
    endpoint: Endpoint,
    headers: Sequence[Tuple[str, str]],
    body: bytes,
    keep_alive: bool,
    user_id: int,
    sequence: int,
) -> bytes:
    path = resolve_path(target, endpoint.path)
    request_body = body if endpoint.method in {"POST", "PUT", "PATCH"} else b""
    connection = "keep-alive" if keep_alive else "close"
    request_headers: List[Tuple[str, str]] = [
        ("Host", target.host_header),
        ("User-Agent", "RescueNetConcurrencyProbe/1.0"),
        ("Accept", "application/json,*/*"),
        ("Connection", connection),
        ("X-Load-Test-User", str(user_id)),
        ("X-Load-Test-Sequence", str(sequence)),
    ]
    request_headers.extend(headers)
    if request_body and not any(name.lower() == "content-type" for name, _ in request_headers):
        request_headers.append(("Content-Type", "application/json"))
    if request_body:
        request_headers.append(("Content-Length", str(len(request_body))))

    head = f"{endpoint.method} {path} HTTP/1.1\r\n"
    header_text = "".join(f"{name}: {value}\r\n" for name, value in request_headers)
    return f"{head}{header_text}\r\n".encode("utf-8") + request_body


async def discard_exactly(reader: asyncio.StreamReader, size: int) -> int:
    remaining = size
    read_total = 0
    while remaining > 0:
        chunk = await reader.read(min(remaining, 65536))
        if not chunk:
            raise asyncio.IncompleteReadError(partial=b"", expected=remaining)
        read_total += len(chunk)
        remaining -= len(chunk)
    return read_total


async def discard_chunked(reader: asyncio.StreamReader) -> int:
    read_total = 0
    while True:
        line = await reader.readline()
        if not line:
            raise asyncio.IncompleteReadError(partial=b"", expected=1)
        size_text = line.split(b";", 1)[0].strip()
        size = int(size_text, 16)
        if size == 0:
            while True:
                trailer = await reader.readline()
                if trailer in {b"\r\n", b"\n", b""}:
                    break
            return read_total
        read_total += await discard_exactly(reader, size)
        crlf = await reader.readexactly(2)
        if crlf != b"\r\n":
            raise ValueError("invalid chunk terminator")


async def read_response(reader: asyncio.StreamReader, method: str) -> Tuple[int, int, bool]:
    header_bytes = await reader.readuntil(b"\r\n\r\n")
    header_text = header_bytes.decode("iso-8859-1")
    lines = header_text.splitlines()
    if not lines:
        raise ValueError("empty HTTP response")
    status_parts = lines[0].split()
    if len(status_parts) < 2:
        raise ValueError(f"invalid HTTP status line: {lines[0]!r}")
    status_code = int(status_parts[1])

    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()

    should_close = headers.get("connection", "").lower() == "close"
    if method == "HEAD" or status_code in {204, 304}:
        return status_code, 0, should_close

    bytes_read = 0
    transfer_encoding = headers.get("transfer-encoding", "").lower()
    if "chunked" in transfer_encoding:
        bytes_read = await discard_chunked(reader)
    elif "content-length" in headers:
        bytes_read = await discard_exactly(reader, int(headers["content-length"]))
    else:
        data = await reader.read()
        bytes_read = len(data)
        should_close = True

    return status_code, bytes_read, should_close


def normalize_error(exc: Exception) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    name = exc.__class__.__name__
    detail = str(exc)
    if detail:
        return f"{name}: {detail}"
    return name


async def virtual_user(
    user_id: int,
    target: Target,
    endpoints: Sequence[Endpoint],
    headers: Sequence[Tuple[str, str]],
    body: bytes,
    args: argparse.Namespace,
    metrics: Metrics,
    start_event: asyncio.Event,
    deadline: Optional[float],
) -> None:
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    sequence = 0
    metrics.mark_user_started()
    await start_event.wait()

    if args.ramp_up > 0 and args.users > 1:
        await asyncio.sleep(args.ramp_up * user_id / (args.users - 1))

    async def ensure_connection():
        nonlocal reader, writer
        if reader is None or writer is None or writer.is_closing():
            reader, writer = await open_connection(target, args.timeout, metrics)

    async def one_request(endpoint: Endpoint) -> None:
        nonlocal reader, writer, sequence
        sequence += 1
        status_code: Optional[int] = None
        error: Optional[str] = None
        bytes_read = 0
        metrics.request_started()
        started_at = time.perf_counter()
        try:
            await ensure_connection()
            assert writer is not None
            assert reader is not None
            request = build_request(target, endpoint, headers, body, args.keepalive, user_id, sequence)
            writer.write(request)
            await asyncio.wait_for(writer.drain(), timeout=args.timeout)
            status_code, bytes_read, should_close = await asyncio.wait_for(
                read_response(reader, endpoint.method),
                timeout=args.timeout,
            )
            if should_close or not args.keepalive:
                await close_connection(writer, metrics)
                reader = None
                writer = None
        except Exception as exc:
            error = normalize_error(exc)
            await close_connection(writer, metrics)
            reader = None
            writer = None
        finally:
            latency_ms = (time.perf_counter() - started_at) * 1000.0
            metrics.request_finished(endpoint, latency_ms, status_code, error, bytes_read)

    try:
        if deadline is not None:
            while time.perf_counter() < deadline:
                for endpoint in endpoints:
                    if time.perf_counter() >= deadline:
                        break
                    await one_request(endpoint)
        else:
            for _ in range(args.iterations_per_user):
                for endpoint in endpoints:
                    await one_request(endpoint)

        if args.hold_open > 0 and writer is not None and not writer.is_closing():
            await asyncio.sleep(args.hold_open)
    finally:
        await close_connection(writer, metrics)
        metrics.mark_user_finished()


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def latency_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "avg": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": min(values),
        "avg": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def build_summary(
    args: argparse.Namespace,
    endpoints: Sequence[Endpoint],
    metrics: Metrics,
    elapsed_s: float,
) -> Dict[str, object]:
    success_rate = metrics.success / metrics.total if metrics.total else 0.0
    latencies = latency_summary(metrics.latencies_ms)
    throughput = metrics.total / elapsed_s if elapsed_s > 0 else 0.0

    pass_reasons: List[str] = []
    passed = True
    if metrics.virtual_users_started < args.users:
        passed = False
        pass_reasons.append(f"only {metrics.virtual_users_started}/{args.users} virtual users started")
    if success_rate < args.min_success_rate:
        passed = False
        pass_reasons.append(
            f"success rate {success_rate:.4f} is below required {args.min_success_rate:.4f}"
        )
    if args.max_p95_ms is not None and latencies["p95"] > args.max_p95_ms:
        passed = False
        pass_reasons.append(f"p95 latency {latencies['p95']:.2f} ms exceeds {args.max_p95_ms:.2f} ms")
    if args.min_open_connections and metrics.max_open_connections < args.min_open_connections:
        passed = False
        pass_reasons.append(
            f"max open connections {metrics.max_open_connections} is below {args.min_open_connections}"
        )
    if not metrics.total:
        passed = False
        pass_reasons.append("no requests completed")

    return {
        "passed": passed,
        "pass_reasons": pass_reasons,
        "config": {
            "base_url": args.base_url,
            "users": args.users,
            "preset": args.preset,
            "endpoints": [endpoint.key for endpoint in endpoints],
            "iterations_per_user": args.iterations_per_user,
            "duration_s": args.duration,
            "ramp_up_s": args.ramp_up,
            "timeout_s": args.timeout,
            "keepalive": args.keepalive,
            "hold_open_s": args.hold_open,
            "min_success_rate": args.min_success_rate,
            "max_p95_ms": args.max_p95_ms,
            "min_open_connections": args.min_open_connections,
        },
        "stats": {
            "elapsed_s": elapsed_s,
            "total_requests": metrics.total,
            "success": metrics.success,
            "failure": metrics.failure,
            "success_rate": success_rate,
            "throughput_rps": throughput,
            "bytes_read": metrics.bytes_read,
            "virtual_users_started": metrics.virtual_users_started,
            "virtual_users_finished": metrics.virtual_users_finished,
            "connection_attempts": metrics.connection_attempts,
            "connection_failures": metrics.connection_failures,
            "max_open_connections": metrics.max_open_connections,
            "max_active_requests": metrics.max_active_requests,
            "status_counts": dict(metrics.status_counts),
            "error_counts": dict(metrics.error_counts.most_common(20)),
            "endpoint_counts": dict(metrics.endpoint_counts),
            "endpoint_success": dict(metrics.endpoint_success),
        },
        "latency_ms": latencies,
    }


def print_summary(summary: Dict[str, object]) -> None:
    config = summary["config"]
    stats = summary["stats"]
    lat = summary["latency_ms"]

    print("\nRescueNet-RL concurrency test result")
    print("=" * 42)
    print(f"Base URL:              {config['base_url']}")
    print(f"Virtual users:         {config['users']}")
    print(f"Endpoints:             {', '.join(config['endpoints'])}")
    print(f"Elapsed:               {stats['elapsed_s']:.2f} s")
    print(f"Requests:              {stats['total_requests']}")
    print(f"Success:               {stats['success']} ({stats['success_rate'] * 100:.2f}%)")
    print(f"Failures:              {stats['failure']}")
    print(f"Throughput:            {stats['throughput_rps']:.2f} req/s")
    print(f"Max active requests:   {stats['max_active_requests']}")
    print(f"Max open connections:  {stats['max_open_connections']}")
    print(f"Connection failures:   {stats['connection_failures']}")
    print(
        "Latency ms:            "
        f"min {lat['min']:.2f}, avg {lat['avg']:.2f}, p50 {lat['p50']:.2f}, "
        f"p90 {lat['p90']:.2f}, p95 {lat['p95']:.2f}, p99 {lat['p99']:.2f}, max {lat['max']:.2f}"
    )
    print(f"HTTP status counts:    {stats['status_counts']}")
    if stats["error_counts"]:
        print(f"Top errors:            {stats['error_counts']}")
    print(f"Conclusion:            {'PASS' if summary['passed'] else 'FAIL'}")
    if summary["pass_reasons"]:
        for reason in summary["pass_reasons"]:
            print(f"  - {reason}")


async def progress_reporter(metrics: Metrics, interval_s: float, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(interval_s)
        if stop_event.is_set():
            break
        print(
            "\r"
            f"requests={metrics.total} "
            f"success={metrics.success} "
            f"fail={metrics.failure} "
            f"active={metrics.active_requests} "
            f"open={metrics.open_connections} "
            f"max_open={metrics.max_open_connections}",
            file=sys.stderr,
            end="",
            flush=True,
        )
    print(file=sys.stderr)


async def run_test(args: argparse.Namespace, endpoints: Sequence[Endpoint]) -> Dict[str, object]:
    target = parse_target(args.base_url)
    headers = parse_headers(args.header)
    body = load_body(args)
    metrics = Metrics()
    start_event = asyncio.Event()
    stop_progress = asyncio.Event()
    started_at = time.perf_counter()
    deadline = started_at + args.duration if args.duration else None

    users = [
        asyncio.create_task(
            virtual_user(
                user_id=i + 1,
                target=target,
                endpoints=endpoints,
                headers=headers,
                body=body,
                args=args,
                metrics=metrics,
                start_event=start_event,
                deadline=deadline,
            )
        )
        for i in range(args.users)
    ]
    progress_task = None
    if not args.quiet:
        progress_task = asyncio.create_task(progress_reporter(metrics, args.progress_interval, stop_progress))

    start_event.set()
    try:
        await asyncio.gather(*users)
    finally:
        stop_progress.set()
        if progress_task:
            await progress_task

    elapsed_s = time.perf_counter() - started_at
    return build_summary(args, endpoints, metrics, elapsed_s)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an asyncio-based concurrent HTTP load test against the RescueNet-RL API. "
            "Success is counted for HTTP 2xx/3xx responses."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API origin, default: %(default)s")
    parser.add_argument("--users", type=int, default=1200, help="number of concurrent virtual users")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="health",
        help="built-in endpoint set to run when --endpoint is not provided",
    )
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        help="endpoint spec such as '/api/health', 'GET /api/health', or 'POST:/api/simulate'; can repeat",
    )
    parser.add_argument("--iterations-per-user", type=int, default=1, help="full endpoint-flow repeats per user")
    parser.add_argument("--duration", type=float, default=0.0, help="run for N seconds instead of fixed iterations")
    parser.add_argument("--ramp-up", type=float, default=0.0, help="spread user starts across N seconds")
    parser.add_argument("--timeout", type=float, default=10.0, help="per request timeout in seconds")
    parser.add_argument(
        "--hold-open",
        type=float,
        default=5.0,
        help="seconds to keep successful keep-alive connections open after the last request",
    )
    parser.add_argument("--no-keepalive", dest="keepalive", action="store_false", help="close each connection")
    parser.set_defaults(keepalive=True)
    parser.add_argument("--header", action="append", default=[], help="extra HTTP header, e.g. 'Authorization: Bearer X'")
    parser.add_argument("--body", help="request body for POST/PUT/PATCH endpoints")
    parser.add_argument("--body-file", help="file containing request body for POST/PUT/PATCH endpoints")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.99,
        help="minimum success ratio required for PASS",
    )
    parser.add_argument("--max-p95-ms", type=float, help="optional p95 latency threshold for PASS")
    parser.add_argument(
        "--min-open-connections",
        type=int,
        default=0,
        help="optional required peak open connections, e.g. 1200 for an access-capacity proof",
    )
    parser.add_argument("--output", help="write JSON summary to this file")
    parser.add_argument("--quiet", action="store_true", help="hide progress output")
    parser.add_argument("--progress-interval", type=float, default=1.0, help="progress refresh interval in seconds")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.users <= 0:
        raise ValueError("--users must be positive")
    if args.iterations_per_user <= 0:
        raise ValueError("--iterations-per-user must be positive")
    if args.duration < 0:
        raise ValueError("--duration cannot be negative")
    if args.ramp_up < 0:
        raise ValueError("--ramp-up cannot be negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.hold_open < 0:
        raise ValueError("--hold-open cannot be negative")
    if not 0 <= args.min_success_rate <= 1:
        raise ValueError("--min-success-rate must be between 0 and 1")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        endpoint_specs = args.endpoint or PRESETS[args.preset]
        endpoints = [parse_endpoint(spec) for spec in endpoint_specs]
        fd_note = raise_fd_limit(args.users)
        if fd_note and not args.quiet:
            print(fd_note, file=sys.stderr)
        summary = asyncio.run(run_test(args, endpoints))
        print_summary(summary)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nJSON summary written to {output_path}")
        return 0 if summary["passed"] else 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
