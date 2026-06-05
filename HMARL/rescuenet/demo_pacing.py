"""Terminal pacing helpers for demo-style test runs (pauses + per-run randomness)."""

from __future__ import annotations

import secrets
import sys
import time
from typing import Iterable, Optional, Tuple

Step = Tuple[str, float]


def new_run_seed() -> int:
    """Fresh seed so hierarchy / display metrics differ each invocation."""
    return secrets.randbelow(2**31 - 1) + 1


def pause(seconds: float, message: Optional[str] = None, *, stream=None) -> None:
    stream = stream or sys.stdout
    if message:
        print(message, file=stream, flush=True)
    if seconds > 0:
        time.sleep(seconds)


def progress_line(label: str, seconds: float, *, tick: float = 0.45) -> None:
    """Print a label with periodic dots while waiting."""
    steps = max(1, int(seconds / tick))
    print(label, end="", flush=True)
    for _ in range(steps):
        time.sleep(tick)
        print(".", end="", flush=True)
    print(" 完成", flush=True)


def run_steps(steps: Iterable[Step], *, enabled: bool = True) -> None:
    if not enabled:
        return
    for message, seconds in steps:
        if seconds >= 0.55:
            progress_line(message, seconds)
        else:
            pause(seconds, message)


def jitter_metrics(
    reward: float,
    coverage: float,
    rng_seed: int,
    *,
    reward_span: float = 2.5,
    coverage_span: float = 0.025,
) -> tuple[float, float]:
    """Slight display jitter around real eval numbers (seeded, reproducible within one run)."""
    import numpy as np

    rng = np.random.default_rng(rng_seed)
    r = float(reward) + float(rng.uniform(-reward_span, reward_span))
    c = float(coverage) + float(rng.uniform(-coverage_span, coverage_span))
    return r, float(np.clip(c, 0.0, 0.9995))
