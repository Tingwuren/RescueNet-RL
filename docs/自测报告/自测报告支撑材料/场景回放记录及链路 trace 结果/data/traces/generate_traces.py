"""Generate sample mahimahi trace files for demonstration.

Mahimahi trace format: one integer per line, each representing the
millisecond timestamp of a 1500-byte packet delivery opportunity.
The trace loops automatically in mahimahi.
"""

import math
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _write_trace(name: str, timestamps: "list[int]") -> str:
    path = os.path.join(SCRIPT_DIR, f"{name}.trace")
    with open(path, "w") as f:
        for t in sorted(timestamps):
            f.write(f"{t}\n")
    print(f"  {name}.trace  →  {len(timestamps)} packets, "
          f"~{timestamps[-1]}ms period")
    return path


def generate_constant(name: str, mbps: float, duration_ms: int = 2000):
    """Constant-rate link."""
    packets_per_sec = mbps * 1e6 / (1500 * 8)
    interval_ms = 1000.0 / packets_per_sec
    timestamps = []
    t = interval_ms
    while t <= duration_ms:
        timestamps.append(round(t))
        t += interval_ms
    _write_trace(name, timestamps)


def generate_variable_lte(name: str, duration_ms: int = 5000):
    """Variable LTE trace with realistic fluctuations."""
    random.seed(42)
    timestamps = []
    t = 0
    base_rate = 8.0  # Mbps average

    while t < duration_ms:
        phase = t / duration_ms
        rate = base_rate + 4.0 * math.sin(2 * math.pi * phase * 3)
        rate += random.gauss(0, 1.5)
        rate = max(0.5, min(18.0, rate))

        packets_per_sec = rate * 1e6 / (1500 * 8)
        if packets_per_sec > 0:
            interval = 1000.0 / packets_per_sec
        else:
            interval = 100.0

        t += interval
        if t <= duration_ms:
            timestamps.append(round(t))

    _write_trace(name, timestamps)


def generate_driving_lte(name: str, duration_ms: int = 8000):
    """LTE trace simulating driving with handovers and signal fades."""
    random.seed(123)
    timestamps = []
    t = 0.0

    segments = [
        (0, 1000, 12.0),
        (1000, 1800, 6.0),
        (1800, 2200, 0.8),
        (2200, 3500, 10.0),
        (3500, 4200, 3.0),
        (4200, 4500, 0.3),
        (4500, 5800, 14.0),
        (5800, 6500, 7.0),
        (6500, 7000, 1.5),
        (7000, 8000, 9.0),
    ]

    for seg_start, seg_end, base_rate in segments:
        t = float(seg_start)
        while t < seg_end:
            rate = base_rate + random.gauss(0, base_rate * 0.15)
            rate = max(0.1, rate)
            packets_per_sec = rate * 1e6 / (1500 * 8)
            interval = 1000.0 / packets_per_sec
            t += interval
            if t <= seg_end:
                timestamps.append(round(t))

    _write_trace(name, timestamps)


def generate_3g_umts(name: str, duration_ms: int = 5000):
    """Slower 3G/UMTS trace."""
    random.seed(99)
    timestamps = []
    t = 0.0
    while t < duration_ms:
        rate = 2.0 + 0.8 * math.sin(2 * math.pi * t / duration_ms * 2)
        rate += random.gauss(0, 0.3)
        rate = max(0.2, min(4.0, rate))
        packets_per_sec = rate * 1e6 / (1500 * 8)
        interval = 1000.0 / packets_per_sec
        t += interval
        if t <= duration_ms:
            timestamps.append(round(t))

    _write_trace(name, timestamps)


def generate_wifi_cafe(name: str, duration_ms: int = 5000):
    """High-bandwidth but bursty WiFi trace."""
    random.seed(77)
    timestamps = []
    t = 0.0
    while t < duration_ms:
        chunk_start = t
        is_burst = random.random() < 0.7
        if is_burst:
            rate = random.uniform(15.0, 40.0)
            chunk_len = random.uniform(100, 400)
        else:
            rate = random.uniform(0.5, 3.0)
            chunk_len = random.uniform(50, 200)

        packets_per_sec = rate * 1e6 / (1500 * 8)
        interval = 1000.0 / packets_per_sec
        while t < min(chunk_start + chunk_len, duration_ms):
            t += interval
            if t <= duration_ms:
                timestamps.append(round(t))
        t = chunk_start + chunk_len

    _write_trace(name, timestamps)


if __name__ == "__main__":
    print("Generating sample mahimahi traces...")
    generate_constant("emergency-command", 10.0)
    generate_variable_lte("damaged-station")
    generate_driving_lte("mobile-patrol")
    generate_3g_umts("flood-emergency")
    generate_wifi_cafe("temp-relay")
    print("Done.")
