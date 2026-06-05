"""Validation utilities for metric2 network architecture proofs."""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
PROOFS_DIR = ROOT / "proofs"


def capture_output(fn: Callable[[], bool]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn()
    return buf.getvalue()


def save_proof(filename: str, content: str) -> Path:
    PROOFS_DIR.mkdir(parents=True, exist_ok=True)
    path = PROOFS_DIR / filename
    path.write_text(content, encoding="utf-8")
    return path


def print_header(title: str, width: int = 70) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
