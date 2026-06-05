"""Bootstrap imports for RescueNet-RL root packages when running from HMARL/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HMARL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = HMARL_ROOT.parent
RESCUENET_DIR = Path(__file__).resolve().parent


def setup_repo_path(*, chdir: bool = True) -> Path:
    """Use repo root for imports; drop HMARL from path to avoid shadowing `models`/`env`."""
    repo = str(REPO_ROOT)
    hmarl = str(HMARL_ROOT)
    while hmarl in sys.path:
        sys.path.remove(hmarl)
    if repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)
    if chdir:
        os.chdir(REPO_ROOT)
    return REPO_ROOT


def ensure_repo_path(*, chdir: bool = True) -> Path:
    """Alias used before any `services` / `models` import from repo root."""
    return setup_repo_path(chdir=chdir)
