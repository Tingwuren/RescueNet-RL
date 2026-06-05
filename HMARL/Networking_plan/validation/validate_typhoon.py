"""Validate super typhoon scenario + both network modes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation._utils import capture_output, save_proof
from validation.validate_scenario_common import validate_scenario


def main() -> None:
    content = capture_output(lambda: validate_scenario("super_typhoon", "超强台风"))
    print(content, end="")
    save_proof("scenario_typhoon.txt", content)
    sys.exit(0 if "[PASS]" in content else 1)


if __name__ == "__main__":
    main()
