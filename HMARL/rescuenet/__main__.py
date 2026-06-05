"""python -m rescuenet.train | python -m rescuenet.validate"""

from __future__ import annotations

import sys
from pathlib import Path

_HMARL_ROOT = Path(__file__).resolve().parents[1]
if str(_HMARL_ROOT) not in sys.path:
    sys.path.insert(0, str(_HMARL_ROOT))


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "Usage (from HMARL/):\n"
            "  python -m rescuenet.train --scenario super_typhoon\n"
            "  python -m rescuenet.validate --scenario super_typhoon\n"
            "Or:\n"
            "  python rescuenet/train.py ...\n"
            "  python rescuenet/validate.py ..."
        )
        raise SystemExit(0 if len(sys.argv) > 1 and sys.argv[1] in {"-h", "--help"} else 1)
    command = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    if command == "train":
        from rescuenet.train import main as train_main

        train_main()
    elif command == "validate":
        from rescuenet.validate import main as validate_main

        validate_main()
    else:
        raise SystemExit(f"Unknown command: {command}. Use train or validate.")


if __name__ == "__main__":
    main()
