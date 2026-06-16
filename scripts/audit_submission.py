#!/usr/bin/env python
"""Content audit wrapper before platform upload."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.build_platform_anchors import audit_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit submission CSV (6977 rows, no garbage)")
    parser.add_argument("--file", required=True, help="Path to submission CSV")
    parser.add_argument("--variant", default=None, help="Optional variant hint for audit")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / path
    return audit_file(path, variant=args.variant)


if __name__ == "__main__":
    sys.exit(main())
