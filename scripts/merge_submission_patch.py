#!/usr/bin/env python
"""Merge patch q_ids into base submission CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def merge(base_path: Path, patch_path: Path, output_path: Path) -> int:
    base = pd.read_csv(base_path)
    patch = pd.read_csv(patch_path)
    out = pd.concat([base[~base.q_id.isin(patch.q_id)], patch]).sort_values("q_id")
    if len(out) != len(base):
        raise SystemExit(f"merge row count mismatch: {len(out)} vs {len(base)}")
    out.to_csv(output_path, index=False)
    print(f"merged {len(out)} rows, patch {len(patch)}")
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    merge(ROOT / args.base, ROOT / args.patch, ROOT / args.output)


if __name__ == "__main__":
    main()
