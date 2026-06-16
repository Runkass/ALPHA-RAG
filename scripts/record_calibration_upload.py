#!/usr/bin/env python
"""Record Phase 12 calibration anchor platform score (journal + calibration json)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.update_state import append_journal_row, patch_run_state, refresh_state_files  # noqa: E402

BASELINE = 76.833

ANCHORS = {
    "K1": {
        "key": "platform_k1_answer_mass",
        "file": "archive/submissions/submission_anchor_k1_answer_mass.csv",
        "changed": 4693,
        "expected": "DOWN",
    },
    "K2": {
        "key": "platform_k2_refuse_wrong",
        "file": "archive/submissions/submission_anchor_k2_refuse_wrong.csv",
        "changed": 2284,
        "expected": "DOWN",
    },
    "K3": {
        "key": "platform_k3_partial_10",
        "file": "archive/submissions/submission_anchor_k3_partial_10.csv",
        "changed": 469,
        "expected": "DOWN",
    },
    "K4": {
        "key": "platform_k4_length_trim",
        "file": "archive/submissions/submission_anchor_k4_length_trim.csv",
        "changed": 3235,
        "expected": "UP_OR_DOWN",
    },
    "K5": {
        "key": "platform_k5_semantic_full",
        "file": "archive/submissions/submission_anchor_k5_semantic_full.csv",
        "changed": 82,
        "expected": "UP_OR_DOWN",
    },
    "K6": {
        "key": "platform_k6_semantic_10",
        "file": "archive/submissions/submission_anchor_k6_semantic_10.csv",
        "changed": 8,
        "expected": "FOLLOWS_K5",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, choices=sorted(ANCHORS.keys()))
    parser.add_argument("--platform", type=float, required=True)
    parser.add_argument("--step", default=None, help="STATE step override, e.g. K2 or C")
    args = parser.parse_args()

    meta = ANCHORS[args.anchor]
    delta = round(args.platform - BASELINE, 3)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = (
        f"{ts} | phase12-{args.anchor} | {meta['file']} | limit=all | RecallL=n/a | "
        f"comment=SUBMITTED platform={args.platform} tag={args.anchor} "
        f"expected={meta['expected']} delta={delta} changed={meta['changed']} review_ok=false"
    )
    append_journal_row(row)

    cal_path = ROOT / "data/cache/platform_calibration.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    cal[meta["key"]] = args.platform
    cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    step = args.step or args.anchor
    patch_run_state(
        phase="phase12",
        step=step,
        next_plan_file="agent/plan/13-phase12-calibration-logic.md",
        comment=f"{args.anchor} platform={args.platform} delta={delta}",
    )
    refresh_state_files()
    print(row)
    print(f"Updated {cal_path} key {meta['key']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
