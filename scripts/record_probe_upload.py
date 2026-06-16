#!/usr/bin/env python
"""Record Phase 13 frontier probe platform score (journal + calibration json)."""

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

PROBES = {
    "E1": {
        "key": "platform_e1_refuse5",
        "file": "archive/submissions/submission_e1_refuse5.csv",
        "expected": "UP_OR_NEUTRAL",
    },
    "E2": {
        "key": "platform_e2_verb5",
        "file": "archive/submissions/submission_e2_verb5.csv",
        "expected": "UP_OR_NEUTRAL",
    },
    "E3": {
        "key": "platform_e3_extract5",
        "file": "archive/submissions/submission_e3_extract5.csv",
        "expected": "UP_OR_NEUTRAL",
    },
    "E1b": {
        "key": "platform_e1_refuse20",
        "file": "archive/submissions/submission_e1_refuse20.csv",
        "expected": "UP",
    },
    "E2b": {
        "key": "platform_e2_verb20",
        "file": "archive/submissions/submission_e2_verb20.csv",
        "expected": "UP",
    },
}


def _count_changed(sub_path: Path) -> int:
    if not sub_path.exists():
        return 0
    import pandas as pd

    sample = pd.read_csv(ROOT / "sample_submission.csv")
    sub = pd.read_csv(sub_path)
    m = sample.merge(sub, on="q_id", suffixes=("_s", "_p"))
    return int((m["answer_new_s"].astype(str) != m["answer_new_p"].astype(str)).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, choices=sorted(PROBES.keys()))
    parser.add_argument("--platform", type=float, default=None)
    parser.add_argument("--status", default="BUILT", choices=("BUILT", "SUBMITTED"))
    parser.add_argument("--step", default=None)
    args = parser.parse_args()

    meta = PROBES[args.probe]
    sub_path = ROOT / meta["file"]
    changed = _count_changed(sub_path)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    if args.status == "BUILT":
        row = (
            f"{ts} | phase13-{args.probe} | {meta['file']} | limit=all | RecallL=n/a | "
            f"comment=BUILT changed={changed} validate=READY UP tag={args.probe} review_ok=false"
        )
    else:
        if args.platform is None:
            raise SystemExit("--platform required for SUBMITTED")
        delta = round(args.platform - BASELINE, 3)
        row = (
            f"{ts} | phase13-{args.probe} | {meta['file']} | limit=all | RecallL=n/a | "
            f"comment=SUBMITTED platform={args.platform} tag={args.probe} "
            f"expected={meta['expected']} delta={delta} changed={changed} review_ok=false"
        )
        cal_path = ROOT / "data/cache/platform_calibration.json"
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        cal[meta["key"]] = args.platform
        cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {cal_path} key {meta['key']}")

    append_journal_row(row)
    step = args.step or args.probe
    patch_run_state(
        phase="phase13",
        step=step,
        next_plan_file="agent/plan/14-phase13-baseline-diagnosis.md",
        comment=f"{args.probe} {args.status} changed={changed}",
    )
    refresh_state_files()
    print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
