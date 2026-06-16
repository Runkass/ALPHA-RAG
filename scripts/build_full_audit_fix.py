#!/usr/bin/env python
"""P3: FULL + replace obvious garbage with «Нет ответа»."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.project_paths import ARCHIVE_SUBMISSIONS, BACKUP_FULL  # noqa: E402
from src.submission_rules import REFUSAL_TEXT, is_garbage_answer  # noqa: E402


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def build_audit_fix(source: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(source).copy()
    ans = df["answer_new"].astype(str)
    mask = ans.map(is_garbage_answer)
    changed = int(mask.sum())
    df.loc[mask, "answer_new"] = REFUSAL_TEXT
    return df[["q_id", "answer_new"]], changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(BACKUP_FULL))
    parser.add_argument(
        "--output",
        default=str(ARCHIVE_SUBMISSIONS / "submission_p3_audit_fix.csv"),
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out

    df, changed = build_audit_fix(source)
    _write_submission(df, out)
    print(f"changed={changed} rows={len(df)} -> {out}")


if __name__ == "__main__":
    main()
