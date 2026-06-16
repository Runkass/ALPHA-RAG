#!/usr/bin/env python
"""Validate submission.csv against questions.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_settings


def validate(submission_path: Path) -> bool:
    settings = get_settings()
    questions = pd.read_csv(settings.questions_path)
    submission = pd.read_csv(submission_path)

    expected_ids = set(questions["q_id"].astype(int))
    got_ids = set(submission["q_id"].astype(int))

    ok = True
    missing = expected_ids - got_ids
    extra = got_ids - expected_ids
    if missing:
        print(f"FAIL: missing q_ids: {len(missing)} (e.g. {sorted(missing)[:5]})")
        ok = False
    if extra:
        print(f"WARN: extra q_ids: {len(extra)}")

    if "answer_new" not in submission.columns:
        print("FAIL: column 'answer_new' not found")
        ok = False
    else:
        empty = submission["answer_new"].isna() | (
            submission["answer_new"].astype(str).str.strip() == ""
        )
        n_empty = int(empty.sum())
        if n_empty:
            print(f"FAIL: {n_empty} empty answers")
            ok = False

        dup = submission["q_id"].duplicated().sum()
        if dup:
            print(f"FAIL: {dup} duplicate q_ids")
            ok = False

        no_answer = submission["answer_new"].astype(str).str.strip().str.lower().eq(
            "нет ответа"
        ).sum()
        avg_len = submission["answer_new"].astype(str).str.len().mean()
        print(f"Rows: {len(submission)}")
        print(f"'Нет ответа' count: {no_answer} ({100*no_answer/len(submission):.1f}%)")
        print(f"Average answer length: {avg_len:.0f} chars")

    if ok:
        print("OK: submission is valid")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="submission.csv")
    args = parser.parse_args()
    path = ROOT / args.file
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    sys.exit(0 if validate(path) else 1)
