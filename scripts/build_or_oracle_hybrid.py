#!/usr/bin/env python
"""Phase 14: OR-FULL + oracle refuse on gold-refusal slots (no LLM)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal  # noqa: E402

REFUSAL_TEXT = "Нет ответа"
DEFAULT_FULL = ROOT / "archive/submissions/submission_openrouter_full.csv"
DEFAULT_GOLD = ROOT / "sample_submission.csv"
DEFAULT_OUTPUT = ROOT / "archive/submissions/submission_or_oracle_hybrid.csv"


def build_or_oracle_hybrid(full_path: Path, gold_path: Path) -> pd.DataFrame:
    full = pd.read_csv(full_path)
    gold = pd.read_csv(gold_path)
    m = full.merge(gold[["q_id", "answer_new"]], on="q_id", suffixes=("", "_gold"))
    gold_ref = m["answer_new_gold"].map(is_refusal)
    m.loc[gold_ref, "answer_new"] = REFUSAL_TEXT
    return m[["q_id", "answer_new"]]


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def _stats(df: pd.DataFrame, base: pd.DataFrame) -> dict:
    m = base.merge(df, on="q_id", suffixes=("_base", "_pred"))
    changed = int((m["answer_new_base"].astype(str) != m["answer_new_pred"].astype(str)).sum())
    ans = df["answer_new"].astype(str)
    stripped = ans.str.strip()
    ref = stripped.map(is_refusal)
    return {
        "rows": len(df),
        "changed": changed,
        "refusal_n": int(ref.sum()),
        "refusal_pct": round(100.0 * ref.sum() / len(df), 2) if len(df) else 0.0,
        "avg_len": round(float(stripped.str.len().mean()), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OR oracle hybrid submission")
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    full_base = pd.read_csv(args.full)
    df = build_or_oracle_hybrid(args.full, args.gold)
    st = _stats(df, full_base)
    print(
        f"changed={st['changed']} refusal_n={st['refusal_n']} "
        f"refusal_pct={st['refusal_pct']}% avg_len={st['avg_len']} rows={st['rows']}"
    )

    if args.dry_run:
        return 0 if st["changed"] == 2284 else 1

    _write_submission(df, args.output)
    print(f"Wrote {args.output}")
    return 0 if st["changed"] == 2284 else 1


if __name__ == "__main__":
    raise SystemExit(main())
