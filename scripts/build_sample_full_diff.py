#!/usr/bin/env python
"""Classify sample vs FULL diffs for Phase 11 blitz queue."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal  # noqa: E402
from src.project_paths import BACKUP_FULL  # noqa: E402
from src.submission_rules import is_protected_short_answer, is_verbose_sample_answer  # noqa: E402


def classify_diff(sample_ans: str, full_ans: str) -> str:
    s = str(sample_ans)
    f = str(full_ans)
    if s == f:
        return "identical"
    if is_refusal(s) and not is_refusal(f):
        return "refusal_only"
    if not is_refusal(s) and is_refusal(f):
        return "full_refuse"
    if is_protected_short_answer(s):
        return "protected_short"
    ls, lf = len(s), len(f)
    if ls > 0 and lf > ls * 1.5:
        return "len_gap_full_longer"
    if lf > 0 and ls > lf * 1.5:
        return "len_gap_sample_longer"
    if is_verbose_sample_answer(s) and not is_verbose_sample_answer(f):
        return "semantic_gap_verbose_sample"
    if s.strip().lower() != f.strip().lower():
        return "semantic_gap"
    return "minor"


def build_diff(
    sample_path: Path,
    full_path: Path,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    full = pd.read_csv(full_path)
    m = sample.merge(full, on="q_id", suffixes=("_sample", "_full"))
    if limit:
        m = m.head(limit)
    rows: list[dict] = []
    for r in m.itertuples(index=False):
        s = str(r.answer_new_sample)
        f = str(r.answer_new_full)
        cls = classify_diff(s, f)
        if cls == "identical":
            continue
        rows.append(
            {
                "q_id": int(r.q_id),
                "sample_answer": s,
                "full_answer": f,
                "diff_class": cls,
                "len_sample": len(s),
                "len_full": len(f),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--full", default=str(BACKUP_FULL.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--output", default="data/cache/sample_full_diff.parquet")
    parser.add_argument("--heuristic-batch", default="data/cache/edit_batches/heuristic_micro15.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    df = build_diff(_p(args.sample), _p(args.full), limit=args.limit)
    out = _p(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    heur = df[df["diff_class"].isin(("refusal_only", "semantic_gap_verbose_sample", "semantic_gap"))]
    heur = heur.head(15)
    batch_out = _p(args.heuristic_batch)
    batch_out.parent.mkdir(parents=True, exist_ok=True)
    heur[["q_id"]].to_csv(batch_out, index=False)

    print(f"saved {len(df)} diffs -> {out}")
    if len(df):
        print(df["diff_class"].value_counts().to_string())
    print(f"heuristic micro15 -> {batch_out} n={len(heur)}")


if __name__ == "__main__":
    main()
