#!/usr/bin/env python
"""Merge patch into base only where local Recall-L improves vs gold."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_max_sample_full import _batch_recall_l  # noqa: E402


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def merge_with_gate(
    base_path: Path,
    patch_path: Path,
    gold_path: Path,
    *,
    min_delta: float = 0.0,
    batch_size: int = 32,
) -> tuple[pd.DataFrame, dict]:
    base = pd.read_csv(base_path)
    patch = pd.read_csv(patch_path)
    gold = pd.read_csv(gold_path)
    patch_ids = set(patch["q_id"].astype(int).tolist())
    m = gold.merge(base, on="q_id", suffixes=("_gold", "_base")).merge(
        patch, on="q_id", suffixes=("", "_patch")
    )
    m = m[m["q_id"].isin(patch_ids)].copy()
    m = m.rename(columns={"answer_new": "answer_new_patch"})

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    golds = m["answer_new_gold"].astype(str).tolist()
    base_scores = _batch_recall_l(
        m["answer_new_base"].astype(str).tolist(), golds, tokenizer, batch_size=batch_size
    )
    patch_scores = _batch_recall_l(
        m["answer_new_patch"].astype(str).tolist(), golds, tokenizer, batch_size=batch_size
    )

    accept_ids: set[int] = set()
    for q_id, b_sc, p_sc in zip(m["q_id"].astype(int), base_scores, patch_scores):
        if p_sc >= b_sc + min_delta:
            accept_ids.add(int(q_id))

    patch_map = dict(zip(patch["q_id"].astype(int), patch["answer_new"].astype(str)))
    out = base.copy()
    applied = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q in accept_ids:
            out.at[i, "answer_new"] = patch_map[q]
            applied += 1

    meta = {
        "patch_rows": len(patch_ids),
        "evaluated": len(m),
        "accepted": applied,
        "rejected": len(patch_ids) - applied,
        "min_delta": min_delta,
    }
    return out[["q_id", "answer_new"]], meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--gold", default="sample_submission.csv")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    df, meta = merge_with_gate(
        _p(args.base),
        _p(args.patch),
        _p(args.gold),
        min_delta=args.min_delta,
        batch_size=args.batch_size,
    )
    out = _p(args.output)
    _write_submission(df, out)
    print(f"meta={meta} -> {out}")


if __name__ == "__main__":
    main()
