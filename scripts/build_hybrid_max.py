#!/usr/bin/env python
"""max(sample, patch) on changed q_id only — Phase 11 Tier 2 hybrid slot."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import length_multiplier, recall_l_pair  # noqa: E402


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def hybrid_max(
    sample_path: Path,
    patch_path: Path,
    *,
    gold_path: Path | None = None,
) -> tuple[pd.DataFrame, int]:
    sample = pd.read_csv(sample_path)
    patch = pd.read_csv(patch_path)
    gold_path = gold_path or sample_path
    gold = pd.read_csv(gold_path)

    smap = dict(zip(sample["q_id"].astype(int), sample["answer_new"].astype(str)))
    pmap = dict(zip(patch["q_id"].astype(int), patch["answer_new"].astype(str)))
    gmap = dict(zip(gold["q_id"].astype(int), gold["answer_new"].astype(str)))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    out = sample.copy()
    changed = 0

    for q, s_ans in smap.items():
        p_ans = pmap.get(q, s_ans)
        if p_ans == s_ans:
            continue
        g_ans = gmap.get(q, s_ans)
        s_score, _, _ = recall_l_pair(s_ans, g_ans, tokenizer)
        p_score, _, _ = recall_l_pair(p_ans, g_ans, tokenizer)
        if p_score > s_score:
            out.loc[out["q_id"].astype(int) == q, "answer_new"] = p_ans
            changed += 1
        elif len(p_ans) > len(s_ans) * 1.5 and len(s_ans) > 0:
            pass
        elif p_score == s_score and len(p_ans) < len(s_ans):
            la = len(tokenizer.encode(p_ans, add_special_tokens=False))
            lr = len(tokenizer.encode(g_ans, add_special_tokens=False)) or 1
            if length_multiplier(la, lr) > length_multiplier(
                len(tokenizer.encode(s_ans, add_special_tokens=False)), lr
            ):
                out.loc[out["q_id"].astype(int) == q, "answer_new"] = p_ans
                changed += 1

    return out, changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--patch", required=True)
    parser.add_argument("--gold", default="sample_submission.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    out, changed = hybrid_max(_p(args.sample), _p(args.patch), gold_path=_p(args.gold))
    out_path = _p(args.output)
    _write_submission(out, out_path)
    print(f"hybrid_max changed={changed} -> {out_path}")


if __name__ == "__main__":
    main()
