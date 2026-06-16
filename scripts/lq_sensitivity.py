#!/usr/bin/env python
"""L(q) sensitivity for short answers (Phase 12 L3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import length_multiplier  # noqa: E402


def _token_len(text: str, tokenizer) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len(tokenizer.encode(t, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--gold", default="sample_submission.csv")
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--output", default="data/cache/phase12_L3_lq_sensitivity.csv")
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    sample = pd.read_csv(_p(args.sample))
    gold = pd.read_csv(_p(args.gold))
    m = sample.merge(gold, on="q_id", suffixes=("_sample", "_gold"))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    rows: list[dict] = []
    for r in m.itertuples(index=False):
        s = str(r.answer_new_sample)
        g = str(r.answer_new_gold)
        if len(s) > args.max_len:
            continue
        lr = _token_len(g, tokenizer) or 1
        for mult in (1.0, 1.5, 2.0, 3.0):
            la = int(lr * mult)
            lm = length_multiplier(float(la), float(lr))
            rows.append(
                {
                    "q_id": int(r.q_id),
                    "len_sample": len(s),
                    "len_gold_tokens": lr,
                    "len_mult_factor": mult,
                    "sim_len_pred_tokens": la,
                    "L_mult": round(lm, 4),
                }
            )

    out = _p(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    at3 = df[df["len_mult_factor"] == 3.0]
    zero_frac = (at3["L_mult"] == 0.0).mean() if len(at3) else 0.0
    at2 = df[df["len_mult_factor"] == 2.0]
    zero2 = (at2["L_mult"] == 0.0).mean() if len(at2) else 0.0
    print(f"short q (len<={args.max_len}): {m['answer_new_sample'].astype(str).str.len().le(args.max_len).sum()}")
    print(f"rows={len(df)} L_mult=0 at 2x: {zero2:.1%} at 3x: {zero_frac:.1%}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
