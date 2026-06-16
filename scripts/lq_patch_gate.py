#!/usr/bin/env python
"""L(q) budget gate for Phase 13 patch candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal, length_multiplier  # noqa: E402
from src.project_paths import BACKUP_FULL  # noqa: E402

_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    return _TOKENIZER


def token_len(text: str) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len(_tokenizer().encode(t, add_special_tokens=False))


def gold_len_proxy(sample_ans: str, error_type: str, corpus_median_tokens: int) -> int:
    if is_refusal(sample_ans):
        return max(8, corpus_median_tokens // 4)
    if error_type == "verbosity":
        return max(token_len(sample_ans) // 2, 12)
    return max(token_len(sample_ans), 12)


def lq_check(pred: str, gold_proxy_tokens: int) -> tuple[float, bool, str]:
    la = token_len(pred)
    lr = max(gold_proxy_tokens, 1)
    lm = length_multiplier(float(la), float(lr))
    if la >= 3 * lr:
        return lm, False, "len_pred>=3x_gold"
    if lm < 0.5:
        return lm, False, "L_mult<0.5"
    return lm, True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", default="data/cache/phase13_manual_review.csv")
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--full", default=str(BACKUP_FULL.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--preview", default=None, help="optional CSV with q_id,answer_new proposals")
    parser.add_argument("--output", default="data/cache/phase13_lq_approved.csv")
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    manual = pd.read_csv(_p(args.manual))
    patch_rows = manual[manual["review_verdict"].astype(str).str.lower() == "patch"].copy()
    if not len(patch_rows):
        print("no review_verdict=patch rows")
        return 1

    sample_map = dict(
        zip(
            pd.read_csv(_p(args.sample))["q_id"].astype(int),
            pd.read_csv(_p(args.sample))["answer_new"].astype(str),
        )
    )
    full_map = dict(
        zip(
            pd.read_csv(_p(args.full))["q_id"].astype(int),
            pd.read_csv(_p(args.full))["answer_new"].astype(str),
        )
    )
    preview_map: dict[int, str] = {}
    if args.preview:
        prev = pd.read_csv(_p(args.preview))
        preview_map = dict(zip(prev["q_id"].astype(int), prev["answer_new"].astype(str)))

    non_refuse = [token_len(a) for a in sample_map.values() if not is_refusal(str(a))]
    non_refuse.sort()
    corpus_median = non_refuse[len(non_refuse) // 2] if non_refuse else 32

    rows: list[dict] = []
    for r in patch_rows.itertuples(index=False):
        q_id = int(r.q_id)
        sample_ans = sample_map.get(q_id, str(r.sample_ans))
        err = str(getattr(r, "error_type", "unknown"))
        pred = preview_map.get(q_id) or full_map.get(q_id, "")
        if not pred or is_refusal(pred):
            rows.append(
                {
                    "q_id": q_id,
                    "error_type": err,
                    "cohort": getattr(r, "cohort", ""),
                    "pred_source": "missing",
                    "L_mult": 0.0,
                    "approved": False,
                    "reject_reason": "no_pred_answer",
                }
            )
            continue
        gold_proxy = gold_len_proxy(str(sample_ans), err, corpus_median)
        lm, ok, reason = lq_check(pred, gold_proxy)
        rows.append(
            {
                "q_id": q_id,
                "error_type": err,
                "cohort": getattr(r, "cohort", ""),
                "pred_source": "preview" if q_id in preview_map else "full",
                "len_pred_tokens": token_len(pred),
                "len_gold_proxy": gold_proxy,
                "L_mult": round(lm, 4),
                "approved": ok,
                "reject_reason": reason if ok else reason,
                "proposed_answer": pred[:200],
            }
        )

    out_df = pd.DataFrame(rows)
    approved = out_df[out_df["approved"]].copy()
    out = _p(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    approved.to_csv(out, index=False)

    rej = (~out_df["approved"]).sum()
    print(f"lq_gate total={len(out_df)} approved={len(approved)} rejected={rej} -> {out}")
    if rej:
        print(out_df[~out_df["approved"]][["q_id", "reject_reason"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
