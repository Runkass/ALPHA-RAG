#!/usr/bin/env python
"""3-axis Recall-L decomposition + patch-only metrics (Phase 11)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal, recall_l_corpus  # noqa: E402
from src.submission_rules import is_protected_q_id, load_protected_q_ids  # noqa: E402


def _load_q_ids(path: Path | None, pred: pd.DataFrame, base: pd.DataFrame | None) -> list[int] | None:
    if path and path.exists():
        df = pd.read_csv(path)
        col = "q_id" if "q_id" in df.columns else df.columns[0]
        return df[col].astype(int).tolist()
    if base is not None:
        m = base.merge(pred, on="q_id", suffixes=("_base", "_pred"))
        changed = m[m["answer_new_base"].astype(str) != m["answer_new_pred"].astype(str)]
        return changed["q_id"].astype(int).tolist()
    return None


def _subset(df: pd.DataFrame, q_ids: list[int] | None) -> pd.DataFrame:
    if not q_ids:
        return df
    s = set(q_ids)
    return df[df["q_id"].astype(int).isin(s)]


def decompose(
    pred_path: Path,
    gold_path: Path,
    *,
    base_path: Path | None = None,
    q_ids_file: Path | None = None,
    limit: int | None = None,
    deny_file: Path | None = None,
) -> dict:
    pred_df = pd.read_csv(pred_path)
    gold_df = pd.read_csv(gold_path)
    base_df = pd.read_csv(base_path) if base_path else None

    merged = pred_df.merge(gold_df, on="q_id", suffixes=("_pred", "_gold"))
    if limit is not None and limit > 0:
        merged = merged.head(limit)

    patch_q_ids = _load_q_ids(q_ids_file, pred_df, base_df)
    if patch_q_ids is not None:
        merged = _subset(merged, patch_q_ids)

    preds = merged["answer_new_pred"].astype(str).tolist()
    golds = merged["answer_new_gold"].astype(str).tolist()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    report = recall_l_corpus(preds, golds, tokenizer, batch_size=32)

    deny = load_protected_q_ids(str(deny_file)) if deny_file else set()
    protected_touched = 0
    if base_df is not None and patch_q_ids:
        bmap = dict(zip(base_df["q_id"].astype(int), base_df["answer_new"].astype(str)))
        pmap = dict(zip(pred_df["q_id"].astype(int), pred_df["answer_new"].astype(str)))
        for q in patch_q_ids:
            if bmap.get(q) != pmap.get(q) and is_protected_q_id(q, deny):
                protected_touched += 1

    len_ratio_violations = 0
    if base_df is not None and patch_q_ids:
        bmap = dict(zip(base_df["q_id"].astype(int), base_df["answer_new"].astype(str)))
        for q in patch_q_ids:
            b = bmap.get(q, "")
            p = dict(zip(pred_df["q_id"].astype(int), pred_df["answer_new"].astype(str))).get(q, b)
            if b != p and len(p) > len(b) * 1.5 and len(b) > 0:
                len_ratio_violations += 1

    return {
        "pred": str(pred_path),
        "gold": str(gold_path),
        "n": report.n,
        "recall_l": round(report.recall_l, 4),
        "r_bert_mean": round(report.r_bert_mean, 4),
        "l_mult_mean": round(report.l_mult_mean, 4),
        "pred_refusal_rate": round(report.pred_refusal_rate, 4),
        "gold_refusal_rate": round(report.gold_refusal_rate, 4),
        "false_refuse": report.false_refuse,
        "false_answer": report.false_answer,
        "answer_only_recall_l": round(report.answer_only_recall_l, 4),
        "refusal_only_recall_l": round(report.refusal_only_recall_l, 4),
        "patch_n": len(patch_q_ids) if patch_q_ids else report.n,
        "protected_touched": protected_touched,
        "len_ratio_violations": len_ratio_violations,
        "patch_only": patch_q_ids is not None and base_path is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gold", default="sample_submission.csv")
    parser.add_argument("--base", default=None, help="sample base for patch-only q_ids")
    parser.add_argument("--q-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--deny-file", default="data/cache/edit_batches/s2_do_not_touch.csv")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    def _p(s: str | None) -> Path | None:
        if not s:
            return None
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    meta = decompose(
        _p(args.pred),
        _p(args.gold),
        base_path=_p(args.base),
        q_ids_file=_p(args.q_ids_file),
        limit=args.limit,
        deny_file=_p(args.deny_file),
    )
    if args.json:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    else:
        print(
            f"decompose n={meta['n']} patch_n={meta['patch_n']} "
            f"RecallL={meta['recall_l']} r_bert={meta['r_bert_mean']} "
            f"l_mult={meta['l_mult_mean']} pred_refusal={meta['pred_refusal_rate']:.1%} "
            f"false_refuse={meta['false_refuse']} false_answer={meta['false_answer']} "
            f"protected_touched={meta['protected_touched']} len_violations={meta['len_ratio_violations']}"
        )


if __name__ == "__main__":
    main()
