#!/usr/bin/env python
"""Sweep REFUSE_RERANK_THRESHOLD on stratified 500 (retrieval only, no LLM)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25Index
from src.config import get_settings
from src.pipeline import load_dense_index
from src.retrieval import hybrid_retrieve

CACHE = ROOT / "data" / "cache" / "stratified_retrieval_scores.csv"
THRESHOLDS = [-5.5, -5.0, -4.8, -4.5, -4.2, -4.0, -3.5]


def _is_refusal(text: str) -> bool:
    return str(text).strip().lower().startswith("нет ответа")


def _load_scores() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE)

    settings = get_settings()
    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    gold = pd.read_csv(ROOT / "sample_submission.csv")
    questions = pd.read_csv(settings.questions_path)
    qids = pd.read_csv(ROOT / "data/cache/eval_stratified_500.csv")["q_id"]
    merged = gold.merge(questions, on="q_id")
    sub = merged[merged["q_id"].isin(qids)].copy()
    sub["gold_refusal"] = sub["answer_new"].map(_is_refusal)

    rows: list[dict] = []
    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="retrieve"):
        ch, low_rrf = hybrid_retrieve(str(row["query"]), dense, bm25)
        top = ch[0] if ch else None
        rows.append(
            {
                "q_id": int(row["q_id"]),
                "gold_refusal": bool(row["gold_refusal"]),
                "top_rrf": top.rrf_score if top else 0.0,
                "top_rerank": top.rerank_score if top else 0.0,
                "low_rrf": low_rrf,
            }
        )
    out = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False)
    return out


def main() -> None:
    df = _load_scores()
    n_ref = int(df["gold_refusal"].sum())
    n_ans = int((~df["gold_refusal"]).sum())
    print(f"pairs={len(df)} gold_refusal={n_ref} gold_answer={n_ans}")
    print("rerank_thr\trefusal_recall\tfalse_refuse\tpred_refusal_pct")
    for thr in THRESHOLDS:
        refuse = df["top_rerank"] < thr
        ref_hit = int((df["gold_refusal"] & refuse).sum())
        false_ref = int((~df["gold_refusal"] & refuse).sum())
        recall = ref_hit / n_ref if n_ref else 0.0
        pred_pct = 100.0 * refuse.sum() / len(df)
        print(f"{thr:.1f}\t{recall:.4f}\t{false_ref}\t{pred_pct:.1f}%")


if __name__ == "__main__":
    main()
