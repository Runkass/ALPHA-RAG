#!/usr/bin/env python
"""Sweep MIN_RRF_SCORE without LLM — estimate refusal metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25Index
from src.config import get_settings
from src.pipeline import load_dense_index
from src.retrieval import hybrid_retrieve

THRESHOLDS = [0.003, 0.005, 0.008, 0.01, 0.015]


def _is_gold_refusal(text: str) -> bool:
    return str(text).strip().lower().startswith("нет ответа")


def _would_refuse(top_rrf: float, threshold: float) -> bool:
    return top_rrf < threshold


def main() -> None:
    settings = get_settings()
    gold = pd.read_csv(ROOT / "sample_submission.csv")
    questions = pd.read_csv(settings.questions_path)
    merged = gold.merge(questions, on="q_id")

    refusal_ids = set(
        pd.read_csv(ROOT / "data" / "cache" / "eval_refusal_only_200.csv")["q_id"]
    )
    strat_ids = set(
        pd.read_csv(ROOT / "data" / "cache" / "eval_stratified_500.csv")["q_id"]
    )

    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)

    scores: dict[int, tuple[float, bool]] = {}
    for q_id in refusal_ids | strat_ids:
        row = merged.loc[merged["q_id"] == q_id].iloc[0]
        chunks, _low = hybrid_retrieve(str(row["query"]), dense, bm25)
        top_rrf = chunks[0].rrf_score if chunks else 0.0
        scores[q_id] = (top_rrf, _is_gold_refusal(row["answer_new"]))

    print("threshold\trefusal_recall_est\tfalse_refuse_est")
    for thr in THRESHOLDS:
        ref_hits = ref_total = 0
        false_refuse = 0
        for q_id in refusal_ids:
            top_rrf, is_ref = scores[q_id]
            if is_ref:
                ref_total += 1
                if _would_refuse(top_rrf, thr):
                    ref_hits += 1
        for q_id in strat_ids:
            top_rrf, is_ref = scores[q_id]
            if not is_ref and _would_refuse(top_rrf, thr):
                false_refuse += 1

        recall_est = ref_hits / ref_total if ref_total else 0.0
        print(f"{thr:.3f}\t{recall_est:.4f}\t{false_refuse}")


if __name__ == "__main__":
    main()
