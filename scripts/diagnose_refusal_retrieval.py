#!/usr/bin/env python
"""Diagnose retrieval scores on refusal vs answer gold subsets."""

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


def _top_rrf(query: str, dense, bm25) -> tuple[float, float, int]:
    chunks, low = hybrid_retrieve(query, dense, bm25)
    top_rrf = chunks[0].rrf_score if chunks else 0.0
    top_rerank = chunks[0].rerank_score if chunks else 0.0
    return top_rrf, top_rerank, len(chunks), low


def main() -> None:
    settings = get_settings()
    gold = pd.read_csv(ROOT / "sample_submission.csv")
    questions = pd.read_csv(settings.questions_path)
    gold = gold.merge(questions, on="q_id")
    gold["is_refusal"] = (
        gold["answer_new"].astype(str).str.strip().str.lower().str.startswith("нет ответа")
    )

    refusal_path = ROOT / "data" / "cache" / "eval_refusal_only_200.csv"
    refusal_ids = set(pd.read_csv(refusal_path)["q_id"].astype(int))

    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)

    from tqdm import tqdm

    rows: list[dict] = []
    refusal_rows = gold[gold["q_id"].isin(refusal_ids)]
    for _, row in tqdm(refusal_rows.iterrows(), total=len(refusal_rows), desc="refusal"):
        q_id = int(row.q_id)
        label = "refusal"
        query = str(row.query)
        top_rrf, top_rerank, n_chunks, low = _top_rrf(query, dense, bm25)
        rows.append(
            {
                "q_id": q_id,
                "label": label,
                "top_rrf": top_rrf,
                "top_rerank": top_rerank,
                "n_chunks": n_chunks,
                "low_confidence": low,
            }
        )

    non_refusal = gold.loc[~gold["is_refusal"]].sample(n=200, random_state=42)
    for _, row in tqdm(non_refusal.iterrows(), total=len(non_refusal), desc="answer"):
        query = str(row.query)
        top_rrf, top_rerank, n_chunks, low = _top_rrf(query, dense, bm25)
        rows.append(
            {
                "q_id": int(row.q_id),
                "label": "answer",
                "top_rrf": top_rrf,
                "top_rerank": top_rerank,
                "n_chunks": n_chunks,
                "low_confidence": low,
            }
        )

    out = pd.DataFrame(rows)
    out_path = ROOT / "data" / "cache" / "refusal_retrieval_diag.csv"
    out.to_csv(out_path, index=False)

    med_ref = out.loc[out["label"] == "refusal", "top_rrf"].median()
    med_ans = out.loc[out["label"] == "answer", "top_rrf"].median()
    print(f"saved {out_path} rows={len(out)}")
    print(f"median_rrf_refusal={med_ref:.6f}")
    print(f"median_rrf_answer={med_ans:.6f}")


if __name__ == "__main__":
    main()
