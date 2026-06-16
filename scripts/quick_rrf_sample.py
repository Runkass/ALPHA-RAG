#!/usr/bin/env python
"""Quick RRF sample for threshold tuning (no LLM)."""

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

THRESHOLDS = [0.010, 0.012, 0.014, 0.015, 0.016, 0.017]


def main() -> None:
    settings = get_settings()
    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    gold = pd.read_csv(ROOT / "sample_submission.csv")
    questions = pd.read_csv(settings.questions_path)
    merged = gold.merge(questions, on="q_id")
    strat = pd.read_csv(ROOT / "data/cache/eval_stratified_500.csv")["q_id"]
    sub = merged[merged["q_id"].isin(strat)]
    sub["is_ref"] = sub["answer_new"].str.strip().str.lower().str.startswith("нет ответа")
    ref = sub[sub["is_ref"]].sample(n=min(50, sub["is_ref"].sum()), random_state=1)
    ans = sub[~sub["is_ref"]].sample(n=min(50, (~sub["is_ref"]).sum()), random_state=2)

    scores: dict[int, tuple[float, float]] = {}
    for frame in (ref, ans):
        for _, row in frame.iterrows():
            ch, _ = hybrid_retrieve(str(row["query"]), dense, bm25)
            scores[int(row["q_id"])] = (
                ch[0].rrf_score if ch else 0.0,
                ch[0].rerank_score if ch else 0.0,
            )

    print("threshold\trefusal_rrf\tanswer_false_rrf")
    for thr in THRESHOLDS:
        r_hit = sum(1 for _, row in ref.iterrows() if scores[int(row["q_id"])][0] <= thr)
        a_false = sum(1 for _, row in ans.iterrows() if scores[int(row["q_id"])][0] <= thr)
        print(f"{thr:.3f}\t{r_hit}/{len(ref)}\t{a_false}/{len(ans)}")

    print("combo rrf<=0.016 & rerank<3.5")
    r_combo = sum(
        1
        for _, row in ref.iterrows()
        if scores[int(row["q_id"])][0] <= 0.016 and scores[int(row["q_id"])][1] < 3.5
    )
    a_combo = sum(
        1
        for _, row in ans.iterrows()
        if scores[int(row["q_id"])][0] <= 0.016 and scores[int(row["q_id"])][1] < 3.5
    )
    print(f"combo\t{r_combo}/{len(ref)}\t{a_combo}/{len(ans)}")
    ref_rr = [scores[int(r['q_id'])][1] for _, r in ref.iterrows()]
    ans_rr = [scores[int(r['q_id'])][1] for _, r in ans.iterrows()]
    import statistics as st

    print(f"rerank median ref={st.median(ref_rr):.3f} ans={st.median(ans_rr):.3f}")
    print("rerank_thr\trefusal_hit\tanswer_false")
    for thr in (-4.5, -4.0, -3.5, -3.0, -2.5, -2.0):
        r_hit = sum(1 for _, row in ref.iterrows() if scores[int(row["q_id"])][1] < thr)
        a_false = sum(1 for _, row in ans.iterrows() if scores[int(row["q_id"])][1] < thr)
        print(f"{thr:.1f}\t{r_hit}/{len(ref)}\t{a_false}/{len(ans)}")


if __name__ == "__main__":
    main()
