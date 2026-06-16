#!/usr/bin/env python
"""White cache dry-run on strat500: count baseline hits without LLM."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("BASELINE_CACHE_ENABLED", "1")
os.environ.setdefault("BASELINE_CACHE_PATH", "sample_submission.csv")
os.environ.setdefault("RERANKER_ENABLED", "false")

from src.baseline_cache import maybe_baseline_answer  # noqa: E402
from src.bm25 import BM25Index  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.metrics.recall_l import is_refusal  # noqa: E402
from src.pipeline import RAGPipeline, load_dense_index  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rrf-threshold", type=float, default=None)
    parser.add_argument("--output", default="archive/submissions/submission_white_cache_smoke500.csv")
    args = parser.parse_args()
    if args.rrf_threshold is not None:
        os.environ["BASELINE_CONFIDENCE_RRF"] = str(args.rrf_threshold)

    qdf = pd.read_csv(ROOT / "data/cache/eval_stratified_500.csv")
    col = "q_id" if "q_id" in qdf.columns else qdf.columns[0]
    q_ids = [int(x) for x in qdf[col].tolist()]
    questions = pd.read_csv(ROOT / "questions.csv")
    qmap = dict(zip(questions["q_id"].astype(int), questions["query"].astype(str)))
    sample = dict(
        zip(
            pd.read_csv(ROOT / "sample_submission.csv")["q_id"].astype(int),
            pd.read_csv(ROOT / "sample_submission.csv")["answer_new"].astype(str),
        )
    )

    settings = get_settings()
    pipeline = RAGPipeline(load_dense_index(), BM25Index.load(settings.bm25_path), cache=None)

    hits = low = high = 0
    out_rows: list[tuple[int, str]] = []
    for q_id in tqdm(q_ids, desc="white_cache_dryrun"):
        query = qmap.get(q_id, "")
        chunks, _ctx, low_confidence = pipeline.retrieve_context(query)
        top_rrf = chunks[0].rrf_score if chunks else 0.0
        cached = maybe_baseline_answer(q_id, low_confidence=low_confidence, top_rrf=top_rrf)
        if low_confidence:
            low += 1
        else:
            high += 1
        if cached is not None:
            hits += 1
            ans = cached
        else:
            ans = sample.get(q_id, "Нет ответа")
        out_rows.append((q_id, ans))

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["q_id", "answer_new"])
        for q_id, ans in sorted(out_rows):
            w.writerow([q_id, ans])

    refusal = sum(1 for _, a in out_rows if is_refusal(a)) / len(out_rows)
    avg_len = sum(len(a) for _, a in out_rows) / len(out_rows)
    sample_refusal = sum(1 for q in q_ids if is_refusal(sample.get(q, ""))) / len(q_ids)
    print(
        f"dryrun n={len(q_ids)} baseline_hits={hits} low_conf={low} high_conf={high} "
        f"refusal={refusal:.1%} avg_len={avg_len:.0f} sample_refusal={sample_refusal:.1%}"
    )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
