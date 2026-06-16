#!/usr/bin/env python
"""Export Phase 13 manual review CSV for cohorts A and B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25Index  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.pipeline import RAGPipeline, load_dense_index  # noqa: E402
from src.retrieval import format_context  # noqa: E402

BATCHES = ROOT / "data" / "cache" / "edit_batches"


def _p(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _refusal_confidence_score(rrf: float, rerank: float, overlap: float) -> float:
    return round(0.5 * min(1.0, rrf / 0.05) + 0.3 * min(1.0, (rerank + 5) / 10) + 0.2 * overlap, 4)


def _overlap(query: str, chunk_text: str) -> float:
    from src.retrieval import _WORD_RE

    q_words = set(_WORD_RE.findall(str(query).lower()))
    c_words = set(_WORD_RE.findall(str(chunk_text).lower()))
    if not q_words or not c_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


def _score_refusal(pipeline: RAGPipeline, query: str, settings) -> tuple[str, float]:
    chunks, _ctx, low_confidence = pipeline.retrieve_context(query)
    chunk_text = chunks[0].text if chunks else ""
    top_rrf = chunks[0].rrf_score if chunks else 0.0
    top_rerank = chunks[0].rerank_score if chunks else -99.0
    high_conf = not low_confidence and bool(chunk_text.strip()) and top_rrf >= settings.min_rrf_score
    conf = _refusal_confidence_score(top_rrf, top_rerank, _overlap(query, chunk_text)) if high_conf else 0.0
    preview = chunk_text[:400] + ("..." if len(chunk_text) > 400 else "")
    return preview, conf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", default="data/cache/phase13_cohorts.parquet")
    parser.add_argument("--candidates", default="data/cache/sample_candidates.parquet")
    parser.add_argument("--top-a", type=int, default=30)
    parser.add_argument("--top-b", type=int, default=30)
    parser.add_argument("--output", default="data/cache/phase13_manual_review.csv")
    parser.add_argument("--with-retrieval", action="store_true", default=True)
    parser.add_argument("--no-retrieval", dest="with_retrieval", action="store_false")
    args = parser.parse_args()

    cohorts = pd.read_parquet(_p(args.cohorts))
    cand = pd.read_parquet(_p(args.candidates)) if _p(args.candidates).exists() else pd.DataFrame()

    a_df = cohorts[cohorts["cohorts"].str.contains("A")].copy()
    if len(cand):
        ref = cand[cand["candidate_type"] == "refusal_high_conf"][
            ["q_id", "refusal_confidence"]
        ].rename(columns={"refusal_confidence": "refusal_confidence_cand"})
        if len(ref):
            a_df = a_df.drop(columns=["refusal_confidence"], errors="ignore").merge(
                ref, on="q_id", how="left"
            )
            a_df["refusal_confidence"] = a_df["refusal_confidence_cand"].fillna(0)
            a_df = a_df.drop(columns=["refusal_confidence_cand"], errors="ignore")
        elif "refusal_confidence" not in a_df.columns:
            a_df["refusal_confidence"] = 0.0
    elif "refusal_confidence" not in a_df.columns:
        a_df["refusal_confidence"] = 0.0
    a_df = a_df.sort_values("refusal_confidence", ascending=False)
    a_pick = a_df.head(args.top_a)

    b_df = cohorts[cohorts["cohorts"].str.contains("B")].copy()
    b_df = b_df.sort_values("verbosity_score", ascending=False)
    b_pick = b_df.head(args.top_b)

    pick = pd.concat([a_pick, b_pick], ignore_index=True).drop_duplicates(subset=["q_id"])

    pipeline = None
    settings = None
    if args.with_retrieval:
        settings = get_settings()
        pipeline = RAGPipeline(load_dense_index(), BM25Index.load(settings.bm25_path), cache=None)

    rows: list[dict] = []
    for r in pick.itertuples(index=False):
        query = str(r.query)
        top_chunk = ""
        conf = float(getattr(r, "refusal_confidence", 0.0) or 0.0)
        if pipeline is not None and settings is not None:
            top_chunk, rconf = _score_refusal(pipeline, query, settings)
            if rconf > conf:
                conf = rconf
        primary = "A" if "A" in str(r.cohorts) else "B"
        rows.append(
            {
                "q_id": int(r.q_id),
                "query": query,
                "sample_ans": str(r.sample_answer),
                "full_ans": str(getattr(r, "full_answer", "")),
                "top_chunk": top_chunk,
                "cohort": primary,
                "refusal_confidence": conf,
                "verbosity_score": float(getattr(r, "verbosity_score", 0.0)),
                "review_verdict": "",
                "error_type": "",
                "review_notes": "",
            }
        )

    out = _p(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in ("review_verdict", "error_type", "review_notes"):
        df[col] = df[col].astype(str)
    df.to_csv(out, index=False)
    print(f"exported n={len(df)} A={len(a_pick)} B={len(b_pick)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
