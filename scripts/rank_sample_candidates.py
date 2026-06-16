#!/usr/bin/env python
"""Rank Phase 10 candidates: false refuse (high-conf retrieval) + verbosity."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25Index  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.metrics.recall_l import is_refusal  # noqa: E402
from src.pipeline import RAGPipeline, load_dense_index  # noqa: E402
from src.retrieval import _WORD_RE, format_context  # noqa: E402
from src.submission_rules import (  # noqa: E402
    is_protected_q_id,
    is_protected_short_answer,
    is_verbose_sample_answer,
    load_protected_q_ids,
    verbosity_score,
)

BATCHES_DIR = ROOT / "data" / "cache" / "edit_batches"
CHECKPOINT = ROOT / "data" / "cache" / "sample_candidates_refusal.checkpoint.parquet"
DENY_DEFAULT = BATCHES_DIR / "s2_do_not_touch.csv"


def _overlap(query: str, chunk_text: str) -> float:
    q_words = set(_WORD_RE.findall(str(query).lower()))
    c_words = set(_WORD_RE.findall(str(chunk_text).lower()))
    if not q_words or not c_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


def _refusal_confidence_score(rrf: float, rerank: float, overlap: float) -> float:
    return round(0.5 * min(1.0, rrf / 0.05) + 0.3 * min(1.0, (rerank + 5) / 10) + 0.2 * overlap, 4)


def _write_batches(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = df[df["candidate_type"] == "refusal_high_conf"].sort_values(
        "refusal_confidence", ascending=False
    )
    ref.head(100).to_csv(out_dir / "refusal_high_conf_top100.csv", index=False)
    ref.head(50).to_csv(out_dir / "refusal_high_conf_top50.csv", index=False)
    verb = df[df["candidate_type"] == "verbosity"].sort_values("verbosity_score", ascending=False)
    verb.head(100).to_csv(out_dir / "verbosity_top100.csv", index=False)
    verb.head(50).to_csv(out_dir / "verbosity_top50.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--questions", default="questions.csv")
    parser.add_argument("--output", default="data/cache/sample_candidates.parquet")
    parser.add_argument("--deny-file", default=str(DENY_DEFAULT.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    os.environ.setdefault("RERANKER_ENABLED", "false")
    deny = load_protected_q_ids(args.deny_file)
    sample = pd.read_csv(ROOT / args.sample)
    questions = pd.read_csv(ROOT / args.questions)
    qmap = dict(zip(questions["q_id"].astype(int), questions["query"].astype(str)))
    sample_map = dict(zip(sample["q_id"].astype(int), sample["answer_new"].astype(str)))

    rows: list[dict] = []

    for q_id, ans in sample_map.items():
        if is_protected_q_id(q_id, deny):
            continue
        if is_verbose_sample_answer(str(ans)):
            rows.append(
                {
                    "q_id": q_id,
                    "query": qmap.get(q_id, ""),
                    "sample_answer": ans,
                    "candidate_type": "verbosity",
                    "verbosity_score": verbosity_score(str(ans)),
                    "answer_len": len(str(ans)),
                }
            )

    verb_n = len(rows)
    print(f"verbosity candidates (no retrieval): {verb_n}")

    refusal_q = [
        int(r.q_id)
        for r in sample.itertuples(index=False)
        if is_refusal(str(r.answer_new)) and not is_protected_q_id(int(r.q_id), deny)
    ]
    print(f"refusal q_id to scan: {len(refusal_q)}")

    done: set[int] = set()
    if CHECKPOINT.exists():
        prev = pd.read_parquet(CHECKPOINT)
        for rec in prev.to_dict("records"):
            rows.append(rec)
        done = set(int(x) for x in prev["q_id"].tolist())
        print(f"resume refusal rows: {len(done)}")

    settings = get_settings()
    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    pipeline = RAGPipeline(dense, bm25, cache=None)

    pending = [q for q in refusal_q if q not in done]
    buf: list[dict] = []

    for q_id in tqdm(pending, desc="refusal retrieval"):
        query = qmap.get(q_id, "")
        chunks, _ctx, low_confidence = pipeline.retrieve_context(query)
        context_len = len(format_context(chunks).strip())
        top_rrf = chunks[0].rrf_score if chunks else 0.0
        top_rerank = chunks[0].rerank_score if chunks else -99.0
        chunk_text = chunks[0].text if chunks else ""
        overlap = _overlap(query, chunk_text)
        high_conf = (
            not low_confidence
            and context_len > 0
            and top_rrf >= settings.min_rrf_score
        )
        if not high_conf:
            continue
        score = _refusal_confidence_score(top_rrf, top_rerank, overlap)
        rec = {
            "q_id": q_id,
            "query": query,
            "sample_answer": sample_map.get(q_id, ""),
            "candidate_type": "refusal_high_conf",
            "refusal_confidence": score,
            "top_rrf": round(top_rrf, 6),
            "top_rerank": round(top_rerank, 4),
            "keyword_overlap": round(overlap, 4),
            "context_len": context_len,
        }
        rows.append(rec)
        buf.append(rec)
        if len(buf) >= args.checkpoint_every:
            pd.DataFrame([r for r in rows if r.get("candidate_type") == "refusal_high_conf"]).to_parquet(
                CHECKPOINT, index=False
            )
            buf.clear()

    df = pd.DataFrame(rows)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink(missing_ok=True)
    _write_batches(df, BATCHES_DIR)

    ref_n = (df["candidate_type"] == "refusal_high_conf").sum() if len(df) else 0
    print(f"saved -> {out} refusal_high_conf={ref_n} verbosity={verb_n}")
    if ref_n:
        top = df[df["candidate_type"] == "refusal_high_conf"].sort_values(
            "refusal_confidence", ascending=False
        ).head(10)
        print("top-10 refusal_high_conf:")
        for r in top.itertuples(index=False):
            print(f"  q_id={r.q_id} conf={r.refusal_confidence} rrf={r.top_rrf} overlap={r.keyword_overlap}")
    if verb_n:
        topv = df[df["candidate_type"] == "verbosity"].sort_values(
            "verbosity_score", ascending=False
        ).head(5)
        print("top-5 verbosity:")
        for r in topv.itertuples(index=False):
            print(f"  q_id={r.q_id} score={r.verbosity_score} len={r.answer_len}")


if __name__ == "__main__":
    main()
