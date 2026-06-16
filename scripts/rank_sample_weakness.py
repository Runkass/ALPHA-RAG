#!/usr/bin/env python
"""Rank sample answer q_id by weakness score for Phase 9 targeted edits."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fallbacks import extractive_fallback  # noqa: E402
from src.metrics.recall_l import is_refusal  # noqa: E402
from src.pipeline import RAGPipeline, load_dense_index  # noqa: E402
from src.bm25 import BM25Index  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.project_paths import BACKUP_FULL  # noqa: E402
from src.retrieval import _WORD_RE  # noqa: E402
from src.submission_rules import (  # noqa: E402
    is_faq_dump,
    is_garbage_answer,
    is_weak_sample_answer,
)

BATCHES_DIR = ROOT / "data" / "cache" / "edit_batches"
CHECKPOINT = ROOT / "data" / "cache" / "sample_weakness.checkpoint.parquet"
STRAT500 = ROOT / "data" / "cache" / "eval_stratified_500.csv"


def _overlap(answer: str, chunk_text: str) -> float:
    a_words = set(_WORD_RE.findall(str(answer).lower()))
    c_words = set(_WORD_RE.findall(str(chunk_text).lower()))
    if not a_words or not c_words:
        return 0.0
    return len(a_words & c_words) / len(a_words)


def _weakness_row(
    q_id: int,
    query: str,
    sample_ans: str,
    full_ans: str,
    overlap: float,
) -> dict:
    low_grounding = 1.0 - min(1.0, overlap)
    weak_heuristic = 1.0 if is_weak_sample_answer(sample_ans) else 0.0
    sample_ne_full = 0.0
    if (
        not is_refusal(sample_ans)
        and not is_refusal(full_ans)
        and full_ans.strip() != sample_ans.strip()
        and not is_garbage_answer(full_ans)
        and len(full_ans.strip()) > len(sample_ans.strip())
    ):
        sample_ne_full = 1.0
    faq_dump = 1.0 if is_faq_dump(sample_ans) else 0.0
    short_answer = 1.0 if len(str(sample_ans).strip()) < 40 else 0.0
    score = (
        0.35 * low_grounding
        + 0.25 * weak_heuristic
        + 0.20 * sample_ne_full
        + 0.15 * faq_dump
        + 0.05 * short_answer
    )
    return {
        "q_id": q_id,
        "query": query,
        "sample_answer": sample_ans,
        "full_answer": full_ans,
        "weakness_score": round(score, 4),
        "keyword_overlap": round(overlap, 4),
        "low_grounding": low_grounding > 0.5,
        "weak_heuristic": bool(weak_heuristic),
        "sample_ne_full": bool(sample_ne_full),
        "faq_dump": bool(faq_dump),
        "short_answer": bool(short_answer),
    }


def _write_batches(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = df.sort_values("weakness_score", ascending=False)
    ranked.head(100).to_csv(out_dir / "top100_grounding.csv", index=False)
    top50_full = ranked[ranked["sample_ne_full"]].head(50)
    top50_full.to_csv(out_dir / "top50_full.csv", index=False)
    answer_q = sorted(df["q_id"].astype(int).unique())
    n = len(answer_q)
    t1, t2 = n // 3, 2 * n // 3
    tert1 = set(answer_q[:t1])
    tert2 = set(answer_q[t1:t2])
    tert3 = set(answer_q[t2:])
    for name, ids in (
        ("start100.csv", tert1),
        ("mid100.csv", tert2),
        ("end100.csv", tert3),
    ):
        sub = ranked[ranked["q_id"].isin(ids)].head(100)
        sub.to_csv(out_dir / name, index=False)
    if STRAT500.exists():
        strat = set(pd.read_csv(STRAT500)["q_id"].astype(int))
        top200 = set(ranked.head(200)["q_id"].astype(int))
        inter = ranked[ranked["q_id"].isin(strat & top200)]
        inter.to_csv(out_dir / "strat50_in_top200.csv", index=False)
        strat_weak = ranked[ranked["q_id"].isin(strat)].head(50)
        strat_weak.to_csv(out_dir / "strat50_high_weak.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--questions", default="questions.csv")
    parser.add_argument("--full", default=str(BACKUP_FULL.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--output", default="data/cache/sample_weakness.parquet")
    parser.add_argument("--batches-dir", default=str(BATCHES_DIR.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()

    os.environ.setdefault("RERANKER_ENABLED", "false")

    sample = pd.read_csv(ROOT / args.sample)
    questions = pd.read_csv(ROOT / args.questions)
    full_path = ROOT / args.full
    full_df = pd.read_csv(full_path) if full_path.exists() else None
    full_map: dict[int, str] = {}
    if full_df is not None:
        full_map = dict(zip(full_df["q_id"].astype(int), full_df["answer_new"].astype(str)))

    qmap = dict(zip(questions["q_id"].astype(int), questions["query"].astype(str)))
    targets = sample[~sample["answer_new"].astype(str).map(is_refusal)].copy()
    if args.limit:
        targets = targets.head(args.limit)

    done: set[int] = set()
    rows: list[dict] = []
    if CHECKPOINT.exists():
        prev = pd.read_parquet(CHECKPOINT)
        rows = prev.to_dict("records")
        done = set(int(x) for x in prev["q_id"].tolist())
        print(f"resume: {len(done)} q_id from checkpoint")

    settings = get_settings()
    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    pipeline = RAGPipeline(dense, bm25, cache=None)

    pending = targets[~targets["q_id"].astype(int).isin(done)]
    buf: list[dict] = []

    for row in tqdm(pending.itertuples(index=False), total=len(pending)):
        q_id = int(row.q_id)
        sample_ans = str(row.answer_new)
        query = qmap.get(q_id, "")
        full_ans = full_map.get(q_id, sample_ans)
        chunks, _ctx, _low = pipeline.retrieve_context(query)
        chunk_text = chunks[0].text if chunks else ""
        overlap = _overlap(sample_ans, chunk_text)
        extractive = extractive_fallback(chunks, max_len=260) if chunks else sample_ans
        rec = _weakness_row(q_id, query, sample_ans, full_ans, overlap)
        rec["top_chunk_text"] = chunk_text[:2000]
        rec["extractive_answer"] = extractive
        rows.append(rec)
        buf.append(rec)
        if len(buf) >= args.checkpoint_every:
            pd.DataFrame(rows).to_parquet(CHECKPOINT, index=False)
            buf.clear()

    if not rows:
        print("no rows")
        return

    df = pd.DataFrame(rows)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink(missing_ok=True)

    batches = ROOT / args.batches_dir
    _write_batches(df, batches)

    top = df.sort_values("weakness_score", ascending=False).head(10)
    print(f"saved {len(df)} answer q_id -> {out}")
    print(f"batches -> {batches}")
    print(f"score>0: {(df['weakness_score'] > 0).sum()}")
    print("top-10 weakness:")
    for r in top.itertuples(index=False):
        flags = []
        if r.low_grounding:
            flags.append("ground")
        if r.weak_heuristic:
            flags.append("weak")
        if r.sample_ne_full:
            flags.append("full")
        if r.faq_dump:
            flags.append("faq")
        print(
            f"  q_id={r.q_id} score={r.weakness_score} overlap={r.keyword_overlap} "
            f"flags={','.join(flags) or '-'} ans={str(r.sample_answer)[:60]}..."
        )


if __name__ == "__main__":
    main()
