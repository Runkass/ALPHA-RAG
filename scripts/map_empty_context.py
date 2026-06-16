#!/usr/bin/env python
"""Retrieval-only scan 6977 q_id -> empty_context_qids.csv + context_len.parquet."""

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
from src.pipeline import RAGPipeline, load_dense_index  # noqa: E402

OUT_QIDS = ROOT / "data" / "cache" / "empty_context_qids.csv"
OUT_PARQUET = ROOT / "data" / "cache" / "context_len.parquet"
CHECKPOINT = ROOT / "data" / "cache" / "context_len.checkpoint.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="questions.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-qids", default=str(OUT_QIDS))
    parser.add_argument("--out-parquet", default=str(OUT_PARQUET))
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--no-rerank", action="store_true", default=True)
    parser.add_argument("--rerank", dest="no_rerank", action="store_false")
    args = parser.parse_args()

    if args.no_rerank:
        os.environ["RERANKER_ENABLED"] = "false"

    settings = get_settings()
    questions = pd.read_csv(ROOT / args.questions)
    if args.limit:
        questions = questions.head(args.limit)

    done: set[int] = set()
    rows: list[dict] = []
    if CHECKPOINT.exists():
        prev = pd.read_parquet(CHECKPOINT)
        rows = prev.to_dict("records")
        done = set(int(x) for x in prev["q_id"].tolist())
        print(f"resume: {len(done)} q_id from checkpoint")

    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    pipeline = RAGPipeline(dense, bm25, cache=None)

    pending = questions[~questions["q_id"].astype(int).isin(done)]
    buf: list[dict] = []

    for row in tqdm(pending.itertuples(index=False), total=len(pending)):
        q_id = int(row.q_id)
        query = str(row.query)
        chunks, context, _low = pipeline.retrieve_context(query)
        ctx_len = len(context.strip())
        top_rrf = chunks[0].rrf_score if chunks else 0.0
        rec = {
            "q_id": q_id,
            "context_len": ctx_len,
            "n_chunks": len(chunks),
            "top_rrf": top_rrf,
            "empty_context": ctx_len == 0,
        }
        rows.append(rec)
        buf.append(rec)
        if len(buf) >= args.checkpoint_every:
            pd.DataFrame(rows).to_parquet(CHECKPOINT, index=False)
            buf.clear()

    if rows:
        pd.DataFrame(rows).to_parquet(CHECKPOINT, index=False)

    df = pd.DataFrame(rows)
    empty_ids = sorted(int(x) for x in df.loc[df["empty_context"], "q_id"].tolist())

    out_q = Path(args.out_qids)
    if not out_q.is_absolute():
        out_q = ROOT / out_q
    out_q.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"q_id": empty_ids}).to_csv(out_q, index=False)

    out_p = Path(args.out_parquet)
    if not out_p.is_absolute():
        out_p = ROOT / out_p
    df.to_parquet(out_p, index=False)

    print(f"questions={len(df)} empty_context={len(empty_ids)} -> {out_q}")
    print(f"context_len parquet -> {out_p}")


if __name__ == "__main__":
    main()
