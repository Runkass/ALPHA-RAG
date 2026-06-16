#!/usr/bin/env python
"""Build BM25 + dense index (fastembed+FAISS or TF-IDF)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
from tqdm import tqdm

from src.bm25 import BM25Index
from src.chunking import Chunk, split_document
from src.config import get_settings
from src.index_store import artifacts_fresh, chunks_to_dataframe
from src.preprocess import prepare_document


def load_and_chunk(websites_path: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    next_id = 0
    with websites_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in tqdm(rows, desc="Chunking"):
        web_id = int(row["web_id"])
        url = row.get("url", "")
        title = row.get("title", "")
        prepared = prepare_document(title, row.get("text", ""))
        if prepared is None:
            continue
        for c in split_document(web_id, url, title, prepared):
            all_chunks.append(
                Chunk(
                    chunk_id=next_id,
                    web_id=c.web_id,
                    url=c.url,
                    title=c.title,
                    chunk_index=c.chunk_index,
                    text=c.text,
                )
            )
            next_id += 1
    return all_chunks


def main(force: bool = False, use_fastembed: bool = True) -> None:
    settings = get_settings()
    websites_path = settings.websites_path
    artifacts_path = settings.artifacts_path
    artifacts_path.mkdir(parents=True, exist_ok=True)

    if not force and artifacts_fresh(websites_path, artifacts_path):
        print("Artifacts are up to date, skipping build.")
        return

    if settings.chunks_path.exists() and not force:
        print(f"Loading chunks from {settings.chunks_path} ...")
        df = pd.read_parquet(settings.chunks_path)
    else:
        print(f"Loading {websites_path} ...")
        chunks = load_and_chunk(websites_path)
        if not chunks:
            raise RuntimeError("No chunks produced from websites.csv")
        print(f"Total chunks: {len(chunks)}")
        df = chunks_to_dataframe(chunks)
        df.to_parquet(settings.chunks_path, index=False)

    texts = df["text"].tolist()

    if not settings.bm25_path.exists() or force:
        print("Building BM25 index ...")
        BM25Index.build(texts).save(settings.bm25_path)
    else:
        print(f"BM25 exists: {settings.bm25_path}")

    if use_fastembed:
        if settings.faiss_path.exists() and not force:
            print(f"FAISS exists: {settings.faiss_path}")
        else:
            print("Building FAISS index (fastembed) ...")
            from src.embeddings import embed_passages
            from src.index_store import VectorIndex

            VectorIndex.build(embed_passages(texts), df).save(
                settings.faiss_path, settings.chunks_path
            )
    elif settings.tfidf_path.exists() and not force:
        print(f"TF-IDF exists: {settings.tfidf_path}")
    else:
        print("Building TF-IDF index ...")
        from src.tfidf_index import TfidfIndex

        TfidfIndex.build(texts, df).save(settings.tfidf_path, settings.chunks_path)

    print(f"Done. Chunks: {settings.chunks_path}")
    print(f"BM25: {settings.bm25_path}")
    if use_fastembed:
        print(f"FAISS: {settings.faiss_path}")
    else:
        print(f"TF-IDF: {settings.tfidf_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild even if fresh")
    parser.add_argument(
        "--fastembed",
        action="store_true",
        default=True,
        help="Use fastembed+FAISS (default)",
    )
    parser.add_argument(
        "--tfidf",
        action="store_true",
        help="Use TF-IDF instead of fastembed",
    )
    args = parser.parse_args()
    main(force=args.force, use_fastembed=not args.tfidf)
