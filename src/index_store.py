"""FAISS vector index and chunks dataframe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .config import get_settings


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    web_id: int
    url: str
    title: str
    chunk_index: int
    text: str


class VectorIndex:
    def __init__(self, index: faiss.Index, chunks: pd.DataFrame) -> None:
        self._index = index
        self.chunks = chunks

    @property
    def size(self) -> int:
        return self._index.ntotal

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, indices = self._index.search(query_vec.astype(np.float32), top_k)
        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx < 0:
                continue
            results.append((int(idx), float(score)))
        return results

    def get_text(self, chunk_idx: int) -> str:
        return str(self.chunks.iloc[chunk_idx]["text"])

    def save(self, index_path: Path, chunks_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        self.chunks.to_parquet(chunks_path, index=False)

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> VectorIndex:
        index = faiss.read_index(str(index_path))
        chunks = pd.read_parquet(chunks_path)
        return cls(index, chunks)

    @classmethod
    def build(cls, embeddings: np.ndarray, chunks: pd.DataFrame) -> VectorIndex:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
        return cls(index, chunks)


def chunks_to_dataframe(chunks: list) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": c.chunk_id,
                "web_id": c.web_id,
                "url": c.url,
                "title": c.title,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            for c in chunks
        ]
    )


def artifacts_fresh(websites_path: Path, artifacts_path: Path) -> bool:
    settings = get_settings()
    dense_ok = settings.faiss_path.exists() or settings.tfidf_path.exists()
    if settings.dense_backend == "faiss" and not settings.faiss_path.exists():
        dense_ok = settings.tfidf_path.exists()
    required = [
        settings.chunks_path,
        settings.bm25_path,
    ]
    if not dense_ok:
        return False
    if not all(p.exists() for p in required):
        return False
    mtime_src = websites_path.stat().st_mtime
    return all(p.stat().st_mtime >= mtime_src for p in required)
