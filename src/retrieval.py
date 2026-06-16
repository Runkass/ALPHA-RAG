"""Hybrid dense + BM25 retrieval with optional cross-encoder rerank."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .bm25 import BM25Index
from .config import get_settings
from .index_store import VectorIndex
from .reranker import rerank_optional
from .tfidf_index import TfidfIndex

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_idx: int
    text: str
    dense_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float


class DenseRetriever:
    def search(self, query: str, top_k: int) -> list[tuple[int, float]]: ...

    def get_text(self, chunk_idx: int) -> str: ...


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _raw) in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


_WORD_RE = re.compile(r"[а-яА-ЯёЁa-zA-Z0-9]{3,}")


def filter_by_keyword_overlap(
    chunks: list[RetrievedChunk],
    query: str,
    min_overlap: int | None = None,
) -> list[RetrievedChunk]:
    """Keep chunks sharing at least min_overlap significant tokens with the query."""
    settings = get_settings()
    need = min_overlap if min_overlap is not None else settings.keyword_min_overlap
    query_words = set(_WORD_RE.findall(query.lower()))
    if not query_words or not chunks:
        return chunks

    filtered: list[RetrievedChunk] = []
    for chunk in chunks:
        chunk_words = set(_WORD_RE.findall(chunk.text.lower()))
        if len(query_words & chunk_words) >= need:
            filtered.append(chunk)
    return filtered if filtered else chunks


def hybrid_retrieve(
    query: str,
    dense: DenseRetriever,
    bm25_index: BM25Index,
    top_k: int | None = None,
) -> tuple[list[RetrievedChunk], bool]:
    settings = get_settings()
    k = top_k or settings.top_k_retrieve

    dense_hits = dense.search(query, k)
    sparse_hits = bm25_index.search(query, k)

    fused = reciprocal_rank_fusion([dense_hits, sparse_hits], settings.rrf_k)[:k]
    low_confidence = not fused or fused[0][1] < settings.min_rrf_score

    dense_map = {idx: score for idx, score in dense_hits}
    sparse_map = {idx: score for idx, score in sparse_hits}

    candidate_indices = [idx for idx, _ in fused]
    candidate_texts = [dense.get_text(idx) for idx in candidate_indices]
    reranked = rerank_optional(query, candidate_texts, settings.top_k_final)

    results: list[RetrievedChunk] = []
    for idx, rerank_score in reranked:
        chunk_idx = candidate_indices[idx]
        results.append(
            RetrievedChunk(
                chunk_idx=chunk_idx,
                text=candidate_texts[idx],
                dense_score=dense_map.get(chunk_idx, 0.0),
                bm25_score=sparse_map.get(chunk_idx, 0.0),
                rrf_score=dict(fused).get(chunk_idx, 0.0),
                rerank_score=rerank_score,
            )
        )
    results = filter_by_keyword_overlap(results, query)
    return results, low_confidence


def format_context(chunks: list[RetrievedChunk], max_chars: int | None = None) -> str:
    settings = get_settings()
    limit = max_chars or settings.max_context_chars
    parts: list[str] = []
    total = 0
    for i, chunk in enumerate(chunks, start=1):
        block = f"# Фрагмент {i}\n{chunk.text.strip()}"
        if total + len(block) > limit and parts:
            break
        if len(block) > limit:
            block = block[:limit]
        parts.append(block)
        total += len(block) + 4
    return "\n---\n".join(parts)
