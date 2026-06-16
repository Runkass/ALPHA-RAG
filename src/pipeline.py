"""End-to-end RAG for a single question."""

from __future__ import annotations

import logging
from typing import Any

from .bm25 import BM25Index
from .cache import AnswerCache
from .config import get_settings
from .answerability import should_refuse
from .fallbacks import extractive_fallback, soft_fallback, try_rule_answer
from .refusal_policy import maybe_refuse_borderline_answer, should_refuse_from_retrieval
from .refusal_rules import should_refuse_before_llm
from .index_store import VectorIndex
from .llm import LLMPaymentRequired, generate_answer, llm_configured, normalize_answer
from .retrieval import RetrievedChunk, format_context, hybrid_retrieve
from .tfidf_index import TfidfIndex

logger = logging.getLogger(__name__)


def compose_answer(
    query: str,
    chunks: list[RetrievedChunk],
    context: str,
    *,
    raw_llm: str | None = None,
    llm_client: Any | None = None,
    q_id: int | None = None,
) -> str:
    """Build final answer: rules -> LLM -> soft/extractive fallback."""
    settings = get_settings()
    top_rrf = chunks[0].rrf_score if chunks else 0.0
    if chunks and top_rrf >= settings.min_rrf_score:
        snippet = chunks[0].text[:400]
        rule = try_rule_answer(query, snippet)
        if rule is not None:
            return rule

    if not context.strip():
        logger.info(
            "REFUSE q_id=%s reason=empty_context ctx_len=0",
            q_id,
        )
        return "Нет ответа"

    if should_refuse_before_llm(query):
        logger.info("REFUSE q_id=%s reason=pre_llm_rules", q_id)
        return "Нет ответа"

    settings = get_settings()
    if settings.refuse_enabled and should_refuse(query, chunks, context):
        logger.info("REFUSE q_id=%s reason=answerability", q_id)
        return "Нет ответа"

    if raw_llm is None:
        if not llm_configured():
            return extractive_fallback(chunks)
        try:
            raw_llm = generate_answer(query, context, client=llm_client)
        except LLMPaymentRequired:
            raise
        except Exception as exc:
            logger.warning("LLM failed q_id=%s: %s", q_id, exc)
            return extractive_fallback(chunks)

    result = normalize_answer(raw_llm)
    result = maybe_refuse_borderline_answer(chunks, result)
    return soft_fallback(result, chunks)


class RAGPipeline:
    def __init__(
        self,
        dense_index: Any,
        bm25_index: BM25Index,
        cache: AnswerCache | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._dense = dense_index
        self._bm25 = bm25_index
        self._cache = cache
        self._llm = llm_client

    def retrieve_context(
        self, query: str
    ) -> tuple[list[RetrievedChunk], str, bool]:
        chunks, low_confidence = hybrid_retrieve(query, self._dense, self._bm25)
        context = format_context(chunks)
        return chunks, context, low_confidence

    def answer(
        self,
        q_id: int,
        query: str,
        *,
        use_cache: bool = True,
        force: bool = False,
    ) -> str:
        if use_cache and not force and self._cache is not None:
            cached = self._cache.get_by_q_id(q_id)
            if cached is not None:
                return cached

        chunks, context, low_confidence = self.retrieve_context(query)
        settings = get_settings()
        top_rrf = chunks[0].rrf_score if chunks else 0.0

        from .baseline_cache import maybe_baseline_answer

        cached_ans = maybe_baseline_answer(
            q_id, low_confidence=low_confidence, top_rrf=top_rrf
        )
        if cached_ans is not None:
            logger.info("BASELINE_CACHE q_id=%s low_confidence=1", q_id)
            result = cached_ans
        elif settings.refuse_enabled and (
            low_confidence or should_refuse_from_retrieval(chunks)
        ):
            reason = "low_rrf" if low_confidence else "rerank_hard"
            logger.info("REFUSE q_id=%s reason=%s", q_id, reason)
            result = "Нет ответа"
        else:
            result = compose_answer(
                query, chunks, context, llm_client=self._llm, q_id=q_id
            )

        if self._cache is not None:
            key = AnswerCache.make_key(q_id, query, context)
            self._cache.set(key, q_id, result)

        return result


class FaissDenseIndex:
    """VectorIndex + query embedding via fastembed."""

    def __init__(self, inner: VectorIndex) -> None:
        self._inner = inner

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        from .embeddings import embed_query

        return self._inner.search(embed_query(query), top_k)

    def get_text(self, chunk_idx: int) -> str:
        return self._inner.get_text(chunk_idx)


def load_dense_index():
    settings = get_settings()
    if settings.faiss_path.exists():
        return FaissDenseIndex(VectorIndex.load(settings.faiss_path, settings.chunks_path))
    if settings.tfidf_path.exists():
        return TfidfIndex.load(settings.tfidf_path, settings.chunks_path)
    raise FileNotFoundError("Run: python scripts/build_index.py --fastembed")
