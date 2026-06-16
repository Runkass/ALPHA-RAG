"""Cross-encoder rerank via fastembed."""

from __future__ import annotations

import logging
from functools import lru_cache

from .config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=get_settings().fastembed_rerank_model)


def rerank(query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
    scores = list(_cross_encoder().rerank(query, texts))
    ranked = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
    return [(idx, float(s)) for idx, s in ranked[:top_k]]


def rerank_optional(query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
    if not get_settings().reranker_enabled:
        return [(i, float(len(texts) - i)) for i in range(min(top_k, len(texts)))]
    try:
        return rerank(query, texts, top_k)
    except Exception as exc:
        logger.warning("rerank failed: %s", exc)
        return [(i, float(len(texts) - i)) for i in range(min(top_k, len(texts)))]
