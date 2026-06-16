"""Retrieval-level refusal policy (Phase 3b anti-false-refuse)."""

from __future__ import annotations

import re

from .config import get_settings
from .retrieval import RetrievedChunk

_DIGITS = re.compile(r"\d{4,}")
_HEDGE_START = re.compile(r"(?i)^(возможно|вероятно|скорее всего|похоже|кажется)\b")
_NO_INFO = re.compile(
    r"(?i)(нет информации|не указан|не содержит|отсутствует|к сожалению|не могу|не удалось найти)"
)


def should_refuse_from_retrieval(chunks: list[RetrievedChunk]) -> bool:
    """Hard refuse only on very low rerank; borderline cases go to LLM."""
    if not get_settings().refuse_enabled:
        return False
    if not chunks:
        return True
    top = chunks[0]
    if top.rerank_score < get_settings().refuse_rerank_hard:
        return True
    return False


def maybe_refuse_borderline_answer(chunks: list[RetrievedChunk], answer: str) -> str:
    """Post-LLM refusal only in narrow rerank band for weak answers."""
    if not get_settings().refuse_enabled:
        return (answer or "").strip() or "Нет ответа"
    if not chunks:
        return "Нет ответа"
    t = (answer or "").strip()
    if not t or t.lower().startswith("нет ответа"):
        return "Нет ответа"
    settings = get_settings()
    top = chunks[0]
    if top.rerank_score < settings.refuse_rerank_hard:
        return "Нет ответа"
    # Only apply extra post-LLM refusal in a narrow uncertain band.
    if top.rerank_score >= -4.8:
        return t
    if top.rerank_score >= settings.refuse_rerank_soft:
        return t
    if _DIGITS.search(t):
        return t
    if len(t) > 240:
        return t
    if _HEDGE_START.match(t):
        return "Нет ответа"
    if _NO_INFO.search(t):
        return "Нет ответа"
    if len(t) < 120 and t.endswith("?"):
        return "Нет ответа"
    return t
