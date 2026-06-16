"""Shared rules for Phase 8 pandas submissions (audit, garbage, FAQ)."""

from __future__ import annotations

import re

from pathlib import Path

from src.metrics.recall_l import is_refusal

REFUSAL_TEXT = "Нет ответа"
_GARBAGE_CHARS = {"−", "-", "—", "–"}
_FAQ_RE = re.compile(r"\] / ")
_WEAK_RE = re.compile(r"(?i)(обратитесь|уточните|техподдерж)")


def is_faq_dump(text: str) -> bool:
    t = str(text)
    return "] / " in t or t.count("?") >= 3


def is_garbage_answer(text: str) -> bool:
    """FULL audit fix: dash, len<=2, FAQ dump."""
    t = str(text).strip()
    if not t or is_refusal(t):
        return False
    if t in _GARBAGE_CHARS:
        return True
    if len(t) <= 2:
        return True
    if is_faq_dump(t):
        return True
    return False


def is_weak_sample_answer(text: str, *, weak_len: int = 80) -> bool:
    t = str(text).strip()
    if is_refusal(t):
        return False
    if len(t) < weak_len:
        return True
    if is_faq_dump(t):
        return True
    if _WEAK_RE.search(t):
        return True
    return False


_REFUSAL_SHORT = re.compile(r"^(да|нет)\.?$", re.I)
_BIK_SHORT = re.compile(r"^\d{9,20}$")
_PROTECTED_DENY: set[int] | None = None


def is_protected_short_answer(text: str, *, max_len: int = 40) -> bool:
    """Phase 10: do not patch short factual sample answers (S2 lesson)."""
    t = str(text).strip()
    if is_refusal(t):
        return False
    if len(t) <= max_len:
        return True
    if _REFUSAL_SHORT.match(t):
        return True
    if len(t) < 20 and (_BIK_SHORT.match(t.replace(" ", "")) or t.isdigit()):
        return True
    return False


def load_protected_q_ids(path: Path | str | None) -> set[int]:
    global _PROTECTED_DENY
    if path is None:
        return set()
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    if not p.exists():
        return set()
    import pandas as pd

    df = pd.read_csv(p)
    col = "q_id" if "q_id" in df.columns else df.columns[0]
    ids = set(int(x) for x in df[col].tolist())
    _PROTECTED_DENY = ids
    return ids


def is_protected_q_id(q_id: int, denylist: set[int] | None = None) -> bool:
    deny = denylist if denylist is not None else (_PROTECTED_DENY or set())
    return int(q_id) in deny


def is_verbose_sample_answer(text: str, *, min_chars: int = 300, min_words: int = 50) -> bool:
    t = str(text).strip()
    if is_refusal(t) or is_protected_short_answer(t):
        return False
    words = len(t.split())
    if len(t) > min_chars or words > min_words:
        return True
    if _WEAK_RE.search(t):
        return True
    if "возможно" in t.lower() or "фрагмент" in t.lower():
        return True
    if is_faq_dump(t):
        return True
    return False


def verbosity_score(text: str) -> float:
    t = str(text).strip()
    if not is_verbose_sample_answer(t):
        return 0.0
    score = min(1.0, len(t) / 1200.0)
    if _WEAK_RE.search(t):
        score += 0.15
    if "возможно" in t.lower():
        score += 0.1
    if is_faq_dump(t):
        score += 0.1
    return round(min(1.0, score), 4)


def fix_to_refusal(text: str) -> str:
    if is_garbage_answer(text):
        return REFUSAL_TEXT
    return str(text)
