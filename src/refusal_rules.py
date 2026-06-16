"""Pre-LLM refusal heuristics (narrow patterns only)."""

from __future__ import annotations

import re

_PERSONAL_RE = re.compile(
    r"(?i)\b(мой|моего|моя|мои|моим|моих)\b.{0,40}\b(номер|договор|баланс|задолжен)",
)
_BRACKET_ONLY = re.compile(r"^\[[^\]]+\]\s*$")


def should_refuse_before_llm(query: str) -> bool:
    """True for bracket-only queries or personal-data requests without context."""
    q = (query or "").strip()
    if not q:
        return False
    if _BRACKET_ONLY.match(q):
        return True
    if _PERSONAL_RE.search(q):
        return True
    return False
