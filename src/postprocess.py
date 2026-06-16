"""Clean and normalize LLM answers for submission."""

from __future__ import annotations

import re

_PREFIXES = [
    r"^согласно фрагменту(?: \d+)?[,:]?\s*",
    r"^согласно фрагментам[,:]?\s*",
    r"^на основе (?:предоставленного )?контекста[,:]?\s*",
    r"^в (?:предоставленных )?фрагментах[^,\n]*[,:]?\s*",
    r"^из (?:контекста|фрагментов)[^,\n]*[,:]?\s*",
    r"^в соответствии с (?:фрагментом|фрагментами|текстом|контекстом)(?: \d+)?[,:]?\s*",
    r"^таким образом[,:]?\s*",
    r"^итак[,:]?\s*",
    r"^на основании (?:предоставленных )?(?:фрагментов|данных)[,:]?\s*",
]

_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_LIST = re.compile(r"^[\*\-]\s+", re.MULTILINE)
_BRACKET_ONLY = re.compile(r"^\[[^\]]+\]\s*$")
_HEDGE_START = re.compile(r"(?i)^(возможно|вероятно|скорее всего)\b")
_NO_INFO_PATTERNS = [
    re.compile(r"(?i)\b(в (?:предоставленных )?фрагментах|в контексте)\b.{0,80}\bнет\b.{0,40}\bинформац"),
    re.compile(r"(?i)\b(информация|данные)\b.{0,30}\b(отсутств|не содерж)"),
    re.compile(r"(?i)\bне удалось\b.{0,40}\bнайти\b.{0,40}\b(информац|данн)"),
    re.compile(r"(?i)\bк сожалению\b.{0,60}\b(нет|не указан|не содерж)"),
    re.compile(r"(?i)^(к сожалению|сожалею)\b"),
]


def _strip_markdown(text: str) -> str:
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_LIST.sub("", text)
    return text


def postprocess_answer(text: str, *, max_chars: int | None = None) -> str:
    from .config import get_settings

    limit = max_chars if max_chars is not None else get_settings().max_answer_chars
    t = text.strip()
    if not t or t.lower().startswith("нет ответа") or t.upper().startswith("ERROR:"):
        return "Нет ответа"
    for pat in _PREFIXES:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    t = _strip_markdown(t).strip()
    if _BRACKET_ONLY.match(t):
        return "Нет ответа"
    if _HEDGE_START.match(t):
        return "Нет ответа"
    for pat in _NO_INFO_PATTERNS:
        if pat.search(t):
            return "Нет ответа"
    if len(t) < 120 and t.endswith("?"):
        if t.count(".") == 0 and t.count("!") == 0:
            return "Нет ответа"
    t = re.sub(r"\n{3,}", "\n\n", t)
    if len(t) > limit:
        cut = t[: limit - 3]
        last = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind(" "))
        if last > limit // 3:
            cut = cut[: last + 1].rstrip()
        t = cut + "..."
    return t or "Нет ответа"
