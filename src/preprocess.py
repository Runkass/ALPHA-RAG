"""Clean and normalize website text before chunking."""

from __future__ import annotations

import re
from html import unescape

from bs4 import BeautifulSoup

_MIN_DOC_LEN = 50
_MAX_DOC_CHARS = 25_000

_QUOTE_MAP = str.maketrans({
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u00ab": '"',
    "\u00bb": '"',
    "\u2013": "-",
    "\u2014": "-",
})


def strip_html(text: str) -> str:
    if not text or "<" not in text:
        return text
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def normalize_whitespace(text: str) -> str:
    text = unescape(text).translate(_QUOTE_MAP)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw: str) -> str:
    return normalize_whitespace(strip_html(raw or ""))


def prepare_document(title: str, text: str) -> str | None:
    """Return document body with title prefix, or None if too short."""
    body = clean_text(text)
    if len(body) > _MAX_DOC_CHARS:
        body = body[:_MAX_DOC_CHARS]
    if len(body) < _MIN_DOC_LEN:
        return None
    title_clean = clean_text(title)
    if title_clean:
        return f"[{title_clean}]\n{body}"
    return body
