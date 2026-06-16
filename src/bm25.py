"""BM25 sparse retrieval over tokenized chunks."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, corpus_tokens: list[list[str]]) -> None:
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as f:
            return pickle.load(f)

    @classmethod
    def build(cls, texts: list[str]) -> BM25Index:
        tokens = [tokenize(t) for t in texts]
        return cls(tokens)
