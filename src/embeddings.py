"""Embeddings via fastembed (no HuggingFace hub hang)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from .config import get_settings


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.fastembed_model)


def _needs_instruct_prefix(model_name: str | None = None) -> bool:
    name = (model_name or get_settings().fastembed_model).lower()
    return "e5" in name or "bge-m3" in name


def embed_passages(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    settings = get_settings()
    bs = batch_size or settings.embed_batch_size
    model = _model()
    if _needs_instruct_prefix(settings.fastembed_model):
        prefixed = [f"passage: {t}" for t in texts]
    else:
        prefixed = list(texts)
    parts: list[np.ndarray] = []
    for i in range(0, len(prefixed), bs):
        batch = prefixed[i : i + bs]
        vecs = list(model.embed(batch))
        parts.append(np.asarray(vecs, dtype=np.float32))
    if not parts:
        probe = list(model.embed(["passage: x"]))[0]
        return np.empty((0, len(probe)), dtype=np.float32)
    out = np.vstack(parts)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def embed_query(text: str) -> np.ndarray:
    settings = get_settings()
    query_text = f"query: {text}" if _needs_instruct_prefix(settings.fastembed_model) else text
    vec = np.asarray(list(_model().embed([query_text]))[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
