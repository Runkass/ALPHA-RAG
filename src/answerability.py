"""Predict when to answer vs refuse (trained on sample_submission gold, dev only)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .retrieval import RetrievedChunk

_MODEL_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "answerability.pkl"


def features_from_retrieval(
    query: str,
    chunks: list[RetrievedChunk],
    context: str,
) -> np.ndarray:
    top_rrf = chunks[0].rrf_score if chunks else 0.0
    top_rerank = chunks[0].rerank_score if chunks else 0.0
    top_dense = chunks[0].dense_score if chunks else 0.0
    top_bm25 = chunks[0].bm25_score if chunks else 0.0
    return np.array(
        [
            len(query),
            len(context),
            len(chunks),
            top_rrf,
            top_rerank,
            top_dense,
            top_bm25,
            float(bool(context.strip())),
        ],
        dtype=np.float64,
    )


def _load_model():
    if not _MODEL_PATH.exists():
        return None
    with _MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def predict_refuse_prob(
    query: str,
    chunks: list[RetrievedChunk],
    context: str,
) -> float:
    """P(refusal) — higher means more likely gold was «Нет ответа»."""
    bundle = _load_model()
    if bundle is None:
        return 0.0
    model, scaler = bundle["model"], bundle["scaler"]
    x = features_from_retrieval(query, chunks, context).reshape(1, -1)
    x = scaler.transform(x)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x)[0, 1])
    return float(model.predict(x)[0])


def should_refuse(
    query: str,
    chunks: list[RetrievedChunk],
    context: str,
) -> bool:
    from .config import get_settings

    settings = get_settings()
    if not settings.answerability_enabled or not _MODEL_PATH.exists():
        return False
    if not context.strip():
        return True
    prob = predict_refuse_prob(query, chunks, context)
    return prob >= settings.answerability_threshold
