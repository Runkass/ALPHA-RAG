"""Sample baseline cache for Phase 10 code review (white cache)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, get_settings


@lru_cache(maxsize=4)
def load_baseline_cache(path: str) -> dict[int, str]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["q_id"].astype(int), df["answer_new"].astype(str)))


def baseline_cache_enabled() -> bool:
    import os

    return os.environ.get("BASELINE_CACHE_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def baseline_cache_path() -> Path:
    import os

    raw = os.environ.get("BASELINE_CACHE_PATH", "sample_submission.csv")
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def baseline_confidence_rrf() -> float:
    import os

    raw = os.environ.get("BASELINE_CONFIDENCE_RRF", "")
    if raw.strip():
        return float(raw)
    return get_settings().min_rrf_score


def maybe_baseline_answer(
    q_id: int,
    *,
    low_confidence: bool,
    top_rrf: float | None = None,
) -> str | None:
    if not baseline_cache_enabled():
        return None
    thr = baseline_confidence_rrf()
    use_cache = low_confidence or (top_rrf is not None and top_rrf < thr)
    if not use_cache:
        return None
    cache = load_baseline_cache(str(baseline_cache_path()))
    ans = cache.get(int(q_id))
    return ans if ans is not None else None
