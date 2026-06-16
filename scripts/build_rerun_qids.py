#!/usr/bin/env python
"""Build q_id list for selective regen of weak FULL answers (Phase 7)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.project_paths import BACKUP_FULL  # noqa: E402

BASE = BACKUP_FULL
GOLD = ROOT / "sample_submission.csv"
OUT = ROOT / "data" / "cache" / "rerun_weak_full.csv"

_WEAK_RE = re.compile(r"(?i)(обратитесь|уточните|техподдерж)")


def _is_gold_refusal(text: str) -> bool:
    return str(text).strip().lower().startswith("нет ответа")


def main() -> None:
    base = pd.read_csv(BASE)
    gold = pd.read_csv(GOLD)
    m = gold.merge(base, on="q_id", suffixes=("_gold", "_pred"))
    col_g = "answer_new_gold"
    col_p = "answer_new_pred"

    weak: list[int] = []
    for row in m.itertuples(index=False):
        q_id = int(row.q_id)
        g = str(getattr(row, col_g))
        p = str(getattr(row, col_p))
        if _is_gold_refusal(g):
            continue
        if len(p.strip()) < 120 or _WEAK_RE.search(p):
            weak.append(q_id)

    weak = sorted(set(weak))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"q_id": weak}).to_csv(OUT, index=False)
    print(f"weak_q_ids={len(weak)} -> {OUT}")


if __name__ == "__main__":
    main()
