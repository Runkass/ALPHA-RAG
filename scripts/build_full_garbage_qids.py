#!/usr/bin/env python
"""Build q_id lists: FULL garbage rows + recon union (strat500 + garbage)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.project_paths import BACKUP_FULL  # noqa: E402
from src.submission_rules import is_garbage_answer  # noqa: E402

GARBAGE_OUT = ROOT / "data" / "cache" / "full_garbage_qids.csv"
RECON_OUT = ROOT / "data" / "cache" / "recon_qids.csv"
STRAT500 = ROOT / "data" / "cache" / "eval_stratified_500.csv"


def main() -> None:
    full = pd.read_csv(BACKUP_FULL)
    ans = full["answer_new"].astype(str)
    mask = ans.map(is_garbage_answer)
    garbage = full.loc[mask, ["q_id"]].astype({"q_id": int}).drop_duplicates()
    garbage = garbage.sort_values("q_id")
    GARBAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    garbage.to_csv(GARBAGE_OUT, index=False)
    print(f"garbage_q_ids={len(garbage)} -> {GARBAGE_OUT}")

    strat: set[int] = set()
    if STRAT500.exists():
        strat = set(pd.read_csv(STRAT500)["q_id"].astype(int).tolist())
        print(f"strat500_q_ids={len(strat)}")
    else:
        print(f"WARN: {STRAT500} missing, recon = garbage only")

    recon_ids = sorted(strat | set(garbage["q_id"].tolist()))
    pd.DataFrame({"q_id": recon_ids}).to_csv(RECON_OUT, index=False)
    print(f"recon_q_ids={len(recon_ids)} -> {RECON_OUT}")


if __name__ == "__main__":
    main()
