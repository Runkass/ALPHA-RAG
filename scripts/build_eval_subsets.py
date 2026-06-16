#!/usr/bin/env python
"""Build stratified eval subsets for Phase 3b."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_settings


def _is_gold_refusal(text: str) -> bool:
    return str(text).strip().lower().startswith("нет ответа")


def main() -> None:
    settings = get_settings()
    gold_path = ROOT / "sample_submission.csv"
    gold = pd.read_csv(gold_path)
    questions = pd.read_csv(settings.questions_path)

    gold = gold.merge(questions[["q_id"]], on="q_id", how="inner")
    gold["is_refusal"] = gold["answer_new"].map(_is_gold_refusal)

    refusal_ids = gold.loc[gold["is_refusal"], "q_id"].sort_values().astype(int)
    answer_ids = gold.loc[~gold["is_refusal"], "q_id"].astype(int)

    refusal_total = int(gold["is_refusal"].sum())
    refusal_only = refusal_ids.head(200)
    strat_answer = answer_ids.sample(n=300, random_state=42)
    strat_refusal = refusal_ids.sample(n=200, random_state=42)
    stratified = (
        pd.concat([strat_refusal, strat_answer])
        .sort_values()
        .astype(int)
        .reset_index(drop=True)
    )

    out_dir = ROOT / "data" / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    refusal_only.to_frame("q_id").to_csv(
        out_dir / "eval_refusal_only_200.csv", index=False
    )
    stratified.to_frame("q_id").to_csv(out_dir / "eval_stratified_500.csv", index=False)

    print(f"refusal_total={refusal_total}")
    print(f"refusal_only={len(refusal_only)}")
    print(f"stratified={len(stratified)}")


if __name__ == "__main__":
    main()
