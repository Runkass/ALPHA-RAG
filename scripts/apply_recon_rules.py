#!/usr/bin/env python
"""R5: apply pandas recon rules to base (no 72B text copy)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal  # noqa: E402
from src.submission_rules import REFUSAL_TEXT, is_faq_dump  # noqa: E402


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def apply_rules(
    base: pd.DataFrame,
    recon: pd.DataFrame,
    rules: dict,
) -> tuple[pd.DataFrame, dict]:
    recon_map = dict(zip(recon["q_id"].astype(int), recon["answer_new"].astype(str)))
    out = base.copy()
    r1 = r2 = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q not in recon_map:
            continue
        base_ans = str(row["answer_new"])
        recon_ans = recon_map[q]
        if is_refusal(recon_ans) and not is_refusal(base_ans):
            out.at[i, "answer_new"] = REFUSAL_TEXT
            r1 += 1
            continue
        if not is_faq_dump(recon_ans) and is_faq_dump(base_ans):
            out.at[i, "answer_new"] = REFUSAL_TEXT
            r2 += 1
    meta = {"rule_recon_refusal": r1, "rule_faq_clean": r2, "rows": len(out)}
    return out[["q_id", "answer_new"]], meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--recon", required=True, help="recon CSV for rule signals only")
    parser.add_argument("--rules", default="data/cache/recon_rules.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    rules_path = _p(args.rules)
    rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else {}

    df, meta = apply_rules(
        pd.read_csv(_p(args.base)),
        pd.read_csv(_p(args.recon)),
        rules,
    )
    out = _p(args.output)
    _write_submission(df, out)
    print(f"meta={meta} rules_n={rules.get('n_patch', 'n/a')} -> {out}")


if __name__ == "__main__":
    main()
