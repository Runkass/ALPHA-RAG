#!/usr/bin/env python
"""Apply strict assisted review heuristics to phase13 manual review CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal  # noqa: E402
from src.submission_rules import is_protected_short_answer, is_verbose_sample_answer  # noqa: E402

_WEAK_MARKERS = (
    "обратитесь",
    "к сожалению",
    "в предоставленной информации",
    "рекомендуем обратиться",
    "свяжитесь",
    "уточните",
    "техподдерж",
)


def _p(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _has_weak_marker(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _WEAK_MARKERS)


def assisted_verdict(row: pd.Series) -> tuple[str, str, str]:
    sample = str(row.get("sample_ans", ""))
    full = str(row.get("full_ans", ""))
    cohort = str(row.get("cohort", ""))
    chunk = str(row.get("top_chunk", ""))
    conf = float(row.get("refusal_confidence", 0.0) or 0.0)

    if is_protected_short_answer(sample):
        return "skip", "unknown", "protected_short"

    if cohort == "A" or is_refusal(sample):
        if conf < 0.35:
            return "skip", "false_refuse", "low_retrieval_conf"
        if not chunk.strip():
            return "skip", "false_refuse", "no_context"
        if full and not is_refusal(full) and len(full) <= 240:
            return "patch", "false_refuse", "full_has_short_answer"
        if len(chunk) >= 40 and conf >= 0.5:
            return "patch", "false_refuse", "high_conf_context"
        return "keep", "false_refuse", "insufficient_evidence"

    if cohort == "B" or is_verbose_sample_answer(sample):
        if not full or is_refusal(full):
            return "skip", "verbosity", "no_full_alternative"
        if len(full) >= len(sample) * 0.7:
            return "skip", "verbosity", "full_not_shorter"
        if _has_weak_marker(sample) or len(sample) > 350:
            return "patch", "verbosity", "cop_out_or_long"
        if is_verbose_sample_answer(sample) and len(full) < len(sample) * 0.7:
            return "patch", "verbosity", "full_shorter"
        return "keep", "verbosity", "borderline"

    if full and full.strip() != sample.strip() and not is_refusal(full):
        if len(full) <= max(len(sample), 240):
            return "patch", "extraction", "full_disagree_shorter"
        return "skip", "extraction", "full_too_long"

    return "keep", "unknown", "default_keep"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/cache/phase13_manual_review.csv")
    parser.add_argument("--output", default=None, help="defaults to overwrite input")
    parser.add_argument("--only-empty", action="store_true", default=True)
    parser.add_argument("--force", dest="only_empty", action="store_false")
    args = parser.parse_args()

    inp = _p(args.input)
    df = pd.read_csv(inp)
    for col in ("review_verdict", "error_type", "review_notes"):
        if col in df.columns:
            df[col] = df[col].astype("object")
    out_path = _p(args.output) if args.output else inp

    patched = 0
    for i, row in df.iterrows():
        verdict_raw = row.get("review_verdict", "")
        if args.only_empty and pd.notna(verdict_raw) and str(verdict_raw).strip():
            continue
        verdict, err_type, notes = assisted_verdict(row)
        df.at[i, "review_verdict"] = verdict
        df.at[i, "error_type"] = err_type
        df.at[i, "review_notes"] = notes
        if verdict == "patch":
            patched += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    counts = df["review_verdict"].value_counts().to_dict()
    print(f"assisted review n={len(df)} patch={patched} counts={counts} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
