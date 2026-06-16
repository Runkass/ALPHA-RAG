#!/usr/bin/env python
"""Top tokens in questions.csv for rule-fallback tuning."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROJECT_ROOT, get_settings

_WORD_RE = re.compile(r"[а-яА-ЯёЁa-zA-Z]{4,}")
_BERT_TOK = "bert-base-multilingual-cased"


def _gold_token_lengths(gold_path: Path) -> pd.Series:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_BERT_TOK)
    df = pd.read_csv(gold_path)
    answers = df["answer_new"].astype(str).str.strip()
    no_refusal = answers.str.lower() != "нет ответа"
    return answers[no_refusal].map(lambda x: len(tok.encode(x, add_special_tokens=False)))


def print_gold_token_stats(gold_path: Path) -> None:
    lengths = _gold_token_lengths(gold_path)
    print(f"Gold answers (excl. refusal): {len(lengths)}")
    print(f"median: {lengths.median():.1f}")
    print(f"P75: {lengths.quantile(0.75):.1f}")
    print(f"P90: {lengths.quantile(0.90):.1f}")
    print(f"mean: {lengths.mean():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--gold-tokens",
        action="store_true",
        help="Token length stats for sample_submission.csv (BERT multilingual)",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.gold_tokens:
        print_gold_token_stats(PROJECT_ROOT / "sample_submission.csv")
        return

    df = pd.read_csv(settings.questions_path)
    words: list[str] = []
    for q in df["query"].dropna().astype(str):
        words.extend(_WORD_RE.findall(q.lower()))

    print(f"Questions: {len(df)}")
    for word, count in Counter(words).most_common(args.top):
        print(f"{word}: {count}")


if __name__ == "__main__":
    main()