#!/usr/bin/env python
"""Build platform anchor submissions (no LLM) + content audit before upload."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal
from src.project_paths import ARCHIVE_SUBMISSIONS, BACKUP_FULL

REFUSAL_TEXT = "Нет ответа"
ALT_PHRASE = (
    "В предоставленной базе знаний нет информации для ответа на этот вопрос"
)

VARIANT_FILES = {
    "oracle_refuse": "submission_anchor_oracle_refuse.csv",
    "alt_phrase": "submission_anchor_alt_phrase.csv",
    "short_refuse": "submission_anchor_short_refuse.csv",
}

_GARBAGE_CHARS = {"−", "-", "—", "–"}
_SINGLE_LETTER_GARBAGE = {"о", "в", "0"}
_ORACLE_REFUSAL_EXPECTED = 2284


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def _stats(df: pd.DataFrame) -> dict:
    ans = df["answer_new"].astype(str)
    stripped = ans.str.strip()
    ref = stripped.map(is_refusal)
    empty = df["answer_new"].isna() | (stripped == "")
    return {
        "rows": len(df),
        "empty": int(empty.sum()),
        "refusal_n": int(ref.sum()),
        "refusal_pct": 100.0 * ref.sum() / len(df) if len(df) else 0.0,
        "avg_len": float(stripped.str.len().mean()),
    }


def build_oracle_refuse(*, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    full = pd.read_csv(BACKUP_FULL)
    gold = pd.read_csv(ROOT / "sample_submission.csv")
    m = full.merge(gold[["q_id", "answer_new"]], on="q_id", suffixes=("", "_gold"))
    gold_ref = m["answer_new_gold"].map(is_refusal)
    changed = int(gold_ref.sum())
    if not dry_run:
        m.loc[gold_ref, "answer_new"] = REFUSAL_TEXT
    meta = {"changed": changed, "variant": "oracle_refuse"}
    return m[["q_id", "answer_new"]], meta


def build_alt_phrase(*, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    q = pd.read_csv(ROOT / "questions.csv")
    df = pd.DataFrame({"q_id": q["q_id"].astype(int), "answer_new": ALT_PHRASE})
    meta = {"variant": "alt_phrase", "phrase": ALT_PHRASE}
    return df, meta


def build_short_refuse(*, short_len: int, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    full = pd.read_csv(BACKUP_FULL)
    df = full.copy()
    ans = df["answer_new"].astype(str).str.strip()
    short_mask = (ans.str.len() < short_len) & (~ans.map(is_refusal))
    replaced = int(short_mask.sum())
    if not dry_run:
        df.loc[short_mask, "answer_new"] = REFUSAL_TEXT
    meta = {
        "variant": "short_refuse",
        "short_len": short_len,
        "short_replaced": replaced,
    }
    return df[["q_id", "answer_new"]], meta


def _detect_variant(path: Path) -> str | None:
    name = path.name
    for variant, fname in VARIANT_FILES.items():
        if name == fname:
            return variant
    return None


def _is_garbage_answer(text: str) -> bool:
    t = str(text).strip()
    if not t or is_refusal(t):
        return False
    if t in _GARBAGE_CHARS:
        return True
    if len(t) <= 2 and t.lower() in _SINGLE_LETTER_GARBAGE:
        return True
    if len(t) <= 2:
        return True
    return False


def audit_file(path: Path, *, variant: str | None = None) -> int:
    if not path.exists():
        print(f"FAIL: file not found: {path}")
        return 1

    df = pd.read_csv(path)
    variant = variant or _detect_variant(path)
    ans = df["answer_new"].astype(str)
    stripped = ans.str.strip()
    st = _stats(df)
    errors: list[str] = []

    print(f"Audit: {path.name} variant={variant or 'unknown'}")
    print(
        f"  rows={st['rows']} empty={st['empty']} "
        f"refusal={st['refusal_n']} ({st['refusal_pct']:.1f}%) avg_len={st['avg_len']:.0f}"
    )

    if st["empty"] > 0:
        errors.append(f"empty answers: {st['empty']}")
    if st["rows"] != 6977:
        errors.append(f"row count {st['rows']} != 6977")

    garbage = df[stripped.map(_is_garbage_answer)]
    if len(garbage):
        if variant == "oracle_refuse":
            full = pd.read_csv(BACKUP_FULL)
            inherited = 0
            for _, row in garbage.iterrows():
                f = full.loc[full.q_id == row.q_id, "answer_new"]
                if len(f) and str(f.iloc[0]).strip() == str(row.answer_new).strip():
                    inherited += 1
            new_n = len(garbage) - inherited
            if new_n:
                errors.append(
                    f"garbage len<=2 or dash (new): {new_n} (e.g. q_id={garbage.q_id.iloc[0]})"
                )
            else:
                print(f"  WARN: {inherited} short answers inherited from FULL (ok)")
        else:
            errors.append(
                f"garbage len<=2 or dash: {len(garbage)} (e.g. q_id={garbage.q_id.iloc[0]})"
            )

    faq = df[ans.str.contains("] / ", regex=False, na=False)]
    if len(faq):
        errors.append(f"FAQ dump '] / ': {len(faq)} (e.g. q_id={faq.q_id.iloc[0]})")

    if variant == "oracle_refuse":
        full = pd.read_csv(BACKUP_FULL)
        gold = pd.read_csv(ROOT / "sample_submission.csv")
        m = df.merge(full, on="q_id", suffixes=("_new", "_full")).merge(
            gold[["q_id", "answer_new"]], on="q_id"
        )
        gold_ref = m["answer_new"].map(is_refusal)
        if st["refusal_n"] != _ORACLE_REFUSAL_EXPECTED:
            errors.append(
                f"oracle refusal {st['refusal_n']} != {_ORACLE_REFUSAL_EXPECTED}"
            )
        non_ref = m[~gold_ref]
        drift = non_ref[non_ref["answer_new_new"] != non_ref["answer_new_full"]]
        if len(drift):
            errors.append(f"oracle: {len(drift)} non-refusal rows differ from FULL")

    elif variant == "alt_phrase":
        unique = stripped.unique()
        if len(unique) != 1 or unique[0] != ALT_PHRASE:
            errors.append(f"alt_phrase: expected single phrase, got {len(unique)} unique")

    elif variant == "short_refuse":
        if st["refusal_pct"] <= 0:
            errors.append("short_refuse: expected some refusals")

    random.seed(42)
    non_ref = df[~stripped.map(is_refusal)]
    sample_ids = random.sample(list(non_ref["q_id"]), min(5, len(non_ref)))
    print("  spot-check random q_id:", sample_ids)
    for qid in sample_ids:
        row = df[df.q_id == qid].iloc[0]
        snippet = str(row.answer_new).replace("\n", " ")[:120]
        print(f"    q={qid}: {snippet}")

    if len(non_ref):
        short = non_ref.assign(_len=stripped.loc[non_ref.index].str.len()).nsmallest(
            10, "_len"
        )
        print("  shortest non-refusal q_id:", list(short.q_id))
        for _, row in short.iterrows():
            snippet = str(row.answer_new).replace("\n", " ")[:120]
            print(f"    q={row.q_id} len={len(str(row.answer_new))}: {snippet}")

    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("OK: content audit passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["oracle_refuse", "alt_phrase", "short_refuse", "all"],
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--short-len", type=int, default=80)
    parser.add_argument("--file", type=str, default=None, help="Audit existing CSV")
    parser.add_argument("--audit", action="store_true", help="With --file: run audit")
    args = parser.parse_args()

    if args.file:
        path = ROOT / args.file if not Path(args.file).is_absolute() else Path(args.file)
        if not args.audit:
            print("Use --audit with --file")
            return 1
        return audit_file(path)

    if not args.variant:
        parser.error("Specify --variant or --file --audit")

    variants = (
        ["oracle_refuse", "alt_phrase", "short_refuse"]
        if args.variant == "all"
        else [args.variant]
    )

    for v in variants:
        if v == "oracle_refuse":
            df, meta = build_oracle_refuse(dry_run=args.dry_run)
        elif v == "alt_phrase":
            df, meta = build_alt_phrase(dry_run=args.dry_run)
        else:
            df, meta = build_short_refuse(short_len=args.short_len, dry_run=args.dry_run)

        st = _stats(df)
        print(f"variant={v} dry_run={args.dry_run} meta={meta} stats={st}")

        if args.dry_run:
            continue

        out = ARCHIVE_SUBMISSIONS / VARIANT_FILES[v]
        _write_submission(df, out)
        print(f"Wrote {out}")
        code = audit_file(out, variant=v)
        if code != 0:
            return code

    return 0


if __name__ == "__main__":
    sys.exit(main())
