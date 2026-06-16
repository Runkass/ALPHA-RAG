#!/usr/bin/env python
"""Phase 12 calibration anchors K1-K6 (mass ablation, no LLM)."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal  # noqa: E402
from src.project_paths import ARCHIVE_SUBMISSIONS, BACKUP_FULL  # noqa: E402

REFUSAL_TEXT = "Нет ответа"
WRONG_TEXT = "ошибка"
DIFF_PATH = ROOT / "data/cache/sample_full_diff.parquet"

VARIANT_FILES = {
    "k1_answer_mass": "submission_anchor_k1_answer_mass.csv",
    "k2_refuse_wrong": "submission_anchor_k2_refuse_wrong.csv",
    "k3_partial_10_refuse": "submission_anchor_k3_partial_10.csv",
    "k4_mass_length_trim": "submission_anchor_k4_length_trim.csv",
    "k5_class_semantic_full": "submission_anchor_k5_semantic_full.csv",
    "k6_partial_semantic_10": "submission_anchor_k6_semantic_10.csv",
}


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
    return {
        "rows": len(df),
        "refusal_n": int(ref.sum()),
        "refusal_pct": round(100.0 * ref.sum() / len(df), 2) if len(df) else 0.0,
        "avg_len": round(float(stripped.str.len().mean()), 1),
    }


def _load_sample() -> pd.DataFrame:
    return pd.read_csv(ROOT / "sample_submission.csv")


def _load_full() -> pd.DataFrame:
    return pd.read_csv(BACKUP_FULL)


def _load_diff() -> pd.DataFrame:
    if not DIFF_PATH.exists():
        raise SystemExit(f"Run build_sample_full_diff.py first: {DIFF_PATH}")
    return pd.read_parquet(DIFF_PATH)


def build_k1(*, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    df = _load_sample().copy()
    mask = ~df["answer_new"].astype(str).map(is_refusal)
    changed = int(mask.sum())
    if not dry_run:
        df.loc[mask, "answer_new"] = REFUSAL_TEXT
    return df, {"variant": "k1_answer_mass", "changed": changed}


def build_k2(*, dry_run: bool = False) -> tuple[pd.DataFrame, dict]:
    df = _load_sample().copy()
    mask = df["answer_new"].astype(str).map(is_refusal)
    changed = int(mask.sum())
    if not dry_run:
        df.loc[mask, "answer_new"] = WRONG_TEXT
    return df, {"variant": "k2_refuse_wrong", "changed": changed}


def build_k3(*, dry_run: bool = False, seed: int = 42, frac: float = 0.10) -> tuple[pd.DataFrame, dict]:
    df = _load_sample().copy()
    non_ref = df[~df["answer_new"].astype(str).map(is_refusal)].copy()
    non_ref["_len"] = non_ref["answer_new"].astype(str).str.len()
    non_ref["_bin"] = pd.qcut(non_ref["_len"], q=5, duplicates="drop")
    picks: list[int] = []
    rng = random.Random(seed)
    for _, grp in non_ref.groupby("_bin", observed=True):
        n = max(1, int(round(len(grp) * frac)))
        picks.extend(rng.sample(grp["q_id"].astype(int).tolist(), min(n, len(grp))))
    target_n = int(round(len(non_ref) * frac))
    if len(picks) > target_n:
        picks = rng.sample(picks, target_n)
    elif len(picks) < target_n:
        rest = [q for q in non_ref["q_id"].astype(int) if q not in picks]
        picks.extend(rng.sample(rest, min(target_n - len(picks), len(rest))))
    pick_set = set(picks)
    changed = len(pick_set)
    if not dry_run:
        df.loc[df["q_id"].astype(int).isin(pick_set), "answer_new"] = REFUSAL_TEXT
    return df, {"variant": "k3_partial_10_refuse", "changed": changed, "seed": seed}


def build_k4(*, dry_run: bool = False, min_len: int = 300, trim_to: int = 150) -> tuple[pd.DataFrame, dict]:
    df = _load_sample().copy()
    lens = df["answer_new"].astype(str).str.len()
    mask = lens > min_len
    changed = int(mask.sum())
    if not dry_run:
        for i in df.index[mask]:
            text = str(df.at[i, "answer_new"])
            df.at[i, "answer_new"] = text[:trim_to]
    return df, {"variant": "k4_mass_length_trim", "changed": changed, "min_len": min_len, "trim_to": trim_to}


def build_k5(*, dry_run: bool = False, max_full_len: int = 120) -> tuple[pd.DataFrame, dict]:
    diff = _load_diff()
    full = _load_full()
    fmap = dict(zip(full["q_id"].astype(int), full["answer_new"].astype(str)))
    sem = diff[diff["diff_class"] == "semantic_gap"]["q_id"].astype(int).tolist()
    df = _load_sample().copy()
    changed = 0
    for q in sem:
        full_ans = fmap.get(q, "")
        if not full_ans or is_refusal(full_ans) or len(full_ans) > max_full_len:
            continue
        if not dry_run:
            df.loc[df["q_id"].astype(int) == q, "answer_new"] = full_ans
        changed += 1
    return df, {"variant": "k5_class_semantic_full", "changed": changed, "max_full_len": max_full_len}


def build_k6(*, dry_run: bool = False, seed: int = 42, frac: float = 0.10, max_full_len: int = 120) -> tuple[pd.DataFrame, dict]:
    diff = _load_diff()
    full = _load_full()
    fmap = dict(zip(full["q_id"].astype(int), full["answer_new"].astype(str)))
    sem = diff[diff["diff_class"] == "semantic_gap"]["q_id"].astype(int).tolist()
    eligible = [q for q in sem if fmap.get(q) and not is_refusal(fmap[q]) and len(fmap[q]) <= max_full_len]
    rng = random.Random(seed)
    n = max(1, int(round(len(eligible) * frac)))
    picks = rng.sample(eligible, min(n, len(eligible)))
    df = _load_sample().copy()
    changed = 0
    for q in picks:
        if not dry_run:
            df.loc[df["q_id"].astype(int) == q, "answer_new"] = fmap[q]
        changed += 1
    return df, {"variant": "k6_partial_semantic_10", "changed": changed, "seed": seed, "eligible": len(eligible)}


BUILDERS = {
    "k1_answer_mass": build_k1,
    "k2_refuse_wrong": build_k2,
    "k3_partial_10_refuse": build_k3,
    "k4_mass_length_trim": build_k4,
    "k5_class_semantic_full": build_k5,
    "k6_partial_semantic_10": build_k6,
}


def audit_calibration(path: Path) -> int:
    from scripts.build_platform_anchors import audit_file

    return audit_file(path, variant=None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=list(BUILDERS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--file", default=None, help="Audit existing CSV")
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = ROOT / path
        return audit_calibration(path)

    variants = list(BUILDERS.keys()) if args.variant == "all" else [args.variant]
    results: list[dict] = []

    for v in variants:
        df, meta = BUILDERS[v](dry_run=args.dry_run)
        st = _stats(df)
        row = {**meta, **st}
        results.append(row)
        print(f"variant={v} dry_run={args.dry_run} {row}")

        if args.dry_run:
            continue

        out = ARCHIVE_SUBMISSIONS / VARIANT_FILES[v]
        _write_submission(df, out)
        print(f"Wrote {out}")
        if args.audit:
            code = audit_calibration(out)
            if code != 0:
                return code

    if args.dry_run and len(results) > 1:
        print("--- summary ---")
        for r in results:
            print(
                f"{r['variant']}: changed_q={r['changed']} refusal%={r['refusal_pct']} avg_len={r['avg_len']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
