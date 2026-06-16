#!/usr/bin/env python
"""Build Phase 13 frontier micro-probe submissions (E1/E2/E3)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.build_sample_patch as bsp  # noqa: E402
from scripts.lq_patch_gate import lq_check, token_len, gold_len_proxy  # noqa: E402
from src.metrics.recall_l import is_refusal  # noqa: E402
from src.project_paths import BACKUP_FULL  # noqa: E402

PROBE_MAP = {
    "E1": {"mode": "false_refuse", "error_type": "false_refuse", "default_out": "archive/submissions/submission_e1_refuse5.csv"},
    "E2": {"mode": "compress", "error_type": "verbosity", "default_out": "archive/submissions/submission_e2_verb5.csv"},
    "E3": {"mode": "full_replace", "error_type": "extraction", "default_out": "archive/submissions/submission_e3_extract5.csv"},
}


def _p(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def _select_e3_from_diff(limit: int) -> set[int]:
    diff_path = ROOT / "data/cache/sample_full_diff.parquet"
    if not diff_path.exists():
        return set()
    diff = pd.read_parquet(diff_path)
    sub = diff[diff["diff_class"].isin(("semantic_gap", "semantic_gap_verbose_sample"))].copy()
    sub = sub[sub["len_full"] < sub["len_sample"] * 0.7]
    sub = sub.sort_values("len_sample", ascending=False)
    return set(sub["q_id"].astype(int).head(limit).tolist())


def _select_q_ids(
    lq_path: Path,
    error_type: str,
    *,
    limit: int,
    probe: str = "",
) -> set[int]:
    if probe == "E3" and error_type == "extraction":
        ids = _select_e3_from_diff(limit)
        if ids:
            return ids
    df = pd.read_csv(lq_path)
    if not len(df):
        return set()
    sub = df
    if "approved" in df.columns:
        sub = df[df["approved"].astype(bool)]
    if error_type:
        if error_type in sub["error_type"].astype(str).values if len(sub) else []:
            sub = sub[sub["error_type"].astype(str) == error_type]
        elif probe != "E3":
            return set()
    if error_type == "false_refuse" and "refusal_confidence" in sub.columns:
        sub = sub.sort_values("refusal_confidence", ascending=False)
    elif error_type == "verbosity" and "verbosity_score" in sub.columns:
        sub = sub.sort_values("verbosity_score", ascending=False)
    return set(sub["q_id"].astype(int).head(limit).tolist())


def _post_lq_filter(
    base: pd.DataFrame,
    out: pd.DataFrame,
    q_ids: set[int],
    sample_map: dict[int, str],
    *,
    corpus_median: int,
    error_type: str,
) -> tuple[pd.DataFrame, int]:
    filtered = out.copy()
    changed = 0
    for q in q_ids:
        old = sample_map.get(q, "")
        new = str(filtered.loc[filtered["q_id"] == q, "answer_new"].iloc[0])
        if new == old:
            continue
        gold_proxy = gold_len_proxy(old, error_type, corpus_median)
        _lm, ok, _reason = lq_check(new, gold_proxy)
        if not ok:
            filtered.loc[filtered["q_id"] == q, "answer_new"] = old
        else:
            changed += 1
    return filtered, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, choices=sorted(PROBE_MAP.keys()))
    parser.add_argument("--base", default="sample_submission.csv")
    parser.add_argument("--lq-approved", default="data/cache/phase13_lq_approved.csv")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--questions", default="questions.csv")
    parser.add_argument("--full", default=str(BACKUP_FULL.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--max-len", type=int, default=240)
    args = parser.parse_args()

    meta = PROBE_MAP[args.probe]
    out_path = _p(args.output or meta["default_out"])
    q_ids = _select_q_ids(_p(args.lq_approved), meta["error_type"], limit=args.limit, probe=args.probe)
    if not q_ids:
        print(f"no approved q_ids for probe {args.probe} type={meta['error_type']}")
        return 1

    base = pd.read_csv(_p(args.base))
    questions = pd.read_csv(_p(args.questions))
    sample_map = dict(zip(base["q_id"].astype(int), base["answer_new"].astype(str)))
    non_refuse = [token_len(a) for a in sample_map.values() if not is_refusal(str(a))]
    non_refuse.sort()
    corpus_median = non_refuse[len(non_refuse) // 2] if non_refuse else 32

    if meta["mode"] == "false_refuse":
        out, _changed = bsp.patch_false_refuse(base, q_ids, questions, max_len=args.max_len)
    elif meta["mode"] == "compress":
        out, _changed = bsp.patch_compress(
            base, q_ids, questions, max_len=args.max_len, min_shrink_ratio=0.7
        )
    elif meta["mode"] == "full_replace":
        out, _changed = bsp.patch_full(base, q_ids, _p(args.full))
    else:
        raise SystemExit(f"unknown mode {meta['mode']}")

    out, changed = _post_lq_filter(
        base,
        out,
        q_ids,
        sample_map,
        corpus_median=corpus_median,
        error_type=meta["error_type"],
    )
    _write_submission(out, out_path)
    print(f"probe={args.probe} mode={meta['mode']} q_ids={len(q_ids)} changed={changed} -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
