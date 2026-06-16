#!/usr/bin/env python
"""Build Phase 13 diagnostic cohorts A/B/C/D and optional frontier score."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal  # noqa: E402
from src.project_paths import BACKUP_FULL  # noqa: E402
from src.submission_rules import (  # noqa: E402
    is_protected_q_id,
    is_protected_short_answer,
    is_verbose_sample_answer,
    load_protected_q_ids,
    verbosity_score,
)

BATCHES = ROOT / "data" / "cache" / "edit_batches"
DENY_DEFAULT = BATCHES / "s2_do_not_touch.csv"


def _p(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _load_candidates(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_diff(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_weakness(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def build_cohorts(
    sample: pd.DataFrame,
    questions: pd.DataFrame,
    *,
    deny: set[int],
    candidates: pd.DataFrame,
    diff: pd.DataFrame,
    weakness: pd.DataFrame,
    full_path: Path,
) -> pd.DataFrame:
    qmap = dict(zip(questions["q_id"].astype(int), questions["query"].astype(str)))
    sample_map = dict(zip(sample["q_id"].astype(int), sample["answer_new"].astype(str)))
    full_map: dict[int, str] = {}
    if full_path.exists():
        fdf = pd.read_csv(full_path)
        full_map = dict(zip(fdf["q_id"].astype(int), fdf["answer_new"].astype(str)))

    cand_ref = {}
    if len(candidates) and "candidate_type" in candidates.columns:
        ref = candidates[candidates["candidate_type"] == "refusal_high_conf"]
        for r in ref.itertuples(index=False):
            cand_ref[int(r.q_id)] = {
                "refusal_confidence": float(getattr(r, "refusal_confidence", 0.0)),
                "keyword_overlap": float(getattr(r, "keyword_overlap", 0.0)),
            }

    diff_map = {}
    if len(diff) and "q_id" in diff.columns:
        for r in diff.itertuples(index=False):
            diff_map[int(r.q_id)] = str(getattr(r, "diff_class", ""))

    weak_map = {}
    if len(weakness) and "q_id" in weakness.columns:
        for r in weakness.itertuples(index=False):
            weak_map[int(r.q_id)] = float(getattr(r, "weak_score", 0.0))

    rows: list[dict] = []
    for q_id, ans in sample_map.items():
        q_id = int(q_id)
        ans_s = str(ans)
        cohorts: list[str] = []

        if is_protected_q_id(q_id, deny) or is_protected_short_answer(ans_s):
            cohorts.append("C")
        elif len(ans_s) <= 80 and not is_refusal(ans_s):
            cohorts.append("C")

        if is_refusal(ans_s) and "C" not in cohorts:
            cohorts.append("A")

        verbose = is_verbose_sample_answer(ans_s) or len(ans_s) > 350
        if verbose and "C" not in cohorts:
            cohorts.append("B")

        cls = diff_map.get(q_id, "")
        if (
            not is_refusal(ans_s)
            and not is_verbose_sample_answer(ans_s)
            and len(ans_s) > 80
            and cls in ("identical", "")
            and q_id in full_map
            and full_map.get(q_id) == ans_s
        ):
            cohorts.append("D")
        elif cls in ("semantic_gap", "len_gap_sample_longer") and not is_refusal(ans_s):
            if "D" not in cohorts and not verbose:
                cohorts.append("D")

        if not cohorts:
            continue

        ref_meta = cand_ref.get(q_id, {})
        full_ans = full_map.get(q_id, "")
        rows.append(
            {
                "q_id": q_id,
                "query": qmap.get(q_id, ""),
                "sample_answer": ans_s,
                "full_answer": full_ans,
                "cohorts": "|".join(sorted(set(cohorts))),
                "cohort_primary": cohorts[0],
                "refusal_confidence": ref_meta.get("refusal_confidence", 0.0),
                "keyword_overlap": ref_meta.get("keyword_overlap", 0.0),
                "verbosity_score": verbosity_score(ans_s) if verbose else 0.0,
                "diff_class": cls,
                "weak_score": weak_map.get(q_id, 0.0),
                "full_disagree": int(
                    bool(full_ans and full_ans.strip() != ans_s.strip() and not is_refusal(full_ans))
                ),
                "is_protected": int("C" in cohorts),
            }
        )

    return pd.DataFrame(rows)


def compute_frontier(df: pd.DataFrame, manual: pd.DataFrame | None) -> pd.DataFrame:
    manual_patch: dict[int, float] = {}
    if manual is not None and len(manual):
        for r in manual.itertuples(index=False):
            verdict = str(getattr(r, "review_verdict", "")).strip().lower()
            if verdict == "patch":
                manual_patch[int(r.q_id)] = 1.0

    scored = df.copy()
    scored["manual_patch"] = scored["q_id"].map(lambda q: manual_patch.get(int(q), 0.0))
    scored["frontier_score"] = (
        0.45 * scored["manual_patch"]
        + 0.25 * scored["refusal_confidence"].clip(0, 1)
        + 0.15 * scored["verbosity_score"].clip(0, 1)
        + 0.10 * scored["full_disagree"]
        + 0.05 * scored["weak_score"].clip(0, 1)
        - 0.20 * scored["is_protected"]
    ).round(4)
    return scored.sort_values("frontier_score", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--questions", default="questions.csv")
    parser.add_argument("--full", default=str(BACKUP_FULL.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--candidates", default="data/cache/sample_candidates.parquet")
    parser.add_argument("--diff", default="data/cache/sample_full_diff.parquet")
    parser.add_argument("--weakness", default="data/cache/sample_weakness.parquet")
    parser.add_argument("--manual", default="data/cache/phase13_manual_review.csv")
    parser.add_argument("--output", default="data/cache/phase13_cohorts.parquet")
    parser.add_argument("--frontier-out", default="data/cache/phase13_frontier_top20.csv")
    parser.add_argument("--frontier", action="store_true")
    args = parser.parse_args()

    deny = load_protected_q_ids(_p(DENY_DEFAULT))
    sample = pd.read_csv(_p(args.sample))
    questions = pd.read_csv(_p(args.questions))
    df = build_cohorts(
        sample,
        questions,
        deny=deny,
        candidates=_load_candidates(_p(args.candidates)),
        diff=_load_diff(_p(args.diff)),
        weakness=_load_weakness(_p(args.weakness)),
        full_path=_p(args.full),
    )

    out = _p(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    BATCHES.mkdir(parents=True, exist_ok=True)
    for letter in ("A", "B", "C", "D"):
        sub = df[df["cohorts"].str.contains(letter, regex=False)]
        sub[["q_id"]].drop_duplicates().to_csv(BATCHES / f"phase13_cohort_{letter}.csv", index=False)

    a_n = df["cohorts"].str.contains("A").sum()
    b_n = df["cohorts"].str.contains("B").sum()
    c_n = df["cohorts"].str.contains("C").sum()
    d_n = df["cohorts"].str.contains("D").sum()
    print(f"cohorts A={a_n} B={b_n} C={c_n} D={d_n} total_rows={len(df)}")
    print(f"saved -> {out}")

    manual_path = _p(args.manual)
    manual = pd.read_csv(manual_path) if manual_path.exists() else None
    if args.frontier or (manual is not None and len(manual)):
        scored = compute_frontier(df, manual)
        top = scored[~scored["is_protected"].astype(bool)].head(20)
        f_out = _p(args.frontier_out)
        top.to_csv(f_out, index=False)
        print(f"frontier top20 -> {f_out} (max score={top['frontier_score'].max() if len(top) else 0})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
