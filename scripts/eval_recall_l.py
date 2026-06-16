#!/usr/bin/env python
"""Evaluate submission against gold CSV using Recall-L."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal, recall_l_corpus


def _load_platform_calibration() -> dict:
    path = ROOT / "data" / "cache" / "platform_calibration.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _load_answer_only_baseline() -> float | None:
    path = ROOT / "data" / "cache" / "run_state.json"
    if not path.exists():
        return None
    import json

    state = json.loads(path.read_text(encoding="utf-8"))
    val = state.get("platform_baseline_answer_only")
    return float(val) if val is not None else None


def _platform_estimate(
    report,
    *,
    cal: dict,
    baseline: float | None,
) -> float | None:
    if report.pred_refusal_rate >= 0.05:
        return None
    if baseline is None or baseline <= 0:
        return None
    floor = float(cal.get("platform_floor", 35.844))
    full = float(cal.get("platform_full", 52.262))
    return floor + report.answer_only_recall_l * (full - floor) / baseline


def _record_eval(
    *,
    pred: str,
    limit: int | None,
    report,
    phase: str,
    step: str,
    comment: str,
    platform_est: float | None = None,
) -> None:
    import runpy

    us = runpy.run_path(str(ROOT / "scripts" / "update_state.py"))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    limit_s = str(limit) if limit is not None else "all"
    extra = ""
    if platform_est is not None:
        extra = f" | answer_only={report.answer_only_recall_l:.4f} | platform_est={platform_est:.3f}"
    row = (
        f"{ts} | {phase}-{step} | {pred} | limit={limit_s} | "
        f"RecallL={report.recall_l:.4f} | r_bert={report.r_bert_mean:.4f} | "
        f"l_mult={report.l_mult_mean:.4f} | "
        f"pred_refusal={100 * report.pred_refusal_rate:.1f}% | "
        f"false_refuse={report.false_refuse} | false_answer={report.false_answer} | "
        f"comment={comment}{extra}"
    )
    us["append_journal_row"](row)

    state = us["read_run_state"]()
    state["phase"] = phase
    state["step"] = step
    state["last_eval"] = f"RecallL={report.recall_l:.4f} (limit={limit_s})"
    best = state.get("best_recall_l")
    if best is None or report.recall_l > float(best):
        state["best_recall_l"] = round(report.recall_l, 4)
        state["best_submission_bak"] = "archive/submissions/submission.csv.bak.phase1-step1"
    if comment:
        state["comment"] = comment
    us["write_run_state"](state)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "update_state.py")], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="submission.csv")
    parser.add_argument("--gold", default="sample_submission.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--phase", default="phase0")
    parser.add_argument("--step", default="eval")
    parser.add_argument("--comment", default="")
    parser.add_argument(
        "--q-ids-file",
        type=str,
        default=None,
        help="CSV with q_id column — evaluate only these pairs",
    )
    args = parser.parse_args()

    pred = pd.read_csv(ROOT / args.pred)
    gold = pd.read_csv(ROOT / args.gold)
    m = gold.merge(pred, on="q_id", suffixes=("_gold", "_pred"))
    if args.q_ids_file:
        qids = pd.read_csv(ROOT / args.q_ids_file)["q_id"].astype(int)
        m = m[m["q_id"].isin(qids)]
    if args.limit:
        m = m.head(args.limit)

    preds = m["answer_new_pred"].astype(str).tolist()
    golds = m["answer_new_gold"].astype(str).tolist()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    print(f"Pairs: {len(m)}")
    report = recall_l_corpus(
        preds, golds, tokenizer, batch_size=args.batch_size
    )
    print(f"Recall-L:        {report.recall_l:.4f}")
    print(f"BERT recall avg: {report.r_bert_mean:.4f}")
    print(f"L mult avg:      {report.l_mult_mean:.4f}")
    print(f"Gold refusals:   {100 * report.gold_refusal_rate:.1f}%")
    print(f"Pred refusals:   {100 * report.pred_refusal_rate:.1f}%")
    print(f"False refuse:    {report.false_refuse}")
    print(f"False answer:    {report.false_answer}")
    print(f"Answer-only RL:  {report.answer_only_recall_l:.4f} (n={report.answer_only_n})")
    print(f"Refusal-only RL: {report.refusal_only_recall_l:.4f} (n={report.refusal_only_n})")

    cal = _load_platform_calibration()
    baseline = _load_answer_only_baseline()
    platform_est = _platform_estimate(report, cal=cal, baseline=baseline)
    if platform_est is not None:
        print(f"Platform est:    {platform_est:.3f}")
    else:
        print("Platform est:    n/a")

    gold_ref = m["answer_new_gold"].map(is_refusal)
    if gold_ref.any():
        hit = (
            m.loc[gold_ref, "answer_new_pred"].map(is_refusal).sum()
        )
        refusal_recall = hit / int(gold_ref.sum())
        print(f"Refusal recall:  {refusal_recall:.4f} ({hit}/{int(gold_ref.sum())})")

    _record_eval(
        pred=args.pred,
        limit=args.limit,
        report=report,
        phase=args.phase,
        step=args.step,
        comment=args.comment or "eval",
        platform_est=platform_est,
    )


if __name__ == "__main__":
    main()
