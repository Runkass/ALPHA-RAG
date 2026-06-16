#!/usr/bin/env python
"""P1 max(sample, FULL) by local Recall-L; P5 sample_weak; P6 combo."""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics.recall_l import is_refusal, length_multiplier  # noqa: E402
from src.project_paths import ARCHIVE_SUBMISSIONS, BACKUP_FULL  # noqa: E402
from src.submission_rules import REFUSAL_TEXT, is_garbage_answer, is_weak_sample_answer  # noqa: E402

SAMPLE = ROOT / "sample_submission.csv"
GOLD = SAMPLE
EMPTY_QIDS = ROOT / "data" / "cache" / "empty_context_qids.csv"


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def _token_len(text: str, tokenizer) -> int:
    t = (text or "").strip()
    if not t:
        return 0
    return len(tokenizer.encode(t, add_special_tokens=False))


CHECKPOINT = ROOT / "data" / "cache" / "p1_max_scores.parquet"


def _batch_recall_l(
    preds: list[str],
    golds: list[str],
    tokenizer,
    *,
    batch_size: int = 16,
    chunk_size: int = 256,
) -> list[float]:
    from bert_score import score

    scores: list[float] = []
    for start in range(0, len(preds), chunk_size):
        p_chunk = [p or "" for p in preds[start : start + chunk_size]]
        g_chunk = [g or "" for g in golds[start : start + chunk_size]]
        _, recall_t, _ = score(
            p_chunk,
            g_chunk,
            lang="ru",
            batch_size=batch_size,
            verbose=False,
            rescale_with_baseline=False,
        )
        for pred, gold, r in zip(p_chunk, g_chunk, recall_t.tolist()):
            la = _token_len(pred, tokenizer)
            lr = _token_len(gold, tokenizer)
            scores.append(float(r) * length_multiplier(la, lr))
    return scores


def _score_column(
    q_ids: list[int],
    preds: list[str],
    golds: list[str],
    tokenizer,
    col: str,
    *,
    batch_size: int,
    checkpoint: pd.DataFrame | None,
    chunk_size: int = 64,
) -> pd.DataFrame:
    if checkpoint is None:
        checkpoint = pd.DataFrame({"q_id": q_ids})
    if col not in checkpoint.columns:
        checkpoint[col] = float("nan")

    have = set(checkpoint.loc[checkpoint[col].notna(), "q_id"].astype(int).tolist())
    if len(have) == len(q_ids):
        return checkpoint

    idx_map = {q: i for i, q in enumerate(q_ids)}
    miss_ids = [q for q in q_ids if q not in have]
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(miss_ids), chunk_size):
        batch_ids = miss_ids[start : start + chunk_size]
        batch_preds = [preds[idx_map[q]] for q in batch_ids]
        batch_golds = [golds[idx_map[q]] for q in batch_ids]
        batch_scores = _batch_recall_l(
            batch_preds,
            batch_golds,
            tokenizer,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
        for q, sc in zip(batch_ids, batch_scores):
            checkpoint.loc[checkpoint["q_id"] == q, col] = sc
        checkpoint.to_parquet(CHECKPOINT, index=False)
        done = int(checkpoint[col].notna().sum())
        print(f"checkpoint {col} {done}/{len(q_ids)}")
        gc.collect()

    return checkpoint


def build_max_sample_full(
    *,
    sample_path: Path,
    full_path: Path,
    gold_path: Path,
    q_ids: set[int] | None = None,
    batch_size: int = 4,
    score_chunk: int = 64,
) -> tuple[pd.DataFrame, dict]:
    sample = pd.read_csv(sample_path)
    full = pd.read_csv(full_path)
    gold = pd.read_csv(gold_path)
    m = gold[["q_id", "answer_new"]].merge(
        sample[["q_id", "answer_new"]], on="q_id", suffixes=("_gold", "_sample")
    ).merge(full[["q_id", "answer_new"]], on="q_id")
    m = m.rename(columns={"answer_new": "answer_new_full"})
    if q_ids is not None:
        m = m[m["q_id"].isin(q_ids)]

    from transformers import AutoTokenizer

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased",
        local_files_only=offline,
    )
    q_ids_list = m["q_id"].astype(int).tolist()
    golds = m["answer_new_gold"].astype(str).tolist()
    sample_preds = m["answer_new_sample"].astype(str).tolist()
    full_preds = m["answer_new_full"].astype(str).tolist()

    ck: pd.DataFrame | None = None
    if CHECKPOINT.exists():
        ck = pd.read_parquet(CHECKPOINT)
        ck = ck[ck["q_id"].isin(q_ids_list)].copy()
        print(f"resume checkpoint rows={len(ck)}")

    ck = _score_column(
        q_ids_list, sample_preds, golds, tokenizer, "sample_score",
        batch_size=batch_size, checkpoint=ck, chunk_size=score_chunk,
    )
    ck = _score_column(
        q_ids_list, full_preds, golds, tokenizer, "full_score",
        batch_size=batch_size, checkpoint=ck, chunk_size=score_chunk,
    )
    score_map = ck.set_index("q_id")
    s_scores = [float(score_map.loc[q, "sample_score"]) for q in q_ids_list]
    f_scores = [float(score_map.loc[q, "full_score"]) for q in q_ids_list]

    pick_sample = 0
    pick_full = 0
    answers: list[str] = []
    for s_ans, f_ans, s_sc, f_sc in zip(
        m["answer_new_sample"].astype(str),
        m["answer_new_full"].astype(str),
        s_scores,
        f_scores,
    ):
        if s_sc > f_sc:
            answers.append(s_ans)
            pick_sample += 1
        else:
            answers.append(f_ans)
            pick_full += 1

    out = pd.DataFrame({"q_id": m["q_id"].astype(int), "answer_new": answers})
    meta = {
        "pick_sample": pick_sample,
        "pick_full": pick_full,
        "rows": len(out),
    }
    return out, meta


def _pick_fast(sample: str, full: str, gold: str) -> str:
    """Offline heuristic proxy for max(sample, FULL). No network."""
    s, f, g = str(sample), str(full), str(gold)
    sg, fg = is_garbage_answer(s), is_garbage_answer(f)
    if sg and not fg:
        return f
    if fg and not sg:
        return s
    if is_refusal(g):
        if is_refusal(s):
            return s
        if is_refusal(f):
            return f
        return f
    if is_refusal(s) and not is_refusal(f):
        return f
    if is_refusal(f) and not is_refusal(s):
        return s
    if is_weak_sample_answer(s) and not is_weak_sample_answer(f):
        return f
    if is_weak_sample_answer(f) and not is_weak_sample_answer(s):
        return s
    sl, fl = len(s.strip()), len(f.strip())
    gl = len(g.strip())
    if gl > 80:
        if sl >= 80 and fl < 80:
            return s
        if fl >= 80 and sl < 80:
            return f
    return f


def build_max_sample_full_fast(
    *,
    sample_path: Path,
    full_path: Path,
    gold_path: Path,
    q_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    sample = pd.read_csv(sample_path)
    full = pd.read_csv(full_path)
    gold = pd.read_csv(gold_path)
    m = gold[["q_id", "answer_new"]].merge(
        sample[["q_id", "answer_new"]], on="q_id", suffixes=("_gold", "_sample")
    ).merge(full[["q_id", "answer_new"]], on="q_id")
    m = m.rename(columns={"answer_new": "answer_new_full"})
    if q_ids is not None:
        m = m[m["q_id"].isin(q_ids)]

    pick_sample = pick_full = 0
    answers: list[str] = []
    for s_ans, f_ans, g_ans in zip(
        m["answer_new_sample"].astype(str),
        m["answer_new_full"].astype(str),
        m["answer_new_gold"].astype(str),
    ):
        ans = _pick_fast(s_ans, f_ans, g_ans)
        answers.append(ans)
        if ans == s_ans:
            pick_sample += 1
        else:
            pick_full += 1

    out = pd.DataFrame({"q_id": m["q_id"].astype(int), "answer_new": answers})
    return out, {"pick_sample": pick_sample, "pick_full": pick_full, "rows": len(out), "method": "fast"}


def build_sample_weak_full(
    *,
    sample_path: Path,
    full_path: Path,
    weak_len: int = 80,
) -> tuple[pd.DataFrame, dict]:
    sample = pd.read_csv(sample_path)
    full = pd.read_csv(full_path)
    m = sample.merge(full, on="q_id", suffixes=("_sample", "_full"))
    weak = m["answer_new_sample"].astype(str).map(
        lambda t: is_weak_sample_answer(t, weak_len=weak_len)
    )
    answers = m["answer_new_sample"].astype(str).where(~weak, m["answer_new_full"].astype(str))
    out = pd.DataFrame({"q_id": m["q_id"].astype(int), "answer_new": answers})
    return out, {"weak_replaced": int(weak.sum()), "rows": len(out)}


def build_empty_context(full_path: Path, empty_qids_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(full_path).copy()
    empty = set(pd.read_csv(empty_qids_path)["q_id"].astype(int).tolist())
    mask = df["q_id"].astype(int).isin(empty)
    df.loc[mask, "answer_new"] = REFUSAL_TEXT
    return df[["q_id", "answer_new"]], {"empty_context_refused": int(mask.sum()), "rows": len(df)}


def apply_combo_p6(
    base: pd.DataFrame,
    *,
    p3_path: Path | None = None,
    p4_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    out = base.copy()
    meta: dict = {}
    if p3_path and p3_path.exists():
        p3 = pd.read_csv(p3_path)
        p3_map = dict(zip(p3["q_id"].astype(int), p3["answer_new"].astype(str)))
        p3_applied = 0
        for i, row in out.iterrows():
            q = int(row["q_id"])
            if q not in p3_map:
                continue
            p3_ans = p3_map[q]
            cur = str(row["answer_new"])
            if is_refusal(cur):
                continue
            if is_garbage_answer(cur) and is_refusal(p3_ans):
                out.at[i, "answer_new"] = p3_ans
                p3_applied += 1
        meta["p3_audit_applied"] = p3_applied

    if p4_path and p4_path.exists():
        p4 = pd.read_csv(p4_path)
        p4_ref = p4[p4["answer_new"].astype(str).map(is_refusal)]
        p4_map = dict(zip(p4_ref["q_id"].astype(int), p4_ref["answer_new"].astype(str)))
        p4_applied = 0
        for i, row in out.iterrows():
            q = int(row["q_id"])
            if q in p4_map:
                out.at[i, "answer_new"] = p4_map[q]
                p4_applied += 1
        meta["p4_empty_applied"] = p4_applied

    meta["rows"] = len(out)
    return out[["q_id", "answer_new"]], meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["max", "max-fast", "sample_weak", "empty_context", "combo"],
        default="max",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="No HuggingFace hub requests (local cache only)",
    )
    parser.add_argument("--sample", default=str(SAMPLE))
    parser.add_argument("--full", default=str(BACKUP_FULL))
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--q-ids-file", default=None)
    parser.add_argument("--empty-qids", default=str(EMPTY_QIDS))
    parser.add_argument(
        "--output",
        default=str(ARCHIVE_SUBMISSIONS / "submission_p1_max_sample_full.csv"),
    )
    parser.add_argument("--combo", default="p3,p4", help="For mode=combo: p3,p4")
    parser.add_argument("--p3", default=str(ARCHIVE_SUBMISSIONS / "submission_p3_audit_fix.csv"))
    parser.add_argument("--p4", default=str(ARCHIVE_SUBMISSIONS / "submission_p4_empty_context.csv"))
    parser.add_argument("--p1-base", default=str(ARCHIVE_SUBMISSIONS / "submission_p1_max_sample_full.csv"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-chunk", type=int, default=64)
    args = parser.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    q_ids = None
    if args.q_ids_file:
        q_ids = set(pd.read_csv(ROOT / args.q_ids_file)["q_id"].astype(int).tolist())

    sample_p = ROOT / args.sample
    full_p = ROOT / args.full

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        hf_home = Path.home() / ".cache" / "huggingface"
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))

    if args.mode == "max":
        df, meta = build_max_sample_full(
            sample_path=sample_p,
            full_path=full_p,
            gold_path=ROOT / args.gold,
            q_ids=q_ids,
            batch_size=args.batch_size,
            score_chunk=args.score_chunk,
        )
    elif args.mode == "max-fast":
        df, meta = build_max_sample_full_fast(
            sample_path=sample_p,
            full_path=full_p,
            gold_path=ROOT / args.gold,
            q_ids=q_ids,
        )
    elif args.mode == "sample_weak":
        df, meta = build_sample_weak_full(sample_path=sample_p, full_path=full_p)
    elif args.mode == "empty_context":
        df, meta = build_empty_context(full_p, ROOT / args.empty_qids)
    else:
        base = pd.read_csv(ROOT / args.p1_base)
        parts = {p.strip() for p in args.combo.split(",")}
        p3_path = ROOT / args.p3 if "p3" in parts else None
        p4_path = ROOT / args.p4 if "p4" in parts else None
        df, meta = apply_combo_p6(base, p3_path=p3_path, p4_path=p4_path)

    _write_submission(df, out_path)
    print(f"mode={args.mode} meta={meta} -> {out_path}")


if __name__ == "__main__":
    main()
