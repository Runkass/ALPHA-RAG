#!/usr/bin/env python
"""Build sample-based submission with targeted q_id patches (Phase 9)."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fallbacks import extractive_fallback  # noqa: E402
from src.llm import generate_slot_answer, llm_configured, make_client, normalize_answer  # noqa: E402
from src.metrics.recall_l import is_refusal  # noqa: E402
from src.pipeline import RAGPipeline, load_dense_index  # noqa: E402
from src.bm25 import BM25Index  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.project_paths import BACKUP_FULL  # noqa: E402
from src.prompts_slot import build_slot1_messages, build_slot2_messages  # noqa: E402
from src.retrieval import format_context  # noqa: E402
from src.submission_rules import (  # noqa: E402
    REFUSAL_TEXT,
    fix_to_refusal,
    is_faq_dump,
    is_garbage_answer,
    is_protected_q_id,
    is_protected_short_answer,
    is_weak_sample_answer,
    load_protected_q_ids,
)


def _write_submission(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["q_id", "answer_new"]].sort_values("q_id")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        for row in out.itertuples(index=False):
            writer.writerow([int(row.q_id), str(row.answer_new)])


def _load_q_ids(path: Path) -> set[int]:
    df = pd.read_csv(path)
    col = "q_id" if "q_id" in df.columns else df.columns[0]
    return set(df[col].astype(int).tolist())


def _faq_trim(text: str) -> str:
    t = str(text).strip()
    if not is_faq_dump(t):
        return t
    if "] / " in t:
        parts = t.split("] / ", 1)
        if len(parts) > 1 and parts[1].strip():
            return parts[1].strip()[:900]
    m = re.search(r"[.!?]\s", t)
    if m:
        return t[: m.end()].strip()
    return REFUSAL_TEXT


def _heuristic_patch(text: str) -> str:
    fixed = fix_to_refusal(text)
    if fixed == REFUSAL_TEXT:
        return fixed
    if is_faq_dump(fixed):
        return _faq_trim(fixed)
    return fixed


def patch_heuristic(base: pd.DataFrame, q_ids: set[int] | None) -> tuple[pd.DataFrame, int]:
    out = base.copy()
    changed = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q_ids is not None and q not in q_ids:
            continue
        ans = str(row["answer_new"])
        if not is_weak_sample_answer(ans) and not is_faq_dump(ans):
            continue
        new = _heuristic_patch(ans)
        if new != ans:
            out.at[i, "answer_new"] = new
            changed += 1
    return out, changed


def patch_extractive(
    base: pd.DataFrame,
    q_ids: set[int],
    *,
    weakness_path: Path | None,
    questions: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    out = base.copy()
    extractive_map: dict[int, str] = {}
    if weakness_path and weakness_path.exists():
        wdf = pd.read_parquet(weakness_path)
        if "extractive_answer" in wdf.columns:
            extractive_map = dict(
                zip(wdf["q_id"].astype(int), wdf["extractive_answer"].astype(str))
            )

    need_retrieve = q_ids - set(extractive_map.keys())
    retrieve_map: dict[int, str] = {}
    if need_retrieve:
        qmap = dict(zip(questions["q_id"].astype(int), questions["query"].astype(str)))
        settings = get_settings()
        dense = load_dense_index()
        bm25 = BM25Index.load(settings.bm25_path)
        pipeline = RAGPipeline(dense, bm25, cache=None)
        for q in need_retrieve:
            query = qmap.get(q, "")
            chunks, _ctx, _low = pipeline.retrieve_context(query)
            retrieve_map[q] = extractive_fallback(chunks, max_len=260)

    changed = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q not in q_ids:
            continue
        new = extractive_map.get(q) or retrieve_map.get(q)
        if not new or is_refusal(new):
            continue
        if str(row["answer_new"]) != new:
            out.at[i, "answer_new"] = new
            changed += 1
    return out, changed


def patch_full(
    base: pd.DataFrame,
    q_ids: set[int],
    full_path: Path,
) -> tuple[pd.DataFrame, int]:
    full_map = dict(
        zip(
            pd.read_csv(full_path)["q_id"].astype(int),
            pd.read_csv(full_path)["answer_new"].astype(str),
        )
    )
    out = base.copy()
    changed = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q not in q_ids:
            continue
        full_ans = full_map.get(q)
        if not full_ans or is_refusal(full_ans):
            continue
        if str(row["answer_new"]) != full_ans:
            out.at[i, "answer_new"] = full_ans
            changed += 1
    return out, changed


def patch_apply(base: pd.DataFrame, patch_path: Path, q_ids: set[int] | None) -> tuple[pd.DataFrame, int]:
    patch = pd.read_csv(patch_path)
    patch_map = dict(zip(patch["q_id"].astype(int), patch["answer_new"].astype(str)))
    out = base.copy()
    changed = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q not in patch_map:
            continue
        if q_ids is not None and q not in q_ids:
            continue
        new = patch_map[q]
        if str(row["answer_new"]) != new:
            out.at[i, "answer_new"] = new
            changed += 1
    return out, changed


def _trim_max_len(text: str, max_len: int) -> str:
    t = str(text).strip()
    if len(t) <= max_len:
        return t
    cut = t[: max_len - 3]
    last = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind(" "))
    if last > max_len // 3:
        cut = cut[: last + 1].rstrip()
    return cut + "..."


def _filter_patch_q_ids(
    q_ids: set[int],
    base: pd.DataFrame,
    *,
    deny: set[int],
    deny_protected: bool,
    require_refusal: bool = False,
    require_verbose: bool = False,
) -> set[int]:
    sample_map = dict(zip(base["q_id"].astype(int), base["answer_new"].astype(str)))
    from src.submission_rules import is_verbose_sample_answer

    out: set[int] = set()
    for q in q_ids:
        if is_protected_q_id(q, deny):
            continue
        ans = sample_map.get(q, "")
        if deny_protected and is_protected_short_answer(ans):
            continue
        if require_refusal and not is_refusal(ans):
            continue
        if require_verbose and not is_verbose_sample_answer(ans):
            continue
        out.add(q)
    return out


def _get_pipeline() -> RAGPipeline:
    settings = get_settings()
    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    return RAGPipeline(dense, bm25, cache=None)


def patch_false_refuse(
    base: pd.DataFrame,
    q_ids: set[int],
    questions: pd.DataFrame,
    *,
    max_len: int = 240,
    client=None,
) -> tuple[pd.DataFrame, int]:
    if not llm_configured():
        raise SystemExit("LLM not configured for false_refuse mode")
    own_client = client is None
    if own_client:
        client = make_client()
    qmap = dict(zip(questions["q_id"].astype(int), questions["query"].astype(str)))
    pipeline = _get_pipeline()
    out = base.copy()
    changed = skipped = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q not in q_ids:
            continue
        if not is_refusal(str(row["answer_new"])):
            continue
        query = qmap.get(q, "")
        chunks, _ctx, low_confidence = pipeline.retrieve_context(query)
        if low_confidence or not chunks:
            skipped += 1
            continue
        context = format_context(chunks)
        raw = generate_slot_answer(build_slot1_messages(query, context), client=client)
        new = _trim_max_len(normalize_answer(raw), max_len)
        if is_refusal(new) or is_garbage_answer(new):
            skipped += 1
            continue
        if str(row["answer_new"]) != new:
            out.at[i, "answer_new"] = new
            changed += 1
    if own_client:
        from src.llm import close_client

        close_client(client)
    print(f"false_refuse skipped={skipped}")
    return out, changed


def patch_compress(
    base: pd.DataFrame,
    q_ids: set[int],
    questions: pd.DataFrame,
    *,
    max_len: int = 240,
    min_shrink_ratio: float = 0.7,
    client=None,
) -> tuple[pd.DataFrame, int]:
    if not llm_configured():
        raise SystemExit("LLM not configured for compress mode")
    own_client = client is None
    if own_client:
        client = make_client()
    qmap = dict(zip(questions["q_id"].astype(int), questions["query"].astype(str)))
    pipeline = _get_pipeline()
    out = base.copy()
    changed = skipped = 0
    for i, row in out.iterrows():
        q = int(row["q_id"])
        if q not in q_ids:
            continue
        sample_ans = str(row["answer_new"])
        if is_refusal(sample_ans) or is_protected_short_answer(sample_ans):
            continue
        query = qmap.get(q, "")
        chunks, _ctx, _low = pipeline.retrieve_context(query)
        context = format_context(chunks) if chunks else ""
        raw = generate_slot_answer(
            build_slot2_messages(query, context, sample_ans), client=client
        )
        new = _trim_max_len(normalize_answer(raw), max_len)
        if is_garbage_answer(new) or is_refusal(new):
            skipped += 1
            continue
        if len(new) >= len(sample_ans) * min_shrink_ratio:
            skipped += 1
            continue
        if sample_ans != new:
            out.at[i, "answer_new"] = new
            changed += 1
    if own_client:
        from src.llm import close_client

        close_client(client)
    print(f"compress skipped={skipped}")
    return out, changed


def patch_combo_slots(base: pd.DataFrame, patch_paths: list[Path]) -> tuple[pd.DataFrame, int]:
    out = base.copy()
    sample_map = dict(zip(base["q_id"].astype(int), base["answer_new"].astype(str)))
    changed = 0
    for pp in patch_paths:
        patch = pd.read_csv(pp)
        patch_map = dict(zip(patch["q_id"].astype(int), patch["answer_new"].astype(str)))
        for i, row in out.iterrows():
            q = int(row["q_id"])
            if q not in patch_map:
                continue
            new = patch_map[q]
            if sample_map.get(q) == new:
                continue
            if str(row["answer_new"]) != new:
                out.at[i, "answer_new"] = new
                changed += 1
    return out, changed


patch_combo = patch_combo_slots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="sample_submission.csv")
    parser.add_argument("--mode", required=True, choices=(
        "heuristic", "extractive", "full", "apply", "combo", "combo_slots",
        "false_refuse", "compress",
    ))
    parser.add_argument("--q-ids-file", default=None)
    parser.add_argument("--weakness", default="data/cache/sample_weakness.parquet")
    parser.add_argument("--full", default=str(BACKUP_FULL.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--patch", default=None, help="patch CSV for apply/combo")
    parser.add_argument("--patches", nargs="*", default=[], help="multiple patches for combo")
    parser.add_argument("--output", required=True)
    parser.add_argument("--questions", default="questions.csv")
    parser.add_argument("--max-len", type=int, default=240)
    parser.add_argument("--deny-protected", action="store_true", default=True)
    parser.add_argument("--no-deny-protected", dest="deny_protected", action="store_false")
    parser.add_argument("--deny-file", default="data/cache/edit_batches/s2_do_not_touch.csv")
    parser.add_argument("--min-shrink", type=float, default=0.7)
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    deny = load_protected_q_ids(_p(args.deny_file)) if args.deny_file else set()
    base = pd.read_csv(_p(args.base))
    q_ids = _load_q_ids(_p(args.q_ids_file)) if args.q_ids_file else None
    questions = pd.read_csv(_p(args.questions))

    if args.mode == "heuristic":
        if q_ids is None:
            wpath = _p(args.weakness)
            if wpath.exists():
                wdf = pd.read_parquet(wpath)
                q_ids = set(wdf[wdf["weak_heuristic"]]["q_id"].astype(int).tolist())
            else:
                q_ids = set(
                    base.loc[
                        base["answer_new"].astype(str).map(is_weak_sample_answer), "q_id"
                    ].astype(int)
                )
        out, changed = patch_heuristic(base, q_ids)
    elif args.mode == "extractive":
        if not q_ids:
            raise SystemExit("--q-ids-file required for extractive mode")
        out, changed = patch_extractive(
            base, q_ids, weakness_path=_p(args.weakness), questions=questions
        )
    elif args.mode == "full":
        if not q_ids:
            raise SystemExit("--q-ids-file required for full mode")
        out, changed = patch_full(base, q_ids, _p(args.full))
    elif args.mode == "apply":
        if not args.patch:
            raise SystemExit("--patch required for apply mode")
        out, changed = patch_apply(base, _p(args.patch), q_ids)
    elif args.mode == "false_refuse":
        if not q_ids:
            raise SystemExit("--q-ids-file required for false_refuse mode")
        q_ids = _filter_patch_q_ids(
            q_ids, base, deny=deny, deny_protected=args.deny_protected, require_refusal=True
        )
        out, changed = patch_false_refuse(
            base, q_ids, questions, max_len=args.max_len
        )
    elif args.mode == "compress":
        if not q_ids:
            raise SystemExit("--q-ids-file required for compress mode")
        q_ids = _filter_patch_q_ids(
            q_ids, base, deny=deny, deny_protected=args.deny_protected, require_verbose=True
        )
        out, changed = patch_compress(
            base,
            q_ids,
            questions,
            max_len=args.max_len,
            min_shrink_ratio=args.min_shrink,
        )
    elif args.mode == "combo_slots":
        paths = [_p(p) for p in (args.patches or ([args.patch] if args.patch else []))]
        if not paths:
            raise SystemExit("--patches or --patch required for combo_slots mode")
        out, changed = patch_combo_slots(base, paths)
    else:
        paths = [_p(p) for p in (args.patches or ([args.patch] if args.patch else []))]
        if not paths:
            raise SystemExit("--patches or --patch required for combo mode")
        out, changed = patch_combo(base, paths)

    out_path = _p(args.output)
    _write_submission(out, out_path)
    print(f"mode={args.mode} changed={changed} q_ids={len(q_ids) if q_ids else 'all'} -> {out_path}")


if __name__ == "__main__":
    main()
