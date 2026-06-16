#!/usr/bin/env python
"""Generate submission.csv (sync or async)."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25Index
from src.cache import AnswerCache
from src.config import get_settings
from src.index_store import artifacts_fresh
from src.llm import (
    LLMPaymentRequired,
    close_client,
    generate_batch_async,
    llm_configured,
    make_async_client,
    make_client,
)
from src.answerability import should_refuse
from src.fallbacks import is_refusal
from src.refusal_policy import should_refuse_from_retrieval
from src.refusal_rules import should_refuse_before_llm
from src.pipeline import RAGPipeline, compose_answer, load_dense_index

_STOP_FLAG = {"value": False}
_RUN_STATE_PATH = ROOT / "data" / "cache" / "run_state.json"


def _graceful_stop(signum, frame) -> None:
    _STOP_FLAG["value"] = True
    print("\n[STOP] SIGINT received, finishing current work then saving...")


def _patch_run_state(**kwargs) -> None:
    state: dict = {}
    if _RUN_STATE_PATH.exists():
        state = json.loads(_RUN_STATE_PATH.read_text(encoding="utf-8"))
    state.update(kwargs)
    _RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RUN_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_one(pipeline: RAGPipeline, q_id: int, query: str, resume: bool) -> tuple[int, str]:
    if resume and pipeline._cache is not None:
        cached = pipeline._cache.get_by_q_id(q_id)
        if cached is not None:
            return q_id, cached
    return q_id, pipeline.answer(q_id, query, use_cache=resume)


def _save_submission(rows: list[tuple[int, str]], out_path: Path) -> None:
    rows.sort(key=lambda x: x[0])
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        writer.writerows(rows)


def _load_q_ids_filter(path: str | None) -> set[int] | None:
    if not path:
        return None
    df = pd.read_csv(ROOT / path)
    col = "q_id" if "q_id" in df.columns else df.columns[0]
    return {int(x) for x in df[col].tolist()}


def _load_existing_rows(path: Path) -> list[tuple[int, str]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "q_id" not in df.columns or "answer_new" not in df.columns:
        raise ValueError(f"{path}: need columns q_id, answer_new")
    rows = [
        (int(r.q_id), str(r.answer_new))
        for r in df.itertuples(index=False)
    ]
    rows.sort(key=lambda x: x[0])
    return rows


def _validate_partial(out_path: Path, questions_path: Path) -> bool:
    try:
        q = pd.read_csv(questions_path)
        s = pd.read_csv(out_path)
        return len(s) >= len(q) * 0.5 and "answer_new" in s.columns
    except Exception:
        return False


async def run_async(
    pipeline: RAGPipeline,
    items: list[tuple[int, str]],
    *,
    use_resume: bool,
    out_path: Path,
    checkpoint_every: int,
) -> list[tuple[int, str]]:
    settings = get_settings()
    rows: list[tuple[int, str]] = []
    llm_queue: list[tuple[int, str, str, list]] = []

    for q_id, query in tqdm(items, desc="Retrieve"):
        if use_resume and pipeline._cache is not None:
            cached = pipeline._cache.get_by_q_id(q_id)
            if cached is not None and not is_refusal(cached):
                rows.append((q_id, cached))
                continue

        chunks, context, low_conf = pipeline.retrieve_context(query)
        if (
            low_conf
            or should_refuse_from_retrieval(chunks)
            or not context.strip()
            or should_refuse_before_llm(query)
            or should_refuse(query, chunks, context)
        ):
            ans = "Нет ответа"
            rows.append((q_id, ans))
            if pipeline._cache is not None:
                pipeline._cache.set(
                    AnswerCache.make_key(q_id, query, context),
                    q_id,
                    ans,
                )
            continue

        if llm_configured():
            llm_queue.append((q_id, query, context, chunks))
        else:
            ans = compose_answer(query, chunks, context, q_id=q_id)
            rows.append((q_id, ans))
            if pipeline._cache is not None:
                pipeline._cache.set(
                    AnswerCache.make_key(q_id, query, context),
                    q_id,
                    ans,
                )

    if llm_queue:
        client = make_async_client()
        try:
            llm_results = await generate_batch_async(
                [(a, b, c) for a, b, c, _ in llm_queue],
                client=client,
                concurrency=settings.llm_concurrency,
            )
        finally:
            await client.close()

        chunk_map = {qid: ch for qid, _, _, ch in llm_queue}
        ctx_map = {qid: ctx for qid, _, ctx, _ in llm_queue}
        query_map = {qid: qu for qid, qu, _, _ in llm_queue}
        for q_id, raw in llm_results:
            ans = compose_answer(
                query_map[q_id],
                chunk_map[q_id],
                ctx_map[q_id],
                raw_llm=raw,
                q_id=q_id,
            )
            rows.append((q_id, ans))
            if pipeline._cache is not None:
                pipeline._cache.set(
                    AnswerCache.make_key(q_id, query_map[q_id], ctx_map[q_id]),
                    q_id,
                    ans,
                    flush=True,
                )

    if pipeline._cache is not None:
        pipeline._cache.flush()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        metavar="CSV",
        help="Skip q_id already in this CSV; merge new answers into --output",
    )
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Async YandexGPT with semaphore",
    )
    parser.add_argument("--no-lock", action="store_true", help="Skip single-instance lock")
    parser.add_argument("--phase", default=None, help="Phase id for run_state.json")
    parser.add_argument("--step", default=None, help="Step id for run_state.json")
    parser.add_argument(
        "--q-ids-file",
        type=str,
        default=None,
        help="CSV with q_id column — process only these questions",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save partial output every N questions (default 100)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _graceful_stop)

    settings = get_settings()
    if not artifacts_fresh(settings.websites_path, settings.artifacts_path):
        if not settings.tfidf_path.exists() and not settings.faiss_path.exists():
            print("Index not found. Run: python scripts/build_index.py --fastembed")
            sys.exit(1)

    if not llm_configured():
        print(f"LLM not configured for provider={settings.llm_provider!r}")
        sys.exit(1)

    def _run() -> None:
        rs = {}
        if _RUN_STATE_PATH.exists():
            rs = json.loads(_RUN_STATE_PATH.read_text(encoding="utf-8"))
        phase = args.phase or rs.get("phase", "phase0")
        step = args.step or rs.get("step", "run")
        _patch_run_state(
            phase=phase,
            step=step,
            last_run=datetime.now().strftime("%Y-%m-%d %H:%M"),
            comment=f"generate started, output={args.output}",
        )

        questions = pd.read_csv(settings.questions_path)
        if args.limit:
            questions = questions.head(args.limit)

        dense_index = load_dense_index()
        bm25_index = BM25Index.load(settings.bm25_path)
        cache = (
            None
            if args.refresh
            else AnswerCache(
                settings.cache_db_path,
                batch_size=settings.cache_batch_size,
            )
            if args.resume
            else None
        )

        all_items = [(int(r.q_id), str(r.query)) for r in questions.itertuples()]
        q_filter = _load_q_ids_filter(args.q_ids_file)
        if q_filter is not None:
            all_items = [(qid, q) for qid, q in all_items if qid in q_filter]
            print(f"Filtered to {len(all_items)} q_id from {args.q_ids_file}")
        out_path = ROOT / args.output
        use_resume = args.resume and not args.refresh
        checkpoint_every = max(1, args.checkpoint_every)

        existing_rows: list[tuple[int, str]] = []
        done_ids: set[int] = set()
        if args.continue_from:
            cont_path = ROOT / args.continue_from
            existing_rows = _load_existing_rows(cont_path)
            done_ids = {qid for qid, _ in existing_rows}
            print(f"Continue from {cont_path.name}: {len(done_ids)} q_id already done")
        items = [(qid, q) for qid, q in all_items if qid not in done_ids]

        def _save_merged(new_part: dict[int, str] | list[tuple[int, str]]) -> None:
            merged = {qid: ans for qid, ans in existing_rows}
            if isinstance(new_part, dict):
                merged.update(new_part)
            else:
                merged.update(new_part)
            _save_submission(sorted(merged.items()), out_path)

        if args.use_async:
            pipeline = RAGPipeline(dense_index, bm25_index, cache=cache, llm_client=None)
            print(f"Async mode, concurrency={settings.llm_concurrency}")
            rows = asyncio.run(
                run_async(
                    pipeline,
                    items,
                    use_resume=use_resume,
                    out_path=out_path,
                    checkpoint_every=checkpoint_every,
                )
            )
        else:
            llm_client = make_client()
            pipeline = RAGPipeline(
                dense_index, bm25_index, cache=cache, llm_client=llm_client
            )
            workers = args.workers or settings.llm_concurrency
            new_rows: list[tuple[int, str]] = []
            print(f"Sync mode, workers={workers}, remaining={len(items)}")

            if not items:
                _save_merged([])
            elif workers <= 1:
                for i, (q_id, query) in enumerate(tqdm(items, desc="Generating"), 1):
                    new_rows.append(process_one(pipeline, q_id, query, use_resume))
                    if i % checkpoint_every == 0:
                        _save_merged(dict(new_rows))
                    if _STOP_FLAG["value"]:
                        _save_merged(dict(new_rows))
                        print(f"[STOP] saved {len(new_rows)} new rows, exiting")
                        sys.exit(130)
                _save_merged(dict(new_rows))
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {
                        pool.submit(process_one, pipeline, q_id, query, use_resume): q_id
                        for q_id, query in items
                    }
                    for i, fut in enumerate(
                        tqdm(as_completed(futures), total=len(futures), desc="Generating"),
                        1,
                    ):
                        q_id, ans = fut.result()
                        new_rows.append((q_id, ans))
                        if i % checkpoint_every == 0:
                            _save_merged(dict(new_rows))
                        if _STOP_FLAG["value"]:
                            _save_merged(dict(new_rows))
                            print(f"[STOP] saved {len(new_rows)} new rows, exiting")
                            sys.exit(130)
                _save_merged(dict(new_rows))
            close_client(llm_client)

        if cache is not None:
            cache.close()

        if args.use_async:
            if args.continue_from:
                _save_merged({qid: ans for qid, ans in rows})
            else:
                _save_submission(rows, out_path)
        final = pd.read_csv(out_path)
        print(f"Saved {len(final)} answers to {out_path}")
        if _validate_partial(out_path, settings.questions_path):
            print("Checkpoint validation: OK")
        else:
            print("Checkpoint validation: WARN (check file manually)")

        _patch_run_state(
            comment=f"generate done, rows={len(final)}, output={args.output}",
        )
        subprocess_mod = __import__("subprocess")
        subprocess_mod.run(
            [sys.executable, str(ROOT / "scripts" / "update_state.py")],
            check=False,
        )

    if args.no_lock:
        _run()
    else:
        from src.process_lock import exclusive_pipeline_lock

        with exclusive_pipeline_lock():
            _run()


if __name__ == "__main__":
    main()
