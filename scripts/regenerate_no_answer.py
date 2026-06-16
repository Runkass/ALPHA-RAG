#!/usr/bin/env python
"""Regenerate answers only for questions with refusal in submission or cache."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Меньше пик RAM и конкуренции с ONNX при массовом прогоне
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.bm25 import BM25Index
from src.cache import AnswerCache
from src.config import get_settings
from src.fallbacks import is_refusal
from src.llm import close_client, llm_configured, make_client
from src.pipeline import RAGPipeline, load_dense_index


def _write_progress(done: int, total: int, *, state: str = "running") -> None:
    path = ROOT / "data" / "regenerate.progress"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{done}/{total} {state}\n", encoding="utf-8")


def _save_submission(rows: list[tuple[int, str]], out_path: Path) -> None:
    rows.sort(key=lambda x: x[0])
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["q_id", "answer_new"])
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--output", default=None, help="Defaults to --submission")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="По умолчанию 1 — стабильнее для SQLite/LLM (избегать OOM)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Сохранять submission.csv каждые N ответов (0 = только в конце)",
    )
    parser.add_argument(
        "--also-cache",
        action="store_true",
        help="Also regenerate q_ids with refusal only in cache",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Отключить cross-encoder (меньше RAM при тысячах запросов)",
    )
    args = parser.parse_args()

    if args.no_rerank:
        os.environ["RERANKER_ENABLED"] = "false"
    get_settings.cache_clear()

    settings = get_settings()
    if not llm_configured():
        print("LLM not configured")
        sys.exit(1)

    sub_path = ROOT / args.submission
    out_path = ROOT / (args.output or args.submission)
    sub = pd.read_csv(sub_path)
    questions = pd.read_csv(settings.questions_path)

    mask = sub["answer_new"].astype(str).apply(is_refusal)
    target_ids = set(sub.loc[mask, "q_id"].astype(int).tolist())

    if args.also_cache:
        cache = AnswerCache(settings.cache_db_path)
        for q_id in questions["q_id"].astype(int):
            ans = cache.get_by_q_id(int(q_id))
            if ans and is_refusal(ans):
                target_ids.add(int(q_id))
        cache.close()

    target_ids = sorted(target_ids)
    if args.limit:
        target_ids = target_ids[: args.limit]

    total = len(target_ids)
    print(f"Regenerating {total} questions (refusal in submission)")
    _write_progress(0, total, state="running")

    progress = {"done": 0}
    try:
        _run_regenerate(args, settings, sub, questions, out_path, target_ids, total, progress)
    except Exception:
        _write_progress(progress["done"], total, state="error")
        raise


def _run_regenerate(
    args: argparse.Namespace,
    settings,
    sub: pd.DataFrame,
    questions: pd.DataFrame,
    out_path: Path,
    target_ids: list[int],
    total: int,
    progress: dict[str, int],
) -> None:
    cache = AnswerCache(settings.cache_db_path, batch_size=settings.cache_batch_size)

    print("Loading indices and models...", flush=True)
    qmap = questions.set_index("q_id")["query"].to_dict()
    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    if settings.reranker_enabled:
        from src.reranker import rerank_optional

        rerank_optional("warmup", ["warmup"], 1)
    llm = make_client()
    pipeline = RAGPipeline(dense, bm25, cache=cache, llm_client=llm)
    print(f"Ready. Processing {total} questions...", flush=True)

    answers: dict[int, str] = dict(zip(sub["q_id"].astype(int), sub["answer_new"].astype(str)))
    items = [(qid, str(qmap[qid])) for qid in target_ids if qid in qmap]

    if args.workers <= 1:
        ck = args.checkpoint_every
        try:
            for i, (q_id, query) in enumerate(tqdm(items, desc="Regenerate"), 1):
                answers[q_id] = pipeline.answer(q_id, query, use_cache=True, force=True)
                progress["done"] = i
                _write_progress(i, total, state="running")
                if ck and i % ck == 0:
                    _save_submission(
                        [(int(x), str(y)) for x, y in answers.items()],
                        out_path,
                    )
                    if pipeline._cache is not None:
                        pipeline._cache.flush()
        finally:
            close_client(llm)
            if pipeline._cache is not None:
                pipeline._cache.flush()
            cache.close()
    else:
        print("WARN: workers>1 может нестабильно грузить RAM/LLM; предпочтительно --workers 1")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        pipeline.answer, q_id, query, use_cache=True, force=True
                    ): q_id
                    for q_id, query in items
                }
                done_count = 0
                ck = args.checkpoint_every
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Regenerate"):
                    q_id = futures[fut]
                    answers[q_id] = fut.result()
                    done_count += 1
                    progress["done"] = done_count
                    _write_progress(done_count, total, state="running")
                    if ck and done_count % ck == 0:
                        _save_submission(
                            [(int(x), str(y)) for x, y in answers.items()],
                            out_path,
                        )
                        if pipeline._cache is not None:
                            pipeline._cache.flush()
        finally:
            close_client(llm)
            if pipeline._cache is not None:
                pipeline._cache.flush()
            cache.close()

    rows = [(int(qid), str(ans)) for qid, ans in answers.items()]
    _save_submission(rows, out_path)
    _write_progress(total, total, state="done")
    refused = sum(1 for _, a in rows if is_refusal(a))
    print(f"Saved {len(rows)} rows to {out_path}")
    print(f"Still refusal: {refused} ({100 * refused / len(rows):.1f}%)")


if __name__ == "__main__":
    main()