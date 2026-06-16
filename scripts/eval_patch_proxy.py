#!/usr/bin/env python
"""Local proxy metrics on changed q_id only (Phase 9 — not Recall-L vs sample gold)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import _WORD_RE  # noqa: E402
from src.submission_rules import is_faq_dump, is_garbage_answer  # noqa: E402


def _overlap(answer: str, chunk_text: str) -> float:
    a_words = set(_WORD_RE.findall(str(answer).lower()))
    c_words = set(_WORD_RE.findall(str(chunk_text).lower()))
    if not a_words or not c_words:
        return 0.0
    return len(a_words & c_words) / len(a_words)


def _load_q_ids(path: Path | None, base: pd.DataFrame, patched: pd.DataFrame) -> list[int]:
    if path and path.exists():
        df = pd.read_csv(path)
        col = "q_id" if "q_id" in df.columns else df.columns[0]
        return df[col].astype(int).tolist()
    m = base.merge(patched, on="q_id", suffixes=("_base", "_patched"))
    changed = m[m["answer_new_base"].astype(str) != m["answer_new_patched"].astype(str)]
    return changed["q_id"].astype(int).tolist()


def eval_proxy(
    base_path: Path,
    patched_path: Path,
    q_ids_file: Path | None,
    weakness_path: Path | None,
) -> dict:
    base = pd.read_csv(base_path)
    patched = pd.read_csv(patched_path)
    q_ids = _load_q_ids(q_ids_file, base, patched)
    if not q_ids:
        return {"changed": 0, "q_ids": 0}

    chunk_map: dict[int, str] = {}
    if weakness_path and weakness_path.exists():
        wdf = pd.read_parquet(weakness_path)
        if "top_chunk_text" in wdf.columns:
            chunk_map = dict(zip(wdf["q_id"].astype(int), wdf["top_chunk_text"].astype(str)))

    bmap = dict(zip(base["q_id"].astype(int), base["answer_new"].astype(str)))
    pmap = dict(zip(patched["q_id"].astype(int), patched["answer_new"].astype(str)))

    overlaps_base: list[float] = []
    overlaps_patch: list[float] = []
    lens_base: list[int] = []
    lens_patch: list[int] = []
    garbage = faq = 0
    examples: list[tuple[int, str, str]] = []

    for q in q_ids:
        b = bmap.get(q, "")
        p = pmap.get(q, b)
        if b == p:
            continue
        chunk = chunk_map.get(q, "")
        overlaps_base.append(_overlap(b, chunk))
        overlaps_patch.append(_overlap(p, chunk))
        lens_base.append(len(b))
        lens_patch.append(len(p))
        if is_garbage_answer(p):
            garbage += 1
        if is_faq_dump(p):
            faq += 1
        if len(examples) < 10:
            examples.append((q, b[:120], p[:120]))

    n = len(overlaps_base)
    if n == 0:
        return {"changed": 0, "q_ids": len(q_ids)}

    med_b = sorted(overlaps_base)[n // 2]
    med_p = sorted(overlaps_patch)[n // 2]
    return {
        "q_ids": len(q_ids),
        "changed": n,
        "median_overlap_base": round(med_b, 4),
        "median_overlap_patched": round(med_p, 4),
        "overlap_delta": round(med_p - med_b, 4),
        "median_len_base": sorted(lens_base)[n // 2],
        "median_len_patched": sorted(lens_patch)[n // 2],
        "garbage": garbage,
        "faq_dump": faq,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="sample_submission.csv")
    parser.add_argument("--patched", required=True)
    parser.add_argument("--q-ids-file", default=None)
    parser.add_argument("--weakness", default="data/cache/sample_weakness.parquet")
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    meta = eval_proxy(_p(args.base), _p(args.patched), _p(args.q_ids_file) if args.q_ids_file else None, _p(args.weakness))
    print(f"proxy: {meta}")
    if meta.get("examples"):
        print("examples (q_id | base | patched):")
        for q, b, p in meta["examples"]:
            print(f"  {q} | {b} | {p}")


if __name__ == "__main__":
    main()
