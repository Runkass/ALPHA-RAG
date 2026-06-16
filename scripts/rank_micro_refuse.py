#!/usr/bin/env python
"""Export non-overlap refuse micro slices for Phase 11 blitz (refuse 2.0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.submission_rules import load_protected_q_ids  # noqa: E402

BATCHES_DIR = ROOT / "data" / "cache" / "edit_batches"
DEFAULT_SOURCE = BATCHES_DIR / "refusal_high_conf_top100.csv"


def _write_slice(df: pd.DataFrame, start: int, end: int, path: Path) -> int:
    sl = df.iloc[start:end].copy()
    sl[["q_id"]].to_csv(path, index=False)
    return len(sl)


def rank_slices(
    source: Path,
    *,
    overlap_min: float = 0.35,
    slice_size: int = 5,
    max_rank: int = 20,
    deny_file: Path | None = None,
    out_dir: Path = BATCHES_DIR,
) -> dict[str, int]:
    df = pd.read_csv(source)
    if "keyword_overlap" in df.columns:
        df = df[df["keyword_overlap"].astype(float) >= overlap_min]
    if "refusal_confidence" in df.columns:
        df = df.sort_values("refusal_confidence", ascending=False)
    elif "q_id" in df.columns:
        df = df.sort_values("q_id")

    deny = load_protected_q_ids(str(deny_file)) if deny_file else set()
    if deny:
        df = df[~df["q_id"].astype(int).isin(deny)]

    df = df.head(max_rank).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for i in range(0, min(len(df), max_rank), slice_size):
        end = min(i + slice_size, max_rank)
        tag = f"r{i + 1}_{end}"
        path = out_dir / f"refusal_micro_{tag}.csv"
        n = _write_slice(df, i, end, path)
        counts[str(path.name)] = n

    verb_src = ROOT / "data/cache/edit_batches/verbosity_top100.csv"
    if verb_src.exists():
        vdf = pd.read_csv(verb_src).head(10)
        for i in range(0, min(len(vdf), 10), slice_size):
            end = min(i + slice_size, 10)
            path = out_dir / f"verbosity_micro_s{i + 1}_{end}.csv"
            vdf.iloc[i:end][["q_id"]].to_csv(path, index=False)
            counts[str(path.name)] = end - i

    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE.relative_to(ROOT)).replace("\\", "/"))
    parser.add_argument("--overlap-min", type=float, default=0.35)
    parser.add_argument("--slice-size", type=int, default=5)
    parser.add_argument("--max-rank", type=int, default=20)
    parser.add_argument("--out-dir", default="data/cache/edit_batches")
    parser.add_argument("--deny-file", default="data/cache/edit_batches/s2_do_not_touch.csv")
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    counts = rank_slices(
        _p(args.source),
        overlap_min=args.overlap_min,
        slice_size=args.slice_size,
        max_rank=args.max_rank,
        deny_file=_p(args.deny_file),
        out_dir=_p(args.out_dir),
    )
    for name, n in counts.items():
        print(f"{name}: {n} q_id")


if __name__ == "__main__":
    main()
