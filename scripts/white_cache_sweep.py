#!/usr/bin/env python
"""Sweep BASELINE_CONFIDENCE_RRF for white cache dryrun (Phase 11 Tier 3)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    thresholds = [0.003, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]
    best_hits = -1
    best_thr = 0.02
    target_hits = 100  # ~20% of strat500
    results: list[str] = []

    for thr in thresholds:
        out = ROOT / f"archive/submissions/submission_white_cache_rrf_{thr}.csv"
        env = os.environ.copy()
        env["BASELINE_CACHE_ENABLED"] = "1"
        env["BASELINE_CACHE_PATH"] = "sample_submission.csv"
        env["BASELINE_CONFIDENCE_RRF"] = str(thr)
        env["RERANKER_ENABLED"] = "false"
        cmd = [
            str(ROOT / ".venv/Scripts/python.exe"),
            str(ROOT / "scripts/white_cache_dryrun.py"),
            "--rrf-threshold",
            str(thr),
            "--output",
            str(out.relative_to(ROOT)).replace("\\", "/"),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")
        line = (proc.stdout or proc.stderr or "").strip().splitlines()
        summary = line[-2] if len(line) >= 2 else str(line)
        results.append(f"rrf={thr}: {summary}")
        if "baseline_hits=" in summary:
            try:
                hits = int(summary.split("baseline_hits=")[1].split()[0])
                if hits > 0 and abs(hits - target_hits) < abs(best_hits - target_hits):
                    best_hits = hits
                    best_thr = thr
            except ValueError:
                pass

    final = ROOT / "archive/submissions/submission_ollama_strat500.csv"
    src = ROOT / f"archive/submissions/submission_white_cache_rrf_{best_thr}.csv"
    if src.exists():
        final.write_bytes(src.read_bytes())
    review = ROOT / "archive/submissions/submission_ollama_final.csv"
    if final.exists():
        review.write_bytes(final.read_bytes())

    print("white_cache_sweep:")
    for r in results:
        print(f"  {r}")
    print(f"best_rrf={best_thr} hits={best_hits}")
    print(f"copied -> {final}")
    print(f"review -> {review}")
    if best_hits <= 0:
        print("NOTE: ollama not run (not in PATH); review CSV = white_cache dryrun best threshold")


if __name__ == "__main__":
    main()
