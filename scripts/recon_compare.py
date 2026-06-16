#!/usr/bin/env python
"""R4: compare recon vs base vs sample -> rules JSON + q_ids_to_patch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_max_sample_full import _batch_recall_l  # noqa: E402
from src.metrics.recall_l import is_refusal  # noqa: E402
from src.submission_rules import is_faq_dump  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--gold", default="sample_submission.csv")
    parser.add_argument("--out-rules", default="data/cache/recon_rules.json")
    parser.add_argument("--out-qids", default="data/cache/q_ids_to_patch.csv")
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    def _p(s: str) -> Path:
        p = Path(s)
        return p if p.is_absolute() else ROOT / p

    recon = pd.read_csv(_p(args.recon))
    base = pd.read_csv(_p(args.base))
    sample = pd.read_csv(_p(args.sample))
    gold = pd.read_csv(_p(args.gold))

    m = gold[["q_id", "answer_new"]].rename(columns={"answer_new": "answer_new_gold"})
    m = m.merge(
        recon[["q_id", "answer_new"]].rename(columns={"answer_new": "answer_new_recon"}),
        on="q_id",
    )
    m = m.merge(
        base[["q_id", "answer_new"]].rename(columns={"answer_new": "answer_new_base"}),
        on="q_id",
    )
    m = m.merge(
        sample[["q_id", "answer_new"]].rename(columns={"answer_new": "answer_new_sample"}),
        on="q_id",
    )

    import os

    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-multilingual-cased",
        local_files_only=offline,
    )
    golds = m["answer_new_gold"].astype(str).tolist()
    recon_sc = _batch_recall_l(
        m["answer_new_recon"].astype(str).tolist(), golds, tokenizer, batch_size=args.batch_size
    )
    base_sc = _batch_recall_l(
        m["answer_new_base"].astype(str).tolist(), golds, tokenizer, batch_size=args.batch_size
    )
    m = m.assign(recon_score=recon_sc, base_score=base_sc, delta=[r - b for r, b in zip(recon_sc, base_sc)])

    patch = m[m["delta"] >= args.min_delta].copy()
    patch_ids = patch["q_id"].astype(int).tolist()

    rules = {
        "min_delta": args.min_delta,
        "rules": [
            {
                "id": "recon_refusal_base_not",
                "desc": "recon=refusal AND base!=refusal -> Нет ответа",
            },
            {
                "id": "recon_faq_clean_base_faq",
                "desc": "recon not FAQ AND base has FAQ dump -> Нет ответа",
            },
        ],
        "patch_q_ids": patch_ids,
        "n_patch": len(patch_ids),
    }

    out_rules = _p(args.out_rules)
    out_rules.parent.mkdir(parents=True, exist_ok=True)
    out_rules.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_q = _p(args.out_qids)
    pd.DataFrame({"q_id": patch_ids}).to_csv(out_q, index=False)

    print(f"compared={len(m)} patch_q_ids={len(patch_ids)} min_delta={args.min_delta}")
    print(f"rules -> {out_rules}")
    print(f"q_ids -> {out_q}")
    if len(patch):
        top = patch.nlargest(10, "delta")
        for row in top.itertuples(index=False):
            print(
                f"  q={row.q_id} delta={row.delta:.3f} "
                f"base={str(row.answer_new_base)[:60]!r} recon={str(row.answer_new_recon)[:60]!r}"
            )


if __name__ == "__main__":
    main()
