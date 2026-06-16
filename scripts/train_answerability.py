#!/usr/bin/env python
"""Train refusal classifier from sample_submission (offline gold labels)."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.answerability import features_from_retrieval
from src.bm25 import BM25Index
from src.config import get_settings
from src.fallbacks import is_refusal
from src.pipeline import RAGPipeline, load_dense_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="sample_submission.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    questions = pd.read_csv(settings.questions_path)
    gold = pd.read_csv(ROOT / args.gold)
    m = questions.merge(gold, on="q_id")
    if args.limit:
        m = m.head(args.limit)

    dense = load_dense_index()
    bm25 = BM25Index.load(settings.bm25_path)
    pipe = RAGPipeline(dense, bm25, cache=None)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for row in tqdm(m.itertuples(), total=len(m), desc="Features"):
        q_id = int(row.q_id)
        query = str(row.query)
        label = 1 if is_refusal(str(row.answer_new)) else 0
        chunks, context, _ = pipe.retrieve_context(query)
        X_list.append(features_from_retrieval(query, chunks, context))
        y_list.append(label)

    X = np.vstack(X_list)
    y = np.array(y_list)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.15, random_state=42)

    model = LogisticRegression(max_iter=500, class_weight="balanced")
    model.fit(X_tr, y_tr)
    print(classification_report(y_te, model.predict(X_te), target_names=["answer", "refuse"]))

    out = settings.artifacts_path / "answerability.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
