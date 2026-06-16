"""TF-IDF vector index (no HuggingFace download required)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class TfidfIndex:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix,
        neighbors: NearestNeighbors,
        chunks: pd.DataFrame,
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.neighbors = neighbors
        self.chunks = chunks

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_vec = self.vectorizer.transform([query])
        distances, indices = self.neighbors.kneighbors(q_vec, n_neighbors=min(top_k, self.matrix.shape[0]))
        results: list[tuple[int, float]] = []
        for idx, dist in zip(indices[0], distances[0], strict=True):
            if idx < 0:
                continue
            # cosine distance -> similarity
            results.append((int(idx), float(1.0 - dist)))
        return results

    def get_text(self, chunk_idx: int) -> str:
        return str(self.chunks.iloc[chunk_idx]["text"])

    def save(self, path: Path, chunks_path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks.to_parquet(chunks_path, index=False)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                    "neighbors": self.neighbors,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path, chunks_path: Path) -> TfidfIndex:
        with path.open("rb") as f:
            data = pickle.load(f)
        chunks = pd.read_parquet(chunks_path)
        return cls(data["vectorizer"], data["matrix"], data["neighbors"], chunks)

    @classmethod
    def build(cls, texts: list[str], chunks: pd.DataFrame) -> TfidfIndex:
        vectorizer = TfidfVectorizer(
            max_features=40_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        neighbors = NearestNeighbors(n_neighbors=min(50, matrix.shape[0]), metric="cosine")
        neighbors.fit(matrix)
        return cls(vectorizer, matrix, neighbors, chunks)
