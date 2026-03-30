"""
In-memory vector store with FAISS-backed similarity search.
"""

from __future__ import annotations
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    index: int
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """
    Simple vector store that keeps texts, metadata, and a FAISS index in memory.
    Supports add, search, save, and load operations.
    """

    def __init__(self, embedding_dim: int, metric: str = "cosine"):
        """
        Args:
            embedding_dim: Dimensionality of the embedding vectors.
            metric:        'cosine' (inner product on normalised vecs) or 'l2'.
        """
        self.embedding_dim = embedding_dim
        self.metric = metric

        if metric == "cosine":
            self.index = faiss.IndexFlatIP(embedding_dim)   # inner-product = cosine for L2-normed vecs
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)

        self.texts: list[str] = []
        self.metadata: list[dict] = []

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """
        Add pre-computed vectors and their source texts to the store.

        Args:
            vectors:  float32 array of shape (N, embedding_dim).
            texts:    List of N source strings.
            metadata: Optional list of N dicts with arbitrary per-item metadata.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        if metadata is None:
            metadata = [{} for _ in texts]

        self.index.add(vectors)
        self.texts.extend(texts)
        self.metadata.extend(metadata)

    def reset(self) -> None:
        """Clear all stored vectors and texts."""
        self.index.reset()
        self.texts.clear()
        self.metadata.clear()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Return the top-k most similar documents to a query vector.

        Args:
            query_vector: 1-D float32 array of length embedding_dim.
            top_k:        Number of results to return.

        Returns:
            List of SearchResult sorted by score descending.
        """
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]

        top_k = min(top_k, len(self.texts))
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                SearchResult(
                    index=int(idx),
                    score=float(score),
                    text=self.texts[idx],
                    metadata=self.metadata[idx],
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save index and texts to disk (creates <path>.faiss and <path>.npz)."""
        faiss.write_index(self.index, f"{path}.faiss")
        np.savez(
            f"{path}.npz",
            texts=np.array(self.texts, dtype=object),
            metadata=np.array(self.metadata, dtype=object),
        )
        print(f"Saved {len(self.texts)} vectors to '{path}'")

    @classmethod
    def load(cls, path: str, embedding_dim: int, metric: str = "cosine") -> VectorStore:
        """Load a previously saved store from disk."""
        store = cls(embedding_dim=embedding_dim, metric=metric)
        store.index = faiss.read_index(f"{path}.faiss")
        data = np.load(f"{path}.npz", allow_pickle=True)
        store.texts = list(data["texts"])
        store.metadata = list(data["metadata"])
        print(f"Loaded {len(store.texts)} vectors from '{path}'")
        return store

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.texts)

    def __repr__(self) -> str:
        return (
            f"VectorStore(items={len(self)}, dim={self.embedding_dim}, metric={self.metric})"
        )
