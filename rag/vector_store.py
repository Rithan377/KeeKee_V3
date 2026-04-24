"""
vector_store.py — In-Memory Vector Store
==========================================
Stores document embeddings and supports cosine similarity search.
Two backends:
  - numpy (default): no extra dependencies, works everywhere
  - FAISS (optional): faster for large collections (>10K docs)

All vectors are expected to be L2-normalized (from TextEmbedder),
so cosine similarity = dot product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
# SearchResult
# ─────────────────────────────────────────────
@dataclass
class SearchResult:
    """Single search result from the vector store."""
    document: str           # the text chunk
    score: float            # cosine similarity score [0, 1]
    index: int              # index in the store
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# VectorStore
# ─────────────────────────────────────────────
class VectorStore:
    """
    In-memory vector store with cosine similarity search.

    Default backend is pure numpy (no extra dependencies).
    Set use_faiss=True for FAISS IndexFlatIP acceleration on large datasets.

    All vectors must be L2-normalized (cosine sim = dot product).

    Usage:
        store = VectorStore(dim=384)
        store.add(vectors, documents=["doc1", "doc2"])
        results = store.search(query_vec, top_k=3)
    """

    def __init__(
        self,
        dim: int = 384,
        use_faiss: bool = False,
    ):
        self.dim = dim
        self.use_faiss = use_faiss

        # Storage
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._documents: list[str] = []
        self._metadata: list[dict] = []

        # FAISS index (lazy init)
        self._faiss_index = None
        if use_faiss:
            self._init_faiss()

    def _init_faiss(self):
        """Initialize FAISS index for inner-product search."""
        try:
            import faiss
            self._faiss_index = faiss.IndexFlatIP(self.dim)
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISS backend. "
                "Install with: pip install faiss-cpu"
            )

    # ── Add vectors ──────────────────────────────────────────────────────
    def add(
        self,
        vectors: np.ndarray,
        documents: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Add vectors and their associated documents to the store.

        Args:
            vectors:    [N, dim] float32 array (L2-normalized)
            documents:  list of N text chunks
            metadata:   optional list of N metadata dicts
        """
        if len(vectors) != len(documents):
            raise ValueError(
                f"vectors ({len(vectors)}) and documents ({len(documents)}) "
                f"must have the same length"
            )

        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of shape [N, {self.dim}], "
                f"got {vectors.shape}"
            )

        # Append to numpy storage
        self._vectors = np.vstack([self._vectors, vectors])
        self._documents.extend(documents)
        self._metadata.extend(metadata or [{}] * len(documents))

        # Add to FAISS index if enabled
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    # ── Search ───────────────────────────────────────────────────────────
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Find the top-k most similar documents to the query.

        Args:
            query_vector: [dim] float32 array (L2-normalized)
            top_k:        number of results to return

        Returns:
            list of SearchResult, sorted by score (highest first)
        """
        if self.size == 0:
            return []

        top_k = min(top_k, self.size)

        if self._faiss_index is not None:
            return self._search_faiss(query_vector, top_k)
        return self._search_numpy(query_vector, top_k)

    def _search_numpy(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """Pure numpy cosine similarity search."""
        # Dot product (= cosine similarity for L2-normalized vectors)
        scores = self._vectors @ query_vector  # [N]

        # Get top-k indices (descending score)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            SearchResult(
                document=self._documents[i],
                score=float(scores[i]),
                index=int(i),
                metadata=self._metadata[i],
            )
            for i in top_indices
        ]

    def _search_faiss(
        self,
        query_vector: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """FAISS inner-product search."""
        query_2d = query_vector.reshape(1, -1)
        scores, indices = self._faiss_index.search(query_2d, top_k)

        return [
            SearchResult(
                document=self._documents[idx],
                score=float(scores[0][i]),
                index=int(idx),
                metadata=self._metadata[idx],
            )
            for i, idx in enumerate(indices[0])
            if idx >= 0  # FAISS returns -1 for empty slots
        ]

    # ── Utilities ────────────────────────────────────────────────────────
    def clear(self) -> None:
        """Remove all vectors and documents."""
        self._vectors = np.empty((0, self.dim), dtype=np.float32)
        self._documents.clear()
        self._metadata.clear()
        if self._faiss_index is not None:
            self._faiss_index.reset()

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self._documents)

    def __repr__(self) -> str:
        backend = "faiss" if self._faiss_index else "numpy"
        return f"VectorStore(dim={self.dim}, size={self.size}, backend='{backend}')"
