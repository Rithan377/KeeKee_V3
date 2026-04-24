"""
embedder.py — Text Embedding for RAG
=======================================
Uses sentence-transformers (all-MiniLM-L6-v2) to convert text chunks
into dense 384-dimensional vectors for similarity search.

This is a SEPARATE model from KeeKee — it's a small encoder used only
for computing document/query embeddings for retrieval.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ─────────────────────────────────────────────
# TextEmbedder
# ─────────────────────────────────────────────
class TextEmbedder:
    """
    Embeds text chunks into dense float vectors using sentence-transformers.

    Uses the all-MiniLM-L6-v2 model by default:
      - 384-dimensional output
      - ~80MB download on first use
      - Fast inference (~14K sentences/sec on CPU)
      - L2-normalized embeddings (cosine similarity = dot product)

    Usage:
        embedder = TextEmbedder()
        vectors = embedder.embed(["Hello world", "Python is great"])
        query_vec = embedder.embed_query("What is Python?")
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: sentence-transformers model name or path
            device:     "cpu", "cuda", or None (auto-detect)
        """
        self._model_name = model_name
        self._device = device
        self._model = None  # lazy initialization

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG embeddings. "
                "Install with: pip install sentence-transformers"
            )

        self._model = SentenceTransformer(
            self._model_name,
            device=self._device,
        )

    @property
    def dim(self) -> int:
        """Embedding dimension of the model."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into dense vectors.

        Args:
            texts:          list of text strings to embed
            batch_size:     batch size for encoding
            show_progress:  show progress bar

        Returns:
            np.ndarray of shape [N, dim] (float32, L2-normalized)
        """
        self._load_model()

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into a dense vector.

        Args:
            query: query text

        Returns:
            np.ndarray of shape [dim] (float32, L2-normalized)
        """
        return self.embed([query])[0]

    def __repr__(self) -> str:
        return f"TextEmbedder(model='{self._model_name}')"
