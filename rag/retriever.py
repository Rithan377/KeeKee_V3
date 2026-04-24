"""
retriever.py — RAG Retriever
===============================
High-level RAG interface: add documents → chunk → embed → store → retrieve.

Usage:
    retriever = RAGRetriever()
    retriever.add_documents(["Python is a programming language.", ...])
    context = retriever.retrieve("What is Python?")
    # context = "[CONTEXT]\nPython is a programming...\n[/CONTEXT]"
"""

from __future__ import annotations

from typing import Optional

from rag.embedder import TextEmbedder
from rag.vector_store import VectorStore, SearchResult


# ─────────────────────────────────────────────
# RAGRetriever
# ─────────────────────────────────────────────
class RAGRetriever:
    """
    End-to-end RAG retriever: chunk documents, embed, store, and retrieve.

    Pipeline:
        1. add_documents() → split into chunks → embed → store in VectorStore
        2. retrieve(query) → embed query → search top-k → format as context string

    Usage:
        retriever = RAGRetriever(chunk_size=512, top_k=3)
        retriever.add_documents(["Full document text...", ...])
        context = retriever.retrieve("What is X?")
        # Returns: "[CONTEXT]\nchunk1\n---\nchunk2\n[/CONTEXT]"
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 3,
        embedder_model: str = "all-MiniLM-L6-v2",
        use_faiss: bool = False,
    ):
        """
        Args:
            chunk_size:     max characters per chunk
            chunk_overlap:  overlap between consecutive chunks (chars)
            top_k:          default number of chunks to retrieve
            embedder_model: sentence-transformers model name
            use_faiss:      use FAISS backend for vector search
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedder = TextEmbedder(model_name=embedder_model)
        self.store = VectorStore(dim=384, use_faiss=use_faiss)

        # Track source documents
        self._doc_count = 0

    def add_documents(
        self,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """
        Chunk, embed, and index documents.

        Args:
            texts:    list of document strings (can be long)
            metadata: optional per-document metadata dicts

        Returns:
            Total number of chunks added
        """
        all_chunks = []
        all_metadata = []

        for i, text in enumerate(texts):
            doc_meta = (metadata[i] if metadata else {}).copy() if metadata else {}
            doc_meta["doc_index"] = self._doc_count + i

            chunks = self._chunk_text(text)
            for j, chunk in enumerate(chunks):
                chunk_meta = doc_meta.copy()
                chunk_meta["chunk_index"] = j
                all_chunks.append(chunk)
                all_metadata.append(chunk_meta)

        if not all_chunks:
            return 0

        # Embed all chunks at once (batched for efficiency)
        vectors = self.embedder.embed(all_chunks, show_progress=len(all_chunks) > 100)

        # Fix dim if needed (first call sets the VectorStore dimension)
        if self.store.size == 0 and vectors.shape[1] != self.store.dim:
            self.store = VectorStore(
                dim=vectors.shape[1],
                use_faiss=self.store._faiss_index is not None,
            )

        # Store
        self.store.add(vectors, all_chunks, all_metadata)
        self._doc_count += len(texts)

        return len(all_chunks)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Retrieve relevant chunks for a query.

        Args:
            query:  search query string
            top_k:  number of chunks to retrieve (default: self.top_k)

        Returns:
            Formatted context string:
            "[CONTEXT]\nchunk1\n---\nchunk2\n---\nchunk3\n[/CONTEXT]"

            Returns empty string if no documents are indexed.
        """
        if self.store.size == 0:
            return ""

        k = top_k or self.top_k

        # Embed query
        query_vec = self.embedder.embed_query(query)

        # Search
        results = self.store.search(query_vec, top_k=k)

        if not results:
            return ""

        return self._format_context(results)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[SearchResult]:
        """
        Retrieve with full SearchResult objects (includes scores + metadata).

        Args:
            query:  search query string
            top_k:  number of chunks to retrieve

        Returns:
            list of SearchResult objects
        """
        if self.store.size == 0:
            return []

        k = top_k or self.top_k
        query_vec = self.embedder.embed_query(query)
        return self.store.search(query_vec, top_k=k)

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results as a context block for prompt injection."""
        chunks = [r.document for r in results]
        inner = "\n---\n".join(chunks)
        return f"[CONTEXT]\n{inner}\n[/CONTEXT]"

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Uses a simple sliding window approach:
          - chunk_size characters per chunk
          - chunk_overlap characters of overlap between consecutive chunks
          - Tries to break at sentence boundaries when possible

        Args:
            text: full document text

        Returns:
            list of text chunks
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # If text fits in one chunk, return as-is
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk — take everything remaining
                chunks.append(text[start:].strip())
                break

            # Try to find a sentence boundary near the end
            chunk = text[start:end]
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")

            # Prefer sentence boundary, then newline, then hard cut
            break_point = max(last_period, last_newline)
            if break_point > self.chunk_size // 2:
                end = start + break_point + 1

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        # Filter out empty chunks
        return [c for c in chunks if c]

    def clear(self) -> None:
        """Remove all indexed documents."""
        self.store.clear()
        self._doc_count = 0

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently indexed."""
        return self.store.size

    def __repr__(self) -> str:
        return (
            f"RAGRetriever(chunks={self.chunk_count}, "
            f"chunk_size={self.chunk_size}, top_k={self.top_k})"
        )
