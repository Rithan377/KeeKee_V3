"""
rag/ — Retrieval-Augmented Generation Pipeline
================================================
Embed documents, store vectors, retrieve relevant chunks.
"""

from rag.embedder import TextEmbedder
from rag.vector_store import VectorStore, SearchResult
from rag.retriever import RAGRetriever

__all__ = ["TextEmbedder", "VectorStore", "SearchResult", "RAGRetriever"]
