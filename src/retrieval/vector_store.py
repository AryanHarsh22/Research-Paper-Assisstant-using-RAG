import os
import json
import pickle
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np


class FAISSVectorStore:
    """
    A lightweight FAISS-backed vector store for dense retrieval.

    Uses an IndexFlatIP (inner-product) index. Because embeddings are
    L2-normalised at embed time, inner product == cosine similarity, so
    higher score = more relevant.

    Persistence layout (one directory):
        <store_dir>/
            index.faiss   – the raw FAISS binary index
            metadata.pkl  – parallel list of chunk dicts (text + metadata)

    Args:
        embedding_dim (int): Dimensionality of embedding vectors (e.g. 384).
        store_dir (str): Directory where index and metadata are saved/loaded.
    """

    def __init__(self, embedding_dim: int, store_dir: str = "data/vector_store"):
        self.embedding_dim = embedding_dim
        self.store_dir = store_dir
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(embedding_dim)
        self._metadata: List[Dict[str, Any]] = []  # parallel to FAISS rows

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        """Number of vectors currently stored in the index."""
        return self._index.ntotal

    def add(self, records: List[Dict[str, Any]]) -> None:
        """
        Adds embedded records to the index.

        Args:
            records: Output of DocumentEmbedder.embed_chunks() — each dict
                     must have keys 'embedding', 'text', and 'metadata'.

        Raises:
            ValueError: If records list is empty or embedding dim mismatches.
        """
        if not records:
            raise ValueError("No records provided to add to the vector store.")

        embeddings = np.array(
            [r["embedding"] for r in records], dtype=np.float32
        )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        self._index.add(embeddings)

        # Store only text + metadata (not the raw embedding vector)
        for r in records:
            self._metadata.append({
                "chunk_id": r.get("chunk_id", ""),
                "text":     r["text"],
                "metadata": r["metadata"],
            })

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-k most similar chunks for a query vector.

        Args:
            query_embedding: Unit-normalised query vector (from DocumentEmbedder.embed_query).
            top_k: Number of results to return.

        Returns:
            List of dicts, each containing:
                - 'chunk_id' (str)
                - 'text'     (str)
                - 'metadata' (dict)
                - 'score'    (float) — cosine similarity in [-1, 1]

        Raises:
            ValueError: If the index is empty.
        """
        if self.size == 0:
            raise ValueError("Vector store is empty. Add documents before searching.")

        k = min(top_k, self.size)
        query = np.array([query_embedding], dtype=np.float32)

        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS padding for insufficient results
                continue
            entry = dict(self._metadata[idx])   # shallow copy
            entry["score"] = float(score)
            results.append(entry)

        return results

    def save(self) -> None:
        """
        Persists the FAISS index and metadata to disk at self.store_dir.
        Creates the directory if it does not exist.
        """
        os.makedirs(self.store_dir, exist_ok=True)
        index_path    = os.path.join(self.store_dir, "index.faiss")
        metadata_path = os.path.join(self.store_dir, "metadata.pkl")

        faiss.write_index(self._index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self._metadata, f)

    def load(self) -> None:
        """
        Loads the FAISS index and metadata from disk.

        Raises:
            FileNotFoundError: If the store directory or required files are missing.
        """
        index_path    = os.path.join(self.store_dir, "index.faiss")
        metadata_path = os.path.join(self.store_dir, "metadata.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found at: {metadata_path}")

        self._index    = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self._metadata = pickle.load(f)

    @classmethod
    def from_disk(cls, embedding_dim: int, store_dir: str) -> "FAISSVectorStore":
        """
        Convenience constructor: creates an instance and immediately loads from disk.

        Args:
            embedding_dim: Must match the dim used when the index was created.
            store_dir: Directory containing index.faiss and metadata.pkl.

        Returns:
            A fully loaded FAISSVectorStore instance.
        """
        store = cls(embedding_dim=embedding_dim, store_dir=store_dir)
        store.load()
        return store
