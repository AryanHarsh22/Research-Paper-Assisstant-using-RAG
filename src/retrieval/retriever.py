from typing import List, Dict, Any, Optional

from src.ingestion.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore


class Retriever:
    """
    Semantic retriever for the RAG pipeline.

    Wraps a DocumentEmbedder and a FAISSVectorStore to provide a single
    high-level interface:  query string → ranked list of relevant chunks.

    Features:
        - Configurable top-k and minimum similarity score threshold.
        - Chunk-level deduplication (same chunk_id returned only once).

    Args:
        embedder  (DocumentEmbedder): Produces query embeddings at retrieval time.
        store     (FAISSVectorStore): Loaded vector store to search against.
        top_k     (int):  Maximum number of results to return (default 5).
        score_threshold (float): Minimum cosine similarity to include a result.
                                  Range [-1, 1].  Default 0.0 (any positive match).
    """

    def __init__(
        self,
        embedder: DocumentEmbedder,
        store: FAISSVectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        self.embedder        = embedder
        self.store           = store
        self.top_k           = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant chunks for a natural-language query.

        Pipeline:
            1. Embed the query using the DocumentEmbedder.
            2. Search the vector store for the top candidates.
            3. Filter by score_threshold.
            4. Deduplicate by chunk_id (keeps the highest-scoring copy).

        Args:
            query (str): The user's natural-language question.

        Returns:
            List[Dict[str, Any]]: Ranked list of result dicts, each containing:
                - 'chunk_id'  (str)
                - 'text'      (str)
                - 'metadata'  (dict): source, page, chunk_index, etc.
                - 'score'     (float): cosine similarity in [-1, 1]

        Raises:
            ValueError: If query is blank or the vector store is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        # Fetch more candidates than top_k to allow for filtering/dedup loss
        fetch_k = max(self.top_k * 3, 15)

        # 1. Embed query
        query_embedding = self.embedder.embed_query(query)

        # 2. Search vector store
        candidates = self.store.search(query_embedding, top_k=fetch_k)

        # 3. Score threshold filter
        candidates = [r for r in candidates if r["score"] >= self.score_threshold]

        # 4. Deduplicate by chunk_id (keep highest score, preserve order)
        seen_ids: set = set()
        deduplicated = []
        for r in candidates:
            cid = r.get("chunk_id", "")
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduplicated.append(r)

        return deduplicated[: self.top_k]

    def retrieve_with_context(
        self, query: str, context_window: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Like retrieve(), but also surfaces which source documents and pages
        contributed results, useful for citation generation.

        Args:
            query          (str): Natural-language question.
            context_window (int): Reserved for future neighbour-chunk expansion
                                  (not yet implemented; pass 0).

        Returns:
            Same as retrieve() — a list of result dicts enriched with a
            top-level 'source_info' field summarising provenance.
        """
        results = self.retrieve(query)
        for r in results:
            meta = r.get("metadata", {})
            r["source_info"] = {
                "file":  meta.get("source", "unknown"),
                "page":  meta.get("page",   "?"),
            }
        return results
