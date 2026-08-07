from typing import List, Dict, Any
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import numpy as np


class DocumentEmbedder:
    """
    Embedder class to generate dense vector embeddings for document chunks.

    Uses a SentenceTransformer model to encode chunk text into fixed-size
    embedding vectors. Designed to consume the output of DocumentChunker
    and produce a list of enriched records ready for vector store ingestion.

    Args:
        model_name (str): HuggingFace model identifier for SentenceTransformer.
                          Defaults to 'all-MiniLM-L6-v2'.
        batch_size (int): Number of chunks to encode in a single forward pass.
                          Larger values are faster but use more memory.
        show_progress (bool): Whether to display a tqdm progress bar during encoding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model = SentenceTransformer(model_name)

    @property
    def embedding_dim(self) -> int:
        """Returns the dimensionality of the embedding vectors produced by the model."""
        return self._model.get_embedding_dimension()

    def embed_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a list of document chunks.

        Each input chunk (a LangChain Document) is encoded into a dense vector.
        The result is a list of dicts containing the original text, metadata,
        and the corresponding embedding — ready to be inserted into a vector store.

        Args:
            chunks (List[Document]): Output from DocumentChunker.chunk_documents().

        Returns:
            List[Dict[str, Any]]: One record per chunk with the following keys:
                - "chunk_id"  (str):   Unique identifier from chunk metadata.
                - "text"      (str):   Raw page content of the chunk.
                - "metadata"  (dict):  All metadata fields from the source chunk.
                - "embedding" (List[float]): Dense embedding vector.

        Raises:
            ValueError: If chunks list is empty.
        """
        if not chunks:
            raise ValueError("No chunks provided for embedding. Check your chunker output.")

        texts = [chunk.page_content for chunk in chunks]

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Unit-norm vectors → cosine similarity = dot product
        )

        records = []
        for chunk, embedding in zip(chunks, embeddings):
            records.append({
                "chunk_id": chunk.metadata.get("chunk_id", ""),
                "text": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding.tolist(),
            })

        return records

    def embed_query(self, query: str) -> List[float]:
        """
        Generates a single embedding for a query string at retrieval time.

        The query is normalized in the same way as document embeddings so that
        cosine similarity comparisons are valid.

        Args:
            query (str): The user's search query.

        Returns:
            List[float]: Dense embedding vector for the query.
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty.")

        embedding: np.ndarray = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()
