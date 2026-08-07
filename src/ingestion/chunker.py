from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentChunker:
    """
    Chunker class to split documents into overlapping chunks with preserved metadata.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Splits a list of parsed pages (from PDFLoader) into chunks.
        
        Args:
            documents (List[Dict[str, Any]]): Pages from PDFLoader.
            
        Returns:
            List[Document]: List of LangChain Document objects representing the chunks.
        """
        langchain_docs = [
            Document(page_content=doc["text"], metadata=doc["metadata"])
            for doc in documents
        ]
        
        chunks = self.splitter.split_documents(langchain_docs)
        
        # Track chunk index per page to generate unique IDs
        page_chunk_counters = {}
        
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            key = (source, page)
            
            chunk_idx = page_chunk_counters.get(key, 0)
            page_chunk_counters[key] = chunk_idx + 1
            
            # Enrich metadata
            chunk.metadata["chunk_index"] = chunk_idx
            
            # Format unique chunk ID: filename_pPageNum_cChunkIdx
            safe_source = source.replace(" ", "_")
            chunk.metadata["chunk_id"] = f"{safe_source}_p{page}_c{chunk_idx}"
            
        return chunks
