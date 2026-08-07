"""
Verification script for full end-to-end integration of app.py modules.
Tests PDF creation/ingestion, chunking, embedding, vector store persistence,
retrieval, RAG prompt construction, mocked LLM generation, and citation verification.
"""

import os
import sys
import fitz  # PyMuPDF
import tempfile

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import PDFLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.generation.prompt import PromptBuilder
from src.generation.llm import get_llm_client
from src.generation.citations import CitationProcessor


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def ok(msg: str):   print(f"  [OK]   {msg}")
def fail(msg: str): print(f"  [FAIL] {msg}")


def create_sample_pdf(file_path: str):
    """Creates a sample 2-page PDF for testing."""
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        fitz.Point(50, 50),
        "Header Notice - Confidential\n\n"
        "Transformers are deep learning models introduced in 2017. "
        "The self-attention mechanism allows models to weigh the significance "
        "of different parts of the input sequence dynamics.\n\n"
        "Page 1"
    )
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        fitz.Point(50, 50),
        "Header Notice - Confidential\n\n"
        "Multi-head attention maps queries and keys to parallel projection subspaces. "
        "This enhances the model capability to jointly attend to information from "
        "different positions.\n\n"
        "Page 2"
    )
    
    doc.save(file_path)
    doc.close()


def test_full_pipeline():
    section("STAGE 1 — End-to-End Pipeline Integration Test")

    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_path = os.path.join(tmp_dir, "test_paper.pdf")
        store_dir = os.path.join(tmp_dir, "vector_store")
        
        # 1. Create sample PDF
        create_sample_pdf(pdf_path)
        ok("Created synthetic research paper PDF.")

        # 2. PDF Loader
        loader = PDFLoader(remove_headers_footers=True)
        documents = loader.load_pdf(pdf_path)
        assert len(documents) == 2, f"Expected 2 pages, got {len(documents)}"
        ok(f"PDFLoader extracted {len(documents)} cleaned pages.")

        # 3. Document Chunker
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=30)
        chunks = chunker.chunk_documents(documents)
        assert len(chunks) > 0, "No chunks created by chunker"
        ok(f"DocumentChunker split pages into {len(chunks)} chunks.")

        # 4. Document Embedder
        embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2", show_progress=False)
        records = embedder.embed_chunks(chunks)
        assert len(records) == len(chunks), "Record count mismatch"
        assert len(records[0]["embedding"]) == 384, f"Unexpected embedding dim: {len(records[0]['embedding'])}"
        ok(f"DocumentEmbedder encoded chunks into {embedder.embedding_dim}-dim vectors.")

        # 5. FAISS Vector Store
        store = FAISSVectorStore(embedding_dim=embedder.embedding_dim, store_dir=store_dir)
        store.add(records)
        assert store.size == len(chunks), "Store size mismatch"
        store.save()
        ok("FAISSVectorStore saved to disk.")

        # Load back from disk
        loaded_store = FAISSVectorStore.from_disk(embedding_dim=embedder.embedding_dim, store_dir=store_dir)
        assert loaded_store.size == store.size, "Loaded store size mismatch"
        ok("FAISSVectorStore loaded successfully from disk.")

        # 6. Retriever
        retriever = Retriever(embedder=embedder, store=loaded_store, top_k=2)
        query = "What is self-attention?"
        retrieved_chunks = retriever.retrieve_with_context(query)
        assert len(retrieved_chunks) > 0, "Retriever returned 0 chunks"
        ok(f"Retriever retrieved {len(retrieved_chunks)} relevant chunks for query: '{query}'.")

        # 7. Prompt Builder
        rag_prompt = PromptBuilder.build_rag_prompt(query, retrieved_chunks)
        assert "System Instructions:" in rag_prompt
        assert query in rag_prompt
        ok("PromptBuilder successfully assembled grounded RAG prompt.")

        # 8. Simulated Generation & Citation Verification
        simulated_response = (
            "The self-attention mechanism allows models to weigh different parts of input "
            "[test_paper.pdf, p. 1]. Multi-head attention maps queries to parallel subspaces [test_paper.pdf, p. 2]."
        )
        verification_res = CitationProcessor.verify_citations(simulated_response, retrieved_chunks)
        citations = verification_res["citations"]
        assert len(citations) == 2, f"Expected 2 citations, got {len(citations)}"
        assert citations[0]["verified"] is True, "First citation should be verified"
        ok("CitationProcessor verified inline citations against retrieved corpus.")


def main():
    try:
        test_full_pipeline()
        section("ALL INTEGRATION TESTS PASSED ✓")
    except AssertionError as e:
        fail(f"Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        fail(f"Unexpected exception during verification: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
