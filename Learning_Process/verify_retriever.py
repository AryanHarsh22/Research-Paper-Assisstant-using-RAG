"""
Verifies the Retriever end-to-end:
  1. Build a vector store from the full Attention paper
  2. Run several semantic queries and inspect results
  3. Verify score ordering, deduplication, threshold filtering
  4. Verify retrieve_with_context() adds source_info
"""
import os
import sys
import shutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import PDFLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever


def section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def ok(msg: str):   print(f"  [OK]   {msg}")
def fail(msg: str): print(f"  [FAIL] {msg}")


QUERIES = [
    "What is multi-head attention?",
    "How does the encoder-decoder architecture work?",
    "What is positional encoding and why is it needed?",
]


def main():
    pdf_path  = os.path.abspath("data/papers/attention_is_all_you_need.pdf")
    store_dir = os.path.abspath("data/vector_store_retriever_test")

    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)

    if not os.path.exists(pdf_path):
        fail("Sample PDF not found. Run verify_ingestion.py first.")
        return

    # ------------------------------------------------------------------ #
    # Build full index from all chunks
    # ------------------------------------------------------------------ #
    section("BUILDING FULL INDEX")
    loader   = PDFLoader()
    chunker  = DocumentChunker(chunk_size=500, chunk_overlap=50)
    embedder = DocumentEmbedder(show_progress=True)

    pages   = loader.load_pdf(pdf_path)
    chunks  = chunker.chunk_documents(pages)
    records = embedder.embed_chunks(chunks)

    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim, store_dir=store_dir)
    store.add(records)
    store.save()

    ok(f"Total chunks indexed : {store.size}")

    # ------------------------------------------------------------------ #
    # 1. Basic retrieval
    # ------------------------------------------------------------------ #
    section("STAGE 1 — retrieve()")
    retriever = Retriever(embedder=embedder, store=store, top_k=3, score_threshold=0.0)

    for query in QUERIES:
        print(f"\n  Query: \"{query}\"")
        try:
            results = retriever.retrieve(query)
        except Exception as e:
            fail(f"retrieve() raised: {e}"); return

        assert len(results) > 0, "retrieve() returned 0 results"

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

        chunk_ids = [r["chunk_id"] for r in results]
        assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk_ids in results"

        for r in results:
            print(f"    [{r['score']:.4f}] {r['chunk_id']}")
            print(f"            {r['text'][:100].strip()}...")

    ok("All queries returned sorted, deduplicated results")

    # ------------------------------------------------------------------ #
    # 2. Score threshold filtering
    # ------------------------------------------------------------------ #
    section("STAGE 2 — score_threshold filtering")
    strict_retriever = Retriever(
        embedder=embedder, store=store, top_k=10, score_threshold=0.9
    )
    results_strict = strict_retriever.retrieve(QUERIES[0])
    ok(f"Results with threshold=0.9 : {len(results_strict)}  (expected 0 or very few)")
    for r in results_strict:
        assert r["score"] >= 0.9, f"Score {r['score']} below threshold"

    lenient_retriever = Retriever(
        embedder=embedder, store=store, top_k=5, score_threshold=0.3
    )
    results_lenient = lenient_retriever.retrieve(QUERIES[0])
    ok(f"Results with threshold=0.3 : {len(results_lenient)}")
    for r in results_lenient:
        assert r["score"] >= 0.3, f"Score {r['score']} below threshold"

    # ------------------------------------------------------------------ #
    # 3. retrieve_with_context()
    # ------------------------------------------------------------------ #
    section("STAGE 3 — retrieve_with_context()")
    try:
        ctx_results = retriever.retrieve_with_context(QUERIES[0])
    except Exception as e:
        fail(f"retrieve_with_context() raised: {e}"); return

    assert len(ctx_results) > 0
    for r in ctx_results:
        assert "source_info" in r, "'source_info' key missing"
        assert "file" in r["source_info"], "'file' missing from source_info"
        assert "page" in r["source_info"], "'page' missing from source_info"

    ok("source_info present on all results")
    print()
    for r in ctx_results:
        print(f"    [{r['score']:.4f}] {r['source_info']['file']}  page {r['source_info']['page']}")

    # ------------------------------------------------------------------ #
    # 4. Empty query guard
    # ------------------------------------------------------------------ #
    section("STAGE 4 — error handling")
    try:
        retriever.retrieve("   ")
        fail("Should have raised ValueError for blank query")
        return
    except ValueError:
        ok("Blank query correctly raises ValueError")

    # Cleanup
    shutil.rmtree(store_dir)

    section("ALL STAGES PASSED — Retriever")
    print(f"  Total chunks in index : {store.size}")
    print(f"  Queries tested        : {len(QUERIES)}")
    print()


if __name__ == "__main__":
    main()
