"""
Verifies the FAISSVectorStore end-to-end:
  1. Load a PDF, chunk it, embed the first N chunks
  2. Add embeddings to the vector store
  3. Save the index to disk
  4. Reload from disk and confirm vector count is preserved
  5. Run a semantic search and inspect results
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


def section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def ok(msg: str):   print(f"  [OK]   {msg}")
def fail(msg: str): print(f"  [FAIL] {msg}")


def main():
    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    pdf_path  = os.path.abspath("data/papers/attention_is_all_you_need.pdf")
    store_dir = os.path.abspath("data/vector_store_test")

    # Clean any leftover test store
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)

    if not os.path.exists(pdf_path):
        fail(f"Sample PDF not found at {pdf_path}. Run verify_ingestion.py first.")
        return

    # ------------------------------------------------------------------ #
    # Build embeddings (reuse ingestion pipeline)
    # ------------------------------------------------------------------ #
    section("BUILDING EMBEDDINGS (first 20 chunks)")
    loader   = PDFLoader()
    chunker  = DocumentChunker(chunk_size=500, chunk_overlap=50)
    embedder = DocumentEmbedder(show_progress=True)

    pages  = loader.load_pdf(pdf_path)
    chunks = chunker.chunk_documents(pages)[:20]   # keep it fast
    records = embedder.embed_chunks(chunks)

    ok(f"Chunks embedded : {len(records)}")
    ok(f"Embedding dim   : {len(records[0]['embedding'])}")

    # ------------------------------------------------------------------ #
    # 1. Add to vector store
    # ------------------------------------------------------------------ #
    section("STAGE 1 — add()")
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim, store_dir=store_dir)

    try:
        store.add(records)
    except Exception as e:
        fail(f"add() raised: {e}"); return

    assert store.size == len(records), f"Expected {len(records)}, got {store.size}"
    ok(f"Vectors in index : {store.size}")

    # ------------------------------------------------------------------ #
    # 2. Save to disk
    # ------------------------------------------------------------------ #
    section("STAGE 2 — save()")
    try:
        store.save()
    except Exception as e:
        fail(f"save() raised: {e}"); return

    index_path = os.path.join(store_dir, "index.faiss")
    meta_path  = os.path.join(store_dir, "metadata.pkl")
    assert os.path.exists(index_path), "index.faiss not written"
    assert os.path.exists(meta_path),  "metadata.pkl not written"
    ok(f"index.faiss size : {os.path.getsize(index_path):,} bytes")
    ok(f"metadata.pkl size: {os.path.getsize(meta_path):,} bytes")

    # ------------------------------------------------------------------ #
    # 3. Reload from disk
    # ------------------------------------------------------------------ #
    section("STAGE 3 — from_disk()")
    try:
        store2 = FAISSVectorStore.from_disk(
            embedding_dim=embedder.embedding_dim,
            store_dir=store_dir
        )
    except Exception as e:
        fail(f"from_disk() raised: {e}"); return

    assert store2.size == len(records), \
        f"After reload: expected {len(records)}, got {store2.size}"
    ok(f"Vectors after reload : {store2.size}  (matches original ✓)")

    # ------------------------------------------------------------------ #
    # 4. Semantic search
    # ------------------------------------------------------------------ #
    section("STAGE 4 — search()")
    query = "What is the attention mechanism in transformers?"
    q_emb = embedder.embed_query(query)

    try:
        results = store2.search(q_emb, top_k=3)
    except Exception as e:
        fail(f"search() raised: {e}"); return

    assert len(results) > 0, "search() returned 0 results"
    assert "score"    in results[0], "'score' key missing"
    assert "text"     in results[0], "'text' key missing"
    assert "chunk_id" in results[0], "'chunk_id' key missing"

    ok(f"Results returned : {len(results)}")
    print()
    for i, r in enumerate(results):
        print(f"  Result #{i+1}")
        print(f"    chunk_id : {r['chunk_id']}")
        print(f"    score    : {r['score']:.4f}  (cosine similarity)")
        print(f"    preview  : {r['text'][:120].strip()}...")
        print()

    # Scores should be in descending order
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted by score"
    ok("Scores are in descending order ✓")

    # ------------------------------------------------------------------ #
    # Cleanup & summary
    # ------------------------------------------------------------------ #
    shutil.rmtree(store_dir)

    section("ALL STAGES PASSED — FAISSVectorStore")
    print(f"  Vectors stored   : {len(records)}")
    print(f"  Embedding dim    : {embedder.embedding_dim}")
    print(f"  Top result score : {results[0]['score']:.4f}")
    print()


if __name__ == "__main__":
    main()
