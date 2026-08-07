import os
import urllib.request
import sys

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the root of the project is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader import PDFLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import DocumentEmbedder


def download_sample_pdf(dest_path: str):
    """
    Downloads the 'Attention Is All You Need' paper as a sample PDF.
    """
    url = "https://arxiv.org/pdf/1706.03762.pdf"
    print(f"Downloading sample PDF from {url} to {dest_path}...")
    
    # Configure headers to prevent standard bot blockers
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download completed successfully!")


def section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def ok(msg: str):
    print(f"  [OK]  {msg}")


def fail(msg: str):
    print(f"  [FAIL] {msg}")


def main():
    # ------------------------------------------------------------------ #
    # 0. Setup paths
    # ------------------------------------------------------------------ #
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/papers"))
    os.makedirs(data_dir, exist_ok=True)
    pdf_path = os.path.join(data_dir, "attention_is_all_you_need.pdf")

    # Download sample if not exists
    if not os.path.exists(pdf_path):
        try:
            download_sample_pdf(pdf_path)
        except Exception as e:
            fail(f"Could not download sample PDF: {e}")
            return

    # ------------------------------------------------------------------ #
    # 1. Loader
    # ------------------------------------------------------------------ #
    section("STAGE 1 — PDFLoader")
    loader = PDFLoader(remove_headers_footers=True, header_threshold=0.3)
    try:
        pages = loader.load_pdf(pdf_path)
    except Exception as e:
        fail(f"PDFLoader raised an exception: {e}")
        return

    assert len(pages) > 0, "PDFLoader returned 0 pages"
    ok(f"Pages extracted         : {len(pages)}")

    first_page = pages[0]
    assert "text" in first_page,     "Page dict missing 'text' key"
    assert "metadata" in first_page, "Page dict missing 'metadata' key"
    assert first_page["metadata"].get("source"), "Metadata 'source' is empty"
    assert isinstance(first_page["metadata"].get("page"), int), "'page' should be an int"
    ok(f"First page metadata     : {first_page['metadata']}")
    ok(f"First page text preview : {first_page['text'][:120].strip()}...")

    # ------------------------------------------------------------------ #
    # 2. Chunker
    # ------------------------------------------------------------------ #
    section("STAGE 2 — DocumentChunker")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    try:
        chunks = chunker.chunk_documents(pages)
    except Exception as e:
        fail(f"DocumentChunker raised an exception: {e}")
        return

    assert len(chunks) > 0, "Chunker returned 0 chunks"
    ok(f"Total chunks generated  : {len(chunks)}")

    sample_chunk = chunks[0]
    assert sample_chunk.metadata.get("chunk_id"),    "chunk_id is missing"
    assert sample_chunk.metadata.get("source"),      "'source' missing from chunk metadata"
    assert isinstance(sample_chunk.metadata.get("page"), int), "'page' should be int"
    assert isinstance(sample_chunk.metadata.get("chunk_index"), int), "'chunk_index' should be int"
    ok(f"Sample chunk_id         : {sample_chunk.metadata['chunk_id']}")
    ok(f"Sample content preview  : {sample_chunk.page_content[:120].strip()}...")

    # ------------------------------------------------------------------ #
    # 3. Embedder
    # ------------------------------------------------------------------ #
    section("STAGE 3 — DocumentEmbedder")
    embedder = DocumentEmbedder(model_name="all-MiniLM-L6-v2", batch_size=64, show_progress=True)
    ok(f"Model loaded            : {embedder.model_name}")
    ok(f"Embedding dimension     : {embedder.embedding_dim}")

    # Embed only the first 10 chunks to keep verification fast
    sample_chunks = chunks[:10]
    try:
        records = embedder.embed_chunks(sample_chunks)
    except Exception as e:
        fail(f"embed_chunks raised an exception: {e}")
        return

    assert len(records) == len(sample_chunks), "Record count mismatch"
    ok(f"Records returned        : {len(records)} (embedded first {len(sample_chunks)} chunks)")

    first_rec = records[0]
    assert "chunk_id"  in first_rec, "Record missing 'chunk_id'"
    assert "text"      in first_rec, "Record missing 'text'"
    assert "metadata"  in first_rec, "Record missing 'metadata'"
    assert "embedding" in first_rec, "Record missing 'embedding'"

    emb = first_rec["embedding"]
    assert isinstance(emb, list),           "Embedding should be a list"
    assert len(emb) == embedder.embedding_dim, (
        f"Expected dim {embedder.embedding_dim}, got {len(emb)}"
    )
    ok(f"Embedding vector length : {len(emb)}")

    # Verify L2 norm ≈ 1.0 (normalized embeddings)
    import math
    norm = math.sqrt(sum(x * x for x in emb))
    assert abs(norm - 1.0) < 1e-4, f"Embedding not unit-normalized (norm={norm:.4f})"
    ok(f"L2 norm (should be 1.0) : {norm:.6f}")

    # Verify embed_query works
    try:
        q_emb = embedder.embed_query("What is the attention mechanism?")
    except Exception as e:
        fail(f"embed_query raised an exception: {e}")
        return

    assert len(q_emb) == embedder.embedding_dim, "Query embedding dim mismatch"
    ok(f"Query embedding length  : {len(q_emb)}")

    # ------------------------------------------------------------------ #
    # 4. Summary
    # ------------------------------------------------------------------ #
    section("ALL STAGES PASSED")
    print(f"  Pages    : {len(pages)}")
    print(f"  Chunks   : {len(chunks)}")
    print(f"  Emb. dim : {embedder.embedding_dim}")
    print()


if __name__ == "__main__":
    main()
