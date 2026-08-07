import os
import sys
import tempfile
import streamlit as st
import pandas as pd
from typing import List, Dict, Any

# Ensure local imports work reliably
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.ingestion.loader import PDFLoader
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import DocumentEmbedder
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import Retriever
from src.generation.prompt import PromptBuilder
from src.generation.llm import get_llm_client, list_ollama_models
from src.generation.citations import CitationProcessor

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Research Paper Assistant | RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Global Styles */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stApp {
        background-color: #0e1117;
        color: #e0e6ed;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Status Badges & Cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Citation Badges */
    .badge-verified {
        background-color: #065f46;
        color: #34d399;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-unverified {
        background-color: #7f1d1d;
        color: #f87171;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Context Passage Box */
    .passage-card {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .passage-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
    .passage-text {
        font-size: 0.93rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resource Loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def get_cached_embedder(model_name: str = "all-MiniLM-L6-v2") -> DocumentEmbedder:
    """Cache the SentenceTransformer model in memory to avoid reloading."""
    return DocumentEmbedder(model_name=model_name, show_progress=False)


# -----------------------------------------------------------------------------
# Helper Functions & Session State
# -----------------------------------------------------------------------------
VECTOR_STORE_DIR = os.path.join("data", "vector_store")
UPLOADS_DIR = os.path.join("data", "uploads")


def get_configured_api_key(*names: str) -> str:
    """Read a provider key server-side without ever rendering it in the UI."""
    for name in names:
        try:
            value = st.secrets.get(name, "")
            if value:
                return str(value).strip()
        except Exception:
            pass

        value = os.environ.get(name, "")
        if value:
            return value.strip()

    return ""


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_files" not in st.session_state:
        st.session_state.ingested_files = []
    if "auto_loaded" not in st.session_state:
        st.session_state.auto_loaded = False


def load_existing_vector_store(embedding_dim: int) -> bool:
    """Attempts to load a pre-existing vector store from disk."""
    index_path = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        try:
            store = FAISSVectorStore.from_disk(
                embedding_dim=embedding_dim,
                store_dir=VECTOR_STORE_DIR
            )
            st.session_state.vector_store = store
            
            # Extract unique filenames from store metadata
            unique_sources = set()
            for item in store._metadata:
                src = item.get("metadata", {}).get("source")
                if src:
                    unique_sources.add(src)
            st.session_state.ingested_files = sorted(list(unique_sources))
            return True
        except Exception as e:
            st.sidebar.error(f"Failed to auto-load existing index: {e}")
            return False
    return False


def process_and_index_pdfs(
    uploaded_files,
    embedder: DocumentEmbedder,
    chunk_size: int,
    chunk_overlap: int,
    remove_headers_footers: bool
):
    """Processes uploaded PDFs through Loader -> Chunker -> Embedder -> VectorStore."""
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    loader = PDFLoader(remove_headers_footers=remove_headers_footers)
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    processed_filenames = []

    progress_bar = st.progress(0, text="Starting PDF processing...")
    total_files = len(uploaded_files)

    for idx, uploaded_file in enumerate(uploaded_files):
        file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        progress_bar.progress(
            int((idx + 0.3) / total_files * 100),
            text=f"Extracting text from `{uploaded_file.name}`..."
        )
        parsed_pages = loader.load_pdf(file_path)

        progress_bar.progress(
            int((idx + 0.6) / total_files * 100),
            text=f"Chunking `{uploaded_file.name}`..."
        )
        file_chunks = chunker.chunk_documents(parsed_pages)
        all_chunks.extend(file_chunks)
        processed_filenames.append(uploaded_file.name)

    if not all_chunks:
        st.error("No valid text chunks could be extracted from the uploaded PDF(s).")
        progress_bar.empty()
        return

    progress_bar.progress(85, text=f"Generating dense embeddings for {len(all_chunks)} chunks...")
    records = embedder.embed_chunks(all_chunks)

    progress_bar.progress(95, text="Building FAISS vector index...")
    store = FAISSVectorStore(embedding_dim=embedder.embedding_dim, store_dir=VECTOR_STORE_DIR)
    store.add(records)
    store.save()

    st.session_state.vector_store = store
    st.session_state.ingested_files = processed_filenames

    progress_bar.progress(100, text="Indexing complete!")
    st.toast(f"Successfully indexed {len(all_chunks)} chunks from {len(processed_filenames)} document(s)!", icon="✅")


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main():
    init_session_state()

    # App Header
    st.markdown("""
        <div class="header-banner">
            <h1 class="header-title">📚 Research Paper Assistant</h1>
            <p class="header-subtitle">
                Retrieval-Augmented Generation (RAG) system with dynamic header/footer removal, 
                FAISS vector retrieval, and automated inline citation verification.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize Embedder Model
    embedder = get_cached_embedder(model_name="all-MiniLM-L6-v2")

    # Try loading existing index on first load if available
    if not st.session_state.auto_loaded:
        load_existing_vector_store(embedder.embedding_dim)
        st.session_state.auto_loaded = True

    # -------------------------------------------------------------------------
    # Sidebar Configuration & Controls
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("📄 Ingestion & Corpus")

        uploaded_files = st.file_uploader(
            "Upload Research Papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or multiple PDF papers to build or rebuild the knowledge base."
        )

        with st.expander("🛠️ Ingestion Options", expanded=False):
            chunk_size = st.slider("Chunk Size (characters)", 256, 1024, 512, step=64)
            chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 256, 64, step=16)
            remove_hf = st.checkbox("Filter Common Headers/Footers", value=True)

        if st.button("🚀 Process & Build Index", type="primary", use_container_width=True):
            if uploaded_files:
                process_and_index_pdfs(
                    uploaded_files=uploaded_files,
                    embedder=embedder,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    remove_headers_footers=remove_hf
                )
            else:
                st.warning("Please upload at least one PDF file before processing.")

        st.divider()

        # System Metrics & Vector Store Status
        st.header("⚡ Vector Index Status")
        if st.session_state.vector_store is not None:
            v_size = st.session_state.vector_store.size
            num_docs = len(st.session_state.ingested_files)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{v_size}</div>
                        <div class="metric-label">Total Chunks</div>
                    </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{num_docs}</div>
                        <div class="metric-label">Documents</div>
                    </div>
                """, unsafe_allow_html=True)

            if st.session_state.ingested_files:
                st.caption("**Indexed Papers:**")
                for fname in st.session_state.ingested_files:
                    st.caption(f"• `{fname}`")

            if st.button("🔄 Reload Index from Disk", use_container_width=True):
                if load_existing_vector_store(embedder.embedding_dim):
                    st.success("Loaded index from disk!")
                else:
                    st.error("No valid index found on disk.")
        else:
            st.info("No active vector store loaded. Upload PDFs or reload existing index.")

        st.divider()

        # LLM & Retrieval Configuration
        st.header("⚙️ Model & Retrieval Settings")

        # Read hosted secrets server-side. Never use these values as widget defaults.
        system_openai_key = get_configured_api_key("OPENAI_API_KEY")
        system_groq_key = get_configured_api_key("GROQ_API_KEY")
        system_gemini_key = get_configured_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")

        # Default to OpenAI if host configured a system key (online mode), else Ollama (local mode)
        default_index = 1 if system_openai_key else 0

        llm_provider = st.selectbox(
            "LLM Provider",
            options=["Ollama (Local LLM)", "OpenAI (Cloud API)", "Groq (Fast Cloud API)", "Google Gemini (Cloud API)"],
            index=0 if not system_openai_key else 1,
            help="Choose Ollama for local execution, or select a Cloud API provider for online hosting."
        )

        provider_api_key = None
        ollama_url = "http://localhost:11434"

        if "Ollama" in llm_provider:
            provider_type = "ollama"
            ollama_url = st.text_input("Ollama Base URL", value="http://localhost:11434", help="Local Ollama server URL")
            
            # Detect installed Ollama models
            detected_models = list_ollama_models(ollama_url)
            if detected_models:
                st.caption(f"🟢 **Ollama Connected**: {len(detected_models)} local model(s) available")
                model_name = st.selectbox(
                    "Select Ollama Model",
                    options=detected_models,
                    index=0,
                    help="Select from models currently installed in your local Ollama instance."
                )
            else:
                st.caption("⚠️ Could not detect installed models automatically. Enter model name manually:")
                model_name = st.text_input("Ollama Model Name", value="mistral:latest", help="e.g., mistral:latest, llama3, gemma:2b")

        elif "OpenAI" in llm_provider:
            provider_type = "openai"
            model_name = st.text_input("OpenAI Model Name", value="gpt-4o-mini", help="e.g., gpt-4o-mini, gpt-4o")
            
            if system_openai_key:
                st.caption("✅ **Default API Key Provided**: Online users don't need to enter a key!")
                custom_key = st.text_input(
                    "Custom OpenAI API Key (Optional)",
                    value="",
                    type="password",
                    help="Leave blank to use the app's default system key, or enter your own key to override."
                )
                provider_api_key = custom_key.strip() if custom_key.strip() else system_openai_key
            else:
                provider_api_key = st.text_input(
                    "OpenAI API Key",
                    value="",
                    type="password",
                    help="Enter your OpenAI API key (sk-...)."
                )

        elif "Groq" in llm_provider:
            provider_type = "groq"
            model_name = st.text_input("Groq Model Name", value="llama-3.3-70b-versatile", help="e.g. llama-3.3-70b-versatile or llama-3.1-8b-instant")
            if system_groq_key:
                st.caption("✅ Groq API key configured securely")
                custom_key = st.text_input(
                    "Custom Groq API key (optional)",
                    value="",
                    type="password",
                    help="Leave blank to use the server-side Streamlit secret."
                )
                provider_api_key = custom_key.strip() or system_groq_key
            else:
                provider_api_key = st.text_input(
                    "Groq API key",
                    value="",
                    type="password",
                    help="Enter a Groq API key for this session."
                )

        elif "Gemini" in llm_provider:
            provider_type = "gemini"
            model_name = st.text_input("Gemini Model Name", value="gemini-2.5-flash", help="Use a model enabled for generateContent in your Gemini project.")
            if system_gemini_key:
                st.caption("✅ Gemini API key configured securely")
                custom_key = st.text_input(
                    "Custom Gemini API key (optional)",
                    value="",
                    type="password",
                    help="Leave blank to use the server-side Streamlit secret."
                )
                provider_api_key = custom_key.strip() or system_gemini_key
            else:
                provider_api_key = st.text_input(
                    "Gemini API key",
                    value="",
                    type="password",
                    help="Enter a Gemini API key for this session."
                )

        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.1)

        with st.expander("🎯 Retrieval Parameters", expanded=False):
            top_k = st.slider("Top-K Chunks", 1, 10, 4)
            score_threshold = st.slider("Min Similarity Score", -1.0, 1.0, 0.0, step=0.05)

        st.divider()

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # -------------------------------------------------------------------------
    # Main Tabs
    # -------------------------------------------------------------------------
    tab_chat, tab_inspector, tab_arch = st.tabs([
        "💬 Research Q&A Assistant",
        "🔍 Vector Search & Chunk Inspector",
        "🏗️ System Architecture"
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Chat Assistant
    # -------------------------------------------------------------------------
    with tab_chat:
        if st.session_state.vector_store is None:
            st.warning("⚠️ Please upload research papers in the sidebar or click **Reload Index from Disk** to begin asking questions.")
        
        # Display existing message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                # Show citations summary & context passages if available
                if "citations" in msg and msg["citations"]:
                    with st.expander("📌 Citation Verification Summary", expanded=False):
                        cit_data = []
                        for c in msg["citations"]:
                            cit_data.append({
                                "Raw Citation": c["raw"],
                                "Document": c["document"],
                                "Page": c["page"],
                                "Status": "✅ Verified" if c["verified"] else "❌ Unverified",
                                "Snippet": c["snippet"][:120] + "..." if c["snippet"] else "N/A"
                            })
                        st.dataframe(pd.DataFrame(cit_data), use_container_width=True)

                if "sources" in msg and msg["sources"]:
                    with st.expander("📖 Retrieved Context Passages", expanded=False):
                        for i, src in enumerate(msg["sources"]):
                            meta = src.get("metadata", {})
                            score = src.get("score", 0.0)
                            st.markdown(f"""
                                <div class="passage-card">
                                    <div class="passage-header">
                                        <span>[Passage {i+1}] {meta.get('source', 'Unknown')} (Page {meta.get('page', '?')})</span>
                                        <span>Similarity: {score:.4f}</span>
                                    </div>
                                    <div class="passage-text">{src.get('text', '')}</div>
                                </div>
                            """, unsafe_allow_html=True)

        # Handle user query input
        if prompt_query := st.chat_input("Ask a question about your uploaded research papers..."):
            if st.session_state.vector_store is None or st.session_state.vector_store.size == 0:
                st.error("Vector store is empty! Please upload and process at least one PDF file.")
                return

            # Render User Query
            st.session_state.messages.append({"role": "user", "content": prompt_query})
            with st.chat_message("user"):
                st.markdown(prompt_query)

            # Generate Assistant Response
            with st.chat_message("assistant"):
                with st.spinner("Retrieving relevant passages and generating answer..."):
                    try:
                        # 1. Retrieve relevant chunks
                        retriever = Retriever(
                            embedder=embedder,
                            store=st.session_state.vector_store,
                            top_k=top_k,
                            score_threshold=score_threshold
                        )
                        retrieved_chunks = retriever.retrieve_with_context(prompt_query)

                        if not retrieved_chunks:
                            response_text = "I couldn't find any relevant passages in the document corpus matching your query with the specified similarity score threshold."
                            st.markdown(response_text)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_text,
                                "sources": [],
                                "citations": []
                            })
                            return

                        # 2. Build Prompt
                        rag_prompt = PromptBuilder.build_rag_prompt(prompt_query, retrieved_chunks)

                        # 3. Instantiate LLM Client & Generate
                        kwargs = {}
                        if provider_type == "ollama":
                            kwargs["base_url"] = ollama_url
                        else:
                            kwargs["api_key"] = provider_api_key

                        llm_client = get_llm_client(
                            provider=provider_type,
                            model_name=model_name,
                            temperature=temperature,
                            **kwargs
                        )
                        
                        raw_llm_response = llm_client.generate(rag_prompt)

                        # 4. Verify Citations
                        verification_res = CitationProcessor.verify_citations(
                            response_text=raw_llm_response,
                            retrieved_chunks=retrieved_chunks
                        )

                        formatted_text = verification_res["text"]
                        parsed_citations = verification_res["citations"]

                        st.markdown(formatted_text)

                        # Citation Verification Expandable Section
                        if parsed_citations:
                            with st.expander("📌 Citation Verification Summary", expanded=True):
                                cit_data = []
                                for c in parsed_citations:
                                    cit_data.append({
                                        "Raw Citation": c["raw"],
                                        "Document": c["document"],
                                        "Page": c["page"],
                                        "Status": "✅ Verified" if c["verified"] else "❌ Unverified",
                                        "Snippet": c["snippet"][:120] + "..." if c["snippet"] else "N/A"
                                    })
                                st.dataframe(pd.DataFrame(cit_data), use_container_width=True)

                        # Context Passages Expandable Section
                        with st.expander("📖 Retrieved Context Passages", expanded=False):
                            for i, src in enumerate(retrieved_chunks):
                                meta = src.get("metadata", {})
                                score = src.get("score", 0.0)
                                st.markdown(f"""
                                    <div class="passage-card">
                                        <div class="passage-header">
                                            <span>[Passage {i+1}] {meta.get('source', 'Unknown')} (Page {meta.get('page', '?')})</span>
                                            <span>Similarity: {score:.4f}</span>
                                        </div>
                                        <div class="passage-text">{src.get('text', '')}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                        # Save to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": formatted_text,
                            "sources": retrieved_chunks,
                            "citations": parsed_citations
                        })

                    except Exception as e:
                        st.error(f"Error generating answer: {e}")

    # -------------------------------------------------------------------------
    # Tab 2: Vector Search & Chunk Inspector
    # -------------------------------------------------------------------------
    with tab_inspector:
        st.subheader("🔍 Semantic Search & Index Inspector")
        st.write("Test semantic retrieval directly against the vector store without sending prompts to an LLM.")

        if st.session_state.vector_store is None or st.session_state.vector_store.size == 0:
            st.info("No active vector store. Ingest PDFs in the sidebar to inspect chunks.")
        else:
            search_query = st.text_input("Enter test search query:", placeholder="e.g. self-attention mechanism, learning rate schedule...")
            search_top_k = st.slider("Results to display", 1, 20, 5, key="inspector_top_k")

            if search_query:
                query_vec = embedder.embed_query(search_query)
                results = st.session_state.vector_store.search(query_vec, top_k=search_top_k)

                st.write(f"Found **{len(results)}** matching chunks:")
                for r in results:
                    meta = r.get("metadata", {})
                    st.markdown(f"""
                        <div class="passage-card">
                            <div class="passage-header">
                                <span>📄 {meta.get('source', 'unknown')} | Page {meta.get('page', '?')} | Chunk #{meta.get('chunk_index', '?')}</span>
                                <span>Score: {r['score']:.4f}</span>
                            </div>
                            <div class="passage-text">{r['text']}</div>
                            <div style="font-size: 0.75rem; color: #64748b; margin-top: 6px;">Chunk ID: <code>{r['chunk_id']}</code></div>
                        </div>
                    """, unsafe_allow_html=True)

            st.divider()
            st.subheader("📊 Indexed Chunks Overview")

            metadata_list = [item["metadata"] for item in st.session_state.vector_store._metadata]
            df_meta = pd.DataFrame(metadata_list)
            st.dataframe(df_meta, use_container_width=True)

    # -------------------------------------------------------------------------
    # Tab 3: System Architecture
    # -------------------------------------------------------------------------
    with tab_arch:
        st.subheader("🏗️ System Architecture & Workflow")
        st.markdown("""
        This application implements an end-to-end modular **Retrieval-Augmented Generation (RAG)** pipeline tailored for academic paper analysis.

        ```
        +-----------------------+     +------------------------+     +-------------------------+
        |   1. PDF Ingestion    | --> |  2. Chunking & Clean   | --> |  3. Dense Embedding     |
        |   (PyMuPDF / fitz)    |     | (LangChain Splitter)   |     | (SentenceTransformers)  |
        +-----------------------+     +------------------------+     +-------------------------+
                                                                                  |
                                                                                  v
        +-----------------------+     +------------------------+     +-------------------------+
        |  6. Citation Verify   | <-- |   5. Grounded LLM Gen  | <-- |   4. FAISS Vector Store |
        | (CitationProcessor)   |     |  (Ollama / OpenAI API) |     |   (Inner Product / Cos) |
        +-----------------------+     +------------------------+     +-------------------------+
        ```

        ### Pipeline Components:
        1. **Document Loader (`src/ingestion/loader.py`)**:
           - Extracts raw text page-by-page using PyMuPDF.
           - Dynamically identifies and removes repeating document headers/footers based on frequency.
           - Handles word hyphenation at line breaks and normalizes Unicode text.
        2. **Document Chunker (`src/ingestion/chunker.py`)**:
           - Splits pages into sliding text chunks using `RecursiveCharacterTextSplitter`.
           - Preserves page provenance and assigns deterministic chunk IDs (`filename_pX_cY`).
        3. **Dense Embedder (`src/ingestion/embedder.py`)**:
           - Encodes chunks into 384-dimensional unit-normalized embedding vectors using `all-MiniLM-L6-v2`.
        4. **Vector Store (`src/retrieval/vector_store.py`)**:
           - Fast vector indexing and nearest-neighbor search using `FAISS` (`IndexFlatIP`).
           - Persistent storage of index (`index.faiss`) and parallel metadata (`metadata.pkl`).
        5. **Retriever & Prompt Builder (`src/retrieval/retriever.py`, `src/generation/prompt.py`)**:
           - Retrieves top-K candidates, applies score threshold filtering, and deduplicates chunks.
           - Formats context passages with explicit `[Context N] (Source, Page)` instructions for strict LLM grounding.
        6. **LLM & Citation Verification (`src/generation/llm.py`, `src/generation/citations.py`)**:
           - Connects to local Ollama models or OpenAI chat completion endpoints.
           - Parses inline citations `[Filename, p. X]` and verifies them against retrieved chunks, tagging invalid citations as `[UNVERIFIED]`.
        """)


if __name__ == "__main__":
    main()
