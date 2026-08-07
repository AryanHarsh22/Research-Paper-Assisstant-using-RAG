# 📚 Research Paper Assistant using RAG

A modular, production-ready **Retrieval-Augmented Generation (RAG)** application designed for academic paper analysis. Features dynamic PDF header/footer cleaning, dense vector indexing with FAISS, support for both local (**Ollama**) and cloud (**OpenAI**) LLMs, and an automated **Inline Citation Verification Engine**.

---

## ✨ Features

- **📄 Smart PDF Ingestion & Cleaning (`fitz` / PyMuPDF)**:
  - Dynamically detects and strips repeating document headers/footers based on frequency analysis.
  - Automatically filters out standalone page numbers and cleans hyphenations at line breaks.
  - Preserves exact page numbers and document provenance.

- **🧩 Chanking & Dense Vector Embeddings**:
  - Overlapping text chunking via LangChain's `RecursiveCharacterTextSplitter`.
  - 384-dimensional dense vector embeddings generated via `SentenceTransformers` (`all-MiniLM-L6-v2`).

- **⚡ FAISS Vector Store & Semantic Retriever**:
  - Fast vector similarity search using `FAISS` (`IndexFlatIP` with L2-normalized vectors).
  - Configurable Top-K retrieval, cosine similarity score thresholding, and chunk deduplication.
  - Persistent vector store serialization (`index.faiss` and `metadata.pkl`).

- **🤖 Dual LLM Backend Support**:
  - **Local Models via Ollama**: Connects to locally running models (`mistral:latest`, `llama3`, `gemma:2b`, `deepseek-r1`) with auto-detection of installed models.
  - **Cloud Models via OpenAI**: Supports `gpt-4o-mini`, `gpt-4o`, and custom API keys.

- **📌 Automated Citation Verification Engine**:
  - Automatically parses inline citations in LLM responses (e.g., `[attention.pdf, p. 4]` or `[Passage 1, p. 2]`).
  - Cross-references cited claims against actual retrieved PDF context chunks.
  - Resolves index aliases (e.g. `Context 1` -> `attention.pdf`) and marks status in real-time as **`✅ Verified`** or **`❌ Unverified`**.

- **🖥️ Interactive Streamlit Dashboard**:
  - **💬 Research Q&A Chat**: Conversational interface with citation badges and expandable context passage views.
  - **🔍 Semantic Search Inspector**: Test vector retrieval directly without generating LLM completions.
  - **🏗️ System Architecture**: Embedded pipeline diagram and workflow summary.

---

## 🏗️ System Architecture

```
+-------------------------+     +--------------------------+     +---------------------------+
|    1. PDF Ingestion     | --> |   2. Clean & Chunking    | --> |  3. Dense Vector Embed    |
|   (PyMuPDF / fitz)      |     | (LangChain Splitter)     |     | (SentenceTransformers)    |
+-------------------------+     +--------------------------+     +---------------------------+
                                                                               |
                                                                               v
+-------------------------+     +--------------------------+     +---------------------------+
|  6. Citation Verifier   | <-- |   5. Grounded LLM Gen    | <-- |   4. FAISS Vector Store   |
|   (CitationProcessor)   |     |  (Ollama / OpenAI API)   |     |   (Inner Product / Cos)   |
+-------------------------+     +--------------------------+     +---------------------------+
```

---

## 📁 Repository Structure

```text
Research-Paper-Assisstant-using-RAG/
├── app.py                         # Streamlit application UI & main workflow
├── requirements.txt               # Python package dependencies
├── README.md                      # Project documentation
├── src/
│   ├── ingestion/
│   │   ├── loader.py              # PDFLoader: PyMuPDF page parsing & header/footer cleaning
│   │   ├── chunker.py             # DocumentChunker: Text splitting & metadata tracking
│   │   └── embedder.py            # DocumentEmbedder: SentenceTransformer embedding engine
│   ├── retrieval/
│   │   ├── vector_store.py        # FAISSVectorStore: FAISS index management & persistence
│   │   └── retriever.py           # Retriever: Top-K search, score filtering & deduplication
│   └── generation/
│       ├── prompt.py              # PromptBuilder: Grounded RAG prompt assembly
│       ├── llm.py                 # LLMClient: Ollama & OpenAI API interfaces + model listing
│       └── citations.py           # CitationProcessor: Inline citation parser & verifier
├── Learning_Process/
│   ├── verify_ingestion.py        # Ingestion pipeline unit tests
│   ├── verify_vector_store.py     # FAISS vector store unit tests
│   ├── verify_retriever.py        # Retriever unit tests
│   ├── verify_generation.py       # Generation & citation engine unit tests
│   └── verify_app_integration.py  # End-to-end integration verification script
└── data/
    ├── uploads/                   # Uploaded raw PDF files
    └── vector_store/              # Serialized FAISS index & metadata storage
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- **Python**: `3.9+` (Python `3.10` or `3.11` recommended)
- **Git**

### 2. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/AryanHarsh22/Research-Paper-Assisstant-using-RAG.git
cd Research-Paper-Assisstant-using-RAG

# Create virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 💻 Running the Application

Launch the Streamlit web app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🤖 LLM Configuration Guide

### Option A: Local Models via Ollama (Free & Private)

1. Download and install [Ollama](https://ollama.com/).
2. Pull your desired open-source model in terminal:
   ```bash
   ollama pull mistral
   # or
   ollama pull llama3
   ```
3. Start the app (`streamlit run app.py`). In the sidebar under **`⚙️ Model & Retrieval Settings`**:
   - Select **LLM Provider**: `Ollama`
   - Select your installed model from the auto-detected dropdown list (e.g. `mistral:latest`).

### Option B: Cloud Models via OpenAI

1. Obtain an API key from [OpenAI Platform](https://platform.openai.com/).
2. Set your environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   *Or enter your API key directly in the app's sidebar configuration.*
3. In the sidebar:
   - Select **LLM Provider**: `OpenAI`
   - Choose your model (e.g., `gpt-4o-mini` or `gpt-4o`).

---

## 🧪 Verification & Unit Testing

Run the test suite to verify pipeline components end-to-end:

```bash
# Run Generation & Citation verification tests
python Learning_Process/verify_generation.py

# Run full end-to-end RAG pipeline integration test
python Learning_Process/verify_app_integration.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check out the [Issues page](https://github.com/AryanHarsh22/Research-Paper-Assisstant-using-RAG/issues).
