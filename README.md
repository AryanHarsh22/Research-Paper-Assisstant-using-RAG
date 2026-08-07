# 📚 Research Paper Assistant using RAG

A modular, deployment-ready **Retrieval-Augmented Generation (RAG)** application for academic paper analysis. It combines PDF cleaning, SentenceTransformer embeddings, FAISS retrieval, grounded generation through local or cloud LLMs, and automated inline citation verification in an interactive Streamlit interface.

---

## ✨ Features

- **📄 Smart PDF Ingestion & Cleaning (`fitz` / PyMuPDF)**:
  - Dynamically detects and strips repeating document headers/footers based on frequency analysis.
  - Automatically filters out standalone page numbers and cleans hyphenations at line breaks.
  - Preserves exact page numbers and document provenance.

- **🧩 Chunking & Dense Vector Embeddings**:
  - Overlapping text chunking via LangChain's `RecursiveCharacterTextSplitter`.
  - 384-dimensional dense vector embeddings generated via `SentenceTransformers` (`all-MiniLM-L6-v2`).

- **⚡ FAISS Vector Store & Semantic Retriever**:
  - Fast vector similarity search using `FAISS` (`IndexFlatIP` with L2-normalized vectors).
  - Configurable Top-K retrieval, cosine similarity score thresholding, and chunk deduplication.
  - Persistent vector store serialization (`index.faiss` and `metadata.pkl`).

- **🤖 Multiple LLM Backends**:
  - **Local Models via Ollama**: Connects to locally running models (`mistral:latest`, `llama3`, `gemma:2b`, `deepseek-r1`) with auto-detection of installed models.
  - **OpenAI**: Uses models such as `gpt-4o-mini` through the OpenAI API.
  - **Groq**: Fast hosted inference with models such as `llama-3.3-70b-versatile`.
  - **Google Gemini**: Uses the current stable `gemini-3.6-flash` model through the Gemini API.
  - Hosted keys are loaded server-side from Streamlit Secrets and are never displayed in the application.

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
|   (CitationProcessor)   |     | (Local / Cloud LLM APIs) |     |   (Inner Product / Cos)   |
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
│       ├── llm.py                 # Ollama, OpenAI, Groq & Gemini clients
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

The sidebar supports Ollama, OpenAI, Groq, and Google Gemini. Hosted API keys should be configured as server-side secrets. A blank optional override field means the app will use the configured secret without sending it to the browser.

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

### Option B: Cloud LLMs

Create `.streamlit/secrets.toml` for local development:

```toml
OPENAI_API_KEY = "sk-your-openai-key"
GROQ_API_KEY = "gsk_your-groq-key"
GEMINI_API_KEY = "your-gemini-key"
```

Add only the providers you intend to use. This file is excluded by `.gitignore` and must never be committed.

| Provider | Default model | API key page | Free API tier |
| :--- | :--- | :--- | :---: |
| OpenAI | `gpt-4o-mini` | [OpenAI Platform](https://platform.openai.com/api-keys) | Usually requires prepaid API credit |
| Groq | `llama-3.3-70b-versatile` | [GroqCloud Console](https://console.groq.com/keys) | Yes, subject to rate limits |
| Google Gemini | `gemini-3.6-flash` | [Google AI Studio](https://ai.google.dev/aistudio) | Yes, for eligible models and regions |

OpenAI API billing is separate from ChatGPT Free, Plus, or Pro subscriptions.

Environment variables can be used instead of `secrets.toml`:

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="..."
```

On PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:GROQ_API_KEY = "gsk_..."
$env:GEMINI_API_KEY = "..."
```

### Streamlit Community Cloud

1. Deploy `app.py` from this repository at [share.streamlit.io](https://share.streamlit.io/).
2. Select Python `3.10` in **Advanced settings**.
3. Paste the TOML keys above into **Advanced settings → Secrets**.
4. Save the settings and reboot the app.
5. Select a cloud provider in **Model & Retrieval Settings**.

Ollama at `localhost:11434` is only available locally. It will not be reachable from Streamlit Community Cloud unless a separately hosted Ollama endpoint is supplied.

For Docker and other hosting options, see [the cloud deployment guide](docs/CLOUD_DEPLOYMENT.md).

### API troubleshooting

- **Gemini `404 NOT_FOUND`**: The selected model is unavailable. Use the current default, `gemini-3.6-flash`, or another model enabled for `generateContent` in your project.
- **Groq/OpenAI `401`**: The key is missing, invalid, or revoked. Create a new key and update Streamlit Secrets.
- **`403`**: The account, project, model, or hosting network is not permitted.
- **`429`**: The provider's request or token limit has been reached. Wait for the quota window to reset or upgrade the provider plan.
- **Groq edge error `1010`**: The request was rejected before reaching the normal Groq API. Reboot the deployment and contact Groq support with the Cloudflare Ray ID if it persists.

Never post API keys in screenshots, issues, logs, or commits. Revoke and rotate a key immediately if it is exposed.

### OpenAI example

1. Obtain an API key from [OpenAI Platform](https://platform.openai.com/).
2. Set your environment variable or add it to Streamlit Secrets:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   *The hosted secret remains server-side; the sidebar field is only for an optional session override.*
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
