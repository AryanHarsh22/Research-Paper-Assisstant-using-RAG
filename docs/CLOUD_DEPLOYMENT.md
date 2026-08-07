# 🌐 Cloud Deployment Guide

This guide details how to deploy the **Research Paper Assistant using RAG** application to various cloud platforms.

---

## 📌 Deployment Options Overview

| Platform | Difficulty | Cost | Recommended For |
| :--- | :---: | :---: | :--- |
| **Streamlit Community Cloud** | 🟢 Easy (1-click) | Free | Demos & Public Portfolio |
| **HuggingFace Spaces** | 🟢 Easy | Free | Public Open-Source Demos |
| **Docker Container** | 🟡 Medium | Custom | Local Server / VPS |
| **GCP Cloud Run / AWS ECS** | 🔴 Advanced | Pay-per-use | Scalable Enterprise Deployments |

---

## 1. Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud offers free 1-click hosting directly connected to your GitHub repository.

### Prerequisites
- Push your latest code to GitHub (`origin/main`).
- Create a free account at [share.streamlit.io](https://share.streamlit.io/).

### Steps
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New App** -> Select your repository: `AryanHarsh22/Research-Paper-Assisstant-using-RAG`.
3. Set **Main file path**: `app.py`.
4. (Optional) Configure **Secrets** for Cloud LLM keys:
   ```toml
   # Advanced settings -> Secrets:
   OPENAI_API_KEY = "sk-..."
   GROQ_API_KEY = "gsk_..."
   GEMINI_API_KEY = "AIzaSy..."
   ```
5. Click **Deploy!** Your app will be live at `https://<your-app-name>.streamlit.app`.

---

## 2. Deploying to HuggingFace Spaces

### Steps
1. Log in to [HuggingFace](https://huggingface.co/) and click **New Space**.
2. Name your space (e.g. `research-paper-assistant`).
3. Select SDK: **Streamlit**.
4. Choose **Public** or **Private** space.
5. Clone the space repository locally or push your files:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/research-paper-assistant
   git push hf main
   ```

---

## 3. Containerized Deployment via Docker

Use the included `Dockerfile` to build and run the container locally or on a virtual private server (VPS).

### Build Docker Image
```bash
docker build -t research-paper-assistant .
```

### Run Docker Container
```bash
docker run -d \
  -p 8501:8501 \
  -e OPENAI_API_KEY="sk-..." \
  -e GROQ_API_KEY="gsk_..." \
  --name rag-assistant \
  research-paper-assistant
```

Access the app at `http://localhost:8501`.

---

## 4. Deploying to GCP Cloud Run (Serverless Container)

### Build & Push to Google Artifact Registry
```bash
# 1. Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Submit build to Cloud Build
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/research-paper-assistant

# 3. Deploy to Cloud Run
gcloud run deploy research-paper-assistant \
  --image gcr.io/YOUR_PROJECT_ID/research-paper-assistant \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi
```

---

## 🔒 Cloud Environment Considerations

- **Ollama in the Cloud**: Local `localhost:11434` is unavailable in serverless environments. When hosted online, users can select **OpenAI**, **Groq**, or **Google Gemini** in the sidebar, or provide a remote Ollama server URL.
- **Storage Persistence**: Local temporary uploads (`data/uploads`) and indices (`data/vector_store`) reset when container instances restart. For persistent storage in enterprise deployments, attach an AWS S3 bucket or Google Cloud Storage.
