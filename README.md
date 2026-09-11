# 🩺 Production-Grade Medical RAG Chatbot

An enterprise-ready, context-aware Medical Assistant powered by Retrieval-Augmented Generation (RAG). Built using **LangChain**, **Groq (OpenAI GPT-OSS 20B)**, **Pinecone Vector DB**, and **BAAI/bge-small-en-v1.5** embeddings, containerized with **Docker**, and deployed on **Google Cloud Run** (`asia-south2`).

🌐 **[Launch Live Application](https://medical-chatbot-1032077414344.asia-south2.run.app)**

---

## 🏗️ Project Architecture

```text
                              +-------------------+
                              |   User Request    |
                              +---------+---------+
                                        |
                                        v
                              +---------+---------+
                              |  Flask Web Server |
                              +---------+---------+
                                        |
                                        v
                            +-----------+-----------+
                            | Lazy Loading Pipeline |
                            +-----------+-----------+
                                        |
                   +--------------------+--------------------+
                   |                                         |
                   v                                         v
 +-----------------+-----------------+     +-----------------+-----------------+
 |  BAAI/bge-small-en-v1.5 Embedding |     |  Pinecone Vector DB (k=2)         |
 +-----------------+-----------------+     +-----------------+-----------------+
                   |                                         |
                   +--------------------+--------------------+
                                        |
                                        v
                              +---------+---------+
                              |  LangChain LCEL   |
                              +---------+---------+
                                        |
                                        v
                              +---------+---------+
                              |  Groq Inference   |
                              +---------+---------+
```

1. **User Query**: Input received via the Flask `/get` endpoint.
2. **Context Retrieval**: Query embedded on-the-fly using `BAAI/bge-small-en-v1.5` and matched against index vectors in Pinecone (k=2).
3. **Prompt Augmentation**: Top relevant passages injected into a strict system prompt preventing non-medical responses and masking system internal phrases.
4. **LLM Generation**: High-throughput inference executed via Groq's LPUs running `openai/gpt-oss-20b`.
5. **Session Memory**: In-memory `RunnableWithMessageHistory` tracks multi-turn conversational context per session.

---

## 📚 Dataset & Ingestion

This RAG chatbot is indexed using content from ***The Gale Encyclopedia of Medicine (2nd Edition)***, edited by Jacqueline L. Longe (Gale Group).

- **Source Material:** Multi-volume medical reference covering diseases, symptoms, tests, and treatments.
- **Storage & Indexing:** Extracted text chunks are embedded using `BAAI/bge-small-en-v1.5` and stored in a vector index on **Pinecone**.
- **Notice:** The raw source PDF is excluded from this repository to respect publisher copyright.

---

## 🚀 Key Features & Backend Innovations

- **Lazy-Initialization Pattern:** Heavy machine learning models (`HuggingFaceEmbeddings`) and Pinecone connections are initialized lazily inside route execution rather than module load time. This enables sub-second Gunicorn startup, satisfying Cloud Run health checks and preventing 503 deployment failures.
- **Stateful Memory Management:** Includes a `/clear` endpoint to purge active chat histories dynamically.
- **Custom Responsive UI:** Dark-themed dashboard designed with CSS flexbox layout, async JavaScript fetch calls, auto-scrolling message streams, and real-time backend status indicators.
- **Production Containerization:** Optimized `Dockerfile` running Python 3.10-slim with single-worker, multi-threaded Gunicorn execution (`--threads 8`).

---

## 🛠️ Tech Stack

| Component | Technology Used |
|---|---|
| **Framework** | Flask, Gunicorn |
| **Orchestration** | LangChain (LCEL) |
| **LLM Provider** | Groq (`openai/gpt-oss-20b`) |
| **Embeddings** | HuggingFace (`BAAI/bge-small-en-v1.5`) |
| **Vector Database** | Pinecone |
| **Deployment** | Google Cloud Run (`asia-south2`) |
| **Secret Management** | Google Secret Manager |
| **Frontend** | HTML5, CSS3, JavaScript (Fetch API) |

---

## 📁 Repository Structure

```text
Medical-RAG-Chatbot/
│
├── src/
│   ├── __init__.py
│   ├── helper.py          # Data loaders, text splitters, & embedding downloads
│   └── prompt.py          # LangChain ChatPromptTemplate definitions
├── static/
│   └── style.css          # UI styling
├── templates/
│   └── chat.html          # Web interface
│
├── env.example             # Local environment template
├── .gitignore              # Excluded data, environments, and secrets
├── app.py                  # Core Flask server & lazy-loading RAG logic
├── Dockerfile               # Container spec for Cloud Run
├── LICENSE                  # Repository license
├── README.md                # Project documentation
├── requirements.txt         # Production Python dependencies
├── store_index.py           # One-time script to chunk PDF and build Pinecone index
├── template.py               # Project scaffolding / boilerplate generator
└── trials.ipynb               # Notebook containing pipeline development trials
```

---

## 🔐 Security & Local Setup

We use **Google Secret Manager** in production to securely inject API keys into the container. However, an `env.example` file is included so developers can easily see which environment variables are required to run the project locally.

**1. Clone the repository:**

```bash
git clone https://github.com/OjasPal/Medical-RAG-Chatbot.git
cd Medical-RAG-Chatbot
```

**2. Create a virtual environment & install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure environment variables:**

Rename `env.example` to `.env` and insert your API credentials.

**4. Run locally:**

```bash
python app.py
```

---

## ☁️ Production Deployment

This application is containerized via `Dockerfile` and deployed to Google Cloud Run using Secret Manager for credential handling:

```bash
gcloud run deploy medical-chatbot \
  --source . \
  --region asia-south2 \
  --memory 2Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --set-secrets GROQ_API_KEY=GROQ_API_KEY:latest,PINECONE_API_KEY=PINECONE_API_KEY:latest
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](https://github.com/OjasPal/Medical-RAG-Chatbot/blob/main/LICENSE) file for details.
