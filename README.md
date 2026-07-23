# 🧠 FinGenius-AI

An AI-powered compliance and financial auditing system. It ingests company documents (PDFs or URLs) and regulatory guidelines, indexes them for retrieval, and uses LLM agents to answer questions, generate structured compliance reports, run SWOT analysis, and produce scored audit reports — pulling in live market data where relevant.

## What it does

- **Document ingestion** — Loads PDFs or web URLs, chunks the text, and embeds it using Hugging Face sentence-transformers.
- **Vector search (RAG)** — Stores embeddings in a FAISS vector index for fast semantic retrieval, so questions are answered using the actual source documents rather than the LLM's general knowledge.
- **Conversational Q&A** — Ask free-form questions and get answers grounded in the ingested documents, with cited sources.
- **Automated compliance reports** — Generates structured, multi-section compliance breakdowns (policies, financial/tax compliance, legal & regulatory obligations, data privacy, risk management, SWOT analysis) for a company against a set of regulatory standards.
- **Regulatory benchmarking** — Extracts and summarizes official regulatory guidelines (e.g. RBI, SEBI, Income Tax Dept.) into the same structured format, so company policy can be benchmarked against it.
- **Multi-agent audit scoring** — A team of LLM agents (using the `agno`/`phidata` agent frameworks) cross-checks company data against regulations, pulls live financial data and news (via `yfinance` and DuckDuckGo search), and produces a compliance score with justification.

## Architecture

| Component | Description |
|---|---|
| **Ingestion pipeline** | `PyPDFLoader` / `UnstructuredURLLoader` → `RecursiveCharacterTextSplitter` → `HuggingFaceEmbeddings` → FAISS index (persisted as `.pkl`) |
| **Retrieval QA** | LangChain `RetrievalQAWithSourcesChain` backed by Groq-hosted LLMs (Qwen 2.5, Llama 3.3, DeepSeek-R1) |
| **Agent layer** | `agno`/`phidata` agents with tool access to DuckDuckGo web search and Yahoo Finance, orchestrated as a multi-agent team for report generation and scoring |
| **Document store** | MongoDB, used to track source document URLs for ingestion |
| **API** | FastAPI service exposing ingestion, Q&A, and report-generation endpoints |

## Tech Stack

Python · FastAPI · LangChain · FAISS (vector search) · Hugging Face sentence-transformers · Groq LLM API · `agno` / `phidata` agent frameworks · MongoDB (PyMongo) · yfinance · DuckDuckGo Search · PyPDF

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)
- A MongoDB connection string (for tracking source document URLs)
- (Optional) Google/Hugging Face API keys, depending on which embedding/LLM backends you use

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/TanviMadani/FinGenius-AI.git
cd FinGenius-AI
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Create a `.env` file in the project root (see `.env.example`):
```
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
HF_TOKEN=your_huggingface_token_here
MONGO_URI=your_mongodb_connection_string_here
```

> ⚠️ Never commit your real `.env` file. Rotate any keys that were previously exposed.

**5. Run the API server**
```bash
python app.py
# or, for the multi-agent audit-scoring service:
python x.py
```

`app.py` serves on `http://127.0.0.1:7000`; `x.py` serves on `http://0.0.0.0:6700`.

## API Endpoints (app.py)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/fetch-urls` | Pulls document URLs from MongoDB and embeds them |
| POST | `/embed` | Embeds a given URL/PDF into the vector store |
| POST | `/query` | Answers a free-form question using the indexed documents |
| GET | `/get-answer` | Query variant using a separate vector store |
| POST | `/company-report` | Generates a structured compliance report for a company |
| POST | `/authority-report` | Generates a structured summary of regulatory guidelines |
| POST | `/agent-report` | Generates a detailed agent-driven report |

## API Endpoints (x.py — multi-agent audit scoring)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Runs the multi-agent audit pipeline: compliance report + compliance score |
| POST | `/ans` | General Q&A agent with web search + market data access |

## Deployment

Includes a `render.yaml` for deployment on [Render](https://render.com/).

## Notes

- Vector indexes are persisted locally as `.pkl` files (`docs.pkl`, `auth.pkl`, `final.pkl`, `authority.pkl`) — these are excluded from version control since they're generated artifacts.
- `x.py` currently uses hardcoded sample compliance data for demo/testing purposes.

## License

MIT
