# Spectre — Legal Document Extraction Pipeline

Extract structured data from legal documents (fee proposals, engagement letters, NDAs, MSAs) using OCR + a local LLM. Runs entirely offline — no API keys required.

## Quick Start

### Prerequisites

- Docker Desktop (running)
- ~8 GB free RAM recommended
- ~5 GB free disk space for Docker images + model

### Run

```bash
docker compose up --build
```

**First run:** The build takes ~5-10 minutes. After containers start, the LLM model (2.1 GB) auto-downloads in the background — this takes 2-10 minutes depending on your connection. The server and UI are accessible immediately during download.

**Subsequent runs:** Instant. No downloads.

### Access

| Service | URL |
|---------|-----|
| **Streamlit UI** | http://localhost:5070 |
| **API Docs** (Scalar) | http://localhost:5060/docs |
| **API Health** | http://localhost:5060/health |

### Test with a Sample Document

```bash
# Upload a PDF for OCR
curl -X POST -F "file=@sample_docs/fee_proposal_test.pdf" http://localhost:5060/upload

# Upload and extract structured fields
curl -X POST -F "file=@sample_docs/fee_proposal_test.pdf" http://localhost:5060/extract
```

Or upload via the Streamlit UI at http://localhost:5070.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│  llama-cpp      │
│  (Port 5070)│     │  (Port 5060) │     │  (Granite 4.1) │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                     ┌─────▼──────┐
                     │  PaddleOCR │
                     │  (Threaded)│
                     └────────────┘
```

1. **Upload** PDF via Streamlit UI or API
2. **OCR** converts pages to text (PaddleOCR, 4 thread pool)
3. **Classifier** detects document type (NDA, MSA, fee proposal, etc.)
4. **LLM** extracts structured fields using Granite 4.1 (runs locally, no API keys)
5. **Draft** generates grounded legal memo with source citations

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Scalar API reference (interactive docs) |
| `/health` | GET | Service health check |
| `/upload` | POST | Upload PDF, get OCR text + classification |
| `/extract` | POST | Upload PDF, get structured extraction |
| `/draft` | POST | Generate grounded legal memo |
| `/feedback` | POST | Submit operator edits for improvement loop |
| `/evaluate` | POST | Run evaluation metrics against ground truth |

Full request/response schemas with examples at http://localhost:5060/docs.

## Configuration

Set environment variables in `docker-compose.yml` or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODE` | `local` | Use `local` (Granite 4.1 via llama-cpp) or `groq` (free API) |
| `MODEL_PATH` | `/models/granite-4.1-3b-Q4_K_M.gguf` | Path to the GGUF model file |
| `DATABASE_URL` | `sqlite:////home/user/app/data/spectre.db` | Database connection string |

## Tech Stack

| Component | Choice |
|-----------|--------|
| OCR | PaddleOCR (PP-OCRv5, threaded) |
| LLM | Granite 4.1 3B via llama-cpp-python |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Database | SQLite |
| Package mgmt | uv (10x faster than pip) |
| Vector Store | ChromaDB (embedded) |

## Project Structure

```
backend/
├── Dockerfile
├── pyproject.toml           # All deps pinned to exact versions
├── entrypoint.sh            # Starts server, downloads model in background
└── app/
    ├── main.py              # FastAPI app + routes
    ├── config.py            # Environment config
    ├── models.py            # Pydantic schemas
    ├── ocr/                 # PaddleOCR engine + classifier
    ├── extraction/          # LLM extraction with schema enforcement
    ├── retrieval/           # ChromaDB vector store
    ├── draft/               # Draft generation with citations
    ├── feedback/            # Operator edit capture + reinforcement
    ├── evaluation/          # LLM-as-judge evaluation harness
    └── telemetry/           # OpenTelemetry tracing

frontend/
├── Dockerfile
├── app.py                   # Streamlit UI
└── requirements.txt

sample_docs/                 # Test PDFs for quick start
```
