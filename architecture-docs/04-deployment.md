# Deployment

## Docker Compose Topology

![Deployment](./diagrams/out/05-deployment.png)

The entire system runs as 3 Docker containers orchestrated by `docker compose`:

### Service Matrix

| Service | Image | Ports | Depends On | Health Check |
|---------|-------|-------|------------|--------------|
| **backend** | `spectre-backend` (build) | `8080:8000` | jaeger | `GET /health` every 30s |
| **frontend** | `spectre-frontend` (build) | `5070:8501` | backend | None (Streamlit) |
| **jaeger** | `jaegertracing/all-in-one:1.63.0` | `16686`, `4317`, `4318` | — | None |

### Environment Variables

#### Backend

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama GPU endpoint (empty = CPU fallback) |
| `MODEL_PATH` | `/models/ibm-granite_granite-4.1-3b-Q4_K_M.gguf` | GGUF model for llama-cpp-python |
| `DATABASE_URL` | `sqlite:////home/user/app/data/spectre.db` | SQLite database path |
| `CHROMA_PERSIST_DIR` | `/home/user/app/data/chroma` | ChromaDB persistence directory |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://jaeger:4317` | Jaeger OTLP gRPC endpoint |
| `OTEL_SERVICE_NAME` | `spectre-backend` | Service name in traces |

#### Frontend

| Variable | Default | Purpose |
|----------|---------|---------|
| `API_BASE_URL` | `http://backend:8000` | Backend API URL |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./models` | `/models` | GGUF model files |
| `./data` | `/home/user/app/data` | SQLite DB + ChromaDB persistence |
| `./sample_docs` | `/home/user/app/sample_docs` | PDF documents for testing |
| `./datasets` | `/home/user/app/datasets` | CUAD evaluation data |
| `./paddlex_cache` | `/home/user/.paddlex` | PaddleOCR model cache |

---

## LLM Backend Detection

```python
# backend/app/extraction/llm_extractor.py:55-85
def get_llm():
    # 1. Try Ollama (GPU)
    if OLLAMA_BASE_URL:
        client = OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
        client.models.list()  # health check
        return client  # GPU mode

    # 2. Fall back to CPU
    return Llama(model_path=MODEL_PATH, n_ctx=16384)  # CPU mode
```

The system auto-detects at startup:
- **GPU path**: Ollama running on the host at `host.docker.internal:11434`
- **CPU path**: llama-cpp-python with Granite 4.1 3B GGUF model (auto-downloaded from HuggingFace)

---

## Jaeger Telemetry

Traces are sent via OTLP gRPC from the backend to Jaeger:

```python
# backend/app/telemetry/tracing.py:22-43
def setup_tracing(service_name=None):
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        return False  # silent no-op
    # ... configures OTLPSpanExporter + BatchSpanProcessor
    FastAPIInstrumentor.instrument_app(app)
```

Tracing is fully optional. If `OTEL_EXPORTER_OTLP_ENDPOINT` is not set, no traces are sent and the app runs identically.

**Jaeger UI**: `http://127.0.0.1:16686` (Search tab for trace details)

---

## Getting Started

```bash
# Prerequisites
# 1. Docker Desktop (8 GB RAM minimum)
# 2. Ollama running on host (optional, GPU path)
# 3. Pull the LLM model: ollama pull granite4.1:3b

# Build and start
cp .env.example .env
docker compose up --build

# Access
# - Frontend:   http://127.0.0.1:5070
# - API docs:   http://127.0.0.1:8080/docs
# - Jaeger UI:  http://127.0.0.1:16686
```

---

## Dockerfile Details

### Backend (`backend/Dockerfile`)

```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 poppler-utils curl
WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system --no-cache -e ".[dev]"
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user . .
ENV FLAGS_enable_pir_api=0
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
```

Key design decisions:
- **`uv` replaces pip** — resolves complex dependency tree (chromadb + paddleocr + onnxruntime) in <1s instead of 10+ minutes
- **`python:3.13-slim`** — minimal base image, ~120 MB
- **`FLAGS_enable_pir_api=0`** — disables PaddlePaddle PIR executor (bug in 3.3.x, safe with pinned 3.2.2)
- **`entrypoint.sh`** — starts uvicorn immediately, downloads GGUF model in background

### Frontend (`frontend/Dockerfile`)

```dockerfile
FROM python:3.13-slim
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

---

## Known Issues

| # | Concern | Location |
|---|---------|----------|
| 1 | Ollama requires host Docker network — `host.docker.internal` is Docker Desktop specific | `docker-compose.yml:15` |
| 2 | Jaeger System Architecture tab requires ES/Cassandra — in-memory storage only supports Search | `docker-compose.yml` |
| 3 | PaddleOCR models download at first runtime (~500 MB downloaded once, cached) | `engine.py:23-29` |
| 4 | Docker Desktop default 3.46 GB RAM is insufficient — must increase to 8 GB | docs |

## Environment Matrix

| Scenario | OLLAMA_BASE_URL | MODEL_PATH | GPU? | Notes |
|----------|----------------|------------|------|-------|
| Docker + Ollama on host | `http://host.docker.internal:11434` | `/models/...gguf` | Yes | Recommended. Fastest inference. |
| Docker + CPU only | (empty) | `/models/...gguf` | No | Slower. ~2 min per extraction vs ~30s on GPU. |
| Standalone (no Docker) | `http://localhost:11434` | `./models/...gguf` | Yes | For development. Run uvicorn directly. |
