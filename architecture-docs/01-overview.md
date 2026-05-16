# System Overview

## System Context

![System context](./diagrams/out/01-context.png)

Spectre is a legal document extraction and grounded draft generation pipeline. It ingests messy legal PDFs, extracts structured fields using OCR + LLM, and generates a grounded memo with inline citations.

### Actors

| Actor | Role |
|-------|------|
| **Operator** | Uploads PDFs, reviews extracted fields, edits corrections, generates memos |
| **System** | The Spectre backend + frontend, runs locally with zero external API calls |

### External Dependencies

| Dependency | Role | Connection |
|------------|------|------------|
| **Ollama** (host, optional) | GPU LLM inference for Granite 4.1 | `host.docker.internal:11434` |
| **HuggingFace** | CUAD dataset for benchmarks (downloaded once) | Internet on first benchmark run |
| **Jaeger** (Docker sidecar) | OpenTelemetry trace collection and visualization | OTLP gRPC on port 4317 |

All LLM inference is local. No API keys, no external model calls.

---

## Request Lifecycle

The happy path for a document extraction request follows these steps:

### 1. Upload and OCR

```ts
// frontend/app.py:84-120
resp = requests.post(
    f"{API_BASE_URL}/extract/stream",
    files={"file": (filename, file_bytes, "application/pdf")},
    stream=True, timeout=300,
)
```

The frontend streams the PDF to `POST /extract/stream`. The backend:

1. Saves the uploaded PDF to a temp path
2. Calls `process_document()` which runs PaddleOCR on each page
3. Classifies the document type via keyword scoring
4. Streams OCR progress events back to the frontend via SSE

```python
# backend/app/main.py:343-362
ocr_result = process_document(upload_path)
yield "event: ocr_result\ndata: " + json.dumps(ocr_result.to_dict()) + "\n\n"
```

### 2. LLM Extraction

After OCR completes, the backend runs LLM-based structured extraction:

```python
# backend/app/main.py:352-356
extracted = extract_fields(
    raw_text=ocr_result.full_text,
    doc_type=ocr_result.doc_type,
)
yield "event: result\ndata: " + json.dumps({"ocr": ocr_result.to_dict(), "extracted": extracted}) + "\n\n"
```

The `extract_fields()` function:

1. Auto-detects the LLM backend (Ollama GPU > llama-cpp-python CPU)
2. Selects the Pydantic schema matching the document type
3. Prompts the LLM with the OCR text and returns structured JSON
4. Sanitizes output to remove hallucinated values

### 3. Correction & Feedback

The operator can edit any extracted field in the Review & Edit tab. Corrections are submitted to `POST /feedback`:

```python
# backend/app/main.py:417-426
pair = store_correction(
    original=payload.get("original", {}),
    corrected=payload.get("corrected", {}),
    document_type=payload.get("document_type", "unknown"),
)
```

Corrections are stored in memory and fed back as few-shot examples in subsequent extraction prompts.

### 4. Draft Generation

The operator clicks "Generate Draft" which calls `POST /draft`:

1. The backend indexes the OCR text by page in ChromaDB
2. Queries the vector store for passages relevant to extracted fields
3. Feeds passages + extracted data to the LLM
4. Generates a 4-section legal memo with `[Source: page X]` citations

### 5. Evaluation

Three benchmark modes run against CUAD contracts from HuggingFace:

| Benchmark | What It Measures |
|-----------|-----------------|
| **Classifier** | Keyword classifier accuracy against 13,155 labeled clauses |
| **OCR** | Character Error Rate (CER) on 10 real CUAD PDFs vs ground truth |
| **Extraction** | Field-level grounding rate (can extracted fields be found in source text?) |

---

## Three Wire Contracts

### 1. Frontend -> Backend (API Contract)

The frontend communicates with the backend via HTTP REST + SSE streaming.

| Endpoint | Method | Input | Output |
|----------|--------|-------|--------|
| `/extract/stream` | POST + SSE | PDF file as multipart | SSE events: progress, ocr_result, result |
| `/draft` | POST | JSON `{extracted_data, ocr_text, pages, doc_id}` | JSON `{draft, evidence_count}` |
| `/feedback` | POST | JSON `{original, corrected, document_type}` | JSON `{status, corrections_count}` |
| `/evaluate` | POST | JSON `{extracted, ground_truth}` | JSON with 4 metric scores |
| `/benchmark` | POST | JSON `{mode, num_samples, max_docs}` | JSON with benchmark results |

### 2. Backend -> LLM (Integration Contract)

The LLM is called with a structured prompt:

```json
{
  "system_prompt": "You are a legal document extraction assistant...",
  "user_prompt": "Extract fields from this nda:\n\n{ocr_text}",
  "response_format": {"type": "json_object"}
}
```

The LLM response must be valid JSON matching the schema for the document type.

### 3. LLM -> Frontend (Response Envelope)

The extraction result is returned as a flat JSON object with field names matching the schema:

```json
{
  "file_name": "nda.pdf",
  "doc_type": "nda",
  "ocr_confidence": 0.92,
  "extracted": {
    "parties": ["Acme Corp", "Beta Inc"],
    "effective_date": "2024-01-15",
    "governing_law": "New York",
    "is_mutual": true,
    "summary": "Mutual NDA..."
  }
}
```

---

## Known Issues

| # | Concern | Location |
|---|---------|----------|
| 1 | `hallucination_rate` in extraction benchmark can be inflated by summary fields | `benchmark.py:446` — summary fields are now auto-exempted |
| 2 | Jaeger System Architecture tab requires ES/Cassandra backend | `docker-compose.yml` — uses in-memory storage by default |
| 3 | PaddlePaddle 3.2.2 pinned to avoid PIR executor bug in 3.3.x | `pyproject.toml:13` |
| 4 | ChromaDB ONNX model downloads on first API call (~80 MB) | `chroma_store.py` — cached in `~/.cache/chroma/` |
