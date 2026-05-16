"""Spectre backend — FastAPI application."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference

from app.config import settings
from app.extraction.llm_extractor import extract_fields
from app.ocr.pipeline import process_document
from app.retrieval.chroma_store import retrieve_for_draft
from app.draft.generator import generate_draft, generate_draft_stream
from app.evaluation.judge import evaluate_extraction as judge_extraction
from app.evaluation.metrics import compute_metrics, EvaluationResult
from app.feedback.edit_capture import store_correction


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup/shutdown hooks."""
    # Upload dir is created by config.py on import
    yield


app = FastAPI(
    title="Spectre — Legal Document Extraction API",
    description=(
        "Extract structured data from legal documents, retrieve relevant "
        "passages, generate grounded drafts, and improve from operator edits."
    ),
    summary="Upload PDFs of legal documents and extract structured fields",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,  # using Scalar instead
    redoc_url=None,  # using Scalar instead
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Scalar API Docs ─────────────────────────────────


@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    """Scalar API reference — clean, modern API documentation
    with built-in request testing (Elysia.js theme)."""
    import json
    from scalar_fastapi import Layout

    return get_scalar_api_reference(
        content=json.dumps(app.openapi()),
        title="Spectre API",
        layout=Layout.MODERN,
        dark_mode=False,
        hide_download_button=True,
        hide_test_request_button=False,
        overrides={"theme": "elysiajs"},
    )


# ─── Health ──────────────────────────────────────────


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint.

    Returns basic service status. Use to verify the backend is running.

    **Response example:**
    ```json
    {
      "status": "ok",
      "service": "spectre-backend",
      "version": "0.1.0"
    }
    ```
    """
    return {
        "status": "ok",
        "service": "spectre-backend",
        "version": "0.1.0",
    }


# ─── Upload ──────────────────────────────────────────


@app.post(
    "/upload",
    tags=["Documents"],
    summary="Upload and OCR a PDF document",
    description=(
        "Upload a legal document (PDF only). The system will:\n"
        "1. Render each page to an image\n"
        "2. OCR all pages in parallel using PaddleOCR\n"
        "3. Classify the document type (NDA, MSA, fee proposal, etc.)\n"
        "4. Return extracted text per page with confidence scores"
    ),
)
async def upload_pdf(
    file: UploadFile = File(
        ...,
        description="PDF file of a legal document. Max size determined by server.",
    ),
):
    """Upload a PDF document for OCR processing.

    Accepts a PDF file, runs OCR on all pages, classifies the document type,
    and returns structured results including per-page text and confidence.

    **Request example:**
    Upload a file using multipart/form-data with field name `file`.

    **Response example:**
    ```json
    {
      "file_name": "contract.pdf",
      "doc_type": "nda",
      "page_count": 3,
      "confidence": 0.92,
      "full_text": "NON-DISCLOSURE AGREEMENT...",
      "pages": [
        {
          "page": 1,
          "text": "NON-DISCLOSURE AGREEMENT...",
          "confidence": 0.94
        }
      ]
    }
    ```
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    # Save uploaded file
    upload_path = settings.upload_dir / file.filename
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    upload_path.write_bytes(content)

    try:
        # Run OCR pipeline
        result = process_document(upload_path)

        return result.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR processing failed: {str(e)}",
        )

    finally:
        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink()


# ─── Extract ────────────────────────────────────────


@app.post(
    "/extract",
    tags=["Documents"],
    summary="Upload, OCR, and extract structured fields from a PDF",
    description=(
        "Upload a legal document (PDF only). The system will:\n"
        "1. OCR all pages using PaddleOCR\n"
        "2. Classify the document type\n"
        "3. Run LLM-based structured extraction using the matching schema\n"
        "4. Return extracted fields with confidence scores\n\n"
        "Supports: NDAs, MSAs, engagement letters, fee proposals, "
        "and generic legal documents."
    ),
)
async def extract_pdf(
    file: UploadFile = File(
        ...,
        description="PDF file of a legal document.",
    ),
):
    """Upload a PDF and extract structured fields.

    Chains OCR → document classification → LLM extraction using the
    schema that matches the detected document type.

    **Request example:**
    Upload a file using multipart/form-data with field name `file`.

    **Response example:**
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
        "summary": "Mutual NDA governing confidential information..."
      }
    }
    ```
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted",
        )

    # Save uploaded file
    upload_path = settings.upload_dir / file.filename
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    upload_path.write_bytes(content)

    try:
        # Step 1: OCR + classify
        ocr_result = process_document(upload_path)

        # Step 2: LLM extraction with matching schema
        extracted = extract_fields(
            raw_text=ocr_result.full_text,
            doc_type=ocr_result.doc_type,
        )

        return {
            "file_name": ocr_result.file_name,
            "doc_type": ocr_result.doc_type,
            "ocr_confidence": round(ocr_result.confidence, 3),
            "extracted": extracted,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}",
        )

    finally:
        if upload_path.exists():
            upload_path.unlink()


# ─── Draft ───────────────────────────────────────────────


@app.post(
    "/draft",
    tags=["Draft"],
    summary="Generate a grounded draft memo",
    description=(
        "Generates a legal memo grounded in extracted data and source passages. "
        "The system retrieves relevant chunks from the vector store and generates "
        "a memo with inline citations to the source material."
    ),
)
async def generate_draft_endpoint(payload: dict):
    """Generate a grounded draft memo with citations.

    Expects:
    ```json
    {
      "extracted_data": {...},
      "ocr_text": "full OCR text from the document",
      "doc_id": "optional-document-id"
    }
    ```

    Returns a draft with inline [Source: page X] citations.
    """
    extracted = payload.get("extracted_data", {})
    ocr_text = payload.get("ocr_text", "")
    pages = payload.get("pages") or payload.get("ocr_pages", [])
    doc_id = payload.get("doc_id")

    if not ocr_text:
        return {"draft": "No OCR text provided. Upload a document first."}

    passages = retrieve_for_draft(
        extracted_data=extracted,
        ocr_text=ocr_text,
        pages=pages,
        doc_id=doc_id,
    )
    if not passages:
        return {"draft": "No relevant passages found in the document."}
    draft = generate_draft(extracted_data=extracted, source_passages=passages)
    return {"draft": draft, "evidence_count": len(passages)}


# ─── Extract SSE Stream ──────────────────────────────────


@app.post(
    "/extract/stream",
    tags=["Documents"],
    summary="Upload, OCR, and extract structured fields (SSE stream)",
)
async def extract_stream(file: UploadFile = File(...)):
    """Extract with SSE progress events for each stage."""
    from fastapi.responses import StreamingResponse

    async def event_stream():
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            yield "event: error\ndata: " + json.dumps({"message": "Only PDF files accepted"}) + "\n\n"
            return

        content = await file.read()
        upload_path = settings.upload_dir / file.filename
        upload_path.write_bytes(content)

        # OCR step
        yield "event: progress\ndata: " + json.dumps({"step": "ocr", "message": "Running OCR (7-60 sec)...", "pct": 10}) + "\n\n"
        try:
            ocr_result = process_document(upload_path)
        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"message": f"OCR failed: {str(e)}"}) + "\n\n"
            upload_path.unlink()
            return

        yield "event: progress\ndata: " + json.dumps({"step": "ocr_done", "message": "OCR complete", "pct": 50}) + "\n\n"
        yield "event: ocr_result\ndata: " + json.dumps(ocr_result.to_dict()) + "\n\n"

        # Extraction step
        yield "event: progress\ndata: " + json.dumps({"step": "extract", "message": "Extracting with Granite 4.1 (30-60 sec)...", "pct": 55}) + "\n\n"
        try:
            extracted = extract_fields(raw_text=ocr_result.full_text, doc_type=ocr_result.doc_type)
        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"message": f"Extraction failed: {str(e)}"}) + "\n\n"
            upload_path.unlink()
            return

        yield "event: progress\ndata: " + json.dumps({"step": "done", "message": "Complete", "pct": 100}) + "\n\n"
        yield "event: result\ndata: " + json.dumps({
            "ocr": ocr_result.to_dict(),
            "extracted": extracted,
        }) + "\n\n"
        upload_path.unlink()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Draft SSE Stream ─────────────────────────────────────


@app.post(
    "/draft/stream",
    tags=["Draft"],
    summary="Generate a grounded draft memo (SSE stream)",
)
async def generate_draft_stream_endpoint(payload: dict):
    """Generate a grounded draft memo with streaming response."""
    from fastapi.responses import StreamingResponse

    extracted = payload.get("extracted_data", {})
    ocr_text = payload.get("ocr_text", "")
    pages = payload.get("pages") or payload.get("ocr_pages", [])
    doc_id = payload.get("doc_id")

    if not ocr_text:
        return {"draft": "No OCR text provided."}

    passages = retrieve_for_draft(
        extracted_data=extracted,
        ocr_text=ocr_text,
        pages=pages,
        doc_id=doc_id,
    )
    if not passages:
        return {"draft": "No relevant passages found."}

    async def event_stream():
        yield "event: meta\ndata: " + json.dumps({"evidence_count": len(passages)}) + "\n\n"
        for chunk in generate_draft_stream(extracted_data=extracted, source_passages=passages):
            yield "data: " + json.dumps({"chunk": chunk}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Feedback (stub) ────────────────────────────────────


@app.post(
    "/feedback",
    tags=["Feedback"],
    summary="Submit operator edits for improvement loop",
    description="Accepts corrections from the operator review sheet. (Stub — full implementation pending.)",
)
async def submit_feedback(payload: dict):
    """Accept operator corrections and store for reinforcement learning."""
    pair = store_correction(
        original=payload.get("original", {}),
        corrected=payload.get("corrected", {}),
        document_type=payload.get("document_type", "unknown"),
    )
    return {
        "status": "accepted",
        "corrections_count": len(pair.changed_fields),
        "changed_fields": pair.changed_fields,
    }


# ─── Evaluate ────────────────────────────────────────────


@app.post(
    "/evaluate",
    tags=["Evaluation"],
    summary="Run LLM-as-judge evaluation metrics",
    description=(
        "Evaluates extraction quality against ground truth using Granite 4.1 as judge. "
        "Returns: context_relevance, answer_faithfulness, answer_relevance, hallucination_rate."
    ),
)
async def evaluate(payload: dict):
    """Run LLM-as-judge evaluation on extracted data vs ground truth."""
    extracted = payload.get("extracted", {})
    ground_truth = payload.get("ground_truth", {})

    if not extracted:
        return {"error": "No extracted data provided.", "num_samples": 0}

    # Run LLM-as-judge
    scores = judge_extraction(extracted=extracted, ground_truth=ground_truth)
    result = compute_metrics([scores])

    return result.to_dict()


# ─── Benchmark ────────────────────────────────────────────


@app.post(
    "/benchmark",
    tags=["Evaluation"],
    summary="Run pipeline benchmarks (classifier, OCR, extraction)",
    description=(
        "Benchmark pipeline quality in three modes:\n\n"

        "**classifier** — Document classifier accuracy against 13,155 CUAD clauses. "
        "Reports per-type accuracy and keyword-coverage upper bounds.\n\n"

        "**ocr** (default) — PaddleOCR accuracy on CUAD PDFs. "
        "Computes CER/WER by comparing OCR output against HuggingFace ground-truth text. "
        "Note: first call downloads dataset (~200 MB for 10 PDFs).\n\n"

        "**extraction** — Full pipeline (OCR → classify → LLM extract) on CUAD PDFs. "
        "Measures field-level grounding rate and hallucination rate.\n\n"

        "**all** — Runs all three benchmarks and aggregates."
    ),
)
async def benchmark(payload: dict):
    """Run pipeline benchmark in the requested mode."""
    mode = payload.get("mode", "classifier")
    from app.evaluation.benchmark import (
        run_classifier_benchmark,
        run_ocr_benchmark,
        run_extraction_benchmark,
        run_all_benchmarks,
    )

    try:
        if mode == "classifier":
            num_samples = payload.get("num_samples", 200)
            result = run_classifier_benchmark(num_samples=num_samples)
        elif mode == "ocr":
            max_docs = payload.get("max_docs", 10)
            result = run_ocr_benchmark(max_docs=max_docs)
        elif mode == "extraction":
            max_docs = payload.get("max_docs", 5)
            result = run_extraction_benchmark(max_docs=max_docs)
        elif mode == "all":
            result = run_all_benchmarks(
                classifier_samples=payload.get("classifier_samples", 200),
                ocr_docs=payload.get("ocr_docs", 10),
                extraction_docs=payload.get("extraction_docs", 5),
            )
        else:
            return {"error": f"Unknown mode: {mode}. Use: classifier, ocr, extraction, or all."}

        if "error" in result:
            return {
                "error": result["error"],
                "hint": "Run: python datasets/download_cuad.py --clauses-only",
            }

        return result

    except Exception as e:
        return {"error": f"Benchmark failed: {str(e)}"}
