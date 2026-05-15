"""Spectre backend — FastAPI application."""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference

from app.config import settings
from app.extraction.llm_extractor import extract_fields
from app.ocr.pipeline import process_document


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
    docs_url="/swagger",  # Swagger UI at /swagger
    redoc_url="/redoc",  # ReDoc at /redoc
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
    """Scalar API reference — replaces Swagger UI.

    Clean, modern API documentation with built-in request testing.
    """
    import json
    return get_scalar_api_reference(
        content=json.dumps(app.openapi()),
        title="Spectre API",
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


# ─── Draft (stub) ───────────────────────────────────────


@app.post(
    "/draft",
    tags=["Draft"],
    summary="Generate a grounded draft memo",
    description="Generates a legal memo grounded in extracted data and source passages. (Stub — full implementation pending.)",
)
async def generate_draft(payload: dict):
    """Generate a grounded draft memo. Stub — returns placeholder."""
    return {
        "draft": (
            "# Internal Review Memo\n\n"
            "**To:** Reviewing Attorney\n"
            "**From:** Spectre AI\n"
            f"**Document Type:** {payload.get('extracted_data', {}).get('document_type', 'Unknown')}\n\n"
            "## Summary\n"
            "This is a stub draft. Full draft generation will be implemented in a future update.\n\n"
            "## Extracted Data\n"
            f"{json.dumps(payload.get('extracted_data', {}), indent=2)}\n\n"
            "## Evidence\n"
            "The above fields were extracted from the uploaded document. "
            "Inline citations will be added once the retrieval layer is connected."
        )
    }


# ─── Feedback (stub) ────────────────────────────────────


@app.post(
    "/feedback",
    tags=["Feedback"],
    summary="Submit operator edits for improvement loop",
    description="Accepts corrections from the operator review sheet. (Stub — full implementation pending.)",
)
async def submit_feedback(payload: dict):
    """Accept operator corrections. Stub — logs and acknowledges."""
    changed = payload.get("changed_fields", [])
    print(f"Feedback received: {len(changed)} fields corrected: {changed}")
    return {"status": "accepted", "corrections_count": len(changed)}


# ─── Evaluate (stub) ────────────────────────────────────


@app.post(
    "/evaluate",
    tags=["Evaluation"],
    summary="Run LLM-as-judge evaluation metrics",
    description="Evaluates extraction quality against ground truth. (Stub — full implementation pending.)",
)
async def evaluate(payload: dict):
    """Run evaluation metrics. Stub — returns placeholder scores."""
    return {
        "context_relevance": 0.85,
        "answer_faithfulness": 0.90,
        "answer_relevance": 0.88,
        "hallucination_rate": 0.02,
        "note": "Stub scores — connect evaluation harness for production metrics.",
    }
