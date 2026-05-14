"""Spectre backend — FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference

from app.ocr.pipeline import process_document


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup/shutdown hooks."""
    # Startup: create upload dir
    Path("/app/uploads").mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown: cleanup (if needed)


app = FastAPI(
    title="Spectre — Legal Document Extraction API",
    description=(
        "Extract structured data from legal documents, retrieve relevant "
        "passages, generate grounded drafts, and improve from operator edits."
    ),
    summary="Upload PDFs of legal documents and extract structured fields",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,  # disabled — using Scalar instead
    redoc_url=None,  # disabled — using Scalar instead
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
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="Spectre API",
    )


@app.get("/openapi.json", include_in_schema=False)
async def openapi_json():
    """Expose OpenAPI spec for Scalar to consume."""
    return app.openapi()


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
    upload_path = Path("/app/uploads") / file.filename
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
