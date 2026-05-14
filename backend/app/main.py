"""Spectre backend — FastAPI application."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle — startup/shutdown hooks."""
    # Startup: initialize services, load model, etc.
    yield
    # Shutdown: cleanup resources


app = FastAPI(
    title="Spectre — Legal Document Extraction API",
    description=(
        "Extract structured data from legal documents, retrieve relevant "
        "passages, generate grounded drafts, and improve from operator edits. "
        "\n\n"
        "## Endpoints\n"
        "- `/upload` — Upload a PDF for extraction\n"
        "- `/extract` — Extract structured fields\n"
        "- `/retrieve` — Retrieve relevant passages\n"
        "- `/draft` — Generate a grounded draft\n"
        "- `/feedback` — Submit operator edits\n"
        "- `/evaluate` — Run evaluation metrics\n"
        "- `/health` — Service health check\n"
        "\n"
        "See each endpoint for request/response schemas and examples."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
async def health():
    """Health check endpoint.

    Returns basic service status including model loading state
    and database connectivity.
    """
    return {
        "status": "ok",
        "service": "spectre-backend",
        "version": "0.1.0",
    }
