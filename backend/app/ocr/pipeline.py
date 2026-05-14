"""OCR pipeline orchestrator — connects PDF ingestion, OCR, and document classification.

Usage:
    from app.ocr.pipeline import process_document

    result = process_document("contract.pdf")
    print(result.doc_type)       # "nda", "msa", "fee_proposal", etc.
    print(result.full_text)      # OCR'd text from all pages
    print(result.confidence)     # average OCR confidence
"""

from pathlib import Path
from typing import Optional

from app.ocr.classifier import classify_document
from app.ocr.engine import DocumentResult, process_pdf


class PipelineResult:
    """Combined result from OCR + document classification."""

    def __init__(
        self,
        ocr_result: DocumentResult,
        doc_type: str,
        file_name: str,
    ):
        self.ocr_result = ocr_result
        self.doc_type = doc_type
        self.file_name = file_name

    @property
    def full_text(self) -> str:
        return self.ocr_result.full_text

    @property
    def confidence(self) -> float:
        return self.ocr_result.average_confidence

    @property
    def page_count(self) -> int:
        return self.ocr_result.page_count

    def to_dict(self) -> dict:
        """Serialize to a dict for API responses."""
        return {
            "file_name": self.file_name,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
            "confidence": round(self.confidence, 3),
            "full_text": self.full_text,
            "pages": [
                {
                    "page": p.page_number,
                    "text": p.text,
                    "confidence": round(p.confidence, 3),
                }
                for p in self.ocr_result.pages
            ],
        }

    def __repr__(self) -> str:
        return (
            f"PipelineResult(file={self.file_name}, "
            f"type={self.doc_type}, "
            f"pages={self.page_count}, "
            f"conf={self.confidence:.2f})"
        )


def process_document(
    file_path: str | Path,
    max_workers: Optional[int] = None,
    dpi: int = 200,
) -> PipelineResult:
    """Run the full OCR pipeline on a document.

    Steps:
        1. Render PDF pages to images (pdf2image)
        2. OCR each page in parallel (PaddleOCR thread pool)
        3. Classify the document type (keyword scoring)

    Args:
        file_path: Path to the PDF file.
        max_workers: Thread count for OCR. Defaults to min(CPU cores, 4).
        dpi: Rendering DPI for page images.

    Returns:
        PipelineResult with OCR text, classification, and metadata.
    """
    file_path = Path(file_path)

    # Step 1 + 2: OCR the document
    ocr_result = process_pdf(file_path, max_workers=max_workers, dpi=dpi)

    # Step 3: Classify the document type
    doc_type = classify_document(ocr_result.full_text)

    return PipelineResult(
        ocr_result=ocr_result,
        doc_type=doc_type,
        file_name=file_path.name,
    )
