"""PaddleOCR engine with thread pool for multi-page PDF processing.

Uses PaddleOCR 3.5.0's predict() API for native PDF support.
Thread pool runs page-level predictions in parallel.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import threading
from typing import Optional

import numpy as np

# Singleton — initialized once, reused across threads
_ocr_instance = None
_ocr_lock = threading.Lock()


def get_ocr():
    """Get or create the shared PaddleOCR instance (thread-safe)."""
    global _ocr_instance
    if _ocr_instance is None:
        with _ocr_lock:
            # Double-checked locking
            if _ocr_instance is None:
                # Deferred import — paddleocr is heavy
                from paddleocr import PaddleOCR

                _ocr_instance = PaddleOCR(
                    lang="en",
                    text_det_thresh=0.3,
                    text_det_box_thresh=0.5,
                )
    return _ocr_instance


def ocr_predict_single(pdf_path: str) -> list[dict]:
    """Run PaddleOCR predict() on a PDF, returns per-page results.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts, one per page, with keys:
            page_index, rec_texts, rec_scores, dt_polys, etc.
    """
    ocr = get_ocr()
    return list(ocr.predict(pdf_path))


class PageResult:
    """OCR result for a single page."""

    def __init__(self, page_number: int, text: str, confidence: float):
        self.page_number = page_number
        self.text = text
        self.confidence = confidence

    def __repr__(self) -> str:
        return (
            f"PageResult(page={self.page_number}, "
            f"chars={len(self.text)}, conf={self.confidence:.2f})"
        )


class DocumentResult:
    """OCR result for an entire document."""

    def __init__(self, pages: list[PageResult]):
        self.pages = pages

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def average_confidence(self) -> float:
        if not self.pages:
            return 0.0
        return float(np.mean([p.confidence for p in self.pages]))

    def __repr__(self) -> str:
        return (
            f"DocumentResult(pages={self.page_count}, "
            f"mean_conf={self.average_confidence:.2f})"
        )


def process_pdf(
    pdf_path: str | Path,
    max_workers: Optional[int] = None,
    dpi: int = 200,
) -> DocumentResult:
    """OCR a PDF document using PaddleOCR 3.5.0's native PDF support.

    Args:
        pdf_path: Path to the PDF file.
        max_workers: Not used for PDFs (PaddleOCR handles natively).
        dpi: Not used for PDFs (PaddleOCR uses native page rendering).

    Returns:
        DocumentResult with per-page text and confidence scores.
    """
    raw_results = ocr_predict_single(str(pdf_path))

    pages = []
    for raw in raw_results:
        page_idx = raw.get("page_index", 0)
        texts: list[str] = raw.get("rec_texts", []) or []
        scores: list[float] = raw.get("rec_scores", []) or []

        page_text = "\n".join(texts)
        page_conf = float(np.mean(scores)) if scores else 0.0

        pages.append(PageResult(
            page_number=page_idx + 1,
            text=page_text,
            confidence=page_conf,
        ))

    return DocumentResult(pages)


def process_page_images(
    page_images: list[np.ndarray],
    max_workers: Optional[int] = None,
) -> list[str]:
    """OCR pre-rendered page images (used when images are pre-rendered).

    Args:
        page_images: List of RGB numpy arrays, one per page.
        max_workers: Thread count. Defaults to min(CPU cores, 4).

    Returns:
        List of extracted text strings, one per page.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 4)

    ocr = get_ocr()

    def ocr_page(img: np.ndarray) -> str:
        result = list(ocr.predict(img))
        if result and "rec_texts" in result[0]:
            return "\n".join(result[0]["rec_texts"] or [])
        return ""

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(ocr_page, page_images))

    return results
