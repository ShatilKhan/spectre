"""PaddleOCR engine with thread pool for multi-page PDF processing.

Usage:
    from app.ocr.engine import process_pdf

    pages = process_pdf("contract.pdf")
    for page in pages:
        print(page.text)
        print(page.confidence)
"""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from typing import Optional

import numpy as np

# Singleton — initialized once, reused across threads
_ocr_instance = None


def get_ocr():
    """Get or create the shared PaddleOCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        # Deferred import — paddleocr is heavy
        from paddleocr import PaddleOCR

        _ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
        )
    return _ocr_instance


def ocr_page(page_image: np.ndarray) -> tuple[str, float]:
    """OCR a single page image. Thread-safe (PaddlePaddle releases GIL).

    Args:
        page_image: RGB numpy array of the page.

    Returns:
        Tuple of (extracted_text, average_confidence).
    """
    ocr = get_ocr()
    result = ocr.ocr(page_image)
    lines: list[str] = []
    confidences: list[float] = []
    if result and result[0]:
        for line in result[0]:
            text, confidence = line[1]
            if confidence > 0.3:
                lines.append(text)
                confidences.append(confidence)
    text = "\n".join(lines)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return text, avg_conf


def pdf_to_images(pdf_path: str | Path, dpi: int = 200) -> list[np.ndarray]:
    """Convert PDF pages to numpy arrays for OCR.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (lower = faster, higher = better OCR).

    Returns:
        List of RGB numpy arrays, one per page.
    """
    from pdf2image import convert_from_path

    pil_images = convert_from_path(str(pdf_path), dpi=dpi)
    return [np.array(img) for img in pil_images]


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
    """OCR a PDF document end-to-end with multi-page parallelism.

    Args:
        pdf_path: Path to the PDF file.
        max_workers: Thread count per page. Defaults to min(CPU cores, 4).
        dpi: Rendering resolution for page images.

    Returns:
        DocumentResult with per-page text and confidence scores.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 4)

    # Step 1: render PDF pages to images
    page_images = pdf_to_images(pdf_path, dpi=dpi)
    if not page_images:
        return DocumentResult([])

    # Step 2: OCR all pages in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(ocr_page, page_images))

    # Step 3: build result objects
    pages = [
        PageResult(page_number=i + 1, text=text, confidence=conf)
        for i, (text, conf) in enumerate(results)
    ]

    return DocumentResult(pages)


def process_page_images(
    page_images: list[np.ndarray],
    max_workers: Optional[int] = None,
) -> list[str]:
    """OCR pre-rendered page images (useful when images are already available).

    Args:
        page_images: List of RGB numpy arrays, one per page.
        max_workers: Thread count. Defaults to min(CPU cores, 4).

    Returns:
        List of extracted text strings, one per page.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(ocr_page, page_images))

    return [text for text, conf in results]
