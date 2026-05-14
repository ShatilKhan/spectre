"""PaddleOCR engine with thread pool for multi-page processing."""

from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
from PIL import Image


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


def ocr_page(page_image: np.ndarray) -> str:
    """OCR a single page image. Thread-safe (PaddlePaddle releases GIL)."""
    ocr = get_ocr()
    result = ocr.ocr(page_image)
    lines = []
    if result and result[0]:
        for line in result[0]:
            text, confidence = line[1]
            if confidence > 0.3:
                lines.append(text)
    return "\n".join(lines)


def ocr_document(page_images: list[np.ndarray], max_workers: int | None = None) -> list[str]:
    """OCR a multi-page document using thread pool.

    Args:
        page_images: List of numpy arrays, one per page.
        max_workers: Thread count. Defaults to min(CPU cores, 4).

    Returns:
        List of strings, one per page.
    """
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(ocr_page, page_images))

    return results
