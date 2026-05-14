"""Image preprocessing for better OCR accuracy on messy documents.

Applies deskewing, binarization, denoising, and illumination correction
before passing images to the OCR engine.
"""

import cv2
import numpy as np


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Correct skew using min area rectangle on all text components."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) > max_angle:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def binarize(image: np.ndarray) -> np.ndarray:
    """Apply Sauvola binarization for variable-illumination documents."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    from skimage.filters import threshold_sauvola

    thresh = threshold_sauvola(gray, window_size=35, k=0.2)
    binary = (gray > thresh).astype(np.uint8) * 255
    return binary


def despeckle(binary: np.ndarray, min_area: int = 30) -> np.ndarray:
    """Remove small connected components (speckles, dust, fold marks)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def correct_illumination(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for non-uniform lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(tophat)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline for a scanned page.

    Order: deskew -> illumination correction -> binarize -> despeckle.
    """
    image = deskew(image)
    image = correct_illumination(image)
    image = binarize(image)
    image = despeckle(image)
    return image
