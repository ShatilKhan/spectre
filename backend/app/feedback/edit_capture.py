"""Capture operator edits and store correction pairs."""

from datetime import datetime
from typing import Any

from app.models import CorrectionPair

# In-memory store for correction pairs (in prod this would be a DB table)
_corrections: list[CorrectionPair] = []


def compute_diff(original: dict, corrected: dict) -> list[str]:
    """Compute which fields changed between original and corrected.

    Args:
        original: The original extracted data.
        corrected: The operator-edited data.

    Returns:
        List of field names that changed.
    """
    changed = []
    for key in set(list(original.keys()) + list(corrected.keys())):
        if original.get(key) != corrected.get(key):
            changed.append(key)
    return changed


def store_correction(
    original: dict,
    corrected: dict,
    document_type: str = "",
) -> CorrectionPair:
    """Store a correction pair for the feedback loop.

    Args:
        original: Original extracted data.
        corrected: Operator-corrected data.
        document_type: Type of document being edited.

    Returns:
        The stored CorrectionPair.
    """
    diff = compute_diff(original, corrected)
    pair = CorrectionPair(
        original=original,
        corrected=corrected,
        changed_fields=diff,
        document_type=document_type,
        timestamp=datetime.utcnow(),
    )
    _corrections.append(pair)
    return pair


def get_recent_corrections(n: int = 5) -> list[CorrectionPair]:
    """Get the most recent correction pairs."""
    return sorted(_corrections, key=lambda c: c.timestamp, reverse=True)[:n]
