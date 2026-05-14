"""Legal document classifier using weighted keyword scoring.

Inspired by the fast-path classification pattern: first determine
whether the document is a legal document at all, then classify
its specific type for routing to the appropriate extraction schema.
"""

import re

# Sentinels
NON_DOCUMENT_SENTINEL = "[NON_DOCUMENT]"

# Weighted keywords for legal document type classification
LEGAL_KEYWORDS: dict[str, int] = {
    # Strong signals (weight 4)
    "engagement letter": 4,
    "fee proposal": 4,
    "master services agreement": 4,
    "msa": 4,
    "non-disclosure agreement": 4,
    "nda": 4,
    "confidentiality agreement": 4,
    # Medium (weight 3)
    "agreement": 3,
    "contract": 3,
    "terms and conditions": 3,
    "scope of work": 3,
    "statement of work": 3,
    "exhibit": 3,
    "schedule": 3,
    # Medium-weak (weight 2)
    "whereas": 2,
    "hereby": 2,
    "indemnify": 2,
    "party": 2,
    "confidential": 2,
    "governing law": 2,
    "effective date": 2,
    "termination": 2,
    "liability": 2,
    # Weak (weight 1)
    "dated": 1,
    "signature": 1,
    "witness": 1,
    "draft": 1,
}

KNOWN_DOC_TYPES: dict[str, str] = {
    "engagement letter": "engagement_letter",
    "fee proposal": "fee_proposal",
    "master services agreement": "msa",
    "msa": "msa",
    "non-disclosure agreement": "nda",
    "nda": "nda",
    "confidentiality agreement": "nda",
    "statement of work": "sow",
    "scope of work": "sow",
    "subscription agreement": "saas",
    "data processing addendum": "dpa",
    "employment agreement": "employment",
    "consulting agreement": "consulting",
    "service agreement": "service",
    "license agreement": "license",
    "settlement agreement": "settlement",
    "amendment": "amendment",
}


def classify_document(text: str) -> str:
    """Classify a document based on OCR text.

    Returns one of: a known doc type key, 'legal_generic', or 'others'.
    """
    text_lower = text.lower()

    # Score against legal keywords
    score = sum(
        weight for keyword, weight in LEGAL_KEYWORDS.items() if keyword in text_lower
    )

    if score < 3:
        return "others"

    # Try to identify specific document type
    text_start = text_lower[:500]  # Title is typically in first ~500 chars
    for keyword, doc_type in KNOWN_DOC_TYPES.items():
        if keyword in text_start:
            return doc_type

    return "legal_generic"
