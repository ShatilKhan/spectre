"""Extraction schemas for different document types.

Each document type has a corresponding extraction schema defining
the fields to extract and their types.
"""

from app.models import ProposalExtraction, ExtractedField

# Map of document type → extraction schema
SCHEMA_REGISTRY: dict[str, type] = {
    "engagement_letter": ProposalExtraction,
    "fee_proposal": ProposalExtraction,
    "msa": ProposalExtraction,
    "nda": ProposalExtraction,
    "legal_generic": ProposalExtraction,
}


def get_schema(doc_type: str) -> type:
    """Get the extraction schema for a document type.

    Falls back to generic proposal extraction for unknown types.
    """
    return SCHEMA_REGISTRY.get(doc_type, ProposalExtraction)
