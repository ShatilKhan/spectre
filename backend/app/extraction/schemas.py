"""Extraction schemas for different document types.

Each document type has a corresponding Pydantic schema defining
the fields to extract. The schema registry maps classifier output
to the right schema for structured extraction.
"""

from pydantic import BaseModel
from app.models import (
    NdaExtraction,
    MsaExtraction,
    EngagementLetterExtraction,
    FeeProposalExtraction,
    GenericLegalExtraction,
)

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "nda": NdaExtraction,
    "msa": MsaExtraction,
    "engagement_letter": EngagementLetterExtraction,
    "fee_proposal": FeeProposalExtraction,
    "legal_generic": GenericLegalExtraction,
}


def get_schema(doc_type: str) -> type[BaseModel]:
    """Get the extraction schema for a document type.

    Falls back to generic legal extraction for unknown or unsupported types,
    ensuring the system degrades gracefully when encountering unfamiliar
    document types.
    """
    return SCHEMA_REGISTRY.get(doc_type, GenericLegalExtraction)
