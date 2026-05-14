"""SQLAlchemy ORM models and Pydantic schemas."""

from datetime import datetime
from pydantic import BaseModel, Field


# ─── Pydantic Schemas ───────────────────────────────────


class ExtractedField(BaseModel):
    """A single extracted field with verification metadata.

    Every extracted value carries its source evidence so operators
    can verify accuracy before accepting the output.
    """

    value: str | float | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_page: int = Field(default=0, ge=0)
    source_snippet: str = ""
    verification_status: str = Field(
        default="unverified", pattern=r"^(verified|unverified|edited)$"
    )


class ProposalExtraction(BaseModel):
    """Structured data extracted from a legal fee proposal."""

    law_firm: ExtractedField = Field(default_factory=ExtractedField)
    client_name: ExtractedField = Field(default_factory=ExtractedField)
    document_date: ExtractedField = Field(default_factory=ExtractedField)
    fee_amount: ExtractedField = Field(default_factory=ExtractedField)
    payment_schedule: list[ExtractedField] = Field(default_factory=list)
    key_clauses: list[ExtractedField] = Field(default_factory=list)
    parties: list[ExtractedField] = Field(default_factory=list)


class CorrectionPair(BaseModel):
    """A correction submitted by an operator.

    Used for the improvement-from-edits feedback loop.
    """

    original: dict = Field(default_factory=dict)
    corrected: dict = Field(default_factory=dict)
    changed_fields: list[str] = Field(default_factory=list)
    document_type: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
