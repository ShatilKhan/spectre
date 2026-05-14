"""Pydantic schemas for extraction models."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class ExtractedField(BaseModel):
    """A single extracted field with verification metadata.

    Every extracted value carries its source evidence so operators
    can verify accuracy before accepting the output.
    """

    value: Optional[str] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_page: int = Field(default=0, ge=0)
    source_snippet: str = ""
    verification_status: str = Field(
        default="unverified", pattern=r"^(verified|unverified|edited)$"
    )


class NdaExtraction(BaseModel):
    """Structured data extracted from a Non-Disclosure Agreement."""

    parties: list[str] = Field(default_factory=list, description="The signing parties")
    effective_date: Optional[str] = Field(None, description="When the NDA takes effect")
    governing_law: Optional[str] = Field(None, description="Governing law jurisdiction")
    confidentiality_term_months: Optional[int] = Field(None, description="Confidentiality obligation term in months")
    is_mutual: Optional[bool] = Field(None, description="Whether the NDA is mutual or one-way")
    has_non_solicit: Optional[bool] = Field(None, description="Whether it includes a non-solicit clause")
    has_non_compete: Optional[bool] = Field(None, description="Whether it includes a non-compete clause")
    permitted_disclosures: list[str] = Field(default_factory=list, description="Exceptions to confidentiality")
    jurisdiction_venue: Optional[str] = Field(None, description="Dispute resolution jurisdiction")
    summary: Optional[str] = Field(None, description="Brief summary of the NDA")


class MsaExtraction(BaseModel):
    """Structured data extracted from a Master Services Agreement."""

    parties: list[str] = Field(default_factory=list, description="The contracting parties")
    effective_date: Optional[str] = Field(None, description="When the agreement starts")
    term: Optional[str] = Field(None, description="Initial term duration")
    auto_renewal: Optional[bool] = Field(None, description="Whether the agreement auto-renews")
    notice_period_days: Optional[int] = Field(None, description="Notice period for termination")
    governing_law: Optional[str] = Field(None, description="Governing law")
    liability_cap: Optional[str] = Field(None, description="Liability limitation amount or multiplier")
    has_indemnification: Optional[bool] = Field(None, description="Whether indemnification clause exists")
    has_confidentiality: Optional[bool] = Field(None, description="Whether confidentiality clause exists")
    payment_terms_days: Optional[int] = Field(None, description="Payment due in number of days")
    termination_for_convenience: Optional[bool] = Field(None, description="Whether either party can terminate without cause")
    summary: Optional[str] = Field(None, description="Brief summary of the MSA")


class EngagementLetterExtraction(BaseModel):
    """Structured data extracted from a legal engagement letter."""

    law_firm: Optional[str] = Field(None, description="Name of the law firm")
    client_name: Optional[str] = Field(None, description="Name of the client")
    matter_description: Optional[str] = Field(None, description="Description of the legal matter")
    effective_date: Optional[str] = Field(None, description="Date of the engagement letter")
    fee_structure: Optional[str] = Field(None, description="Fee arrangement (hourly, flat, contingency, etc.)")
    hourly_rates: list[str] = Field(default_factory=list, description="Hourly rates by attorney role")
    retainer_amount: Optional[str] = Field(None, description="Retainer fee if any")
    fee_estimate: Optional[str] = Field(None, description="Estimated total fee if stated")
    payment_terms: Optional[str] = Field(None, description="Payment terms and schedule")
    scope_of_work: Optional[str] = Field(None, description="Scope of legal services")
    governing_law: Optional[str] = Field(None, description="Governing law")
    has_termination_clause: Optional[bool] = Field(None, description="Whether termination clause is included")
    has_conflicts_clause: Optional[bool] = Field(None, description="Whether conflicts of interest clause is included")
    summary: Optional[str] = Field(None, description="Brief summary of the engagement letter")


class FeeProposalExtraction(BaseModel):
    """Structured data extracted from a fee proposal."""

    firm_name: Optional[str] = Field(None, description="Name of the proposing firm")
    client_name: Optional[str] = Field(None, description="Name of the prospective client")
    project_description: Optional[str] = Field(None, description="Description of the proposed work")
    proposal_date: Optional[str] = Field(None, description="Date of the proposal")
    total_fee: Optional[str] = Field(None, description="Total proposed fee amount")
    fee_breakdown: list[str] = Field(default_factory=list, description="Breakdown of fees by phase or service")
    payment_schedule: list[str] = Field(default_factory=list, description="Milestone payment schedule")
    validity_period_days: Optional[int] = Field(None, description="How long the proposal is valid")
    additional_terms: list[str] = Field(default_factory=list, description="Other key terms or conditions")
    summary: Optional[str] = Field(None, description="Brief summary of the fee proposal")


class GenericLegalExtraction(BaseModel):
    """Generic extraction for legal documents without a specific schema."""

    document_title: Optional[str] = Field(None, description="The title or heading of the document")
    parties_mentioned: list[str] = Field(default_factory=list, description="Names or entities mentioned")
    dates_mentioned: list[str] = Field(default_factory=list, description="Dates referenced in the document")
    monetary_amounts: list[str] = Field(default_factory=list, description="Monetary amounts referenced")
    key_topics: list[str] = Field(default_factory=list, description="Main subjects or clauses discussed")
    summary: Optional[str] = Field(None, description="Brief summary of the document")


class CorrectionPair(BaseModel):
    """A correction submitted by an operator.

    Used for the improvement-from-edits feedback loop.
    """

    original: dict = Field(default_factory=dict)
    corrected: dict = Field(default_factory=dict)
    changed_fields: list[str] = Field(default_factory=list)
    document_type: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
