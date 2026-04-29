"""Pydantic models for the report generator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FacilityItem(BaseModel):
    """A clinic, hospital dermatology service, or diagnostic lab entry."""

    name: str = Field(..., description="Official or commonly used name")
    area_or_city: str = Field(..., description="Area, city, or neighborhood")
    relevance: str = Field(..., description="Why this option may be relevant")
    url: str | None = Field(
        None,
        description="https? URL intended to be copied from web search results",
    )


class StructuredReport(BaseModel):
    """Structured sections returned by the model (informational only, not diagnosis)."""

    summary: str = Field(..., description="Brief summary of inputs and context")
    consistency_check: str = Field(
        ...,
        description="How symptoms may or may not align with the ML prediction; cautious wording",
    )
    ml_limitations: str = Field(..., description="What the ML confidence does and does not mean")
    precautions: str = Field(..., description="General non-prescriptive precautions")
    urgent_care: list[str] = Field(
        ...,
        description="Red-flag signs that warrant prompt or emergency care",
    )
    nearby_facilities: list[FacilityItem] = Field(
        default_factory=list,
        description="Nearby options; model is instructed to include URLs only from search",
    )
    disclaimer: str = Field(..., description="Non-diagnostic disclaimer")

    @field_validator("nearby_facilities", mode="before")
    @classmethod
    def empty_if_null(cls, v: list | None) -> list:
        return v if v is not None else []


class ReportEnvelope(BaseModel):
    """API success body: which model ran plus the structured report."""

    model_config = ConfigDict(protected_namespaces=())

    model_used: str
    maps_search_url: str = Field(
        ...,
        description="Google Maps search URL for dermatology/skin care near the request location",
    )
    report: StructuredReport


class ReportRequest(BaseModel):
    """Input payload for report generation."""

    ml_condition: str | None = Field(None, description="ML-predicted condition name")
    ml_confidence: int | None = Field(None, ge=0, le=100, description="Classifier confidence percent")
    symptoms: str = Field(..., min_length=1, description="User-described symptoms")
    duration: str | None = Field(None, description="How long symptoms have been present")
    patient_name: str | None = None
    gender: str | None = None
    age: int | None = Field(None, ge=1, le=120)
    location: str = Field(..., min_length=1, description="City, region, country for nearby facilities")
    model: str | None = Field(None, description="LLM model id override")

    @field_validator("symptoms", "location", mode="before")
    @classmethod
    def strip_required(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("ml_condition", "duration", "patient_name", "gender", mode="before")
    @classmethod
    def strip_optional(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            return s or None
        return v


class HealthResponse(BaseModel):
    status: str = "ok"
