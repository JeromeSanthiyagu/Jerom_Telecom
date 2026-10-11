"""Pydantic request/response models for the API (Phase 27)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Metrics(BaseModel):
    """Optional operator-supplied measurements. Ranges match src/config/schema.yaml validity limits."""
    model_config = ConfigDict(extra="forbid")

    cell_availability_pct: float | None = Field(None, ge=0, le=100)
    throughput_mbps: float | None = Field(None, ge=0)
    latency_ms: float | None = Field(None, ge=0)
    packet_loss_pct: float | None = Field(None, ge=0, le=100)
    call_drop_pct: float | None = Field(None, ge=0, le=100)
    handover_success_pct: float | None = Field(None, ge=0, le=100)
    signal_strength_dbm: float | None = Field(None, ge=-140, le=-44)
    jitter_ms: float | None = Field(None, ge=0)
    setup_success_pct: float | None = Field(None, ge=0, le=100)
    user_count: int | None = Field(None, ge=0)
    data_usage_gb: float | None = Field(None, ge=0)
    alarm_count: int | None = Field(None, ge=0)
    critical_alarm_count: float | None = Field(None, ge=0)
    parameter_changes: int | None = Field(None, ge=0)
    config_success_pct: float | None = Field(None, ge=0, le=100)
    security_incidents: int | None = Field(None, ge=0)
    auth_failures: int | None = Field(None, ge=0)
    temperature_c: float | None = Field(None, ge=-60, le=60)
    humidity_pct: float | None = Field(None, ge=0, le=100)
    season: Literal["Winter", "Spring", "Summer", "Fall"] | None = None
    weather: Literal["Clear", "Clouds", "Rain", "Snow"] | None = None
    issue_reported: bool | None = None

    def values(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language network query", max_length=5000)
    filters: dict[str, Any] | None = Field(None, examples=[{"state": "Minnesota", "season": "Winter"}])
    top_k: int = Field(5, ge=1, le=20)


class AnalyzeRequest(SearchRequest):
    metrics: Metrics | None = None
    use_llm: bool = True


class RecordOrMetrics(BaseModel):
    record_id: str | None = Field(None, pattern=r"^ROW-\d{6}$")
    metrics: Metrics | None = None


class RecommendRequest(SearchRequest):
    metrics: Metrics | None = None


class FeedbackRequest(BaseModel):
    analysis_id: str | None = None
    query: str = Field(..., min_length=1, max_length=5000)
    retrieved_record_ids: list[str] = Field(default_factory=list)
    diagnosis: dict[str, Any] | None = None
    recommendations: list[str] | None = None
    risk_prediction: dict[str, Any] | None = None
    rating: Literal["useful", "not_useful"]
    comment: str | None = Field(None, max_length=2000)


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
