"""Structured state passed between agents (Phase 23)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentTrace(BaseModel):
    agent: str
    status: str                 # ok | error
    duration_s: float
    summary: str = ""


class AnalysisState(BaseModel):
    # request
    query: str
    filters: dict[str, Any] | None = None
    top_k: int = 5
    metrics: dict[str, Any] | None = None
    use_llm: bool = True
    # accumulated result (same keys as src.rag.pipeline.analyze output)
    result: dict[str, Any] = Field(default_factory=dict)
    trace: list[AgentTrace] = Field(default_factory=list)

    def has(self, *keys: str) -> bool:
        return all(k in self.result for k in keys)
