"""Orchestrator (Phase 23): a fixed, linear, state-based workflow.

    User -> guardrails -> Retrieval -> Classification -> Risk -> Diagnosis
         -> Recommendation -> Validation -> final answer

No autonomous loops: each agent runs exactly once, in order, on a shared
`AnalysisState`. (LangGraph was considered; for a linear six-step flow a plain
orchestrator is simpler to explain and test.)
"""

from __future__ import annotations

import time
from typing import Any

from src.agents.classification_agent import ClassificationAgent
from src.agents.diagnosis_agent import DiagnosisAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.risk_agent import RiskAgent
from src.agents.state import AnalysisState
from src.agents.validation_agent import ValidationAgent
from src.rag.pipeline import LIMITATIONS, step_guardrails

AGENTS = [RetrievalAgent(), ClassificationAgent(), RiskAgent(), DiagnosisAgent(), RecommendationAgent(),
          ValidationAgent()]


def run(query: str, filters: dict | None = None, top_k: int = 5, metrics: dict | None = None,
        use_llm: bool = True) -> dict[str, Any]:
    t0 = time.time()
    clean = step_guardrails(query)            # raises GuardrailRejection
    state = AnalysisState(query=clean, filters=filters, top_k=top_k, metrics=metrics, use_llm=use_llm)
    state.result = {"query": clean, "limitations": list(LIMITATIONS)}
    for agent in AGENTS:
        state = agent(state)
    state.result["agent_trace"] = [t.model_dump() for t in state.trace]
    state.result["timing_s"] = round(time.time() - t0, 2)
    return state.result
