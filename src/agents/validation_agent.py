"""Validation Agent: grounding + LLM-as-judge, provenance, final report."""

from src.agents.base import Agent
from src.agents.state import AnalysisState
from src.rag.judge import validate
from src.services.report import render_report
from src.rag.pipeline import build_provenance


class ValidationAgent(Agent):
    name = "validation"
    requires = ("narrative", "recommendations")

    def run(self, state: AnalysisState) -> str:
        r = state.result
        r["validation"] = validate(r, use_llm=state.use_llm)
        r["provenance"] = build_provenance(r)
        r["report_text"] = render_report(r)
        v = r["validation"]
        return (f"grounded={v['evidence_grounded']} supported={v['diagnosis_supported']} "
                f"hallucination={v['hallucination_detected']} score={v['score']}")
