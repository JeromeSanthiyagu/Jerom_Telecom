"""Classification Agent: rule-based category + severity, ML category as a cross-check."""

from src.agents.base import Agent
from src.agents.state import AnalysisState
from src.rag.pipeline import step_classify


class ClassificationAgent(Agent):
    name = "classification"
    requires = ("evidence",)

    def run(self, state: AnalysisState) -> str:
        c = step_classify(state.result)
        state.result["classification"] = c
        return f"{c['category']} / {c['severity']} ({c['severity_basis']})"
