"""Diagnosis Agent: ranked root-cause hypotheses from the playbook."""

from src.agents.base import Agent
from src.agents.state import AnalysisState
from src.rag.pipeline import step_diagnose


class DiagnosisAgent(Agent):
    name = "diagnosis"
    requires = ("evidence",)

    def run(self, state: AnalysisState) -> str:
        rca = step_diagnose(state.result)
        state.result["root_cause"] = rca
        return f"primary hypothesis: {rca.get('primary_hypothesis')} ({rca.get('confidence')})"
