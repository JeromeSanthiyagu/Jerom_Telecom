"""Risk Agent: predicted fault risk (ML) vs. historical fault rate, then priority."""

from src.agents.base import Agent
from src.agents.state import AnalysisState
from src.rag.pipeline import step_risk


class RiskAgent(Agent):
    name = "risk"
    requires = ("evidence", "classification")

    def run(self, state: AnalysisState) -> str:
        risk, priority = step_risk(state.result)
        state.result["risk"], state.result["priority"] = risk, priority
        p = risk["predicted"]
        level = p.get("risk_level", "n/a") if p.get("available") else "model unavailable"
        return f"predicted risk {level}; priority {priority['priority']}"
