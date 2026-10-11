"""Recommendation Agent: evidence-linked actions + the (grounding-checked) LLM explanation."""

from src.agents.base import Agent
from src.agents.state import AnalysisState
from src.rag.narrative import write_narrative
from src.rag.pipeline import step_recommend


class RecommendationAgent(Agent):
    name = "recommendation"
    requires = ("root_cause", "classification", "risk", "priority")

    def run(self, state: AnalysisState) -> str:
        state.result["recommendations"] = step_recommend(state.result)
        state.result["narrative"] = write_narrative(state.result, use_llm=state.use_llm)
        n = state.result["narrative"]
        return f"{len(state.result['recommendations']['actions'])} actions; narrative from {n['source']}"
