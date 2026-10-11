"""Retrieval Agent: query understanding, filters, hybrid retrieval, evidence extraction."""

from src.agents.base import Agent
from src.agents.state import AnalysisState
from src.rag.pipeline import step_retrieve


class RetrievalAgent(Agent):
    name = "retrieval"

    def run(self, state: AnalysisState) -> str:
        state.result.update(step_retrieve(state.result["query"], state.filters, state.top_k, state.metrics))
        r = state.result["retrieval"]
        obs = state.result["evidence"]["observed"]
        return (f"{len(r['results'])} records from {r['matching_records']} matching filters; "
                f"conditions present: {obs['present'] or 'none'}")
