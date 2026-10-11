"""Agent base class: each agent reads the shared state, adds its part, and is timed."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from src.agents.state import AgentTrace, AnalysisState

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    def __init__(self, agent: str, cause: Exception):
        super().__init__(f"{agent} failed: {cause}")
        self.agent = agent
        self.cause = cause


class Agent(ABC):
    name: str = "agent"
    requires: tuple[str, ...] = ()       # result keys that must exist before this agent runs

    @abstractmethod
    def run(self, state: AnalysisState) -> str:
        """Update `state.result` in place; return a one-line summary for the trace."""

    def __call__(self, state: AnalysisState) -> AnalysisState:
        missing = [k for k in self.requires if k not in state.result]
        if missing:
            raise AgentError(self.name, RuntimeError(f"missing inputs {missing}"))
        t0 = time.time()
        try:
            summary = self.run(state)
        except Exception as exc:  # recorded in the trace, then surfaced to the caller
            state.trace.append(AgentTrace(agent=self.name, status="error",
                                          duration_s=round(time.time() - t0, 3), summary=str(exc)))
            logger.exception("%s failed", self.name)
            raise AgentError(self.name, exc) from exc
        state.trace.append(AgentTrace(agent=self.name, status="ok", duration_s=round(time.time() - t0, 3),
                                      summary=summary))
        return state
