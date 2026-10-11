"""Structured API errors: every failure returns {"error": {code, message, details, request_id}}."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.agents.base import AgentError
from src.rag.filters import FilterError
from src.rag.pipeline import GuardrailRejection

logger = logging.getLogger("netops.api")


class NotFound(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _error(request: Request, status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {
        "code": code, "message": message, "details": details,
        "request_id": getattr(request.state, "request_id", None)}})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GuardrailRejection)
    async def _guardrail(request: Request, exc: GuardrailRejection):
        return _error(request, 422, f"guardrail_{exc.category}", exc.message)

    @app.exception_handler(FilterError)
    async def _filter(request: Request, exc: FilterError):
        return _error(request, 422, "invalid_filter", str(exc), exc.details)

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound):
        return _error(request, 404, "not_found", exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errs = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _error(request, 422, "validation_error", "Request validation failed.", {"errors": errs})

    @app.exception_handler(AgentError)
    async def _agent(request: Request, exc: AgentError):
        logger.error("agent failure: %s", exc)
        return _error(request, 500, "agent_failure", f"The {exc.agent} agent failed.")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error")
        return _error(request, 500, "internal_error", "Internal server error.")
