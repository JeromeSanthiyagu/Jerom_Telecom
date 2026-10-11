"""FastAPI application entry point (Phase 27).

    uvicorn src.main:app --reload --port 8000      (OpenAPI docs at /docs)

The endpoints live in src/api/routes.py and the structured error handlers in
src/api/errors.py. Set WARMUP=0 to skip loading the indexes/models at start-up
(tests do this).
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.errors import register_error_handlers
from src.api.routes import VERSION, router
from src.api.schemas import ErrorResponse
from src.rag.hybrid import get_retriever
from src.utils.config import PROJECT_ROOT

logger = logging.getLogger("netops.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("WARMUP", "1") != "0":
        t = time.time()
        r = get_retriever()
        r.hybrid_search("warm up latency", None, 1)       # loads embedding model + Chroma HNSW
        logger.info("warm-up done in %.1fs", time.time() - t)
    yield


app = FastAPI(
    title="AI-Powered Telecom Network Operations Assistant",
    version=VERSION,
    description=("Evidence-grounded decision support: hybrid retrieval over 100,000 network-performance records, "
                 "derived conditions/category/severity, fault-risk prediction, root-cause hypotheses and "
                 "recommendations. Derived labels are NOT ground truth; hypotheses are NOT confirmed causes."),
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    logger.info("%s %s -> %s (%.2fs) rid=%s", request.method, request.url.path, response.status_code,
                time.time() - t0, rid)
    return response


register_error_handlers(app)
app.include_router(router)


# ----------------------------------------------------------------------------- optional React web UI
# `npm run build` in web/ writes web/dist; when present it is served at /ui (same origin, no CORS).
# The Streamlit front end (frontend.py) is the primary UI and talks to this API over HTTP.
UI_DIST = PROJECT_ROOT / "web" / "dist"
if (UI_DIST / "index.html").exists():
    app.mount("/ui/assets", StaticFiles(directory=UI_DIST / "assets"), name="ui-assets")

    @app.get("/ui", include_in_schema=False)
    @app.get("/ui/{path:path}", include_in_schema=False)
    def web_ui(path: str = "") -> FileResponse:
        target = (UI_DIST / path).resolve()
        if path and target.is_file() and UI_DIST.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(UI_DIST / "index.html")      # client-side routes
