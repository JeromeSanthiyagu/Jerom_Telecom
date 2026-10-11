"""LLM provider abstraction (Phase 15).

The LLM only *interprets* evidence that deterministic code has already
assembled. Every provider exposes one method:

    structured(system, user, schema) -> (instance of schema | None, info)

* NvidiaProvider    - NVIDIA API Catalog (OpenAI-compatible endpoint, e.g.
  openai/gpt-oss-20b) with JSON-schema structured output, validated by Pydantic.
* AnthropicProvider - Claude via the official SDK, structured outputs
  (`messages.parse`), model from LLM_MODEL in .env.
* OfflineProvider   - returns None; callers fall back to deterministic
  templates, so the whole system works without an API key or network.

Configuration (.env, never hard-coded):
    NVIDIA_API_KEY      key for the NVIDIA API Catalog (build.nvidia.com)
    NVIDIA_BASE_URL     optional, default https://integrate.api.nvidia.com/v1
    LLM_REASONING_EFFORT  optional for reasoning models: low | medium | high
                        (default low for gpt-oss; set empty to send no setting)
    ANTHROPIC_API_KEY   key for the Anthropic API
    LLM_MODEL           e.g. openai/gpt-oss-20b or claude-haiku-4-5
    LLM_PROVIDER        nvidia | anthropic | offline
                        (default: nvidia if NVIDIA_API_KEY is set, else anthropic if
                        ANTHROPIC_API_KEY is set, else offline)
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any, Protocol, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from src.utils.config import PROJECT_ROOT

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_MODELS = {"nvidia": "openai/gpt-oss-20b", "anthropic": "claude-haiku-4-5"}
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class LLMProvider(Protocol):
    name: str
    model: str | None

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4096) -> tuple[T | None, dict]:
        ...


class OfflineProvider:
    name = "offline"
    model = None

    def structured(self, system, user, schema, max_tokens=4096):
        return None, {"provider": "offline", "status": "skipped",
                      "reason": "No LLM configured; deterministic template used."}


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None, timeout: float = 60.0):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4096) -> tuple[T | None, dict]:
        a = self._anthropic
        info: dict[str, Any] = {"provider": self.name, "model": self.model}
        t0 = time.time()
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
        except a.AuthenticationError:
            return None, {**info, "status": "error", "reason": "authentication failed (check ANTHROPIC_API_KEY)"}
        except a.NotFoundError:
            return None, {**info, "status": "error", "reason": f"model '{self.model}' not found"}
        except a.RateLimitError:
            return None, {**info, "status": "error", "reason": "rate limited"}
        except a.BadRequestError as e:
            return None, {**info, "status": "error", "reason": f"bad request: {e.message}"}
        except a.APIStatusError as e:
            return None, {**info, "status": "error", "reason": f"API error {e.status_code}"}
        except a.APIConnectionError:
            return None, {**info, "status": "error", "reason": "network error"}

        info.update({
            "latency_s": round(time.time() - t0, 2),
            "stop_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        })
        if resp.stop_reason == "refusal":
            return None, {**info, "status": "refused"}
        if resp.stop_reason == "max_tokens":
            return None, {**info, "status": "truncated"}
        if resp.parsed_output is None:
            return None, {**info, "status": "unparsed"}
        return resp.parsed_output, {**info, "status": "ok"}


class NvidiaProvider:
    """NVIDIA API Catalog through its OpenAI-compatible chat-completions endpoint.

    The schema is sent as a JSON-schema `response_format`; the reply is validated
    with Pydantic, so callers receive the same typed object as with Claude."""

    name = "nvidia"
    # Reasoning models (gpt-oss) spend completion tokens on their reasoning before
    # the answer, so the limit is raised to leave room for both.
    min_tokens = 4096

    def __init__(self, model: str, api_key: str, base_url: str = NVIDIA_BASE_URL, timeout: float = 90.0,
                 reasoning_effort: str | None = None):
        import openai

        self._openai = openai
        self.model = model
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        # gpt-oss: "low" measured fastest (5-16 s) with clean grounding on the evaluation queries;
        # "medium" once produced runaway whitespace and invalid JSON. Other models get no setting.
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else (
            "low" if "gpt-oss" in model else "")

    def structured(self, system: str, user: str, schema: type[T], max_tokens: int = 4096) -> tuple[T | None, dict]:
        o = self._openai
        info: dict[str, Any] = {"provider": self.name, "model": self.model}
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max(max_tokens, self.min_tokens),
                temperature=0.2,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()}},
                **extra,
            )
        except o.AuthenticationError:
            return None, {**info, "status": "error", "reason": "authentication failed (check NVIDIA_API_KEY)"}
        except o.NotFoundError:
            return None, {**info, "status": "error", "reason": f"model '{self.model}' not found"}
        except o.RateLimitError:
            return None, {**info, "status": "error", "reason": "rate limited"}
        except o.BadRequestError as e:
            return None, {**info, "status": "error", "reason": f"bad request: {e.message}"}
        except o.APIStatusError as e:
            return None, {**info, "status": "error", "reason": f"API error {e.status_code}"}
        except o.APIConnectionError:
            return None, {**info, "status": "error", "reason": "network error"}
        choice = resp.choices[0]
        info.update({
            "latency_s": round(time.time() - t0, 2),
            "stop_reason": choice.finish_reason,
            "input_tokens": getattr(resp.usage, "prompt_tokens", None),
            "output_tokens": getattr(resp.usage, "completion_tokens", None),
        })
        if getattr(choice.message, "refusal", None):
            return None, {**info, "status": "refused"}
        if choice.finish_reason == "length":
            return None, {**info, "status": "truncated"}
        try:
            return schema.model_validate_json(choice.message.content or ""), {**info, "status": "ok"}
        except ValueError:                      # empty or not matching the schema
            return None, {**info, "status": "unparsed"}


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    choice = os.getenv("LLM_PROVIDER", "").strip().lower()
    nvidia_key = os.getenv("NVIDIA_API_KEY", "").strip()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not choice:
        choice = "nvidia" if nvidia_key else "anthropic" if anthropic_key else "offline"
    key = {"nvidia": nvidia_key, "anthropic": anthropic_key}.get(choice, "")
    if choice not in DEFAULT_MODELS or not key:
        if choice not in ("offline", *DEFAULT_MODELS):
            logger.warning("Unknown LLM_PROVIDER '%s'; using offline templates", choice)
        elif choice != "offline":
            logger.warning("LLM_PROVIDER=%s but no API key is set; using offline templates", choice)
        logger.info("LLM provider: offline")
        return OfflineProvider()
    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODELS[choice]
    logger.info("LLM provider: %s (%s)", choice, model)
    if choice == "nvidia":
        effort = os.getenv("LLM_REASONING_EFFORT")
        return NvidiaProvider(model=model, api_key=key,
                              base_url=os.getenv("NVIDIA_BASE_URL", "").strip() or NVIDIA_BASE_URL,
                              reasoning_effort=effort.strip().lower() if effort is not None else None)
    return AnthropicProvider(model=model, api_key=key)
