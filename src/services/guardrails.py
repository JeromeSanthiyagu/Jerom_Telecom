"""Input guardrails (Phase 14).

Deterministic checks run before any retrieval or LLM call:

  empty            nothing to analyse
  too_long         longer than max_chars
  prompt_injection attempts to override instructions / change the role
  secret_request   asks for keys, passwords, tokens, environment, prompts
  unsupported      asks the assistant to execute or change something
                   (this is a decision-support tool; it never acts on the network)
  off_topic        no telecom-network vocabulary at all

Rules are regular expressions so every decision is explainable and testable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_CHARS = 1000

INJECTION = [
    r"\bignore\b.{0,40}\b(instruction|prompt|rule|previous|above|system)",
    r"\bdisregard\b.{0,40}\b(instruction|prompt|rule|previous|above)",
    r"\bforget\b.{0,30}\b(instruction|prompt|rule|everything)",
    r"\byou are now\b", r"\bact as\b", r"\bpretend (to be|you)\b", r"\bjailbreak\b", r"\bdeveloper mode\b",
    r"\bnew instructions?\b", r"\boverride\b.{0,30}\b(instruction|rule|safety|guard)",
    r"<\s*/?\s*(system|assistant|instructions?)\s*>", r"\bsystem prompt\b",
]
SECRETS = [
    r"\bapi[\s_-]?keys?\b", r"\bpasswords?\b", r"\bsecret(s| key)?\b", r"\baccess tokens?\b",
    r"\bbearer\b", r"\bcredentials?\b", r"\.env\b", r"\benvironment variables?\b",
    r"\b(reveal|show|print|leak|dump)\b.{0,30}\b(prompt|instructions|config|configuration|keys?)\b",
]
UNSUPPORTED = [
    r"\b(delete|drop|truncate|wipe|erase)\b.{0,30}\b(record|records|data|database|table|index|logs?)\b",
    r"\b(shut ?down|reboot|restart|power off|disable|lock)\b.{0,30}\b(cell|site|node|network|server|base station|gnodeb|enodeb)\b",
    r"\b(execute|run)\b.{0,20}\b(command|script|shell|code|sql)\b",
    r"\b(apply|push|deploy|change|set|modify)\b.{0,40}\b(on|to) the (live|production) network\b",
    r"\brm\s+-rf\b", r"\bsudo\b", r";\s*drop\s+table\b",
]
DOMAIN_TERMS = [
    "network", "cell", "site", "5g", "4g", "lte", "nr", "ran", "radio", "latency", "packet", "loss", "jitter",
    "throughput", "handover", "handoff", "signal", "rsrp", "coverage", "call drop", "drop", "alarm", "fault",
    "outage", "availability", "kpi", "congestion", "load", "users", "traffic", "bandwidth", "mttr", "incident",
    "authentication", "security", "setup", "connection", "configuration", "parameter", "backhaul", "transport",
    "telecom", "mobile", "base station", "gnodeb", "enodeb", "degradation", "interference", "weather",
]


@dataclass
class GuardrailResult:
    allowed: bool
    category: str          # ok | empty | too_long | prompt_injection | secret_request | unsupported | off_topic
    message: str
    sanitized: str = ""


def sanitize(text: str) -> str:
    """Normalise Unicode, drop control/zero-width characters, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return " ".join(text.split())


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in patterns)


def check_query(text: str | None, max_chars: int = MAX_CHARS) -> GuardrailResult:
    clean = sanitize(text or "")
    if not clean:
        return GuardrailResult(False, "empty", "The query is empty. Describe the network problem to analyse.")
    if len(clean) > max_chars:
        return GuardrailResult(False, "too_long", f"The query is longer than {max_chars} characters. "
                               "Summarise the network symptoms more briefly.")
    if _any(INJECTION, clean):
        return GuardrailResult(False, "prompt_injection",
                               "The query tries to change the assistant's instructions. Only telecom "
                               "network-operations questions are supported.")
    if _any(SECRETS, clean):
        return GuardrailResult(False, "secret_request",
                               "Requests for credentials, keys, configuration or prompts are not supported.")
    if _any(UNSUPPORTED, clean):
        return GuardrailResult(False, "unsupported",
                               "This assistant provides decision support only. It cannot execute commands, "
                               "change network elements or modify data.")
    low = clean.lower()
    if not any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low) for t in DOMAIN_TERMS):
        return GuardrailResult(False, "off_topic",
                               "The query does not describe a telecom network-operations issue.")
    return GuardrailResult(True, "ok", "Query accepted.", clean)
