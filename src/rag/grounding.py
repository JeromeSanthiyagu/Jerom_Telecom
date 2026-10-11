"""Deterministic grounding checks for LLM-written text.

An LLM sentence is rejected if it contains:
  * a number not present in the evidence package (after rounding tolerance),
  * a record ID that was not retrieved,
  * a telecom specific the dataset cannot support (X2/Xn, RACH, AMF/MME, A3 ...)
    unless that term appears in the evidence itself,
  * a guaranteed outcome ("will increase to 95%", "within 24 hours", "guaranteed").
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?(?![\w.]*[a-zA-Z_]\d)")
RECORD_ID = re.compile(r"\bROW-\d{6}\b")
# tokens whose digits are not quantities
NON_QUANTITY = [re.compile(p) for p in (r"\bROW-\d{6}\b", r"\b[2-6]G\b", r"\bP[1-4]\b", r"\bPhase \d+\b",
                                        r"\b\d+(st|nd|rd|th)\b")]

UNSUPPORTED_TERMS = {
    "X2": r"\bX2\b", "Xn": r"\bXn\b", "RACH": r"\bP?RACH\b", "AMF": r"\bAMF\b", "MME": r"\bMME\b",
    "A3 event/offset": r"\bA3\b", "S1": r"\bS1\b", "NG interface": r"\bNG-?AP\b",
}
GUARANTEES = [
    r"\bwill (increase|improve|decrease|reduce|restore|rise|drop|fall|recover)\b[^.]{0,40}\b(to|by)\s*-?\d",
    r"\bguarantee(d|s)?\b", r"\bwithin \d+\s*(minutes?|hours?|days?|weeks?)\b", r"\bdefinitely\b",
    r"\bconfirmed (root )?cause\b", r"\bwill (fix|solve|resolve)\b",
]


def _walk(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).replace("_", " ")
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def allowed_numbers(evidence: Any) -> list[float]:
    vals: list[float] = []
    for v in _walk(evidence):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            vals.append(float(v))
        elif isinstance(v, str):
            vals.extend(float(n.replace(",", "")) for n in NUMBER.findall(_strip_non_quantities(v)))
    out = set(vals)
    for v in list(out):
        if 0 <= v <= 1:
            out.add(round(v * 100, 2))        # shares/confidences quoted as percentages
    return sorted(out)


def _strip_non_quantities(text: str) -> str:
    for p in NON_QUANTITY:
        text = p.sub(" ", text)
    return text


UNIT_AFTER = re.compile(r"\s*(%|percent|percentage|pp\b|points?\b|ms\b|dbm\b|mbps\b|gb\b|hours?\b|"
                        r"users?\b|db\b|°|-\s*\d|–\s*\d|to\s+\d)", re.IGNORECASE)


def numbers_in(text: str) -> list[tuple[str, bool]]:
    """(numeric token as written, followed-by-a-unit?) - precision and units both matter."""
    clean = _strip_non_quantities(text)
    return [(m.group(0).replace(",", ""), bool(UNIT_AFTER.match(clean, m.end())))
            for m in NUMBER.finditer(clean)]


def is_supported(token: str, allowed: list[float], has_unit: bool = False) -> bool:
    """A quoted number is supported if some evidence value, rounded to the precision
    the text uses, equals it ("74.3" matches 74.336; "99.9" does not match 99.5).
    Bare small integers ("two hypotheses", "3 records") are allowed; with a unit or as
    part of a range ("4-5 percentage points") they must be in the evidence too."""
    try:
        n = float(token)
    except ValueError:
        return True
    if n.is_integer() and 0 <= n <= 10 and "." not in token and not has_unit:
        return True
    decimals = len(token.split(".")[1]) if "." in token else 0
    half_step = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(n - a) <= half_step for a in allowed)


def check_text(text: str, evidence: Any, retrieved_ids: set[str]) -> list[str]:
    """Return a list of grounding violations (empty = grounded)."""
    issues = []
    allowed = allowed_numbers(evidence)
    for n, has_unit in numbers_in(text):
        if not is_supported(n, allowed, has_unit):
            issues.append(f"unsupported number: {n}")
    for rid in RECORD_ID.findall(text):
        if rid not in retrieved_ids:
            issues.append(f"record ID not in retrieved evidence: {rid}")
    evidence_text = json.dumps(evidence, default=str)
    for name, pat in UNSUPPORTED_TERMS.items():
        if re.search(pat, text) and not re.search(pat, evidence_text):
            issues.append(f"unsupported specific not in evidence: {name}")
    for pat in GUARANTEES:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            issues.append(f"guaranteed outcome / unsupported certainty: '{m.group(0)}'")
    return sorted(set(issues))
