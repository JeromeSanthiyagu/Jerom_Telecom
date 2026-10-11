"""Derived incident category (Phase 4) - deterministic, rule-based, explainable.

See src/config/taxonomy.yaml for the rules and why scoring is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.utils.config import load_yaml


@dataclass(frozen=True)
class CategoryRule:
    name: str
    requires_any: tuple[str, ...]
    weights: dict[str, float]


def load_taxonomy() -> tuple[list[CategoryRule], str, str]:
    cfg = load_yaml("taxonomy.yaml")
    rules = [CategoryRule(c["name"], tuple(c["requires_any"]), dict(c["weights"]))
             for c in cfg["categories"]]
    return rules, cfg["normal_category"], cfg["fallback_category"]


def all_categories() -> list[str]:
    rules, normal, fallback = load_taxonomy()
    return [r.name for r in rules] + [normal, fallback]


def classify_conditions(conditions: Iterable[str]) -> dict:
    """Map a set of active conditions to primary/secondary categories with reasons."""
    active = set(conditions)
    rules, normal, fallback = load_taxonomy()
    if not active:
        return {"primary_category": normal, "secondary_categories": [], "scores": {},
                "reasons": ["No derived network condition is active."]}

    scores: dict[str, float] = {}
    contributing: dict[str, list[str]] = {}
    for rule in rules:
        if not active.intersection(rule.requires_any):
            continue
        hits = [c for c in rule.weights if c in active]
        scores[rule.name] = round(sum(rule.weights[c] for c in hits), 3)
        contributing[rule.name] = hits

    if not scores:
        return {"primary_category": fallback, "secondary_categories": [], "scores": {},
                "reasons": [f"Active conditions {sorted(active)} are not mapped to any category."]}

    order = {r.name: i for i, r in enumerate(rules)}
    ranked = sorted(scores, key=lambda n: (-scores[n], order[n]))
    primary = ranked[0]
    return {
        "primary_category": primary,
        "secondary_categories": ranked[1:],
        "scores": {n: scores[n] for n in ranked},
        "reasons": [f"{primary}: score {scores[primary]} from {', '.join(contributing[primary])}"]
                   + [f"{n}: score {scores[n]} from {', '.join(contributing[n])}" for n in ranked[1:]],
    }
