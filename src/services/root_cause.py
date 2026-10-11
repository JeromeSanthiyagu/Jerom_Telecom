"""Root-cause hypotheses (Phase 17) from src/config/playbook.yaml.

Hypotheses are scored with the documented heuristic in playbook.yaml and are
always reported as hypotheses with supporting evidence, contradicting
evidence, alternatives and the extra evidence needed to confirm them.
"""

from __future__ import annotations

from typing import Any

from src.preprocessing.conditions import load_conditions
from src.utils.config import load_yaml


def _playbook() -> dict:
    return load_yaml("playbook.yaml")


def hypothesis_by_id(hid: str) -> dict | None:
    return next((h for h in _playbook()["hypotheses"] if h["id"] == hid), None)


def score_hypotheses(state: dict[str, Any], history: dict[str, Any]) -> list[dict]:
    """Score every triggered hypothesis. `state` from evidence.observed_state."""
    pb = _playbook()
    cfg = pb["confidence"]
    labels = {n: c.label for n, c in load_conditions().items()}
    present, absent = set(state["present"]), set(state["absent"])
    context = {**state.get("context", {})}
    freq = history.get("condition_frequency", {})
    n_hist = history.get("n_records", 0)
    weather_hist = history.get("weather_distribution", {})

    out = []
    for h in pb["hypotheses"]:
        if not present.intersection(h["triggers_any"]):
            continue
        weights = {**h.get("supporting", {}), **h.get("key_evidence", {})}
        matched = {c: w for c, w in weights.items() if c in present}
        supporting = [{"condition": c, "label": labels[c], "source": state["condition_sources"].get(c, "stated")}
                      for c in matched]

        ctx_match = False
        for key, allowed in (h.get("context_key") or {}).items():
            if context.get(key) in allowed:
                ctx_match = True
                supporting.append({"context": key, "value": context[key], "source": "stated"})
        total_w = sum(weights.values()) + (1.0 if h.get("context_key") else 0.0)
        matched_w = sum(matched.values()) + (1.0 if ctx_match else 0.0)

        contradicting = []
        for c in h.get("contradicted_if_present", []):
            if c in present:
                contradicting.append(f"{labels[c]} is present, which points to a different cause")
        for c in h.get("contradicted_if_absent", []):
            if c in absent:
                contradicting.append(f"{labels[c]} is reported/measured as within threshold")

        # historical support: share of similar records showing the key evidence
        key = list(h.get("key_evidence", {}))
        if key and n_hist:
            hist_share = max(freq.get(c, {}).get("share", 0.0) for c in key)
        elif h.get("context_key") and n_hist:
            allowed = next(iter(h["context_key"].values()))
            hist_share = sum(v for k, v in weather_hist.items() if k in allowed) / n_hist
        else:
            hist_share = 0.0

        conf = (cfg["base"] + cfg["evidence_weight"] * (matched_w / total_w if total_w else 0)
                + cfg["historical_weight"] * hist_share - cfg["contradiction_penalty"] * len(contradicting))
        conf = round(min(cfg["max"], max(cfg["min"], conf)), 2)

        missing_key = [c for c in key if c in state["unknown"]]
        additional = [f"{labels[c]} ({load_conditions()[c].metric}) for the affected cells - not stated in the query"
                      for c in missing_key] + list(h.get("additional_evidence", []))
        out.append({
            "id": h["id"],
            "title": h["title"],
            "confidence": conf,
            "confidence_breakdown": {
                "base": cfg["base"],
                "evidence_match": round(matched_w / total_w, 2) if total_w else 0.0,
                "historical_share": round(hist_share, 2),
                "contradictions": len(contradicting),
            },
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "historical_support": (f"{round(hist_share * 100)}% of the {n_hist} retrieved similar records show "
                                   f"the key evidence" if n_hist and (key or h.get('context_key')) else None),
            "additional_evidence_required": additional,
            "note": h.get("note"),
            "checks": list(h["checks"]),
            "actions": list(h["actions"]),
        })
    out.sort(key=lambda x: (-x["confidence"], x["id"]))
    return out


def root_cause_analysis(state: dict[str, Any], history: dict[str, Any]) -> dict:
    """Phase 17 output structure."""
    cfg = _playbook()["confidence"]
    scored = score_hypotheses(state, history)
    if not scored:
        return {
            "primary_hypothesis": None,
            "confidence": 0.0,
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "alternative_hypotheses": [],
            "additional_evidence_required": ["Describe the affected KPIs (e.g. latency, handover success, "
                                             "call drops) so a hypothesis can be formed."],
            "hypotheses": [],
            "statement": "No degraded condition was identified, so no root-cause hypothesis is proposed.",
        }
    top = scored[0]
    shown = scored[: cfg["max_hypotheses"]]
    return {
        "primary_hypothesis": top["title"],
        "primary_hypothesis_id": top["id"],
        "confidence": top["confidence"],
        "confidence_method": "heuristic score from playbook.yaml (not a calibrated probability)",
        "supporting_evidence": top["supporting_evidence"],
        "contradicting_evidence": top["contradicting_evidence"],
        "alternative_hypotheses": [{"id": h["id"], "title": h["title"], "confidence": h["confidence"]}
                                   for h in shown[1:]],
        "additional_evidence_required": top["additional_evidence_required"],
        "hypotheses": shown,
        "statement": "These are engineering hypotheses ranked by the available evidence, not confirmed causes.",
    }
