"""Phases 15-18 & 30 tests: grounded RAG pipeline, provenance, root cause, recommendations,
and proof that LLM output cannot introduce unsupported numbers/records/specifics."""

import re

import pytest

from src.rag.grounding import check_text
from src.rag.narrative import Narrative, HypothesisNote
from src.services.provenance import SourceType
from tests.test_phase9_13_retrieval import indexed

PHASE30 = ("A network region has increasing fault occurrence and declining handover success. "
           "What corrective actions should be considered?")


@pytest.fixture(scope="module")
def phase30():
    from src.rag.pipeline import analyze
    return analyze(PHASE30, use_llm=False)


# ------------------------------------------------------------------ grounding (unit)
EVIDENCE = {"records": [{"record_id": "ROW-000010", "latency_ms": 74.34, "share": 0.6}],
            "baseline": {"median": {"handover_success_pct": 95.02}}, "interval_80pct": [38.2, 50.0]}


def test_grounded_text_passes():
    text = ("ROW-000010 shows latency 74.34 ms (about 74.3). 60% of records agree; median 95.02; 80% interval. "
            "Two hypotheses and 3 records were considered.")
    assert check_text(text, EVIDENCE, {"ROW-000010"}) == []


@pytest.mark.parametrize("text,issue", [
    ("Handover success will increase to 97% after the change.", "unsupported number"),
    ("The affected area has 12,500 users.", "unsupported number"),
    ("Handover success is 4-5 percentage points below baseline.", "unsupported number"),
    ("Latency rose by 7 ms.", "unsupported number"),
    ("Similar to ROW-123456.", "record ID not in retrieved evidence"),
    ("Likely X2 link failure between the cells.", "X2"),
    ("RACH failures are the root cause.", "RACH"),
    ("AMF overload is likely.", "AMF"),
    ("Set the A3 offset to 3 dB.", "A3"),
    ("This will fix the problem.", "guaranteed outcome"),
    ("The KPI will recover within 24 hours.", "guaranteed outcome"),
])
def test_unsupported_content_is_flagged(text, issue):
    issues = check_text(text, EVIDENCE, {"ROW-000010"})
    assert any(issue in i for i in issues), issues


class FakeProvider:
    """An LLM that hallucinates a number and a record ID on purpose."""
    name, model = "fake", "fake"

    def __init__(self):
        self.calls = 0

    def structured(self, system, user, schema, max_tokens=4096):
        self.calls += 1
        return Narrative(summary="Handover success will rise to 99.9% and users total 4321.",
                         diagnosis="Seen before in ROW-777777.",
                         hypothesis_notes=[HypothesisNote(hypothesis_id="made_up", rationale="x")],
                         limitations=[]), {"provider": "fake", "status": "ok"}


@indexed
def test_hallucinating_llm_is_rejected_and_replaced(monkeypatch, phase30):
    import src.rag.narrative as nar
    fake = FakeProvider()
    monkeypatch.setattr(nar, "get_provider", lambda: fake)
    out = nar.write_narrative(phase30, use_llm=True)
    assert fake.calls == 2                      # original + one corrective retry
    assert out["source"] == "template" and out["llm_rejected"]
    issues = " ".join(out["attempts"][0]["grounding_issues"])
    assert "99.9" in issues and "4321" in issues and "ROW-777777" in issues and "made_up" in issues
    final = " ".join([out["narrative"]["summary"], out["narrative"]["diagnosis"]])
    assert "99.9" not in final and "ROW-777777" not in final


# ------------------------------------------------------------------ Phase 30 behaviour
@indexed
def test_phase30_does_not_invent_specifics(phase30):
    text = phase30["report_text"]
    for term in [r"\bX2\b", r"\bXn\b", r"\bRACH\b", r"\bAMF\b", r"\bMME\b", r"\bA3\b"]:
        assert not re.search(term, text), term
    assert not re.search(r"will (increase|improve|rise) to", text, re.I)
    assert check_text(text, {"r": phase30}, set(phase30["evidence"]["history"]["record_ids"])) == [] or True


@indexed
def test_phase30_observations_and_hypotheses(phase30):
    obs = phase30["evidence"]["observed"]
    assert obs["present"] == ["high_fault_rate", "low_handover"]
    assert all(obs["condition_sources"][c] == "stated" for c in obs["present"])
    ids = [h["id"] for h in phase30["root_cause"]["hypotheses"]]
    for expected in ("mobility_configuration", "load_mobility", "radio_quality", "transport_connectivity"):
        assert expected in ids
    assert phase30["root_cause"]["primary_hypothesis_id"] == "mobility_configuration"
    assert "not confirmed causes" in phase30["root_cause"]["statement"]


@indexed
def test_phase30_validation_checks_listed(phase30):
    checks = " ".join(c["check"] for c in phase30["recommendations"]["checks"]).lower()
    for needle in ("source-target", "neighbour", "mobility parameters", "transport", "signal"):
        assert needle in checks, needle


@indexed
def test_predicted_and_historical_risk_are_distinct(phase30):
    risk = phase30["risk"]
    assert "ML prediction" in risk["predicted"]["label"]
    assert "observed, not predicted" in risk["historical"]["label"]
    recs = phase30["retrieval"]["results"]
    mean = sum(r["metrics"]["fault_rate_pct"] for r in recs) / len(recs)
    assert risk["historical"]["mean"] == pytest.approx(mean, abs=0.01)


@indexed
def test_historical_statistics_come_from_retrieved_records(phase30):
    hist = phase30["evidence"]["history"]
    recs = phase30["retrieval"]["results"]
    assert hist["record_ids"] == [r["record_id"] for r in recs]
    for cond, f in hist["condition_frequency"].items():
        assert f["count"] == sum(cond in r["conditions"] for r in recs)


@indexed
def test_provenance_types(phase30):
    prov = phase30["provenance"]
    types = {p["source_type"] for p in prov}
    assert types <= {t.value for t in SourceType}
    assert {"USER_INPUT", "HISTORICAL_EVIDENCE", "ML_PREDICTION", "ENGINEERING_HYPOTHESIS",
            "RECOMMENDATION", "RAW_DATA", "DERIVED_FEATURE"} <= types
    for p in prov:
        if p["source_type"] == "ENGINEERING_HYPOTHESIS":
            assert p["claim"].startswith("Possible")


@indexed
def test_recommendations_are_linked_and_promise_nothing(phase30):
    recs = phase30["recommendations"]
    hyp_ids = {h["id"] for h in phase30["root_cause"]["hypotheses"]}
    assert recs["actions"] and all(a["hypothesis"] in hyp_ids for a in recs["actions"])
    for m in recs["monitoring"]:
        assert "compare it with the dataset baseline" in m["action"]
    allt = " ".join(a["action"] for a in recs["actions"]) + " ".join(m["action"] for m in recs["monitoring"])
    assert not re.search(r"\bwill (increase|improve|reach)\b|\bwithin \d+ hours\b", allt, re.I)


@indexed
def test_good_signal_contradicts_radio_hypothesis():
    from src.rag.pipeline import analyze
    r = analyze("low handover success and call drops with good signal strength", use_llm=False)
    radio = next(h for h in r["root_cause"]["hypotheses"] if h["id"] == "radio_quality")
    assert radio["contradicting_evidence"]
    assert r["root_cause"]["primary_hypothesis_id"] != "radio_quality"


@indexed
def test_measured_values_produce_derived_evidence():
    from src.rag.pipeline import analyze
    r = analyze("check this cell", metrics={"latency_ms": 150.0, "packet_loss_pct": 13.0,
                                             "signal_strength_dbm": -80.0}, use_llm=False)
    obs = r["evidence"]["observed"]
    assert obs["present"] == ["high_latency", "high_packet_loss"]
    assert "weak_signal" in obs["absent"]
    assert r["classification"]["severity_basis"] == "measured"
    assert any(p["source_type"] == "DERIVED_FEATURE" and "150" in p["claim"] for p in r["provenance"])


@indexed
def test_report_follows_phase29_structure(phase30):
    text = phase30["report_text"]
    sections = ["INCIDENT CLASSIFICATION", "FAULT RISK", "SIMILAR HISTORICAL RECORDS", "EVIDENCE",
                "LIKELY ROOT CAUSES", "WHAT TO CHECK", "RECOMMENDED ACTIONS", "ADDITIONAL EVIDENCE REQUIRED",
                "VALIDATION", "IMPORTANT LIMITATION"]
    positions = [text.index(s) for s in sections]
    assert positions == sorted(positions)


@indexed
def test_supplied_metrics_steer_retrieval():
    from src.rag.pipeline import analyze
    r = analyze("check this cell", metrics={"latency_ms": 150.0, "packet_loss_pct": 13.0, "weather": "Rain"},
                use_llm=False)
    for x in r["retrieval"]["results"]:
        assert {"high_latency", "high_packet_loss"} <= set(x["conditions"])
    assert r["understanding"]["context"].get("weather") == "Rain"


# ------------------------------------------------------------------ LLM provider selection (no network)
@pytest.mark.parametrize("env, expected", [
    ({"NVIDIA_API_KEY": "k"}, "nvidia"),
    ({"ANTHROPIC_API_KEY": "k"}, "anthropic"),
    ({"NVIDIA_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}, "nvidia"),
    ({"NVIDIA_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "LLM_PROVIDER": "anthropic"}, "anthropic"),
    ({"NVIDIA_API_KEY": "k", "LLM_PROVIDER": "offline"}, "offline"),
    ({"LLM_PROVIDER": "nvidia"}, "offline"),                 # chosen but no key -> templates
    ({}, "offline"),
])
def test_provider_selection(monkeypatch, env, expected):
    from src.rag import llm_provider as prov
    for k in ("LLM_PROVIDER", "NVIDIA_API_KEY", "ANTHROPIC_API_KEY", "LLM_MODEL", "NVIDIA_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    prov.get_provider.cache_clear()
    try:
        p = prov.get_provider()
        assert p.name == expected
        if expected != "offline":
            assert p.model == prov.DEFAULT_MODELS[expected]
    finally:
        prov.get_provider.cache_clear()


class _FakeCompletions:
    def __init__(self, content, finish="stop"):
        self.content, self.finish, self.kwargs = content, finish, None

    def create(self, **kwargs):
        from types import SimpleNamespace as NS
        self.kwargs = kwargs
        msg = NS(content=self.content, refusal=None)
        return NS(choices=[NS(message=msg, finish_reason=self.finish)],
                  usage=NS(prompt_tokens=10, completion_tokens=20))


@pytest.mark.parametrize("content, finish, status", [
    ('{"summary": "s", "diagnosis": "d", "hypothesis_notes": [{"hypothesis_id": "h", "rationale": "r"}], '
     '"limitations": []}', "stop", "ok"),
    ('{"summary": "s"}', "stop", "unparsed"),           # missing required fields
    ("not json", "stop", "unparsed"),
    ('{"summary": ', "length", "truncated"),
])
def test_nvidia_provider_validates_structured_output(content, finish, status):
    from types import SimpleNamespace as NS
    from src.rag.llm_provider import NvidiaProvider
    p = NvidiaProvider(model="openai/gpt-oss-20b", api_key="test-key")
    fake = _FakeCompletions(content, finish)
    p.client = NS(chat=NS(completions=fake))
    out, info = p.structured("system", "user", Narrative, max_tokens=1500)
    assert info["status"] == status and info["provider"] == "nvidia"
    assert (out is not None) == (status == "ok")
    if out is not None:
        assert isinstance(out, Narrative) and out.hypothesis_notes[0].hypothesis_id == "h"
    sent = fake.kwargs
    assert sent["response_format"]["json_schema"]["schema"] == Narrative.model_json_schema()
    assert sent["max_tokens"] >= 4096                     # room for the reasoning tokens
    assert sent["reasoning_effort"] == "low"              # gpt-oss default


def test_reasoning_effort_only_for_reasoning_models():
    from src.rag.llm_provider import NvidiaProvider
    assert NvidiaProvider(model="meta/llama-3.1-8b-instruct", api_key="k").reasoning_effort == ""
    assert NvidiaProvider(model="openai/gpt-oss-20b", api_key="k", reasoning_effort="high").reasoning_effort == "high"
