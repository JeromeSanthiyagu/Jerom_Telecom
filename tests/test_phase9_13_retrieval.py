"""Phases 9-13 tests: vector search, BM25, hybrid search, metadata filters, metric similarity."""

import math

import numpy as np
import pytest

from src.rag.bm25 import BM25Index, tokenize
from src.rag.filters import FilterError, normalize_filters
from src.rag.query_parser import parse_query
from src.utils.config import PROJECT_ROOT

INDEXED = (PROJECT_ROOT / "data/processed/chroma/manifest.json").exists()
indexed = pytest.mark.skipif(not INDEXED, reason="indexes not built (python -m src.preprocessing.build_index)")


# ------------------------------------------------------------------ BM25 (unit)
def test_tokenizer_adds_bigrams_and_snake_parts():
    toks = tokenize("High latency (high_latency) 74.3 ms")
    assert {"high", "latency", "high_latency"} <= set(toks)
    assert "74" not in toks and "ms" in toks


def test_bm25_matches_reference_formula():
    docs = ["high latency packet loss", "low latency", "packet loss packet loss jitter", "signal weak"]
    idx = BM25Index.build(docs, [f"D{i}" for i in range(4)], k1=1.5, b=0.75)
    q = "packet loss"
    lens = [len(tokenize(d)) for d in docs]
    avgdl = sum(lens) / len(lens)
    expected = []
    for d, dl in zip(docs, lens):
        toks = tokenize(d)
        s = 0.0
        for t in set(tokenize(q)):
            n = sum(t in tokenize(x) for x in docs)
            idf = math.log(1 + (4 - n + 0.5) / (n + 0.5))
            tf = toks.count(t)
            s += idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl)) if tf else 0
        expected.append(s)
    assert np.allclose(idx.scores(q), expected)
    assert idx.search(q, 2)[0]["record_id"] == "D2"


def test_bm25_mask_excludes_rows():
    idx = BM25Index.build(["packet loss", "packet loss"], ["A", "B"])
    hits = idx.search("packet loss", 5, mask=np.array([False, True]))
    assert [h["record_id"] for h in hits] == ["B"]


# ------------------------------------------------------------------ filters (unit)
def test_filters_normalise_names_and_case():
    f = normalize_filters({"state": "Minnesota", "season": "winter", "severity": "high",
                           "weather": ["rain", "Snow"], "issue_reported": "yes",
                           "conditions": ["HIGH_LATENCY"], "category": "mobility / handover"})
    assert f.values["state"] == ["MN"] and f.values["season"] == ["Winter"] and f.values["severity"] == ["HIGH"]
    assert f.values["weather"] == ["Rain", "Snow"] and f.values["issue_reported"] == [True]
    assert f.conditions == ["high_latency"] and f.values["category"] == ["Mobility / Handover"]


@pytest.mark.parametrize("bad", [{"planet": "Mars"}, {"state": "Narnia"}, {"season": "Monsoon"},
                                 {"conditions": ["very_bad"]}, {"severity": "EXTREME"}, {"issue_reported": "maybe"}])
def test_invalid_filters_raise(bad):
    with pytest.raises(FilterError):
        normalize_filters(bad)


# ------------------------------------------------------------------ query understanding (unit)
def test_qualitative_directions():
    q = parse_query("High latency, high packet loss, good signal strength, normal user load.")
    d = {c.metric: c.direction for c in q.constraints}
    assert d == {"latency_ms": "adverse", "packet_loss_pct": "adverse",
                 "signal_strength_dbm": "favorable", "user_count": "normal"}
    assert q.adverse_conditions() == ["high_latency", "high_packet_loss"]
    assert "weak_signal" in q.favorable_conditions()
    assert "signal" not in q.retrieval_text and "latency" in q.retrieval_text


def test_qualifier_does_not_cross_clauses():
    q = parse_query("low latency and high packet loss")
    d = {c.metric: c.direction for c in q.constraints}
    assert d == {"latency_ms": "favorable", "packet_loss_pct": "adverse"}


def test_declining_success_metric_is_adverse():
    q = parse_query("increasing fault occurrence and declining handover success")
    assert q.adverse_conditions() == ["high_fault_rate", "low_handover"]


def test_numeric_values_and_context():
    q = parse_query("latency 150 ms, packet loss above 12%, signal -115 dBm in Minnesota during winter snow")
    vals = {c.metric: (c.value, c.comparator) for c in q.constraints}
    assert vals["latency_ms"] == (150.0, "eq") and vals["packet_loss_pct"] == (12.0, "gt")
    assert vals["signal_strength_dbm"] == (-115.0, "eq")
    assert q.context == {"season": "Winter", "weather": "Snow", "state": "MN"}
    assert {"high_latency", "weak_signal", "high_packet_loss"} <= set(q.adverse_conditions())
    # "above" a value below the threshold proves nothing
    assert "high_packet_loss" not in parse_query("packet loss above 5%").adverse_conditions()


# ------------------------------------------------------------------ built indexes
@pytest.fixture(scope="module")
def retriever():
    from src.rag.hybrid import get_retriever
    return get_retriever()


@indexed
def test_vector_ids_map_to_real_records(retriever):
    from src.rag.vector_store import semantic_search
    hits = semantic_search("weak signal strength and high call drop rate", top_k=10)
    assert len(hits) == 10
    for h in hits:
        assert h["record_id"] in retriever.row_of
        assert -1.0 <= h["semantic_score"] <= 1.0
        assert h["metadata"]["record_id"] == h["record_id"]


@indexed
def test_keyword_search_finds_condition_records(retriever):
    from src.rag.bm25 import keyword_search
    for h in keyword_search("high jitter", top_k=10):
        assert retriever.df.loc[h["row"], "cond_high_jitter"]


FILTER_CASES = [
    {"state": "Minnesota", "season": "Winter"},
    {"weather": ["Rain", "Snow"]},
    {"category": "Mobility / Handover"},
    {"severity": ["HIGH", "CRITICAL"], "issue_reported": True},
    {"conditions": ["high_latency", "high_packet_loss"]},
    {"city": "Lake Michael"},
    {"priority": "P1", "season": "Summer"},
]


@indexed
@pytest.mark.parametrize("filters", FILTER_CASES)
def test_results_never_violate_filters(retriever, filters):
    from src.rag.filters import COLUMN
    res = retriever.hybrid_search("high latency and packet loss with call drops", filters, top_k=20)
    f = normalize_filters(filters, retriever.cities)
    assert res["results"], filters
    for r in res["results"]:
        row = retriever.df.iloc[retriever.row_of[r["record_id"]]]
        for key, allowed in f.values.items():
            assert row[COLUMN[key]] in allowed, (r["record_id"], key)
        for c in f.conditions:
            assert row[f"cond_{c}"]


@indexed
def test_returned_values_equal_stored_records(retriever):
    res = retriever.hybrid_search("low handover success", None, top_k=5)
    for r in res["results"]:
        row = retriever.df.iloc[retriever.row_of[r["record_id"]]]
        for m, v in r["metrics"].items():
            assert v == pytest.approx(float(row[m]))
        assert r["conditions"] == list(row["conditions"])


@indexed
def test_hybrid_scores_and_ranking(retriever):
    res = retriever.hybrid_search("high latency and packet loss", None, top_k=5)
    scores = [r["hybrid_score"] for r in res["results"]]
    assert scores == sorted(scores, reverse=True)
    assert abs(sum(res["weights"].values()) - 1) < 1e-6
    for r in res["results"]:
        assert {"high_latency", "high_packet_loss"} <= set(r["conditions"])


@indexed
def test_normal_and_good_statements_steer_away(retriever):
    res = retriever.hybrid_search("high latency and packet loss with good signal strength and normal user load",
                                  None, top_k=5)
    for r in res["results"]:
        assert "weak_signal" not in r["conditions"] and "high_load" not in r["conditions"]


@indexed
def test_unmatched_filter_returns_empty(retriever):
    res = retriever.hybrid_search("latency", {"season": "Summer", "weather": "Snow"}, top_k=5)
    assert res["results"] == [] and res["matching_records"] == 0


@indexed
def test_metric_values_rank_close_records(retriever):
    res = retriever.hybrid_search("latency 150 ms", None, top_k=5)
    assert all(abs(r["metrics"]["latency_ms"] - 150) < 25 for r in res["results"])
