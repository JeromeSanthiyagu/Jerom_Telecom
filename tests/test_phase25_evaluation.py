"""Phase 25 tests: the deterministic evaluation set must keep passing (regression gate).

The LLM-judged DeepEval metrics run separately (python -m tests.evaluation.run_eval --deepeval)
because they need an API key and cost money.
"""

import pytest

from tests.evaluation.run_eval import deterministic, load_cases
from tests.test_phase9_13_retrieval import indexed


def test_cases_are_well_formed():
    from src.preprocessing.conditions import load_conditions
    from src.preprocessing.taxonomy import all_categories
    from src.utils.config import load_yaml
    hyp_ids = {h["id"] for h in load_yaml("playbook.yaml")["hypotheses"]}
    cases = load_cases()
    assert len(cases) >= 8
    for c in cases:
        assert set(c["expected_conditions"]) <= set(load_conditions())
        assert c["expected_category"] in all_categories()
        assert set(c["expected_hypotheses"]) <= hyp_ids


@indexed
@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["id"])
def test_eval_case(case):
    from src.agents.orchestrator import run
    m = deterministic(case, run(case["query"], use_llm=False))
    assert m["condition_recall"] == 1.0
    assert m["retrieval_precision"] >= 0.8
    assert m["category_match"] and m["hypothesis_hit"] and m["grounded"]
