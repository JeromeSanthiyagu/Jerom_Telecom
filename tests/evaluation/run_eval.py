"""Evaluation runner (Phase 25).

    python -m tests.evaluation.run_eval              # deterministic metrics (no API key needed)
    python -m tests.evaluation.run_eval --deepeval   # + DeepEval metrics, judged by the configured LLM

Deterministic metrics (computed from the actual retrieved records):
  condition_recall       expected conditions recognised in the query
  retrieval_precision    share of retrieved records that show ALL expected conditions
  average_precision      rank-aware precision of those relevant records (contextual precision)
  category_match         derived category equals the expected one
  hypothesis_hit         an expected hypothesis is ranked in the top 2
  grounded               narrative passed the grounding checks, no hallucination detected

DeepEval metrics (LLM-judged, need NVIDIA_API_KEY or ANTHROPIC_API_KEY):
  answer relevancy, faithfulness, contextual relevancy, contextual precision
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from src.utils.config import PROJECT_ROOT

CASES = Path(__file__).with_name("eval_cases.json")
OUT = PROJECT_ROOT / "docs" / "evaluation"
JUDGE_ATTEMPTS = 3          # a judge reply that is not valid JSON is retried before the metric counts as unscored


def load_cases() -> list[dict]:
    return json.loads(CASES.read_text(encoding="utf-8"))["cases"]


def deterministic(case: dict, result: dict) -> dict:
    expected = set(case["expected_conditions"])
    present = set(result["evidence"]["observed"]["present"])
    recs = result["retrieval"]["results"]
    rel = [expected <= set(r["conditions"]) for r in recs]
    hits, ap = 0, 0.0
    for i, r in enumerate(rel, 1):
        if r:
            hits += 1
            ap += hits / i
    top2 = [h["id"] for h in result["root_cause"].get("hypotheses", [])[:2]]
    return {
        "condition_recall": round(len(expected & present) / len(expected), 3),
        "retrieval_precision": round(sum(rel) / len(rel), 3) if rel else 0.0,
        "average_precision": round(ap / hits, 3) if hits else 0.0,
        "category_match": result["classification"]["category"] == case["expected_category"],
        "hypothesis_hit": any(h in top2 for h in case["expected_hypotheses"]),
        "grounded": result["validation"]["evidence_grounded"] and not result["validation"]["hallucination_detected"],
    }


def actual_output(result: dict) -> str:
    n = result["narrative"]["narrative"]
    acts = [a["action"] for a in result["recommendations"]["actions"]]
    return "\n".join([n["summary"], n["diagnosis"], "Recommended actions: " + "; ".join(acts)])


def run(use_llm: bool = False, deepeval: bool = False) -> dict:
    from src.agents.orchestrator import run as analyze

    rows = []
    for case in load_cases():
        t = time.time()
        result = analyze(case["query"], top_k=5, use_llm=use_llm)
        row = {"id": case["id"], "query": case["query"], **deterministic(case, result),
               "category": result["classification"]["category"],
               "top_hypotheses": [h["id"] for h in result["root_cause"].get("hypotheses", [])[:3]],
               "narrative_source": result["narrative"]["source"], "seconds": round(time.time() - t, 1)}
        if deepeval:
            row["answer"] = actual_output(result)
            row["deepeval"] = deepeval_scores(case, result)
        rows.append(row)
        print(json.dumps(row))

    keys = ["condition_recall", "retrieval_precision", "average_precision", "category_match", "hypothesis_hit", "grounded"]
    summary = {k: round(sum(float(r[k]) for r in rows) / len(rows), 3) for k in keys}
    if deepeval:
        names = sorted({m for r in rows for m in r["deepeval"]})
        # A metric the judge could not score (e.g. invalid JSON after retries) is excluded, not counted as 0.
        scored = {m: [r["deepeval"][m]["score"] for r in rows
                      if m in r["deepeval"] and r["deepeval"][m]["score"] is not None] for m in names}
        summary["deepeval"] = {m: round(sum(v) / len(v), 3) if v else None for m, v in scored.items()}
        summary["deepeval_scored"] = {m: f"{len(v)}/{len(rows)}" for m, v in scored.items()}
    report = {"summary": summary, "cases": rows, "use_llm": use_llm, "deepeval": deepeval}
    if deepeval or use_llm:
        from src.rag.llm_provider import get_provider
        p = get_provider()
        report["judge"] = f"{p.name} {p.model}" if deepeval else None
        report["llm"] = f"{p.name} {p.model}"
    OUT.mkdir(parents=True, exist_ok=True)
    name = "eval_results_deepeval.json" if deepeval else "eval_results.json"
    (OUT / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def deepeval_judge():
    """DeepEval judge model for the active LLM provider (same key and model as the application)."""
    from src.rag.llm_provider import NVIDIA_BASE_URL, get_provider

    p = get_provider()
    if p.name == "nvidia":
        from deepeval.models import LocalModel        # any OpenAI-compatible endpoint
        effort = getattr(p, "reasoning_effort", "")
        return LocalModel(model=p.model, api_key=os.getenv("NVIDIA_API_KEY"),
                          base_url=os.getenv("NVIDIA_BASE_URL") or NVIDIA_BASE_URL, temperature=0,
                          generation_kwargs={"reasoning_effort": effort} if effort else None)
    from deepeval.models import AnthropicModel
    return AnthropicModel(model=p.model, api_key=os.getenv("ANTHROPIC_API_KEY"))


def deepeval_scores(case: dict, result: dict) -> dict:
    from deepeval.metrics import (AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRelevancyMetric,
                                  FaithfulnessMetric)
    from deepeval.test_case import LLMTestCase

    judge = deepeval_judge()
    tc = LLMTestCase(
        input=case["query"],
        actual_output=actual_output(result),
        expected_output=case["expected_output"],
        retrieval_context=[r["search_text"] for r in result["retrieval"]["results"]],
    )
    out = {}
    for metric in (AnswerRelevancyMetric(model=judge, async_mode=False),
                   FaithfulnessMetric(model=judge, async_mode=False),
                   ContextualRelevancyMetric(model=judge, async_mode=False),
                   ContextualPrecisionMetric(model=judge, async_mode=False)):
        name = type(metric).__name__
        for attempt in range(1, JUDGE_ATTEMPTS + 1):
            try:
                metric.measure(tc)
                out[name] = {"score": round(float(metric.score), 3), "reason": metric.reason, "attempts": attempt}
                break
            except Exception as exc:  # e.g. the judge returned invalid JSON: retry, then record as unscored
                out[name] = {"score": None, "reason": f"error: {exc}", "attempts": attempt}
    return out


if __name__ == "__main__":
    from src.rag.llm_provider import get_provider  # loads .env

    want_deepeval = "--deepeval" in sys.argv
    if want_deepeval and get_provider().name == "offline":
        sys.exit("DeepEval metrics need an LLM: set NVIDIA_API_KEY (or ANTHROPIC_API_KEY) in .env")
    rep = run(use_llm="--llm" in sys.argv or want_deepeval, deepeval=want_deepeval)
    print(json.dumps(rep["summary"], indent=2))
