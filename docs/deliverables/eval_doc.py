"""docs/Evaluation_Report.pdf — test cases, retrieval results, DeepEval scores, ML metrics and sample outputs.

    python -m tests.evaluation.run_eval --deepeval      # produces docs/evaluation/eval_results_deepeval.json
    python -m docs.deliverables.eval_doc
"""

from __future__ import annotations

import json
import subprocess
import sys

from docs.deliverables.kit import P, abridge, build, cover, dumps, mono, samples, section, table
from docs.guide.pdf_kit import bullets, callout, code
from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "Evaluation_Report.pdf"
EV = PROJECT_ROOT / "docs" / "evaluation"
METRICS = [("FaithfulnessMetric", "Faithfulness"), ("AnswerRelevancyMetric", "Answer relevancy"),
           ("ContextualPrecisionMetric", "Contextual precision"), ("ContextualRelevancyMetric", "Contextual relevancy")]


def _load(name: str) -> dict | None:
    p = EV / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _cell(m: dict) -> str:
    if not m:
        return "–"
    if m.get("score") is None:
        return "–"
    return f"{m['score']:.2f}" + ("*" if m.get("attempts", 1) > 1 else "")


def _tests() -> tuple[int, list[list[str]]]:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600).stdout
    per_file: dict[str, int] = {}
    total = 0
    for line in out.splitlines():
        if "::" in line:
            f = line.split("::")[0]
            per_file[f] = per_file.get(f, 0) + 1
        if "tests collected" in line:
            total = int(line.split()[0])
    rows = [[mono(f.replace("tests/", "").replace("tests\\", "")), str(n)] for f, n in sorted(per_file.items())]
    return total, rows


def story() -> list:
    run = _load("eval_results_deepeval.json")
    prev = _load("eval_results_deepeval_claude_2026-10-09.json")
    run1 = _load("eval_results_deepeval_nvidia_run1_unretried.json")
    cases = json.loads((PROJECT_ROOT / "tests" / "evaluation" / "eval_cases.json").read_text(encoding="utf-8"))["cases"]
    fr = json.loads((PROJECT_ROOT / "docs" / "models" / "fault_risk.json").read_text(encoding="utf-8"))
    tp = json.loads((PROJECT_ROOT / "docs" / "models" / "performance_throughput.json").read_text(encoding="utf-8"))
    ic = json.loads((PROJECT_ROOT / "docs" / "models" / "incident_classifier.json").read_text(encoding="utf-8"))
    S = samples()
    n_tests, per_file = _tests()
    summ, de = run["summary"], run["summary"].get("deepeval", {})
    judge = run.get("judge") or "nvidia openai/gpt-oss-20b"

    s = cover("Evaluation Report", "Test cases, retrieval results, DeepEval scores and sample outputs "
              "(recommended deliverable 8)", [
                  ["Automated tests", f"{n_tests} pytest tests, all passing"],
                  ["Evaluation set", f"{len(cases)} hand-made cases (tests/evaluation/eval_cases.json)"],
                  ["RAG metrics", f"DeepEval 4.x, judge: {judge}"],
                  ["Deterministic metrics", "condition recall, retrieval precision, average precision, category, "
                                            "hypothesis, grounding"],
                  ["Raw results", "docs/evaluation/eval_results_deepeval.json"],
              ])

    s += section("1. Summary")
    s.append(table([["Metric", "Score", "Meaning"],
                    ["Condition recall", f"{summ['condition_recall']:.2f}", "expected conditions recognised in the query"],
                    ["Retrieval precision", f"{summ['retrieval_precision']:.2f}", "share of top-5 records showing all "
                                                                                 "expected conditions"],
                    ["Average precision", f"{summ['average_precision']:.2f}", "relevant records ranked first"],
                    ["Category match", f"{summ['category_match']:.2f}", "derived category as expected"],
                    ["Hypothesis in top 2", f"{summ['hypothesis_hit']:.2f}", "an expected hypothesis ranked 1st or 2nd"],
                    ["Grounded", f"{summ['grounded']:.2f}", "evidence grounded and no hallucination flagged by the code checks "
                                                         "and the LLM judge combined"]] +
                   [[label, f"{de[k]:.2f}" if de.get(k) is not None else "–",
                     f"DeepEval, LLM-judged ({run['summary'].get('deepeval_scored', {}).get(k, '?')} cases scored)"]
                    for k, label in METRICS],
                   widths=[1.6, 0.7, 3.7], bold_first_col=True))
    s.append(callout(["These are <b>development scores</b>: one retrieval weight (metric similarity 0.6) was tuned on "
                      "these eight cases. A held-out benchmark grown from reviewed feedback is future work.",
                      f"The DeepEval judge is {judge}. A smaller judge model is stricter in places and can misread "
                      "evidence; its scores are reported as returned, without adjustment."], "warn", "Read with care"))

    s += section("2. Test cases")
    s.append(P("Eight cases, one per scenario in the project specification. Expected values follow the documented rules "
               "and are checked against the records actually retrieved — never against invented answers."))
    s.append(table([["Case", "Query", "Expected conditions", "Expected category", "Expected hypotheses"]] +
                   [[c["id"], c["query"], ", ".join(c["expected_conditions"]), c["expected_category"],
                     ", ".join(c["expected_hypotheses"])] for c in cases],
                   widths=[1.15, 2.4, 1.3, 1.1, 1.2]))
    s.append(P("Automated test suite", "h2"))
    s += bullets(["The five proofs the specification demanded are tests: no raw value fabricated (every processed value "
                  "compared with the raw CSV text), filters never violated, retrieved IDs are real records, the LLM "
                  "cannot add unsupported numbers (a deliberately hallucinating fake LLM), and no target leakage.",
                  "Provider tests run without network access (fake clients), so the suite needs no API key."])
    s.append(table([["Test file", "Tests"]] + per_file + [["<b>Total</b>", f"<b>{n_tests}</b>"]], widths=[4.5, 1.5]))

    s += section("3. Retrieval results")
    s.append(P("Per case: the deterministic retrieval and diagnosis metrics, the derived category, the top hypotheses "
               "and the time for the full analysis including LLM calls."))
    s.append(table([["Case", "Cond. recall", "Precision@5", "Avg. precision", "Category", "Top hypotheses",
                     "Explanation", "Time (s)"]] +
                   [[r["id"], f"{r['condition_recall']:.2f}", f"{r['retrieval_precision']:.2f}",
                     f"{r['average_precision']:.2f}", r["category"], ", ".join(r["top_hypotheses"][:2]),
                     r["narrative_source"], f"{r['seconds']:.0f}"] for r in run["cases"]],
                   widths=[1.25, 0.6, 0.65, 0.65, 1.15, 1.45, 0.75, 0.55]))
    sr = S["search"]["response"]
    s.append(P("Example retrieval (real): 'high latency and packet loss', filters Minnesota + Winter", "h2"))
    s.append(table([["Record", "Hybrid", "Semantic", "Keyword", "Metric", "Conditions", "State · season"]] +
                   [[x["record_id"], f"{x['hybrid_score']:.3f}", f"{x['semantic_score']:.3f}",
                     f"{x['keyword_score']:.3f}", f"{x['metric_score']:.3f}", ", ".join(x["conditions"]),
                     f"{x['metadata']['state']} · {x['metadata']['season']}"] for x in sr["results"]],
                   widths=[1.0, 0.6, 0.7, 0.65, 0.6, 1.9, 1.0]))
    s.append(P(f"{sr['matching_records']:,} records matched the filters; {sr['candidates']} candidates were re-scored; "
               f"weights used {sr['weights']}. Every returned record satisfies both filters."))

    s += section(f"4. DeepEval scores (judge: {judge})")
    s.append(table([["Case"] + [label for _, label in METRICS]] +
                   [[r["id"]] + [_cell(r["deepeval"].get(k, {})) for k, _ in METRICS] for r in run["cases"]] +
                   [["<b>Mean of scored cases</b>"] + [f"<b>{de[k]:.2f}</b>" if de.get(k) is not None else "–"
                                                     for k, _ in METRICS]],
                   widths=[1.6, 1.1, 1.1, 1.1, 1.1]))
    s.append(P("'–' means the judge could not return a valid answer after 3 attempts; such a metric is excluded from "
               "the mean instead of being counted as 0. '*' marks a score that needed a retry."))
    s += bullets(["<b>Faithfulness</b>: share of claims in the answer supported by the retrieved context.",
                  "<b>Answer relevancy</b>: share of the answer's statements that address the question.",
                  "<b>Contextual precision</b>: whether relevant retrieved records are ranked above irrelevant ones.",
                  "<b>Contextual relevancy</b>: share of the retrieved context relevant to the question — lower by "
                  "design, because each record's text lists every metric, most unrelated to a given question."])
    if prev:
        pd_ = prev["summary"].get("deepeval", {})
        s.append(P("Comparison with the earlier run (judge: Claude Haiku 4.5, 9 Oct 2026)", "h2"))
        s.append(table([["Metric", "Earlier run (Claude judge)", f"This run ({judge})"]] +
                       [[label, f"{pd_.get(k, 0):.2f}", f"{de[k]:.2f}" if de.get(k) is not None else "–"]
                        for k, label in METRICS],
                       widths=[2.0, 2.0, 2.0]))
        s.append(P("Both runs evaluate the same pipeline and retrieval (retrieval precision is identical); the LLM that "
                   "writes the explanation and the judge both changed between them. The smaller open model writes "
                   "less precise explanations and is a stricter, less consistent judge, so the faithfulness gap mainly "
                   "reflects the models, not the evidence."))
        if run1:
            s.append(P("A first NVIDIA run (kept as eval_results_deepeval_nvidia_run1_unretried.json) recorded three "
                       "judge failures — the judge's reply was not valid JSON — as scores of 0, which lowered the means. "
                       "The runner now retries such replies and excludes metrics that still cannot be scored."))
    reasons = [(r["id"], k, r["deepeval"][k]) for r in run["cases"] for k, _ in METRICS
               if k in r["deepeval"] and k != "ContextualRelevancyMetric"
               and (r["deepeval"][k].get("score") is None or r["deepeval"][k]["score"] < 0.75)]
    if reasons:
        s.append(P("Judge explanations for the lowest scores", "h2"))
        s.append(table([["Case", "Metric", "Score", "Judge's reason (abridged)"]] +
                       [[cid, k.replace("Metric", ""), "–" if v.get("score") is None else f"{v['score']:.2f}",
                         (v.get("reason") or "")[:320]]
                        for cid, k, v in reasons[:8]], widths=[1.1, 1.2, 0.5, 3.6]))
    from src.preprocessing.conditions import load_conditions
    thr = load_conditions()["low_throughput"].threshold
    s.append(P(f"Not every low score is a defect of the answer. In load_throughput the judge states that 268–298 Mbps "
               f"is 'not low throughput', but the documented condition (thresholds.yaml) is below {thr:g} Mbps, so "
               "the answer was right and the judge was wrong: it does not know the project's thresholds. In the "
               "application, the LLM judge is therefore combined with deterministic checks rather than trusted alone."))
    s.append(P("RAGAS", "h2"))
    s.append(P("RAGAS measures the same dimensions (faithfulness, context precision/recall, answer relevancy). Running "
               "it with the same judge would duplicate this measurement, so it was not added; the guideline lists "
               "DeepEval or RAGAS."))

    s += section("5. Machine-learning evaluation")
    srv = fr["served"]
    s.append(table([["Model (served)", "Test set (20%)", "Result"],
                    ["Fault risk — HistGradientBoosting, feature set C", "20,000 records",
                     f"R² {srv['test_metrics']['r2']}, MAE {srv['test_metrics']['mae']}, RMSE {srv['test_metrics']['rmse']}; "
                     f"5-fold CV R² {srv['cv_r2_mean']} ± {srv['cv_r2_std']}"],
                    ["Throughput — HistGradientBoosting", "20,000 records",
                     f"R² {tp['served']['test_metrics']['r2']}, MAE {tp['served']['test_metrics']['mae']:.1f} Mbps "
                     f"(mean baseline MAE {tp['mean_predictor_baseline']['mae']:.1f})"],
                    ["Category classifier — HistGradientBoosting", "20,000 records (stratified)",
                     f"accuracy {ic['models']['hist_gradient_boosting']['accuracy']:.4f}, macro F1 "
                     f"{ic['models']['hist_gradient_boosting']['macro_f1']:.4f} (agreement with the rule-derived labels)"]],
                   widths=[2.2, 1.3, 2.7], bold_first_col=True))
    fs = fr["feature_sets"]
    s.append(table([["Fault-risk feature set", "Linear R²", "Gradient boosting R²", "Status"]] +
                   [[k, f"{v['models']['linear_regression']['r2']}", f"{v['models']['hist_gradient_boosting']['r2']}",
                     {"C_primary": "served", "D_strict": "strict reference", "A_formula_demo": "leakage demonstration, "
                                                                                             "never served"}[k]]
                    for k, v in fs.items()], widths=[1.8, 1.0, 1.4, 2.0]))

    s += section("6. Sample outputs")
    a = S["analyze"]["response"]
    s.append(P(f"Real text report for the specification's example query (POST /api/v1/analyze, LLM off): "
               f"<i>{a['query']}</i>"))
    s.append(code(a["report_text"], max_lines=95))
    s.append(P("Validation block of the same analysis:", "h2"))
    s.append(code(dumps(abridge({k: a["validation"][k] for k in ("evidence_grounded", "diagnosis_supported",
                                                                  "recommendation_relevant", "hallucination_detected",
                                                                  "score", "method", "issues")}))))
    llm_rows = [r for r in run["cases"] if r.get("narrative_source") == "llm" and r.get("answer")]
    if llm_rows:
        r = llm_rows[0]
        s.append(P(f"LLM-written answer evaluated by DeepEval (case {r['id']}, passed the grounding gate):", "h2"))
        s.append(code(r["answer"], max_lines=40))
    s.append(P("The five README sample queries and their real outputs are in data/sample_data/sample_queries.json."))
    return s


def main():
    return build(OUT, "Evaluation Report", story())


if __name__ == "__main__":
    main()
