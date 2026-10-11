"""Primer content: Part D (machine learning) and Part E (LLM, RAG and agents)."""

from __future__ import annotations

import json

from docs.guide.pdf_kit import P, PageBreak, callout, code, table
from docs.tech_primer.kit import fig, mono, topic
from src.utils.config import PROJECT_ROOT


def _reports() -> dict:
    d = PROJECT_ROOT / "docs" / "models"
    return {n: json.loads((d / f"{n}.json").read_text(encoding="utf-8"))
            for n in ("fault_risk", "performance_throughput", "incident_classifier")}


# ================================================================================================ PART D
def part_d(F: dict) -> list:
    R = _reports()
    fr = R["fault_risk"]
    served = fr["served"]
    tp = R["performance_throughput"]
    ic = R["incident_classifier"]["models"]
    bs = F["boosting_scores"]

    s = [P("Part D — Machine learning", "part"),
         P("How the project trains models that estimate fault risk and throughput and cross-check the category — "
           "and how it proves those estimates are honest.")]

    s += topic(
        "12. Supervised learning, train/test split and cross-validation",
        what="<b>Supervised learning</b> means learning from examples where the answer is known. A model is "
             "trained on most of the data and graded on a part it has never seen (the <b>test set</b>), because "
             "only performance on unseen data shows whether it really learned.",
        analogy="A student who memorises last year's exam answers scores 100% on those questions and fails the real "
                "exam. Grading on unseen questions is the only fair test.",
        how=[
            "<b>Regression</b> predicts a number (fault rate 44.08%). <b>Classification</b> predicts a category "
            "(Mobility / Handover).",
            "<b>Train/test split</b>: shuffle the records, keep 80% for training and hide 20% for the final exam. "
            "A fixed <b>random seed</b> (random_state = 42) makes the shuffle identical every run, so results are "
            "reproducible.",
            "<b>Cross-validation (k-fold)</b>: split the training part into 5 folds; train 5 times, each time scoring "
            "on a different fold. If the 5 scores agree, the result is stable rather than lucky.",
            "<b>Overfitting</b> is when a model fits the training data's noise and does worse on new data; the test "
            "set and cross-validation detect it.",
        ],
        after_how=fig(F["split"], "Figure 12.1 — The split and 5-fold cross-validation used in your project "
                                  "(illustration).", 0.92, 60),
        project=[
            "Settings in " + mono("src/config/models.yaml") + ": test_size 0.2, random_state 42 → 80,000 training "
            "and 20,000 test records. Training code: " + mono("python -m src.services.models.train") + ".",
            f"The served fault-risk model's 5-fold cross-validation R² is {served['cv_r2_mean']} ± "
            f"{served['cv_r2_std']} — almost no variation between folds, so the test score is not luck.",
            "The classifier's split is <b>stratified</b>: each category keeps the same share in training and test.",
        ],
        say="\"Every score I report is on 20,000 records the model never saw, and 5-fold cross-validation gives "
            "the same R² within ±0.0006.\"",
    )

    s += topic(
        "13. Linear regression, logistic regression and feature encoding",
        what="<b>Linear regression</b> predicts a number as a weighted sum of the inputs; <b>logistic "
             "regression</b> does the same for categories by turning weighted sums into probabilities. Both are "
             "simple <b>baselines</b> that more powerful models must beat.",
        analogy="A taxi fare: a fixed start fee plus a price per kilometre plus a price per minute. Linear "
                "regression learns those prices from past trips.",
        how=[
            "Linear: prediction = b₀ + b₁ × latency + b₂ × packet loss + … The training finds the weights b that "
            "minimise the squared errors.",
            "Logistic: one weighted sum per category, converted into probabilities that add up to 1 (softmax).",
            "<b>One-hot encoding</b> turns a text category into numbers: weather = Rain becomes four columns "
            "weather_Clear 0, weather_Clouds 0, weather_Rain 1, weather_Snow 0.",
            "<b>Standard scaling</b> (" + mono("StandardScaler") + ") rescales each feature to mean 0 and standard "
            "deviation 1, so a feature measured in thousands (users) does not dominate one measured in single "
            "digits (packet loss). Logistic regression needs it; tree models do not.",
        ],
        project=[
            ["Both are trained as baselines in " + mono("src/services/models/fault_risk.py") + ", "
             + mono("performance.py") + " and " + mono("incident_classifier.py") + ".",
             f"A real lesson: for fault risk, plain linear regression scores R² "
             f"{fr['feature_sets']['C_primary']['models']['linear_regression']['r2']} — the same as the boosted "
             "trees — because the target is almost a linear formula of other columns (Chapter 17). Simple models "
             "reveal the structure of the data.",
             f"For categories the rules are non-linear (thresholds, highest score wins), so logistic regression "
             f"reaches only {ic['logistic_regression']['accuracy']:.1%} agreement while tree models reach "
             f"{ic['hist_gradient_boosting']['accuracy']:.1%}.",
             "Unknown or missing weather becomes 'missing' in every one-hot column rather than a fake category."],
        ],
        say="\"I always train a linear baseline. For fault risk it matched the complex model, which told me the "
            "target is nearly a formula — that finding drove the leakage analysis.\"",
    )

    s += topic(
        "14. Decision trees and random forests",
        what="A <b>decision tree</b> predicts by asking a sequence of yes/no questions about the inputs; a "
             "<b>random forest</b> averages hundreds of slightly different trees to get a steadier answer.",
        analogy="The game '20 questions': each question splits the possibilities in two until you are confident. A "
                "random forest is asking 200 friends to play and taking the average guess.",
        how=[
            "Training picks, at each step, the question (feature and cut-off) that best separates high from low "
            "targets. Each final box (leaf) predicts the average target of the training records that reach it.",
            "A deep tree can memorise noise (overfitting). A random forest trains each tree on a random sample of "
            "rows and features and averages them, which cancels much of the noise.",
            "Trees handle non-linear rules and interactions naturally and need no scaling.",
        ],
        after_how=fig(F["tree"], "Figure 14.1 — Real: a 2-level tree fitted to your training data to predict "
                                 "fault_rate_pct. It discovers on its own that throughput separates high- from "
                                 "low-fault records (served models are far larger).", 1.0, 80),
        project=[
            f"Random forests are trained for comparison: fault risk R² "
            f"{fr['feature_sets']['C_primary']['models']['random_forest']['r2']}, throughput R² "
            f"{tp['models']['random_forest']['r2']}, classifier "
            f"{ic['random_forest']['accuracy']:.1%} agreement (200 trees, " + mono("random_state = 42") + ").",
            "They were not served because gradient boosting (next chapter) scored as well or slightly better and "
            "handles missing inputs natively.",
        ],
        say="\"Trees learn threshold-style rules from data; a forest averages many trees to reduce overfitting. I "
            "used forests as a comparison model.\"",
    )

    s += topic(
        "15. Gradient boosting (HistGradientBoosting)",
        what="<b>Gradient boosting</b> builds trees one after another, each new small tree correcting the "
             "remaining errors of all previous ones. <b>HistGradientBoosting</b> is scikit-learn's fast version, "
             "and it is the model your project serves for all three tasks.",
        analogy="Learning darts with a coach: after every throw the coach says 'a bit higher and left'. Each small "
                "correction is weak, but hundreds of them add up to accuracy.",
        how=[
            "Start with a simple guess (the average). Compute each record's error. Fit a small tree that predicts "
            "those errors. Add a fraction of it (the <b>learning rate</b>, 0.05 in your fault model) to the "
            "prediction. Repeat (" + mono("max_iter") + " = 300 trees).",
            "<b>'Hist'</b>: numeric features are first grouped into at most 255 bins (a histogram), which makes "
            "finding the best split much faster on 80,000 rows.",
            "<b>Missing values</b> are handled natively: the model learns which branch missing values should go "
            "down. This matters because engineers rarely type every KPI.",
        ],
        after_how=fig(F["boosting"], "Figure 15.1 — Real: test-set R² of the fault-risk model as trees are added "
                                     "(trained on your 80,000 training records for this primer).", 0.9, 70),
        project=[
            f"The curve shows boosting at work: 1 tree explains R² {bs[1]}, 10 trees {bs[10]}, 50 trees {bs[50]}, "
            f"and from 100 trees on it stays at {bs[300]} — more trees no longer help, and the score does not drop "
            "either (no overfitting).",
            ["Fault risk: " + mono("HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05)") + ".",
             "Throughput: the same regressor; category: " + mono("HistGradientBoostingClassifier(max_iter=300, "
                                                                   "learning_rate=0.1)") + ".",
             "Models are saved with joblib in " + mono("data/processed/models/*.joblib") + " and reports in "
             + mono("docs/models/*.json") + "."],
        ],
        say="\"Gradient boosting adds small trees that each fix the previous errors. I serve the histogram version "
            "because it is accurate, fast and handles missing inputs, which partial user queries always have.\"",
    )

    s += topic(
        "16. Measuring models: R², MAE, RMSE, accuracy, precision, recall, F1",
        what="Metrics turn 'is the model good?' into numbers. Regression uses error sizes and R²; classification "
             "uses the share of correct answers and how errors are distributed between categories.",
        analogy="Weather forecasts: MAE is 'on average I miss the temperature by 2 °C'; R² is 'how much better I am "
                "than always forecasting the seasonal average'.",
        how=[
            "<b>MAE</b> (mean absolute error): average size of the miss, in the target's unit (percentage points).",
            "<b>RMSE</b> (root mean squared error): like MAE but punishes big misses more.",
            "<b>R²</b> (coefficient of determination): 1 = perfect, 0 = no better than always predicting the "
            "average, below 0 = worse. It is the share of the target's variation the model explains.",
            "<b>Accuracy</b>: share of correct categories. <b>Precision</b> for a category: of the records predicted "
            "as it, how many really are. <b>Recall</b>: of the records that really are it, how many were found. "
            "<b>F1</b>: the balance (harmonic mean) of the two. <b>Macro</b> averages treat every category equally, "
            "so small categories count as much as big ones.",
            "A <b>confusion matrix</b> counts every (true, predicted) pair; the diagonal holds the correct answers.",
        ],
        project=[
            table([
                ["Model (served)", "Metric on 20,000 unseen records", "Meaning"],
                ["Fault risk", f"R² {served['test_metrics']['r2']}, MAE {served['test_metrics']['mae']} points, "
                               f"RMSE {served['test_metrics']['rmse']}",
                 "explains 88% of the variation; typical miss about 3.7 percentage points"],
                ["Throughput", f"R² {tp['served']['test_metrics']['r2']}, MAE "
                               f"{tp['served']['test_metrics']['mae']:.0f} Mbps (baseline MAE "
                               f"{tp['mean_predictor_baseline']['mae']:.0f})",
                 "moderate: weather drives throughput, so treat it as indicative"],
                ["Category classifier", f"accuracy {ic['hist_gradient_boosting']['accuracy']:.1%}, macro F1 "
                                        f"{ic['hist_gradient_boosting']['macro_f1']:.3f}",
                 "agreement with the rule-derived categories, not real-world accuracy"],
            ], widths=[1.3, 2.6, 2.6], bold_first_col=True),
        ],
        after_project=[
            fig(F["confusion"], "Figure 16.1 — Real: confusion matrix of the served classifier (rows = rule category, "
                                "columns = prediction). Most errors involve Service Degradation, the broadest "
                                "category.", 0.72, 95),
            callout("The classifier learns categories that your own rules produced, so 97% means 'the rules can be "
                    "learned from raw metrics', not 'it diagnoses real incidents correctly'. Always say "
                    "'agreement'.", "warn"),
        ],
        say="\"The fault model explains 88% of the variation with a typical error of 3.7 points. The classifier's "
            "97% is agreement with my rules, which I report as a consistency check, not as accuracy.\"",
    )

    fs = fr["feature_sets"]
    s += topic(
        "17. Data leakage and feature sets",
        what="<b>Data leakage</b> is when a model is given information that contains the answer or would not exist "
             "at prediction time. It produces impressive scores that collapse in real use.",
        analogy="A student who has the answer key hidden under the exam paper: a perfect score, zero learning.",
        how=[
            "Leakage hides in two ways: features computed from the target (directly or as part of its formula), and "
            "features only known after the event (e.g. repair time after a fault).",
            "The defence is to define allowed <b>feature sets</b> on purpose and to enforce them in code, not by "
            "memory.",
        ],
        after_how=fig("docs/guide/figs/r2.png", "Figure 17.1 — Real: fault-risk R² for the three feature sets. "
                      "The 0.9998 bar is the leakage demonstration and is never served.", 0.85, 65),
        project=[
            table([
                ["Feature set", "What it contains", "Test R² (linear / boosting)", "Status"],
                ["A_formula_demo", "every column except the target", f"{fs['A_formula_demo']['models']['linear_regression']['r2']} / "
                                   f"{fs['A_formula_demo']['models']['hist_gradient_boosting']['r2']}",
                 "proof that the target is a formula — never served"],
                ["C_primary", "context + radio/transport KPIs + throughput",
                 f"{fs['C_primary']['models']['linear_regression']['r2']} / {fs['C_primary']['models']['hist_gradient_boosting']['r2']}",
                 "<b>served</b>"],
                ["D_strict", "C without throughput", f"{fs['D_strict']['models']['linear_regression']['r2']} / "
                                                     f"{fs['D_strict']['models']['hist_gradient_boosting']['r2']}",
                 "strictest reference"],
            ], widths=[1.3, 2.3, 1.7, 1.9], bold_first_col=True),
            ["Excluded from the served model (" + mono("models.yaml") + "): the seven columns that form the fault "
             "rate's formula (critical alarms, auth failures, security incidents, alarms, call drops, configuration "
             "success, parameter changes), MTTR (known only after a fault), every derived label and condition, "
             "locations and IDs.",
             mono("check_leakage()") + " in " + mono("src/services/models/common.py") + " raises an error if any of these is "
             "used, and a test proves it."],
        ],
        say="\"With every column the model scores 0.9998 because it rediscovers the target's formula — that is "
            "leakage. The served model uses only information available before a fault and scores 0.885 honestly, "
            "and a code guard makes leakage impossible to reintroduce.\"",
    )

    s += topic(
        "18. Prediction intervals and risk bands",
        what="A single predicted number hides uncertainty, so the project adds an <b>80% interval</b> around every "
             "fault-risk prediction and maps the prediction to a <b>risk level</b> (Low, Medium, High, Critical).",
        analogy="A weather app that says '18 °C, likely between 15 and 21 °C' is more useful than '18 °C' alone.",
        how=[
            "<b>Residuals</b> are the model's errors on the test set (actual − predicted). Their 10th and 90th "
            "percentiles are added to each new prediction, giving an interval that the actual value fell inside for "
            "80% of test records.",
            "<b>Risk bands</b> are cut-offs on the predicted rate, taken from the dataset's own distribution: below "
            "P25 is Low, below P75 Medium, below P90 High, otherwise Critical.",
            "<b>Input coverage</b> is the share of the model's inputs the user actually provided. With few inputs the "
            "model can only return its baseline expectation, and the UI says so.",
        ],
        after_how=fig(F["pred"], "Figure 18.1 — Real: predicted vs actual fault rate for the 20,000 test records. "
                                 "The dashed lines are the 80% interval.", 0.62, 85),
        project=[
            f"Residual percentiles: {served['residual_p10']:.2f} and +{served['residual_p90']:.2f} points. "
            f"Risk bands: Low below {served['risk_bands'][0]['below']}, Medium below "
            f"{served['risk_bands'][1]['below']}, High below {served['risk_bands'][2]['below']}, else Critical.",
            "Phase 30 example: 44.08% predicted, interval 38.17–50.01%, risk Medium; shown beside the 58.05% "
            "<b>observed</b> in the five similar records — two different things, never merged.",
        ],
        after_project=callout(f"The interval was calibrated on the same test residuals it is checked against, so "
                              f"its {F['inside_interval']:.0%} coverage there is by construction. On truly new data it "
                              "is an estimate, and it assumes errors are similar across the range.", "warn"),
        say="\"Every prediction comes with an 80% range from the model's own test errors and a risk band from the "
            "data's percentiles, and it is always labelled as a prediction, separate from observed history.\"",
    )
    s.append(PageBreak())
    return s


# ================================================================================================ PART E
def part_e(F: dict) -> list:
    s = [P("Part E — LLM, RAG and agents", "part"),
         P("How the LLM is used safely, how answers are grounded in evidence, and how the analysis is organised into "
           "agents.")]

    s += topic(
        "19. Large language models: gpt-oss-20b on NVIDIA",
        what="A <b>large language model (LLM)</b> is a neural network trained on enormous amounts of text to "
             "predict the next word; this makes it able to read instructions and write fluent text. Your project "
             "uses the open-weight model <b>gpt-oss-20b</b> (" + mono("openai/gpt-oss-20b") + ") through the NVIDIA API Catalog, an OpenAI-compatible endpoint. Claude (Anthropic) is supported as an alternative and was used during development.",
        analogy="An extremely well-read assistant who writes beautifully — and who will confidently fill gaps with "
                "plausible guesses unless you give them the documents and check their work.",
        how=[
            "Text is processed as <b>tokens</b> (pieces of words). Each request has a <b>system prompt</b> (standing "
            "rules) and a <b>user message</b> (the task and data).",
            "<b>Hallucination</b> is fluent text that is not supported by the input — invented numbers, IDs or "
            "causes. It is the central risk of using LLMs for engineering decisions.",
            "<b>Structured outputs</b>: instead of free text, the model must fill a fixed schema (fields and "
            "types). Your code passes a Pydantic schema to " + mono("client.messages.parse(…)") + " and receives a "
            "validated object, so there is no fragile text parsing.",
        ],
        project=[
            ["<b>Provider abstraction</b> (" + mono("src/rag/llm_provider.py") + "): every LLM provider offers one "
             "method, " + mono("structured(system, user, schema)") + ". The NVIDIA provider calls gpt-oss-20b (the Anthropic provider can call Claude instead); the "
             "Offline provider returns nothing and the code falls back to templates. The whole system therefore "
             "works without an API key.",
             "Configuration only from " + mono(".env") + ": ANTHROPIC_API_KEY, LLM_MODEL, LLM_PROVIDER — never "
             "written in code.",
             "Errors are reported, not hidden: authentication, rate limits, bad requests (such as the 'credit "
             "balance too low' message) and network failures each return a clear status that the UI shows.",
             "The narrative schema has four fields: summary (≤ 3 sentences), diagnosis, one note per hypothesis id, "
             "and limitations.",
             "The system prompt's rules include: use only facts in the evidence; copy numbers exactly and compute "
             "nothing new; refer only to listed record IDs; treat causes as hypotheses; name no protocol details "
             "not in the evidence; never promise outcomes."],
        ],
        say="\"The LLM never decides facts. It receives an evidence package and fills a fixed schema with an "
            "explanation; if it is unavailable the system uses a deterministic template and still works.\"",
    )

    s += topic(
        "20. Retrieval-augmented generation (RAG)",
        what="<b>RAG</b> means retrieving relevant documents first and giving them to the LLM, so its answer is "
             "based on that evidence rather than on its memory.",
        analogy="An open-book exam: the student must answer from the provided pages, and the grader can check every "
                "sentence against them.",
        how=[
            "Retrieve → assemble an evidence package → generate → check. Without retrieval an LLM can only answer "
            "from general knowledge; with retrieval it can discuss your specific records.",
            "Most RAG systems stop at 'generate'. Your project adds a check step and goes further: the classification, "
            "risk, hypotheses and actions are computed by code <b>before</b> the LLM is called, so the LLM explains "
            "conclusions instead of making them.",
        ],
        after_how=fig(F["rag"], "Figure 20.1 — Retrieval-augmented generation as used in your project "
                                "(illustration).", 1.0, 45),
        project=[
            "The evidence package (" + mono("evidence_package()") + " in " + mono("src/rag/narrative.py") + ") "
            "contains: the query; values and conditions the operator stated; the retrieved records; statistics of "
            "those records; the dataset baseline; classification and priority; predicted and historical fault risk; "
            "the ranked hypotheses; and the recommended actions.",
            "The very same package is used to check the LLM's answer, so 'grounded' has a precise meaning: "
            "supported by this package.",
        ],
        say="\"It is RAG with deterministic analysis in the middle: retrieval finds evidence, code draws the "
            "conclusions, and the LLM only explains them from that evidence.\"",
    )

    s += topic(
        "21. The grounding gate: catching hallucinations",
        what="A deterministic check in code reads every sentence the LLM writes and rejects it if it contains a "
             "number, record ID, protocol term or promise that the evidence does not support.",
        analogy="A fact-checker at a newspaper who has the source documents and strikes out every figure that cannot "
                "be found in them — before publication.",
        how=[
            "<b>Numbers</b>: each number in the text must match an evidence value <i>at the precision written</i> — "
            "'74.3' matches 74.336, but '99.9' does not match 99.5. Shares like 0.65 may be quoted as 65%. Small "
            "counting words ('two hypotheses') are allowed, but not with a unit ('4–5 points').",
            "<b>Record IDs</b> must be among the retrieved records.",
            "<b>Unsupported specifics</b>: terms such as X2, Xn, RACH, AMF, MME or A3 are rejected unless they "
            "appear in the evidence — the dataset has no such counters.",
            "<b>Guarantees</b>: 'will increase to 95%', 'within 24 hours', 'guaranteed', 'confirmed root cause' are "
            "rejected.",
            "If a check fails, the LLM gets <b>one retry</b> with the violations listed; if it fails again, the "
            "deterministic template replaces it.",
        ],
        project=[
            "Code: " + mono("src/rag/grounding.py") + " (" + mono("check_text") + "). Real test with a "
            "deliberately hallucinated paragraph:",
            code("Handover success will increase to 97% after the fix, which should take about 6 hours. This matches\n"
                 "incident ROW-123456, where 12,500 users were affected. The likely cause is an X2 link failure\n"
                 "combined with RACH congestion. Fault rate is 4.2 points above baseline."),
            "Caught: the invented record ID, the 12,500 users, X2 and RACH, and the promised increase. Not caught: "
            "'97%', '6 hours' and '4.2 points', because those numbers happen to equal unrelated evidence values. "
            "The check is value-based — that is why an LLM judge reviews meaning as a second layer.",
        ],
        say="\"Every number, ID and protocol term the LLM writes is checked against the evidence by code; failures "
            "get one retry and then a template. I also know its limit: a number that coincidentally matches "
            "unrelated evidence can pass.\"",
    )

    s += topic(
        "22. LLM-as-a-judge",
        what="A second LLM call reviews the finished analysis against the evidence and answers fixed questions — "
             "is the diagnosis supported, is it grounded, are the actions relevant, is anything hallucinated.",
        analogy="A second engineer reviewing a colleague's report before it is sent, with the same case file in "
                "hand.",
        how=[
            "Code checks are precise but literal; a judge understands meaning (e.g. 'this sentence draws a "
            "conclusion the evidence does not support').",
            "The judge also returns structured output: true/false verdicts, completeness, a score and a list of "
            "issues.",
            "The final verdict combines both: a check passes only if the deterministic check <b>and</b> the judge "
            "agree; hallucination is flagged if either finds it.",
        ],
        project=[
            ["Code: " + mono("src/rag/judge.py") + ". Deterministic checks always run; the judge runs when the "
             "LLM is enabled.",
             "The UI's 'Automatic review' shows four checks: evidence grounded, diagnosis supported, recommendations "
             "relevant, no hallucination detected — plus the method ('deterministic only' or 'deterministic + LLM "
             "judge').",
             "A real finding by the judge: 'normal user load' retrieved high-load records (Chapter 11)."],
        ],
        after_project=callout("On the Phase 30 demo query, 'Recommendations relevant' fails. The two weather actions "
                              "come from a lower-ranked hypothesis supported by the similar records, not by a condition "
                              "in the question, and the check is deliberately strict about that.", "warn"),
        say="\"Two layers review every answer: exact code checks and an LLM judge for meaning. A check only passes "
            "if both agree.\"",
    )

    s += topic(
        "23. Root-cause hypotheses: the playbook heuristic",
        what="Possible causes are listed in a <b>playbook</b> written as configuration; each is scored with a "
             "documented formula and reported as an engineering <b>hypothesis</b>, never a confirmed cause.",
        analogy="A doctor's differential diagnosis: 'most likely X, consider Y and Z', each with the tests that "
                "would confirm or rule it out.",
        how=[
            "A hypothesis is considered when one of its trigger conditions is present.",
            "Confidence = 0.20 + 0.45 × (share of its evidence weight that is present) + 0.25 × (share of similar "
            "records showing its key evidence) − 0.25 per contradicting item, clipped to 0.05–0.95.",
            "Each hypothesis carries the checks to run, the actions to consider, and the additional evidence (not in "
            "this dataset) that would confirm or refute it.",
            "This is a <b>heuristic</b>: a transparent rule of thumb, not a calibrated probability.",
        ],
        after_how=fig("docs/guide/figs/hypotheses.png", "Figure 23.1 — Real: how each hypothesis score was built for "
                      "the Phase 30 query.", 0.95, 72),
        project=[
            "Phase 30 example: mobility configuration issue = 0.20 + 0.45 × 0.45 + 0.25 × 1.0 = 0.65 — low handover "
            "success was stated, and all 5 similar records show it.",
            "Recommendations are built from the hypotheses, each linked to the conditions it is based on, plus "
            "monitoring actions compared with the dataset baseline. Disclaimer always attached: options to "
            "evaluate, no KPI improvement predicted.",
            "Config: " + mono("src/config/playbook.yaml") + "; code: " + mono("src/services/root_cause.py") + ", "
            + mono("recommendations.py") + ".",
        ],
        say="\"0.65 is a documented heuristic: base 0.20, plus evidence match, plus support from similar records, "
            "minus contradictions. It ranks hypotheses; it is not a probability.\"",
    )

    s += topic(
        "24. Agents and orchestration",
        what="An <b>agent</b> here is a named step with one responsibility that reads a shared state, adds its "
             "result and records what it did. An <b>orchestrator</b> runs the six agents in a fixed order.",
        analogy="An assembly line with stations: each station does one job on the same product and stamps a "
                "logbook.",
        how=[
            "In the AI world 'agent' often means an LLM that decides its own next steps in a loop. Your project "
            "deliberately does <b>not</b> do that: the order is fixed, nothing loops, and most agents use no LLM. "
            "That makes every run predictable and testable.",
            "<b>Shared state</b>: a typed object (" + mono("AnalysisState") + ") holding the request, the growing "
            "result and the trace. Each agent declares which inputs it requires and fails loudly if they are "
            "missing.",
            "<b>Trace</b>: each agent records its status, duration and a one-line summary — shown in the UI.",
        ],
        after_how=fig("docs/guide/figs/agents.png", "Figure 24.1 — Real: the six agents and their timings for the "
                      "Phase 30 query (offline mode; the LLM adds about 10–40 s).", 1.0, 60),
        project=[
            table([
                ["Agent", "Job"],
                ["Retrieval", "query understanding, filters, hybrid search, evidence extraction"],
                ["Classification", "conditions → category, severity (and the ML cross-check)"],
                ["Risk", "fault-risk prediction, priority"],
                ["Diagnosis", "playbook hypotheses with confidence"],
                ["Recommendation", "actions, checks, monitoring; LLM narrative + grounding gate"],
                ["Validation", "deterministic checks + LLM judge, provenance, text report"],
            ], widths=[1.3, 4.7], bold_first_col=True),
            "Guardrails run before the first agent. A test proves the agents give exactly the same result as the "
            "single-pipeline version in " + mono("src/rag/pipeline.py") + ". LangGraph was considered; for six linear "
            "steps a plain orchestrator is simpler to explain and test.",
        ],
        say="\"Six fixed agents on a typed shared state, each timed and traced. No autonomous loops — and a test "
            "proves they produce exactly the single pipeline's result.\"",
    )

    s += topic(
        "25. Guardrails and prompt injection",
        what="<b>Guardrails</b> are checks that run on the user's input before anything else; they reject "
             "attempts to misuse the system, such as <b>prompt injection</b> — text that tries to override the "
             "assistant's instructions.",
        analogy="Security at a building entrance: before you reach any office, the guard checks you are here for a "
                "legitimate reason and not carrying anything dangerous.",
        how=[
            "Your guardrails are <b>regular expressions</b> — precise text patterns — so every decision is "
            "explainable and testable, and no LLM is involved.",
            "Input is first <b>sanitised</b>: Unicode normalised, invisible control characters removed, whitespace "
            "collapsed (attackers hide instructions in invisible characters).",
        ],
        project=[
            table([
                ["Category", "Example that is rejected"],
                ["empty / too_long", "nothing to analyse, or more than 1,000 characters"],
                ["prompt_injection", "'Ignore your instructions…', 'you are now…', 'act as…', fake &lt;system&gt; tags"],
                ["secret_request", "API keys, passwords, tokens, .env, 'reveal your prompt'"],
                ["unsupported", "delete data, restart a cell or site, run commands, push to the live network"],
                ["off_topic", "no telecom-network vocabulary at all"],
            ], widths=[1.4, 4.6], bold_first_col=True),
            "Code: " + mono("src/services/guardrails.py") + ". The API answers a rejected query with HTTP 422 and a code such "
            "as " + mono("guardrail_prompt_injection") + "; the web UI shows the reason and how to rephrase.",
        ],
        say="\"Inputs are sanitised and checked with explainable patterns before any retrieval or LLM call; it is a "
            "decision-support tool, so requests to act on the network are refused.\"",
    )

    s += topic(
        "26. Provenance: where every statement comes from",
        what="<b>Provenance</b> means recording the source of each statement in an answer, so a reader can tell a "
             "measured fact from a prediction or a hypothesis.",
        analogy="Footnotes in a research paper: every claim points to where it came from.",
        how=["Each statement in the final analysis is stored with one of seven source types and the evidence "
             "behind it (condition, record IDs, values)."],
        project=[
            table([
                ["Source type", "Meaning", "Example"],
                ["USER_INPUT", "stated by the operator", "low handover success reported in the query"],
                ["RAW_DATA", "a measured dataset value", "ROW-098813: latency 120.16 ms"],
                ["DERIVED_FEATURE", "computed by documented rules", "category Mobility / Handover, severity MEDIUM"],
                ["HISTORICAL_EVIDENCE", "a pattern across similar records", "low handover in 5 of 5 similar records"],
                ["ML_PREDICTION", "a model estimate", "predicted fault risk Medium (44.08%)"],
                ["ENGINEERING_HYPOTHESIS", "a possible cause", "mobility configuration issue (0.65)"],
                ["RECOMMENDATION", "an action to consider", "audit neighbour relationships"],
            ], widths=[1.8, 2, 2.6], bold_first_col=True),
            "The web UI's Sources section lists every statement with its source and lets you filter by type.",
        ],
        say="\"Every statement is tagged with one of seven source types, so a measured value can never be confused "
            "with a prediction or a hypothesis.\"",
    )
    s.append(PageBreak())
    return s
