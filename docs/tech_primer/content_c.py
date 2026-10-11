"""Primer content: Part F (quality), Part G (serving and interface), cheat sheet and glossary."""

from __future__ import annotations

import json

from docs.guide.pdf_kit import P, PageBreak, callout, code, table
from docs.tech_primer.kit import fig, mono, topic
from src.utils.config import PROJECT_ROOT


def _deepeval() -> dict:
    p = PROJECT_ROOT / "docs" / "evaluation" / "eval_results_deepeval.json"
    summary = json.loads(p.read_text(encoding="utf-8"))["summary"]
    summary["deepeval"] = {k: (v if v is not None else float("nan")) for k, v in summary["deepeval"].items()}
    return summary


# ================================================================================================ PART F
def part_f() -> list:
    ev = _deepeval()
    de = ev["deepeval"]
    s = [P("Part F — Quality: testing and evaluation", "part"),
         P("How the project proves that it works and that it does not invent things.")]

    s += topic(
        "27. Automated testing with pytest",
        what="An <b>automated test</b> is a small program that runs part of the system and checks the result; "
             "<b>pytest</b> is the standard Python tool that finds and runs them all with one command.",
        analogy="A pre-flight checklist: every item is checked the same way before every flight, so nothing "
                "depends on someone remembering.",
        how=[
            "<b>Unit tests</b> check one function (e.g. the BM25 formula). <b>Integration tests</b> run several parts "
            "together (e.g. a full API request). <b>Regression tests</b> pin a fixed bug so it can never return.",
            "A test <b>asserts</b> something that must be true; if it is false, the test fails and names the "
            "problem.",
            "<b>Parametrised</b> tests run the same check on many inputs, which is why 130 test functions produce 222 "
            "test runs.",
        ],
        project=[
            "Run with " + mono("python -m pytest") + ": 222 tests, one file per phase group in " + mono("tests/")
            + ", about 2–3 minutes. All pass.",
            table([
                ["The five proofs the specification demanded", "How a test proves it"],
                ["No raw value fabricated", "every processed value is compared with the raw CSV text"],
                ["Filters never violated", "seven filter combinations; every returned record is checked"],
                ["Retrieved IDs are real records", "every ID returned by search exists in the dataset"],
                ["LLM cannot add unsupported numbers", "a deliberately lying fake LLM is rejected by the grounding gate"],
                ["No target leakage", "the leakage guard rejects forbidden features; the served model is checked"],
            ], widths=[2.4, 3.6], bold_first_col=True),
            "The React UI has its own 6 unit tests (Vitest) for formatting, filters and input validation: "
            + mono("npm test") + " in " + mono("web/") + ".",
        ],
        say="\"222 automated tests, including the five proofs the specification asked for — for example, every "
            "processed value is compared with the original file.\"",
    )

    s += topic(
        "28. Evaluating retrieval and answers",
        what="An <b>evaluation set</b> is a fixed list of questions with the expected outcomes; running the system "
             "on it and scoring the results shows how well the whole pipeline works, not just single functions.",
        analogy="A driving test route: the same streets and situations for every candidate, scored the same way.",
        how=[
            "<b>Precision@k</b>: of the top k results, how many are relevant. Your retrieval precision counts a "
            "record as relevant if it shows <i>all</i> the expected conditions.",
            "<b>Average precision</b>: rewards putting relevant results at the top, not just somewhere in the list "
            "(also called contextual precision).",
            "<b>Recall</b>: of everything that should be found, how much was found. <b>Condition recall</b>: did the "
            "query parser recognise every expected condition?",
            "<b>Development set</b>: a set used while building. Tuning on it makes scores optimistic; a separate "
            "held-out set would be needed for an unbiased number.",
        ],
        project=[
            "8 cases in " + mono("tests/evaluation/eval_cases.json") + ", one per scenario in the specification "
            "(e.g. 'high latency and high packet loss' → expected category Service Degradation, expected hypothesis "
            "transport connectivity). Run: " + mono("python -m tests.evaluation.run_eval") + ".",
            table([
                ["Deterministic metric", "Result", "Meaning"],
                ["condition_recall", f"{ev['condition_recall']:.2f}", "every expected condition recognised"],
                ["retrieval_precision", f"{ev['retrieval_precision']:.2f}", "every retrieved record shows all expected conditions"],
                ["average_precision", f"{ev['average_precision']:.2f}", "relevant records ranked first"],
                ["category_match", f"{ev['category_match']:.2f}", "derived category as expected"],
                ["hypothesis_hit", f"{ev['hypothesis_hit']:.2f}", "an expected hypothesis in the top 2"],
                ["grounded", f"{ev['grounded']:.2f}", "explanation passed the grounding gate, no hallucination"],
            ], widths=[1.8, 0.8, 3.4], bold_first_col=True),
        ],
        after_project=callout("These are development scores: one retrieval weight was tuned on these same 8 cases. "
                              "Say so — it shows you understand evaluation.", "warn"),
        say="\"Eight specification scenarios score 1.00 on every deterministic metric. They are development scores, "
            "because I tuned one weight on them; a held-out set grown from feedback is future work.\"",
    )

    s += topic(
        "29. DeepEval: LLM-judged quality metrics",
        what="<b>DeepEval</b> is an open-source library of evaluation metrics for LLM applications; an LLM acts as "
             "an independent judge and scores each answer against its question and retrieved context.",
        analogy="An external examiner who reads both the student's essay and the sources it was meant to use.",
        how=[
            "<b>Faithfulness</b>: share of the answer's claims that are supported by the retrieved context.",
            "<b>Answer relevancy</b>: share of the answer's statements that address the question.",
            "<b>Contextual precision</b>: are the relevant retrieved items ranked above the irrelevant ones?",
            "<b>Contextual relevancy</b>: share of the retrieved context that is relevant to the question.",
        ],
        project=[
            "Run with " + mono("python -m tests.evaluation.run_eval --deepeval") + " (uses API calls), with the configured LLM "
            "as the judge.",
            table([
                ["DeepEval metric", "Score", "Reading"],
                ["Faithfulness", f"{de['FaithfulnessMetric']:.2f}", "share of the answer's claims supported by the retrieved evidence"],
                ["Answer relevancy", f"{de['AnswerRelevancyMetric']:.2f}", "answers stay on the question"],
                ["Contextual precision", f"{de['ContextualPrecisionMetric']:.2f}", "relevant records come first"],
                ["Contextual relevancy", f"{de['ContextualRelevancyMetric']:.2f}",
                 "lower by design: each record's text lists every metric, much of it unrelated to a given question"],
            ], widths=[1.6, 0.7, 3.7], bold_first_col=True),
        ],
        say=f"\"With the LLM as an independent judge, faithfulness is {de['FaithfulnessMetric']:.2f} and contextual "
            f"precision {de['ContextualPrecisionMetric']:.2f}. Contextual relevancy is lower "
            f"({de['ContextualRelevancyMetric']:.2f}) because each record description includes all its metrics, not "
            "only the relevant ones.\"",
    )
    s.append(PageBreak())
    return s


# ================================================================================================ PART G
def part_g() -> list:
    s = [P("Part G — Serving and interface", "part"),
         P("How the analysis reaches the engineer: an API, a web interface, and the supporting tools.")]

    s += topic(
        "30. APIs: REST, FastAPI and Pydantic",
        what="An <b>API</b> (application programming interface) lets one program ask another for something over "
             "the network. Your backend is a <b>REST</b> API built with <b>FastAPI</b>; <b>Pydantic</b> checks that "
             "every request and response has the right shape.",
        analogy="A restaurant: the menu lists what you can order (endpoints), the waiter takes a written order in a "
                "fixed format (JSON request) and brings back a dish (JSON response) — or tells you clearly what "
                "cannot be served (an error code).",
        how=[
            "<b>HTTP</b> is the protocol of the web. A request has a method (GET to read, POST to send data), a path "
            "(" + mono("/api/v1/analyze") + ") and often a <b>JSON</b> body — structured text of keys and values.",
            "<b>Status codes</b>: 200 OK, 201 created, 404 not found, 422 the request was understood but is invalid, "
            "500 server error.",
            "FastAPI turns Python functions into endpoints and generates interactive documentation at "
            + mono("/docs") + " automatically.",
            "Pydantic models declare fields, types and limits (e.g. packet loss between 0 and 100); invalid requests "
            "are rejected with a precise message before any analysis runs.",
        ],
        project=[
            table([
                ["Endpoint", "Purpose"],
                ["POST /api/v1/analyze", "the full multi-agent analysis (main endpoint)"],
                ["POST /api/v1/search", "hybrid retrieval with filters"],
                ["POST /api/v1/classify  ·  /risk  ·  /recommend", "parts of the analysis on their own"],
                ["GET /api/v1/records/{id}", "one full record, raw and derived, labelled"],
                ["GET /api/v1/reference", "labels, units, ranges and thresholds for the web UI"],
                ["POST /api/v1/feedback", "store a useful / not-useful rating"],
                ["GET /health", "status of data, indexes, models and the LLM provider"],
            ], widths=[2.4, 3.6], bold_first_col=True),
            "Code: " + mono("src/main.py") + " and " + mono("src/api/schemas.py") + ". Every error has the same "
            "structure (code, message, details, request ID), and every response carries an X-Request-ID header for "
            "tracing.",
        ],
        say="\"FastAPI exposes the analysis as REST endpoints; Pydantic validates every request, so a value like "
            "packet loss 150% is rejected with a clear 422 error before any work is done.\"",
    )

    s += topic(
        "31. The web interface: React, TypeScript and Vite (and Streamlit)",
        what="<b>React</b> is a JavaScript library for building interactive web pages from reusable components; "
             "<b>TypeScript</b> adds types to JavaScript so mistakes are caught before the page runs; <b>Vite</b> "
             "builds the code into small files a browser can load.",
        analogy="LEGO: React components are bricks (a badge, a table, a form field) that snap together into pages; "
                "TypeScript is the shape of each brick that prevents wrong pieces from fitting.",
        how=[
            "A <b>component</b> is a function that returns what to show for given data. When the data changes "
            "(<b>state</b>), React updates only the affected parts of the page.",
            "A <b>single-page application (SPA)</b> loads once and then switches pages in the browser; the "
            "<b>router</b> maps addresses like " + mono("/ui/record/ROW-000001") + " to pages.",
            "The browser calls the API with " + mono("fetch") + " and shows the JSON it gets back. The UI never "
            "computes analysis results itself.",
            "<b>Streamlit</b> is a Python library that turns a script into a web page with very little code; it is "
            "ideal for prototypes and is the interface your specification asked for.",
        ],
        project=[
            ["React app in " + mono("web/") + " with four pages: Analyze a case, Search records, Record "
             "details, Assess measurements. FastAPI serves the built files at " + mono("/ui") + ".",
             "Response types are declared in " + mono("web/src/lib/types.ts") + ", so the code cannot read a "
             "field the API does not return.",
             "Labels, units and thresholds come from " + mono("GET /api/v1/reference") + ", which reads the same "
             "YAML files as the backend — the browser never restates a number.",
             "Design: one accent colour, status always shown as text plus colour, light and dark themes, keyboard "
             "focus, and layouts that work at phone width.",
             "The Streamlit app (" + mono("frontend.py") + ") remains as the specified Phase 28 "
             "interface."],
        ],
        after_project=fig("docs/guide/figs/ui_results.png", "Figure 31.1 — The React interface for the Phase 30 "
                          "query: the answer first, with the assessment panel pinned on the right.", 1.0, 125),
        say="\"The React interface is a client of the same API: it shows the answer first, keeps measured, derived "
            "and predicted values visibly apart, and takes every threshold from the backend.\"",
    )

    s += topic(
        "32. Supporting tools: YAML, .env, SQLite and Docker",
        what="Four small tools keep the project configurable, safe, reproducible and easy to run.",
        analogy="The settings menu (YAML), a locked drawer for keys (.env), a notebook for comments (SQLite), and a "
                "shipping container that holds everything needed to run (Docker).",
        how=[
            "<b>YAML</b> is a human-readable configuration format. Keeping thresholds, weights and rules in YAML "
            "means behaviour can change without touching code.",
            "<b>.env</b> holds secrets such as the API key outside the code; it is never committed or shared.",
            "<b>SQLite</b> is a complete database stored in a single file — no server needed.",
            "<b>Docker</b> packages the application and its exact dependencies into an image that runs the same on "
            "any machine; <b>docker-compose</b> starts several containers (API, UI) together.",
        ],
        project=[
            ["YAML: " + mono("src/config/") + " — schema, thresholds, taxonomy, severity, priority, retrieval, "
             "models, playbook.",
             ".env: " + mono("ANTHROPIC_API_KEY") + ", " + mono("LLM_MODEL") + ", " + mono("LLM_PROVIDER") + "; a "
             "template without secrets is in " + mono(".env.example") + ".",
             "SQLite: " + mono("data/feedback/feedback.db") + " stores ratings with the query, retrieved IDs, "
             "diagnosis and comment. Nothing is retrained automatically; " + mono("python -m src.services.feedback export")
             + " exports cases for human review before they join the evaluation set.",
             "Docker: one image builds the React UI and installs the Python stack; " + mono("docker compose up")
             + " starts the API (with the UI at /ui) and the Streamlit app."],
        ],
        say="\"Every rule is in YAML, secrets are only in .env, feedback goes to SQLite for review rather than "
            "automatic retraining, and Docker makes the setup reproducible.\"",
    )
    s.append(PageBreak())
    return s


# ================================================================================================ BACK MATTER
def cheat_sheet() -> list:
    rows = [["Technology", "Say it in one sentence"],
            ["Rules vs ML vs LLM", "Rules decide facts, ML estimates with measured error, the LLM only explains."],
            ["Percentile conditions", "A condition means the worst 10% of this network's own history."],
            ["Category / severity / priority", "Documented point systems in YAML: what kind, how bad, how urgent."],
            ["Embeddings (MiniLM)", "Text turned into 384 numbers so similar meanings are close."],
            ["ChromaDB + HNSW", "A vector database that finds close vectors fast, with metadata filters."],
            ["BM25", "Keyword ranking that rewards rare words and damps repetition."],
            ["Metric similarity", "Compares the record's actual numbers with what the user stated."],
            ["Hybrid retrieval", "Normalised, weighted mix of meaning, keywords and numbers; filters enforced twice."],
            ["Train/test split, CV", "Scores come from 20,000 unseen records; 5-fold CV confirms stability."],
            ["Gradient boosting", "Many small trees, each correcting the previous errors; handles missing inputs."],
            ["R², MAE", "R² = share of variation explained (0.885); MAE = typical miss (3.7 points)."],
            ["Leakage", "0.9998 rediscovers the target's formula; the served 0.885 uses only pre-fault data."],
            ["Prediction interval", "An 80% range from the model's own test errors."],
            ["LLM + structured output", "The LLM fills a fixed schema from the evidence package only."],
            ["RAG", "Retrieve evidence, compute conclusions in code, let the LLM explain them."],
            ["Grounding gate", "Code rejects any number, ID, protocol term or promise not in the evidence."],
            ["LLM-as-a-judge", "A second review of meaning; a check passes only if code and judge agree."],
            ["Playbook confidence", "0.20 + evidence match + similar-record support − contradictions; a heuristic."],
            ["Agents", "Six fixed steps on a shared state, traced; no autonomous loops."],
            ["Guardrails", "Explainable patterns reject injection, secret requests and off-topic input."],
            ["Provenance", "Seven source types so facts, predictions and hypotheses never blur."],
            ["Testing / evaluation", "222 tests incl. five proofs; 8 dev cases at 1.00 on retrieval; DeepEval with an LLM judge."],
            ["FastAPI + Pydantic", "REST endpoints with validated inputs and structured errors."],
            ["React + TypeScript", "A typed web client that shows the answer first and restates no numbers."]]
    return [P("Cheat sheet — one sentence per technology", "h1"),
            P("Read this the morning of the presentation."),
            table(rows, widths=[1.8, 4.2], bold_first_col=True), PageBreak()]


GLOSSARY = [
    ("API", "A defined way for one program to request something from another over the network."),
    ("Average precision", "A ranking score that rewards placing relevant results at the top."),
    ("Baseline", "A simple reference result a better method must beat (e.g. always predicting the average)."),
    ("Bigram", "Two neighbouring words treated as one token, e.g. high_latency."),
    ("BM25", "A keyword-ranking formula based on term frequency, rarity and document length."),
    ("Classification", "Predicting a category."),
    ("Condition", "A named true/false flag derived from one metric and a documented threshold."),
    ("Confusion matrix", "A table counting every (true category, predicted category) pair."),
    ("Cosine similarity", "How closely two vectors point the same way; 1 = same direction."),
    ("Cross-validation", "Training and scoring several times on different splits to check stability."),
    ("Data leakage", "Giving a model information that contains the answer or arrives after the event."),
    ("Decision tree", "A model that predicts through a sequence of yes/no questions."),
    ("Derived", "Computed from raw data by your own rules — an interpretation, not a measurement."),
    ("Deterministic", "Always gives the same output for the same input."),
    ("Embedding", "A list of numbers representing the meaning of a text."),
    ("Endpoint", "One address of an API, e.g. POST /api/v1/analyze."),
    ("F1 score", "The balance of precision and recall."),
    ("Feature", "An input column used by a model."),
    ("Filter (metadata)", "A restriction such as state = MN applied before ranking."),
    ("Gradient boosting", "Building trees one after another, each correcting the remaining errors."),
    ("Grounding", "Checking that generated text is supported by the given evidence."),
    ("Guardrail", "A check on input or output that blocks misuse."),
    ("Hallucination", "Fluent model output that is not supported by its input."),
    ("Heuristic", "A transparent rule of thumb, not a calibrated probability."),
    ("HNSW", "A layered graph index for fast approximate nearest-neighbour search."),
    ("Hybrid retrieval", "Combining several search signals into one ranking."),
    ("Hypothesis", "A possible cause that the evidence supports but does not confirm."),
    ("IDF", "Inverse document frequency: rare words get higher weight."),
    ("Inference", "Using a trained model on a new case."),
    ("JSON", "A text format of keys and values used by web APIs."),
    ("Label", "The known answer attached to a training example."),
    ("LLM", "Large language model: a neural network trained on text to read and write language."),
    ("MAE / RMSE", "Average size of prediction errors; RMSE punishes large errors more."),
    ("Min-max normalisation", "Rescaling scores so the lowest is 0 and the highest is 1."),
    ("One-hot encoding", "Turning a category into several 0/1 columns."),
    ("Overfitting", "Fitting noise in the training data and doing worse on new data."),
    ("Parquet", "A compact, typed, column-based file format for tables."),
    ("Percentile", "The value below which a given share of the data falls (P90: 90%)."),
    ("Precision / recall", "Of predicted positives, how many are right / of real positives, how many were found."),
    ("Prompt injection", "Input that tries to override an LLM system's instructions."),
    ("Provenance", "The recorded source of a statement."),
    ("Pydantic", "A Python library that validates data against declared types and limits."),
    ("R²", "Share of the target's variation a regression model explains (1 = perfect)."),
    ("RAG", "Retrieval-augmented generation: retrieve evidence, then generate from it."),
    ("Random forest", "An average of many decision trees trained on random samples."),
    ("Regex", "Regular expression: a precise pattern for matching text."),
    ("Regression", "Predicting a number."),
    ("Residual", "Actual value minus predicted value."),
    ("REST", "A common style of web API using HTTP methods on resource paths."),
    ("Schema", "The written contract for data: fields, types, units and valid ranges."),
    ("Structured output", "LLM output forced into a fixed schema instead of free text."),
    ("Supervised learning", "Learning from examples whose answers are known."),
    ("System prompt", "Standing instructions given to an LLM for every request."),
    ("Target", "The column a model learns to predict."),
    ("Test set", "Data hidden from training and used only to score the model."),
    ("Token", "A piece of a word; the unit LLMs read and write."),
    ("Transformer", "The neural-network architecture behind modern language models and embeddings."),
    ("Vector database", "A store that finds the vectors closest to a query vector."),
    ("YAML", "A human-readable configuration file format."),
]


def glossary() -> list:
    rows = [["Term", "Meaning"]] + [[f"<b>{t}</b>", d] for t, d in GLOSSARY]
    return [P("Glossary", "h1"), table(rows, widths=[1.6, 4.4])]
