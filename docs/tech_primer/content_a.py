"""Primer content: cover, how to read, Part A (foundations), Part B (data and rules), Part C (search)."""

from __future__ import annotations

from docs.guide.pdf_kit import (KeepTogether, NextPageTemplate, P, PageBreak, Spacer, bullets, callout, code, mm,
                                numbered, table, toc)
from docs.tech_primer.kit import fig, mono, topic


def cover() -> list:
    return [
        Spacer(1, 18 * mm),
        P('<font color="#ffffff">The technology behind<br/>your NetOps Assistant</font>', "title"),
        P('<font color="#cfd8e3">Every model, agent, method and algorithm in the project, explained from zero — '
          'what it is, how it works, and exactly how your project uses it</font>', "subtitle"),
        Spacer(1, 40 * mm),
        table([
            ["For", "Someone new to AI, machine learning and web technology who needs to understand and defend the project"],
            ["Covers", "32 technologies in 7 parts: foundations · data and rules · search · machine learning · "
                       "LLM and agents · quality · serving and interface"],
            ["Every topic", "In one sentence → everyday analogy → how it works → how your project uses it → "
                            "what to say if the panel asks"],
            ["Numbers", "Taken from your project: its configuration files, trained models, evaluation results and "
                        "real outputs. Charts marked 'real' were computed from your data for this document."],
        ], widths=[1, 5], bold_first_col=True),
        Spacer(1, 10 * mm),
        P("Companion to docs/Project_Guide.pdf (what was built, phase by phase). This primer explains the ideas "
          "underneath. Rebuild it with python -m docs.tech_primer.build_primer.", "small"),
        NextPageTemplate("normal"),
        PageBreak(),
    ]


def how_to_read() -> list:
    return [
        P("How to read this primer", "h1"),
        P("You do not need any background. Each topic starts with a one-sentence definition and an everyday "
          "analogy, then explains the mechanics, then shows the exact place, settings and numbers in your project. "
          "If you only have ten minutes before the panel, read the one-sentence definitions and the green "
          "<b>If the panel asks</b> boxes."),
        table([
            ["Box", "Meaning"],
            ["<b>Everyday analogy</b> (purple)", "An intuition from daily life. Useful for explaining, never a "
                                                "substitute for the real definition."],
            ["<b>If the panel asks</b> (green)", "A short, accurate answer you can say out loud."],
            ["<b>Watch out</b> (orange)", "A common misunderstanding, or a limitation you should state honestly."],
        ], widths=[1.6, 4.4]),
        P("Words in " + mono("this font") + " are names of files, settings or code in your project, for example "
          + mono("src/config/thresholds.yaml") + ". The glossary at the end defines every technical term."),
        P("Contents", "h2"),
        toc(),
        PageBreak(),
    ]


# ================================================================================================ PART A
def part_a() -> list:
    s = [P("Part A — Foundations", "part"),
         P("The vocabulary every later chapter uses, and the big picture of how the pieces fit together."),
         Spacer(1, 4)]

    s += topic(
        "1. The big picture: three kinds of intelligence",
        what="Your project combines hand-written <b>rules</b>, trained <b>machine-learning models</b> and a "
             "<b>large language model (LLM)</b>, and gives each one only the job it is best at.",
        analogy="A hospital: written clinical guidelines (rules) decide what counts as a fever; lab equipment "
                "calibrated on past patients (ML) estimates risk; a doctor (the LLM) explains the results to you in "
                "plain words — but the doctor is not allowed to invent lab values.",
        how=[
            "<b>Rules</b> are instructions a person writes: \"latency above 129.92 ms is high latency\". They are "
            "exact, explainable and easy to test, but someone must decide every threshold.",
            "<b>Machine learning (ML)</b> lets a computer find patterns in examples. You show it 80,000 past records "
            "together with their fault rate, and it learns a function that estimates the fault rate of a new record.",
            "<b>A large language model</b> is a very large neural network trained on huge amounts of text. It is "
            "excellent at reading and writing language, and also good at sounding confident about things that are "
            "not true. That is why your project never lets it decide facts.",
        ],
        after_how=fig("docs/tech_primer/figs/pipeline.png", "Figure 1.1 — How the pieces connect. Steps 1–4 run "
                      "once when the data is built; steps 5–12 run for every question (illustration).", 1.0, 112),
        project=[
            ["<b>Rules</b> decide the facts about each record: the 18 conditions, the category, severity and "
             "priority (Part B), and the root-cause hypotheses (Chapter 23).",
             "<b>ML</b> makes estimates the rules cannot: fault risk, throughput and a cross-check of the category "
             "(Part D).",
             "<b>The LLM (NVIDIA gpt-oss-20b)</b> only writes the explanation and reviews it (Part E). Code checks every number "
             "it writes.",
             "<b>Search</b> (Part C) connects them: it finds the historical records that serve as evidence."],
        ],
        say="\"Facts come from the data and documented rules, estimates come from ML models with measured error, "
            "and the LLM only explains — a deterministic check rejects anything it writes that is not in the "
            "evidence.\"",
    )

    s += topic(
        "2. Data words: record, feature, target, label",
        what="Machine learning and search both work on <b>tables</b>: each row is one example (a <b>record</b>), "
             "each column is one measured property (a <b>feature</b>), and the column you want to predict is the "
             "<b>target</b>.",
        analogy="A spreadsheet of house sales: each row is a house; size, rooms and location are features; the sale "
                "price is the target a model learns to predict.",
        how=[
            "<b>Raw data</b> is what was measured. <b>Derived data</b> is computed from it by your own rules. "
            "Keeping them apart is the most important habit in your project: raw values are facts, derived values "
            "are interpretations.",
            "A <b>label</b> is the correct answer attached to a training example. In many projects people write "
            "labels by hand. Your dataset has <b>no</b> human labels, so every category and severity is derived by "
            "rules — and the project says so everywhere.",
            "<b>Training</b> means fitting a model on examples; <b>inference</b> (or prediction) means using the "
            "trained model on a new case.",
        ],
        project=[
            "Dataset: " + mono("fenar/5g_netops") + ", 100,000 records × 27 raw columns, downloaded at a pinned "
            "version and checked with a SHA-256 fingerprint so it can never change silently.",
            ["Raw features: KPIs such as " + mono("latency_ms") + ", " + mono("packet_loss_pct") + ", "
             + mono("handover_success_pct") + "; context such as season, weather, temperature, state and city.",
             "Targets: " + mono("fault_rate_pct") + " (fault-risk model), " + mono("throughput_mbps")
             + " (throughput model) and the derived " + mono("primary_category") + " (classifier).",
             "Each record gets a stable ID, " + mono("ROW-000001") + " to " + mono("ROW-100000") + ", so every "
             "statement in an answer can point to a real row."],
        ],
        after_project=callout("The data is synthetic: columns are uniformly distributed, locations are random, and "
                              "the fault rate is almost a linear formula of seven other columns. Say this yourself "
                              "before anyone else does — it is why causes stay hypotheses.", "warn"),
        say="\"There are no human labels, so I derived them with documented rules and I always call them derived. "
            "Raw values are never modified, and each record keeps an ID back to its row in the original file.\"",
    )
    s.append(PageBreak())
    return s


# ================================================================================================ PART B
def part_b(F: dict) -> list:
    s = [P("Part B — Data and rules", "part"),
         P("How raw measurements become clean, typed data and then the conditions, categories, severity and "
           "priority that everything else relies on.")]

    s += topic(
        "3. pandas, Parquet and data validation",
        what="<b>pandas</b> is the standard Python library for working with tables; <b>Parquet</b> is a compact "
             "file format for storing them; <b>validation</b> means checking every value against written rules "
             "before using it.",
        analogy="Receiving goods at a warehouse: you check every box against the delivery note (validation), "
                "then store it on labelled shelves (Parquet) so it is quick to find later.",
        how=[
            "pandas loads a table into a <b>DataFrame</b> — rows and named columns — and lets you filter, compute and "
            "summarise with a few lines of code.",
            "Parquet stores each column together and remembers its type (number, text, true/false), so tables load "
            "quickly and a number never comes back as text.",
            "A <b>schema</b> is the written contract for the data: column names, types, units and valid ranges.",
        ],
        project=[
            "The schema lives in " + mono("src/config/schema.yaml") + ": for example signal strength must lie "
            "between −140 and −44 dBm and percentages between 0 and 100.",
            [mono("src/preprocessing/validator.py") + " checks types, ranges, missing values and unknown categories. On your data "
             "it found 0 errors in 100,000 rows; nothing was dropped and nothing was filled in (no imputation).",
             mono("src/preprocessing/cleaner.py") + " standardises names and types; four " + mono("dq_*") + " columns flag "
             "suspicious but valid patterns (for example more critical alarms than total alarms) instead of "
             "changing them.",
             "Output: " + mono("data/processed/records.parquet") + " — 27 typed raw fields + record_id + 4 quality "
             "flags."],
        ],
        after_project=callout("A real bug found by testing: pandas' fast number parser changed about 15% of values "
                              "in their last decimal place. A test that compares every processed value with the raw "
                              "CSV text caught it, and the fix parses numbers exactly with Python's own float().",
                              "bad"),
        say="\"Every processed value is proven identical to the raw text by a test, so preprocessing cannot "
            "fabricate or alter a measurement.\"",
    )

    s += topic(
        "4. Percentiles and derived conditions",
        what="A <b>percentile</b> tells you where a value stands among all values: the 90th percentile (P90) is the "
             "value that 90% of records are below. Your project marks the worst 10% of each KPI as a named "
             "<b>condition</b> such as " + mono("high_latency") + ".",
        analogy="Growth charts for children: a child above the 90th percentile for height is tall <i>compared with "
                "other children of the same age</i>, not by an absolute rule.",
        how=[
            "Sort all 100,000 latency values; the value at position 90,000 is P90. Anything above it is in the worst "
            "decile (tenth).",
            "For metrics where low is bad (throughput, handover success) the worst decile is below P10 instead.",
            "Why not industry rules of thumb? Your Phase 1 analysis showed that on this dataset common operator "
            "targets would flag 80–88% of rows — a condition that is true for almost everyone carries no "
            "information.",
            "One exception is a <b>domain threshold</b>: weak signal is fixed at −110 dBm, an established radio "
            "engineering boundary that is also selective on this data.",
        ],
        after_how=fig(F["percentile"], "Figure 4.1 — Real: latency of all 100,000 records. The P90 threshold "
                                       "(129.92 ms) marks exactly the worst 10% as 'high latency'.", 0.95, 62),
        project=[
            "18 conditions in " + mono("src/config/thresholds.yaml") + ", computed once from the data and "
            "re-checked by " + mono("python -m src.preprocessing.build_thresholds --check") + " so the file cannot drift from the "
            "data.",
            "Each record gets 18 true/false columns (" + mono("cond_high_latency") + " …). A missing value never "
            "triggers a condition; it is reported as unavailable.",
            "The same function is used for the whole dataset and for values typed into the web UI, so there is only "
            "one implementation of every rule.",
        ],
        say="\"A condition means 'in the worst 10% of this network's own history', computed from the data and "
            "documented in YAML — except weak signal, which uses the standard −110 dBm boundary.\"",
    )

    s += topic(
        "5. Rule-based scoring: category, severity and priority",
        what="Three transparent point systems turn a record's conditions into a <b>category</b> (what kind of "
             "problem), a <b>severity</b> (how badly the network behaves) and a <b>priority</b> (how urgently to "
             "act).",
        analogy="A hospital triage nurse: symptoms decide the department (category), how abnormal the vital signs "
                "are decides severity, and severity plus how many people are affected decides who is seen first "
                "(priority).",
        how=[
            "<b>Category — weighted scoring.</b> Each category lists condition weights in "
            + mono("taxonomy.yaml") + ", e.g. Mobility / Handover = 1.2 × low_handover + 0.4 × high_call_drop. The "
            "highest score wins; other positive scores become secondary categories; no condition means Normal. "
            "Scoring was chosen over 'A AND B' rules because conditions rarely occur together on this data.",
            "<b>Severity — points × badness.</b> Each condition has points (low availability 30, call drops 22, … "
            "parameter changes 4). Badness is 0.5 when a value just crosses its threshold and rises to 1.0 at the "
            "dataset's 1% extreme, so a slightly bad value earns half the points. Levels: LOW below 15, MEDIUM from "
            "15, HIGH from 35, CRITICAL from 55, with overrides so one extreme problem is never diluted.",
            "<b>Priority — severity plus impact.</b> Priority adds up to 15 points for affected users (scaled by the "
            "dataset's P99 of 9,902 users), 10 if a customer reported an issue, 10 for low availability, 5 for many "
            "critical alarms, and 0–15 for the predicted fault-risk level. P1 from 75, P2 from 50, P3 from 28, "
            "otherwise P4.",
        ],
        project=[
            "Worked example, record " + mono("ROW-000001") + " (real):",
            table([
                ["Step", "Calculation", "Result"],
                ["Conditions", "weak signal (−110.12 dBm vs −110), many parameter changes (18 vs ≥ 18), high data usage",
                 "3 conditions"],
                ["Category", "weak_signal 1.0 beats Congestion / Load 0.8 and Configuration 0.7",
                 "Radio / Signal Quality"],
                ["Severity", "weak signal 12 points × badness 0.51 = 6.1; parameter changes 2.0; customer issue 5",
                 "13.1 → LOW"],
                ["Priority", "13.1 + users 2,442 / 9,902 × 15 = 3.7 + customer issue 10", "26.8 → P4"],
            ], widths=[1, 3.6, 1.4], bold_first_col=True),
            "Badness here is only 0.51 because −110.12 dBm has barely crossed the −110 threshold; at the extreme "
            "(−119.41 dBm) it would earn all 12 points.",
        ],
        say="\"Category, severity and priority are documented point systems in YAML, not ML and not an LLM. "
            "Severity measures how badly the network behaves; priority adds impact, so they answer different "
            "questions.\"",
    )
    s.append(PageBreak())
    return s


# ================================================================================================ PART C
def part_c(F: dict) -> list:
    sim = F["sim"]
    s = [P("Part C — Search and retrieval", "part"),
         P("How the system finds the historical records most similar to a question. This is the 'R' in RAG and "
           "the foundation of every answer.")]

    s += topic(
        "6. Turning rows into searchable text",
        what="Search engines and embedding models read text, but your dataset is numbers, so every record is "
             "described in a deterministic sentence built only from its own fields and derived features.",
        analogy="A library catalogue card: it does not contain the book, but a short, consistent description that "
                "lets you find the right book quickly.",
        how=[
            "<b>Deterministic</b> means the same record always produces exactly the same text — no randomness and no "
            "LLM, so no invented content.",
            "Two versions are produced because two search methods want different things (next chapters).",
        ],
        project=[
            [mono("search_text") + " — the full description: conditions, category, severity, priority, location, "
             "season, weather and all metrics rounded to two decimals. Shown to users and the LLM, and indexed by "
             "BM25 keyword search.",
             mono("embedding_text") + " — only the qualitative part (no numbers), for the embedding model. Real "
             "example for ROW-000001:"],
            code("Derived conditions: weak signal strength (weak_signal), high data usage (high_data_usage), many\n"
                 "parameter changes (high_parameter_changes). Derived category: Radio / Signal Quality (secondary:\n"
                 "Congestion / Load, Configuration). Derived severity: LOW. Derived priority: P4. Weather Clouds,\n"
                 "season Winter. Customer issue reported: yes."),
            "Numbers are left out of the embedding text because embedding models read only the first 256 word "
            "pieces and capture meaning poorly from digits. Numeric closeness is handled separately by metric "
            "similarity (Chapter 10).",
            "Many records share the same qualitative description: only 27,117 distinct embedding texts exist for "
            "100,000 records, so each is embedded once and reused.",
        ],
        say="\"Records are described by a fixed template from their own fields, so the search text can never "
            "contain anything that is not in the data.\"",
    )

    s += topic(
        "7. Embeddings and Sentence-Transformers",
        what="An <b>embedding</b> turns a piece of text into a list of numbers (a <b>vector</b>) so that texts with "
             "similar meaning get similar vectors. Your project uses the model " + mono("all-MiniLM-L6-v2") + " from "
             "the Sentence-Transformers library, which produces 384 numbers per text.",
        analogy="GPS coordinates for meaning: 'long network delay' and 'high latency' are parked on the same street, "
                "while 'snowfall in winter' is in a different town.",
        how=[
            "MiniLM is a small <b>transformer</b> neural network (6 layers). It was trained on over a billion pairs "
            "of related sentences to place related sentences close together — this is called <b>contrastive "
            "learning</b>.",
            "Closeness is measured with <b>cosine similarity</b>: 1 means pointing the same way (same meaning), 0 "
            "means unrelated. Your code normalises every vector to length 1, so cosine similarity is simply the dot "
            "product (multiply matching numbers and add them up).",
            "It runs on the CPU, needs no internet once downloaded, and embeds a short text in milliseconds.",
        ],
        after_how=[
            fig(F["emb_scatter"], "Figure 7.1 — Real: ten phrases embedded by your model, compressed from 384 to 2 "
                                  "directions with PCA. Phrases with the same meaning cluster.", 0.88, 80),
        ],
        project=[
            "Settings in " + mono("src/config/retrieval.yaml") + ": model all-MiniLM-L6-v2, batch size 128. Code: "
            + mono("src/rag/embeddings.py") + ".",
            f"The real similarities show both the strength and the blind spot: 'high latency' vs 'long network "
            f"delay' = {sim[('high latency', 'long network delay')]:.2f} (different words, same meaning — good), but "
            f"'weak signal strength' vs 'good signal strength' = {sim[('weak signal strength', 'good signal strength')]:.2f}, "
            f"higher than 'weak signal strength' vs 'poor radio coverage' = "
            f"{sim[('weak signal strength', 'poor radio coverage')]:.2f}. Embeddings notice the topic (signal) more "
            "than the direction (good or weak).",
            "That is exactly why your project does not rely on embeddings alone: keyword search and metric "
            "similarity cover this blind spot, and the query parser removes 'good/normal' phrases before semantic "
            "search (Chapter 11).",
        ],
        after_project=fig(F["emb_heat"], "Figure 7.2 — Real: cosine similarity between every pair of phrases "
                                         "(1.00 = identical meaning).", 0.8, 110),
        say="\"Embeddings capture meaning, so 'declining handover' finds 'low handover success'. They are weak at "
            "negation and numbers, which is why they are only one of three signals in hybrid retrieval.\"",
    )

    s += topic(
        "8. Vector database: ChromaDB and HNSW",
        what="A <b>vector database</b> stores millions of embeddings and quickly finds the ones closest to a query "
             "vector. Your project uses <b>ChromaDB</b>, which uses an index called <b>HNSW</b> to avoid comparing "
             "the query with every record.",
        analogy="Finding the nearest café in a city: you take the motorway to the right district, then main roads, "
                "then side streets — instead of checking every café in the country.",
        how=[
            "Comparing a query with 100,000 vectors one by one works but becomes slow at scale. <b>Approximate "
            "nearest-neighbour</b> search accepts a tiny chance of missing the very best match in exchange for "
            "speed.",
            "<b>HNSW (Hierarchical Navigable Small World)</b> builds layers of graphs. The top layer has few points "
            "linked by long jumps; each lower layer has more points and shorter links; the bottom layer contains "
            "every vector. A search starts at the top and greedily moves closer at every layer.",
            "Each stored item can carry <b>metadata</b> (season, weather, state …). A <b>where filter</b> limits the "
            "search to items whose metadata match, e.g. only Minnesota in winter.",
        ],
        after_how=fig(F["hnsw"], "Figure 8.1 — How an HNSW search narrows down layer by layer (illustration).",
                      0.85, 70),
        project=[
            [mono("src/rag/vector_store.py") + ": collection " + mono("network_records") + ", cosine "
             "distance, stored in " + mono("data/processed/chroma") + " (100,000 vectors × 384 numbers).",
             "Each item's ID <b>is</b> the record_id, so a search hit maps directly to a real record — no separate "
             "lookup table that could drift.",
             "Metadata stored with every vector: season, weather, state, city, category, severity, priority, "
             "customer issue and the 18 condition flags, so filters run inside the vector search.",
             "The first search after start-up loads the index into memory, which is why the API 'warms up' for "
             "about 40 seconds before it accepts questions."],
        ],
        say="\"ChromaDB stores one 384-number vector per record with its metadata. HNSW finds close vectors without "
            "scanning all 100,000, and metadata filters restrict the search to the user's state, season or "
            "weather.\"",
    )

    s += topic(
        "9. Keyword search: BM25",
        what="<b>BM25</b> is the classic ranking formula behind search engines: a record scores high when it "
             "contains the query's words, especially rare words, but repeating a word many times gives diminishing "
             "returns.",
        analogy="Searching a library for 'monsoon flooding': a book that mentions 'flooding' ten times is relevant, "
                "but not ten times more relevant than one mentioning it five times — and the rare word 'monsoon' "
                "matters more than the common word 'the'.",
        how=[
            "<b>Tokenisation</b> splits text into words (tokens), lower-cases them and removes stop words such as "
            "'the' and 'and'.",
            "<b>Term frequency (tf)</b> — how often the word appears in this record. BM25 makes it <b>saturate</b> "
            "with the parameter k1: the first mentions count a lot, later ones less.",
            "<b>Inverse document frequency (IDF)</b> — words that appear in few records get a high weight; a word in "
            "every record gets almost zero.",
            "<b>Length normalisation</b> with parameter b — long documents are slightly penalised so they do not "
            "win just by containing more words.",
            "Score for one word = IDF × tf × (k1 + 1) / (tf + k1 × (1 − b + b × length / average length)); the "
            "record's score is the sum over the query's words.",
        ],
        after_how=fig(F["bm25"], "Figure 9.1 — Real parameters (k1 = 1.5, b = 0.75, 100,000 records): repetition "
                                 "saturates, long documents are damped, and rare words weigh more.", 1.0, 70),
        project=[
            ["Implemented in " + mono("src/rag/bm25.py") + " (not a library), as a sparse matrix of 100,000 "
             "records × 139,815 terms, so a query is one fast column sum. Settings k1 = 1.5, b = 0.75 in "
             + mono("retrieval.yaml") + ".",
             "<b>Bigrams</b> (pairs of neighbouring words) are added: every record mentions 'latency' in its metric "
             "line, but only records with the condition contain the bigram " + mono("high_latency") + ", so phrase "
             "queries discriminate.",
             "BM25 shines on exact names (conditions, places, categories) — where embeddings are vague."],
        ],
        say="\"BM25 rewards rare query words and damps repetition and long documents. I added word pairs so "
            "'high latency' matches records that actually have the condition, not every record that mentions "
            "latency.\"",
    )

    s += topic(
        "10. Metric similarity and percentile rank",
        what="Metric similarity compares <b>numbers</b>: if the question says 'latency 150 ms' or 'high latency', "
             "records whose actual latency fits that statement score close to 1.",
        analogy="Shopping for a laptop under €800: a €790 laptop fits perfectly, an €850 one nearly fits, a €2,000 "
                "one does not — closeness on a number, not on words.",
        how=[
            "<b>Stated value</b> (\"latency 150 ms\"): similarity = 1 − |record value − 150| ÷ (P99 − P01), so it "
            "falls linearly as the record moves away from 150 ms across the metric's normal range.",
            "<b>Above / below</b> (\"latency above 120 ms\"): 1 for every record beyond the value, decaying for "
            "records short of it.",
            "<b>Qualitative 'high'</b>: half the record's <b>percentile rank</b> (0 = best in the dataset, 1 = "
            "worst) plus half for actually crossing the condition threshold, so records that truly have the "
            "condition jump ahead.",
            "<b>'Normal'</b>: 1 at the median, falling towards both extremes. <b>Context</b> (weather, season, "
            "state mentioned in the text): 1 if equal, else 0.",
            "The record's metric score is the average over everything the question mentioned.",
        ],
        after_how=fig(F["metric"], "Figure 10.1 — Real: similarity of a record to three phrasings about latency, "
                                   "using your data's percentiles and the 129.92 ms threshold (dotted).", 0.95, 75),
        project=[
            mono("src/rag/metric_similarity.py") + ". Percentile ranks are computed against all 100,000 "
            "records, so 'high' always means high for this network.",
            "The threshold half was added after testing: with rank alone, 'low throughput' preferred records just "
            "above the threshold, and precision@5 on one evaluation case was only 0.2. With the threshold indicator "
            "it reached 1.0. You should mention that this weight was tuned on your development cases.",
        ],
        say="\"Embeddings and keywords do not understand '150 ms', so a third score compares the record's actual "
            "values with what the user stated, using the dataset's own percentiles and thresholds.\"",
    )

    s += topic(
        "11. Hybrid retrieval and query understanding",
        what="<b>Hybrid retrieval</b> combines the three signals — semantic, keyword and metric — into one "
             "ranking. <b>Query understanding</b> first reads the question with rules to find which metrics are "
             "described as bad, good or normal.",
        analogy="Three witnesses describe a suspect: one remembers the face (meaning), one the name tag (keywords), "
                "one the height (numbers). A detective weighs all three statements instead of trusting one.",
        how=[
            "<b>1. Understand the query.</b> " + mono("src/rag/query_parser.py") + " uses <b>regular "
            "expressions</b> (text patterns) to find phrases like 'declining handover success' (adverse), 'good "
            "signal strength' (favourable), 'normal user load' (normal), values ('above 12%') and context ('during "
            "snow'). Good/normal phrases are removed from the text sent to semantic and keyword search.",
            "<b>2. Apply filters</b> (state, season, weather, category …) as a mask over all records.",
            "<b>3. Collect candidates:</b> the top 300 from each method inside the filter.",
            "<b>4. Re-score every candidate exactly</b> with all methods, and <b>min-max normalise</b> the semantic "
            "and keyword scores over the candidates (lowest → 0, highest → 1) so they are comparable.",
            "<b>5. Weighted sum.</b> Weights 0.6 semantic, 0.4 keyword, 0.6 metric, re-normalised to the methods "
            "active for this query (e.g. 0.375 / 0.25 / 0.375). Subtract 0.15 for each condition the user called "
            "good that the record nevertheless has.",
            "<b>6. Final guarantee:</b> every result is checked against the filters again before it is returned.",
        ],
        after_how=fig("docs/guide/figs/hybrid.png", "Figure 11.1 — Real: 'high latency and packet loss', filtered to "
                      "Minnesota in winter (412 eligible records, 386 candidates). Each bar shows how the three "
                      "signals add up.", 0.9, 75),
        project=[
            "Top result of that search, " + mono("ROW-033707") + ": semantic 0.568, keyword 1.0 (raw BM25 4.83), "
            "metric 0.99 → hybrid 0.996.",
            "A test checks seven filter combinations and proves that no returned record ever violates a filter.",
        ],
        after_project=callout("Real problem found by the LLM judge: 'normal user load' used to retrieve high-load "
                              "records, because embeddings matched the topic 'user load'. The fix removes good/normal "
                              "phrases from the search text and adds the 0.15 penalty.", "bad"),
        say="\"Each method has a blind spot — meaning, exact words or numbers — so I combine all three with "
            "configurable weights after normalising them, and filters are enforced twice.\"",
    )
    s.append(PageBreak())
    return s
