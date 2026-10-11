# AI-Powered Telecom Network Operations Assistant

**Repository:** `Jerom_Telecom` · **Domain:** Telecom · **Participant:** [Your name] · **Team details:** [team / contact]

An evidence-grounded GenAI assistant for network operations engineers. You describe a network problem in plain
language. The assistant then:

- retrieves similar historical cases from 100,000 records;
- classifies the case and predicts fault risk;
- ranks root-cause hypotheses and recommends what to check and do.

Every statement is traceable to its source, and nothing is invented.

| Deliverable | File |
|---|---|
| Architecture diagram | [architecture/Architecture_Diagram.pdf](architecture/Architecture_Diagram.pdf) |
| Design document | [design/Design_Document.pdf](design/Design_Document.pdf) |
| API documentation | [docs/API_Documentation.pdf](docs/API_Documentation.pdf) |
| Dataset details | [docs/Dataset_Details.pdf](docs/Dataset_Details.pdf) |
| Evaluation report | [docs/Evaluation_Report.pdf](docs/Evaluation_Report.pdf) |
| Presentation deck | [presentation/Project_Presentation.pptx](presentation/Project_Presentation.pptx) |
| Sample data and queries | [data/sample_data/](data/sample_data/) |
| Test cases | [tests/](tests/) · [tests/evaluation/eval_cases.json](tests/evaluation/eval_cases.json) |

---

## 1. Project overview

The assistant combines three kinds of intelligence and gives each one only the job it is best at:

- **Rules** decide the facts about each record: 18 derived conditions, a category, severity and priority. They come from documented YAML configuration.
- **Machine learning** makes estimates the rules cannot: fault-occurrence risk (with an 80% interval), throughput, and a category cross-check.
- **An LLM** (NVIDIA `openai/gpt-oss-20b`) only *explains* the evidence. A deterministic grounding gate rejects any number, record ID, protocol term or promise in its text that is not in the evidence. An LLM judge then reviews the result.

The analysis runs as six agents (retrieval → classification → risk → diagnosis → recommendation → validation) behind a FastAPI REST API, with a Streamlit front end.

## 2. Problem statement

Network operations centre (NOC) engineers face a constant stream of KPI degradations and alarms. Examples are rising latency and packet loss, falling handover success, and alarm storms. For each one they must decide:

- what kind of problem it is;
- how urgent it is;
- what probably caused it;
- what to do.

They usually decide by searching past incidents and from experience. That is slow and inconsistent. General-purpose chatbots are unsafe here: they invent measurements, incident IDs and confident root causes, and a wrong "fact" costs outage time.

## 3. Business objective

- **Shorter time to a first diagnosis:** similar cases, category, severity and priority in seconds.
- **Consistent decisions:** the same documented rules are applied to every case.
- **Reuse of historical knowledge:** hybrid search over past records serves as evidence.
- **Trust and auditability:** every statement is tagged with one of seven source types, and LLM output is checked by code.
- **Safe operation:** decision support only. The system refuses requests to act on the network and never promises KPI improvements.

## 4. Solution summary

1. **Preprocessing (once):**
   - the pinned dataset is validated and cleaned;
   - 18 conditions are derived from worst-decile thresholds;
   - a category, severity and priority are derived by rules.
2. **Chunking:** each record becomes one chunk.
   - `search_text` is the full description and is indexed by BM25.
   - `embedding_text` is the qualitative part, embedded with `all-MiniLM-L6-v2` into **ChromaDB**.
3. **Retrieval (per question):**
   - query understanding (bad / good / normal metrics, values, context);
   - candidates from semantic, keyword and metric similarity;
   - exact re-scoring, then a weighted hybrid score;
   - filters enforced twice.
4. **Agents** build the answer:
   - classification;
   - ML fault risk next to the rate observed in similar cases;
   - ranked hypotheses from a documented playbook;
   - recommendations;
   - an LLM explanation that must pass the grounding gate;
   - validation (code checks plus an LLM judge), with provenance for every statement.
5. **API and UI:** FastAPI serves the result, and Streamlit shows it. An optional React web UI is also included.

## 5. Technology stack

| Layer | Technology |
|---|---|
| Programming language | Python 3.12 (3.11+) |
| LLM | NVIDIA API Catalog — `openai/gpt-oss-20b` (OpenAI-compatible endpoint, JSON-schema output). Claude is supported by configuration. |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, CPU) |
| Vector database | ChromaDB (persistent, HNSW index, cosine) |
| Keyword search | BM25 (own sparse implementation, k1 1.5, b 0.75) |
| ML | scikit-learn HistGradientBoosting (fault risk, throughput, classifier) |
| API | FastAPI + Uvicorn, Pydantic validation |
| UI | Streamlit (`frontend.py`); optional React + TypeScript (`web/`) |
| Data | pandas, PyArrow (Parquet), PyYAML configuration |
| Evaluation | pytest, DeepEval |

### Programming language
Python 3.12 for everything. The optional React UI uses TypeScript.

### LLM used
**`openai/gpt-oss-20b` through the NVIDIA API Catalog** (`https://integrate.api.nvidia.com/v1`).
- **Output format:** a JSON schema, validated with Pydantic.
- **Settings:** temperature 0.2, reasoning effort `low`. In measurements this was 5–16 s with clean grounding.
- **Where it is used:** for the explanation and as the LLM-as-judge.
- **Alternatives:** `LLM_PROVIDER=anthropic` switches to Claude. With no key, the system runs fully offline with template explanations.

### Embedding model
`sentence-transformers/all-MiniLM-L6-v2`, with L2-normalised 384-dimensional vectors. 100,000 records share 27,117 distinct chunk texts, and each is embedded once.

### Vector database
**ChromaDB.** It holds the persistent collection `network_records`, uses cosine distance, and the item id equals the `record_id`. Metadata (state, city, season, weather, category, severity, priority, issue flag, 18 condition flags) supports `where` filtering.

### Frameworks used
- **Serving:** FastAPI, Uvicorn, Pydantic, Streamlit.
- **Search and ML:** Sentence-Transformers, ChromaDB, scikit-learn, pandas.
- **LLM access:** OpenAI SDK (for the NVIDIA endpoint), Anthropic SDK (optional).
- **Testing and evaluation:** DeepEval, pytest.
- **Optional web UI:** React, Vite.

---

## 6. Installation instructions

Requirements:
- Python 3.11+ (developed on 3.12);
- about 3 GB of disk space for PyTorch, the data and the indexes;
- internet access for the first dataset and model download.

```bash
git clone <repository-url> Jerom_Telecom
cd Jerom_Telecom
```

## 7. Environment setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

## 8. Dependency installation

```bash
pip install -r requirements.txt
```

`requirements.txt` lists every package the code imports. The last group is only needed to regenerate the PDF and PPTX deliverables.

## 9. API key configuration

Copy the template and add your key. `.env` is git-ignored; **never commit it**.

```bash
copy .env.example .env          # Windows   (cp .env.example .env on macOS / Linux)
```

```
NVIDIA_API_KEY=nvapi-...         # from https://build.nvidia.com (API Catalog)
LLM_MODEL=openai/gpt-oss-20b
# LLM_PROVIDER=nvidia            # optional: nvidia | anthropic | offline (auto-detected from the keys)
# LLM_REASONING_EFFORT=low       # optional (gpt-oss default)
# ANTHROPIC_API_KEY=...          # alternative provider (Claude)
```

Without any key the system still works end to end. The explanation then uses a deterministic template, and the LLM judge is skipped.

## 10. Local deployment instructions

**Step 1: build the data, indexes and models once** (about 12 minutes on a laptop CPU):

```bash
python -m src.preprocessing.build_all
```

This downloads the pinned dataset (with a SHA-256 check), validates and cleans it, derives the conditions and labels, builds the chunks, builds the BM25 and ChromaDB indexes, and trains the models.

**Step 2: start the backend (FastAPI / Uvicorn):**

```bash
uvicorn src.main:app --reload --port 8000
```

Wait for `warm-up done` in the log (about 40–100 s while the index loads). The interactive API docs are at http://localhost:8000/docs.

**Step 3: start the front end (Streamlit)** in a second terminal:

```bash
streamlit run frontend.py
```

It opens http://localhost:8501. Set `API_URL` if the API is not on `http://localhost:8000`.

The front end has four pages: **Analyze a case** (main), **Search records**, **Record details** and **Assess measurements**.

**Optional React web UI:**

```bash
cd web && npm install && npm run build
```

After the build, the API also serves the React UI at http://localhost:8000/ui.

Docker is optional; see `Dockerfile` and `docker-compose.yml`.

## 11. Sample queries

Below are real outputs of `POST /api/v1/analyze` (`use_llm=false`, top-5) for five sample queries. The full JSON is in [data/sample_data/sample_queries.json](data/sample_data/sample_queries.json). With the LLM enabled, the same analysis also gets a written explanation.

**1. "A network region has increasing fault occurrence and declining handover success. What corrective actions should be considered?"**
- **Conditions:** high fault rate, low handover (stated).
- **Labels:** category Mobility / Handover (also Fault / Alarm), severity MEDIUM (estimated), priority P4.
- **Fault risk:** 44.08% predicted (Medium); 58.05% observed in the 5 similar records.
- **Leading hypothesis:** mobility configuration issue (neighbour relations or handover parameters), confidence 0.65. Alternatives: weather-related, load-related mobility, equipment fault.
- **Top action:** audit neighbour relationships and correct missing or incorrect neighbours.
- **Similar records:** ROW-098813 (0.98), ROW-058468 (0.98), ROW-069805 (0.97).

**2. "Cell has high latency and packet loss with good signal strength and normal user load. What similar network conditions occurred before?"**
- **Conditions:** high latency, high packet loss. Signal and load are treated as fine.
- **Labels:** category Service Degradation, MEDIUM, P4.
- **Fault risk:** 44.08% predicted (Medium); 46.63% observed.
- **Leading hypothesis:** transport or inter-cell connectivity degradation, 0.68.
- **Top action:** investigate transport and inter-cell connectivity for the affected sites.
- **Similar records:** ROW-036379, ROW-083117, ROW-067231 (all 0.96).

**3. "Poor signal strength and frequent call drops during snow"**
- **Conditions:** weak signal, high call drop.
- **Labels:** category Radio / Signal Quality (also Service Degradation), MEDIUM, P4.
- **Fault risk:** 47.61% predicted; 46.0% observed.
- **Leading hypothesis:** radio coverage or signal-quality degradation, 0.72.
- **Top action:** investigate radio quality (coverage, interference, antenna configuration).

**4. "High user load with reduced throughput and rising latency"**
- **Conditions:** high load, low throughput, high latency.
- **Labels:** category Congestion / Load, MEDIUM, P4.
- **Fault risk:** 44.08% predicted; 49.77% observed.
- **Leading hypothesis:** capacity congestion, 0.71.
- **Top action:** review capacity of the affected cells (load balancing, carrier/capacity expansion).

**5. "Many authentication failures and security incidents on a site"**
- **Conditions:** authentication failure, security risk.
- **Labels:** category Security, LOW, P4.
- **Fault risk:** 44.08% predicted; 50.61% observed.
- **Leading hypothesis:** security or authentication anomaly, 0.84.
- **Top action:** escalate to the security operations team for investigation of the incidents.

Try them with curl:

```bash
curl -X POST http://localhost:8000/api/v1/analyze -H "Content-Type: application/json" -d "{\"query\": \"Poor signal strength and frequent call drops during snow\", \"top_k\": 5, \"use_llm\": true}"
```

How to read these outputs:
- **Derived labels** (category, severity, priority, conditions) come from documented rules, not ground truth.
- **Risk:** the predicted value is an ML estimate, while the observed value is the dataset's own history of similar records.
- **Hypotheses** are not confirmed causes.

---

## 12. Repository structure

```
Jerom_Telecom/
├── architecture/      Architecture_Diagram.pdf / .png
├── design/            Design_Document.pdf
├── src/
│   ├── main.py        FastAPI application (uvicorn src.main:app)
│   ├── api/           routes, Pydantic schemas, structured errors
│   ├── agents/        6 agents, shared state, orchestrator
│   ├── rag/           embeddings, ChromaDB store, BM25, filters, query parser, metric similarity,
│   │                  hybrid retriever, RAG pipeline, LLM provider, narrative, grounding gate, LLM judge
│   ├── services/      classification, risk, root cause, recommendations, evidence, provenance, report,
│   │                  guardrails, feedback, models/ (fault risk, throughput, classifier, training)
│   ├── preprocessing/ download, inspect, validate, clean, conditions, taxonomy, severity, priority,
│   │                  chunking, index build, build_all
│   ├── utils/         configuration loading, text helpers, state names
│   └── config/        every threshold, weight and rule (YAML)
├── frontend.py        Streamlit front end
├── web/               optional React + TypeScript web UI
├── data/
│   └── sample_data/   500 raw rows, 200 processed records, sample queries with outputs
├── docs/              API_Documentation.pdf, Dataset_Details.pdf, Evaluation_Report.pdf,
│                      Project_Guide.pdf, Technology_Primer.pdf, evaluation/ and model reports
├── tests/             pytest suites per phase + evaluation/ (cases + DeepEval runner)
├── presentation/      Project_Presentation.pptx
├── requirements.txt   .env.example   .gitignore   README.md
```

`data/raw/`, `data/processed/` and `data/feedback/` are generated locally and git-ignored.

## 13. Testing and evaluation

```bash
python -m pytest                                  # all automated tests
python -m tests.evaluation.run_eval               # deterministic evaluation (no API key needed)
python -m tests.evaluation.run_eval --deepeval    # + DeepEval metrics, NVIDIA model as judge
```

- **Tests:** the suite includes the five required proofs: no raw value fabricated, filters never violated, retrieved IDs are real, the LLM cannot add unsupported numbers, and no target leakage.
- **Evaluation set:** 8 hand-made cases cover the specification's scenarios. Results are in [docs/Evaluation_Report.pdf](docs/Evaluation_Report.pdf) and `docs/evaluation/`.

## 14. Limitations

- **Synthetic data:** uniform distributions and random locations, with no timestamps, cell IDs, neighbour lists or transport counters. Root causes therefore remain **hypotheses**, and no time-series forecasting is done.
- **Derived labels:** categories, severity and priority are derived by rules. Confidence is a documented heuristic, not a calibrated probability.
- **Small evaluation set:** the set is small and one retrieval weight was tuned on it, so the scores are development scores.
- **LLM judge:** the smaller LLM judge can raise false alarms. Its verdict is combined with deterministic checks.

## 15. Further documentation

- [docs/PROJECT_DETAILS.md](docs/PROJECT_DETAILS.md): the full technical reference (phases, data, features, retrieval, models, agents, API).
- [docs/Project_Guide.pdf](docs/Project_Guide.pdf): the complete project guide, phase by phase, with real worked examples.
- [docs/Technology_Primer.pdf](docs/Technology_Primer.pdf): every model and method explained for newcomers.
- `docs/*.md`: architecture, data dictionary, feature engineering, retrieval and RAG, models, evaluation, design decisions, troubleshooting.
