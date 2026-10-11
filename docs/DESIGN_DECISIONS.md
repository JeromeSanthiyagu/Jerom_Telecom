# Design Decisions

Each decision lists what was chosen, why, and what was rejected.

| # | Decision | Why | Rejected alternative |
|---|---|---|---|
| 1 | **The raw CSV is the source of truth**, pinned by revision + SHA-256 | Results are reproducible, and a silent upstream change fails loudly | Downloading "latest" on every run |
| 2 | **Never impute, clip or round a measurement**; invalid → missing; parse floats with Python's exact `float()` | Imputing invents data. `pd.to_numeric` was off by 1 ULP for ~15 % of values | Mean/median imputation, clipping to ranges |
| 3 | **Worst-decile thresholds**, domain threshold only for signal (−110 dBm) | Phase 1: metrics are uniform in narrow ranges, so operator rules of thumb flag 80–88 % of rows | Fixed 3GPP-style KPI targets for every metric |
| 4 | **Score-based taxonomy** with `requires_any` + weights | Condition pairs co-occur at chance level (lift ≈ 1), so AND-rules would give empty categories | "low_handover AND high_call_drop → Mobility" rules |
| 5 | **Points-based severity** (half points at the threshold, full at P99) + overrides | Explainable, monotone, configurable; one extreme value is not diluted | Weighted average of all 18 indicators (almost always tiny) |
| 6 | **Priority separate from severity** (users, customer impact, availability, risk) | How bad vs how urgent are different questions | One combined score |
| 7 | **Two texts per record**: full `search_text` for BM25/LLM, qualitative `embedding_text` for vectors | 256-token embedding limit; numbers carry little semantic meaning; 27k unique texts → 3.7× faster embedding | Embedding the full numeric text |
| 8 | **ChromaDB** with `record_id` as the vector id and condition booleans as metadata | Persistent; filtering via `where`; no separate id-mapping table to drift | FAISS + an id map (the user asked for Chroma) |
| 9 | **Own sparse BM25** (verified against the formula in a test) | Persistable `.npz`, ~ms queries over 100k docs; `rank_bm25` rebuilds in memory every start-up | `rank_bm25` at runtime |
| 10 | **Exact re-scoring of the candidate union** across all methods before combining | Scores are comparable; no "missing score = 0" artefacts | Merging the methods' top-k lists by rank |
| 11 | **Final filter post-check** on every result | Guarantees the "never violate explicit filters" rule even if an index misbehaves | Trusting the vector store's filtering alone |
| 12 | **The LLM only explains**; facts, hypotheses and actions are deterministic | Hypotheses and actions are testable and reproducible; the LLM cannot invent them | Asking the LLM for the root cause |
| 13 | **Deterministic grounding gate** on LLM text + 1 retry + template fallback | Catches invented numbers, IDs and specifics regardless of how convincing the text sounds | Trusting the LLM judge alone |
| 14 | **Offline provider** | Tests, demos and CI work without an API key; the LLM is an enhancement, not a dependency | A hard dependency on the API |
| 15 | **Regression** for fault risk, then bands | The target is continuous; bands from P25/P75/P90 of the historical rate | Classifying arbitrary bins |
| 16 | **Three feature sets** (formula demo / primary / strict) and a code-level leakage guard | Makes the formulaic target visible and defends the feature choice in front of a panel | Silently using all columns (R² 0.9998) |
| 17 | **HistGradientBoosting served** | As accurate as the alternatives and handles missing inputs natively (queries rarely state every KPI) | Linear regression (needs imputation) |
| 18 | **Classifier labelled as rule-learnability**, rules stay authoritative | There are no ground-truth labels; claiming accuracy against the rules as real accuracy would mislead | Presenting 97 % as real-world accuracy |
| 19 | **Heuristic hypothesis confidence**, documented formula | Transparent and explainable; no calibration data exists | Pseudo-probabilities from an LLM |
| 20 | **USER_INPUT provenance type** added | Operator statements are not dataset facts | Labelling them RAW_DATA |
| 21 | **Linear agent orchestrator** sharing step functions with the pipeline | Six sequential steps need no graph engine; a test proves agents = pipeline | LangGraph with autonomous loops |
| 22 | **Feedback → review → evaluation set**, no automatic retraining | One rating is weak evidence and could poison the models | Online learning from 👍/👎 |
| 23 | **Every threshold, weight and rule in YAML** | A panel or engineer can change a rule without touching code | Constants spread through the code |
| 24 | **Hybrid metric weight 0.6** (tuned on the dev set) | The explicit condition signal should dominate when stated; documented as tuned | Keeping 0.3 (load + throughput precision@5 was 0.2) |

## Known limitations
See the README's Limitations section and `src/rag/pipeline.py::LIMITATIONS`, which are shown in every analysis.
