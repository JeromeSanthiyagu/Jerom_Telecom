# ML Models (Phases 19–22)

```bash
python -m src.services.models.train      # ~2 minutes on a laptop CPU
```

| Item | Location |
|---|---|
| Configuration | [`src/config/models.yaml`](../src/config/models.yaml) |
| Code | [`src/services/models/`](../src/services/models/) |
| Saved models | `data/processed/models/*.joblib` |
| Full metric reports | [`docs/models/*.json`](models/) |

Split: 80 / 20 random train/test, seed 42. A random split is valid here: there are no duplicates, no cell IDs and no time order (Phase 1).

## 1. Network fault-risk model (Phase 19)

**Target:** `fault_rate_pct` is continuous (99,976 distinct values), so the task is **regression**. Binning first would discard information. The *predicted* rate is then mapped to risk bands taken from the historical distribution:

| Risk level | Predicted fault occurrence rate | Source |
|---|---|---|
| Low | < 28.34 % | P25 |
| Medium | 28.34 – 48.15 % | P25–P75 |
| High | 48.15 – 55.51 % | P75–P90 |
| Critical | ≥ 55.51 % | P90 (= `high_fault_rate` threshold) |

Two kinds of number are always kept apart:
- **Predicted fault occurrence risk** — the model output, labelled *ML prediction*.
- **Historical fault occurrence rate** — the value recorded in a dataset row, or a statistic over retrieved records, labelled *observed, not predicted*.

### Feature sets and results (test set, 20,000 rows)

| Feature set | Features | Linear R² | Random Forest R² | Gradient Boosting R² | GB MAE / RMSE |
|---|---|---|---|---|---|
| **C_primary (served)** | season, weather, temperature, humidity, latency, packet loss, jitter, signal, handover, availability, setup success, user count, data usage, **throughput** | 0.8848 | 0.8831 | **0.8849** | 3.72 / 4.77 pp |
| D_strict | as C without throughput | 0.4800 | 0.4734 | 0.4783 | 8.39 / 10.15 pp |
| A_formula_demo | every column except the target and derived labels | **0.9998** | — | 0.9997 | 0.19 / 0.26 pp |

- **Served model:** HistGradientBoosting on C_primary. It handles **missing inputs natively**, which matters because queries rarely state every KPI.
- **Stability:** 5-fold CV R² 0.8848 ± 0.0006.
- **Prediction interval:** an 80 % interval is derived from the test residuals (P10 −5.9 pp, P90 +5.9 pp).

**Why three sets?** Set A proves that the target is almost a formula of other columns. Set D shows how much is predictable from context and radio/transport KPIs alone. Throughput mirrors critical alarms (r = −0.958), so including it (set C) is a documented choice, not an accident.

## 2. Data-leakage prevention (Phase 20)

`src/services/models/common.py::check_leakage` runs on every feature list at training time and raises `LeakageError` if any of the following appears:

| Excluded | Why |
|---|---|
| `fault_rate_pct` | the target |
| `critical_alarm_count`, `auth_failures`, `security_incidents`, `alarm_count`, `call_drop_pct`, `config_success_pct`, `parameter_changes` | components of the target's linear formula (R² 0.9992, Phase 1) |
| `mttr_hours` | post-incident: repair time is only known after the fault |
| `cond_*`, `conditions`, `n_conditions`, `primary_category`, `severity*`, `priority*`, `search_text`, `embedding_text`, `dq_*` | derived labels; `high_fault_rate`, category and severity are built *from* the target |
| `city`, `state`, `zip`, `record_id`, `source_row` | identifiers / random locations: the model would memorise instead of generalising |
| `issue_reported` | no relationship with any metric (AUC 0.506) |

Only the A_formula_demo set may use the formula components, and it is never served. Tests prove the guard rejects every forbidden column and that the saved model's features contain none of them.

Unknown `season` / `weather` values are encoded as **missing** in every one-hot column, not as the reference category, so "not stated" never silently means "Fall" or "Clear".

## 3. Performance (KPI) prediction (Phase 21)

**Target: throughput_mbps.** Phase 1 showed it is the only KPI with a meaningful, non-leaking signal. Latency (R² 0.13), call drop (0.19) and handover success (0.00) are not predictable from the other columns. No time-series model is used because there are no timestamps.

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Mean predictor (baseline) | 197.3 | 239.7 | 0.00 |
| Linear Regression | 161.5 | 196.1 | 0.331 |
| Random Forest | 162.1 | 197.1 | 0.324 |
| **Gradient Boosting (served)** | 161.8 | 196.5 | 0.328 |

Features are context plus radio/transport KPIs. `critical_alarm_count` (throughput's mirror) and `fault_rate_pct` are excluded. Explanatory power is moderate and comes mostly from weather, so predictions are shown as indicative ranges.

## 4. Incident classification model (Phase 22)

The dataset has **no** category labels. The classifier is trained on the **rule-derived** `primary_category`, so its scores measure **agreement with the rules**, not real-world accuracy. The rule-based category stays authoritative in the application. The ML prediction is shown beside it as a consistency check and provides class probabilities when inputs are partial.

**Features:** radio/transport KPIs, throughput, alarm counts, configuration and security metrics, and fault rate. Using fault rate is allowed here because the target is the category, not fault risk.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| Logistic Regression | 0.651 | 0.652 | 0.657 | 0.651 |
| Random Forest | 0.972 | 0.975 | 0.978 | 0.976 |
| **Gradient Boosting (served)** | **0.974** | 0.974 | 0.978 | **0.976** |

Per-class F1 (Gradient Boosting) ranges from 0.953 (Service Degradation) to 0.992 (Configuration). The full confusion matrix is in `docs/models/incident_classifier.json`.

**Interpretation:** the rules are step functions (thresholds) with score-based tie-breaks. Tree models learn them almost exactly, but a linear model cannot represent thresholds, hence its 0.65.
