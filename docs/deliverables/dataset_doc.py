"""docs/Dataset_Details.pdf — dataset name, source, link, format, key fields, purpose and preprocessing.

    python -m docs.deliverables.dataset_doc
"""

from __future__ import annotations

import json

import pandas as pd

from docs.deliverables.kit import P, PageBreak, abridge, build, cover, mono, section, table
from docs.guide.pdf_kit import bullets, callout, code
from src.preprocessing.conditions import load_conditions
from src.preprocessing.download_dataset import REPO_ID, REVISION, SHA256
from src.utils.config import PROJECT_ROOT, load_schema

OUT = PROJECT_ROOT / "docs" / "Dataset_Details.pdf"
LINK = f"https://huggingface.co/datasets/{REPO_ID}"


def story() -> list:
    stats = pd.read_csv(PROJECT_ROOT / "docs" / "phase1" / "numeric_stats.csv", index_col=0)
    insp = json.loads((PROJECT_ROOT / "docs" / "phase1" / "inspection_summary.json").read_text(encoding="utf-8"))
    clean = json.loads((PROJECT_ROOT / "docs" / "phase2" / "cleaning_report.json").read_text(encoding="utf-8"))
    cols = load_schema()["columns"]
    conds = load_conditions()
    inc = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "incidents.parquet")

    s = cover("Dataset Details", "Dataset documentation (submission deliverable 6)", [
        ["Dataset name", "5G Network Operations dataset (5G_netops_data_100K)"],
        ["Source", f"Hugging Face Hub, dataset {mono(REPO_ID)}"],
        ["Link", LINK],
        ["Format", "CSV compressed with xz (5G_netops_data_100K.csv.xz), 100,000 rows × 27 columns"],
        ["Licence", "Apache-2.0 (as published on the dataset page)"],
        ["Pinned version", f"revision {REVISION[:12]}…, SHA-256 {SHA256[:16]}…"],
    ])

    s += section("1. Dataset name, source and link")
    s += [table([
        ["Field", "Value"],
        ["Dataset name", "5G Network Operations dataset — file 5G_netops_data_100K.csv.xz"],
        ["Dataset source", "Hugging Face Hub, publisher fenar (public dataset)"],
        ["Dataset link", LINK],
        ["Exact version used", f"commit {mono(REVISION)}"],
        ["Integrity check", f"SHA-256 {mono(SHA256)} — verified on every download"],
        ["How it is obtained", f"{mono('python -m src.preprocessing.download_dataset')} downloads the pinned file to "
                               f"{mono('data/raw/')}; the raw file is never modified"],
        ["Sample in the repository", f"{mono('data/sample_data/5G_netops_sample_500.csv')} — the first 500 rows, "
                                     "byte-for-byte as in the dataset"],
    ], widths=[1.5, 4.5], bold_first_col=True)]

    s += section("2. Data format")
    s += [table([
        ["Property", "Value"],
        ["File type", "Comma-separated values (CSV), header row, UTF-8, xz-compressed (6.3 MB; 44.8 MB in memory)"],
        ["Rows × columns", f"{insp['source']['rows']:,} × {insp['source']['columns']}"],
        ["Missing values", "0 in every column"],
        ["Duplicates", f"{insp['duplicates']['full_row_duplicates']} exact duplicate rows"],
        ["Unit of observation", "one row = one network-performance observation for a cell/site: KPIs, alarms, "
                                "configuration, security counters, weather context and a location"],
        ["Identifiers", "none in the source; the project adds record_id ROW-000001 … ROW-100000 in file order"],
        ["Time information", "season only — no timestamps, so no time-series analysis is possible"],
    ], widths=[1.5, 4.5], bold_first_col=True),
        P("Three real raw rows (selected columns):"),
        table([["Season", "Latency (ms)", "Packet Loss Rate (%)", "Signal Strength (dBm)", "Weather", "City, State",
                "Fault Occurrence Rate (%)"]] +
              [[r["season"], f"{r['latency_ms']:.3f}", f"{r['packet_loss_pct']:.4g}", f"{r['signal_strength_dbm']}",
                r["weather"], f"{r['city']}, {r['state']}", f"{r['fault_rate_pct']:.4f}"] for _, r in inc.head(3).iterrows()],
              widths=[0.8, 0.9, 1.1, 1.1, 0.8, 1.4, 1.2])]

    s += section("3. Key fields")
    s.append(P("All 27 columns. 'Name in project' is the canonical name used in the code "
               f"({mono('src/config/schema.yaml')}); statistics are computed on all 100,000 rows."))
    rows = [["Column in dataset", "Name in project", "Type", "Role", "Min", "Median", "Max"]]
    for c in cols:
        st = stats.loc[c["raw"]] if c["raw"] in stats.index else None
        rows.append([c["raw"], mono(c["name"]), c["type"], c.get("role", ""),
                     f"{st['min']:g}" if st is not None else "–", f"{st['50%']:g}" if st is not None else "–",
                     f"{st['max']:g}" if st is not None else "–"])
    s.append(table(rows, widths=[2.1, 1.7, 0.6, 0.85, 0.65, 0.7, 0.7]))
    s.append(P("Categorical fields: Season (Winter, Spring, Summer, Fall), Weather (Clear, Clouds, Rain, Snow), "
               "Issue Reported (yes/no), State (59 US states and territories), City and Zip (random locations)."))

    s += section("4. Dataset purpose in this project")
    s += bullets([
        "<b>Evidence base for retrieval-augmented analysis.</b> Each of the 100,000 records is a historical "
        "observation the assistant can retrieve as evidence when an engineer describes a problem.",
        "<b>Training data for the ML models.</b> Fault occurrence risk (regression on Fault Occurrence Rate), "
        "throughput (regression) and a category classifier.",
        "<b>Source of truth.</b> Every number shown to a user is either a raw value from this file or computed from "
        "it by a documented rule; the LLM is never allowed to introduce values that are not in the evidence.",
    ])
    s.append(callout("The dataset has no incident labels, free text, timestamps or cell identifiers. Categories, "
                     "severity and priority are therefore derived by documented rules and labelled as derived. "
                     "Phase 1 analysis shows the data is synthetic (uniform distributions, random locations, fault rate "
                     "almost a linear formula of seven columns), so root causes are reported as hypotheses.", "warn",
                     "Limits of the data"))

    s += section("5. Data preprocessing performed")
    s.append(P(f"Run with {mono('python -m src.preprocessing.build_all')} (steps 1–7) — "
               f"rows in {clean['rows_in']:,}, rows out {clean['rows_out']:,}, rows dropped {clean['rows_dropped']}."))
    s.append(table([
        ["Step", "What is done", "Result"],
        ["1. Download", "pinned revision, SHA-256 verification", "raw file in data/raw/, never modified"],
        ["2. Validate (raw)", "types, ranges (e.g. signal −140…−44 dBm, % within 0–100), allowed categories, "
                              "missing values, whitespace", "0 errors in 100,000 rows"],
        ["3. Clean", "snake_case names, typed values (exact float parsing), yes/no → boolean, ZIP as 5-character "
                     "text, exact duplicates dropped, no imputation", "data/processed/records.parquet"],
        ["4. Quality flags", "suspicious but valid patterns are flagged, never changed",
         ", ".join(f"{k} {v:,}" for k, v in clean["quality_flags"].items())],
        ["5. Record IDs", "stable ID per row in file order", "ROW-000001 … ROW-100000"],
        ["6. Derived features", f"{len(conds)} conditions from worst-decile thresholds (signal: −110 dBm domain "
                                "threshold); rule-based category, points-based severity, priority",
         "data/processed/incidents.parquet (62 columns)"],
        ["7. Chunking", "one chunk per record: search_text (full, for BM25) and embedding_text (qualitative, "
                        "for embeddings), built only from the record's own fields",
         f"{inc['embedding_text'].nunique():,} distinct embedding texts"],
        ["8. Indexing", "all-MiniLM-L6-v2 embeddings into ChromaDB; BM25 sparse matrix",
         "data/processed/chroma/, data/processed/bm25/"],
    ], widths=[1.1, 3.2, 1.9], bold_first_col=True))
    s.append(P("A test compares every processed value with the raw CSV text, proving preprocessing neither alters nor "
               "fabricates a measurement."))
    s.append(P("Derived conditions (first 8 of 18):", "h3"))
    s.append(table([["Condition", "Metric", "Rule", "Method"]] +
                   [[c.label, mono(c.metric), f"{c.operator} {c.threshold:g}", c.method] for c in list(conds.values())[:8]],
                   widths=[1.8, 1.8, 1.1, 1]))
    s.append(P("Example chunk (embedding_text) for ROW-000001:", "h3"))
    s.append(code(inc.loc[0, "embedding_text"]))
    return s


def main():
    return build(OUT, "Dataset Details", story())


if __name__ == "__main__":
    main()
