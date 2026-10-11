"""Build the derived incident table (Phases 3-8).

    records.parquet  ->  conditions -> category -> severity -> priority -> text
                     ->  data/processed/incidents.parquet

Usage:
    python -m src.preprocessing.build_incidents
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import pandas as pd

from src.preprocessing.conditions import condition_frame, condition_lists, load_conditions
from src.preprocessing.chunking import build_embedding_text, build_search_text
from src.preprocessing.loader import load_processed
from src.preprocessing.priority import priority_frame
from src.preprocessing.severity import severity_frame
from src.preprocessing.taxonomy import classify_conditions
from src.utils.config import DOCS_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

INCIDENTS_PATH = PROCESSED_DIR / "incidents.parquet"
SUMMARY_PATH = DOCS_DIR / "phase3_6" / "derived_label_summary.json"


@lru_cache(maxsize=None)
def _classify(conds: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    res = classify_conditions(conds)
    return res["primary_category"], tuple(res["secondary_categories"])


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add every derived column to a processed-records frame."""
    conditions = load_conditions()
    flags = condition_frame(df, conditions)
    lists = condition_lists(flags)
    cats = lists.map(lambda l: _classify(tuple(l)))
    sev = severity_frame(df)
    pr = priority_frame(df, sev, flags)

    pts = sev[[c for c in sev.columns if c.startswith("pts_")]]
    drivers = pts.apply(lambda r: [c[4:] for c, v in r.sort_values(ascending=False).items() if v > 0], axis=1)

    out = df.copy()
    out = pd.concat([out, flags.add_prefix("cond_")], axis=1)
    out["conditions"] = lists
    out["n_conditions"] = flags.sum(axis=1).astype("int64")
    out["primary_category"] = cats.map(lambda t: t[0])
    out["secondary_categories"] = cats.map(lambda t: list(t[1]))
    out["severity"] = sev["severity"].astype(str)
    out["severity_score"] = sev["severity_score"]
    out["severity_drivers"] = drivers
    out["priority"] = pr["priority"].astype(str)
    out["priority_score"] = pr["priority_score"]

    records = out.to_dict("records")
    out["search_text"] = [build_search_text(r) for r in records]
    out["embedding_text"] = [build_embedding_text(r) for r in records]
    return out


def summarize(inc: pd.DataFrame) -> dict:
    def dist(s):
        vc = s.value_counts()
        return {k: {"count": int(v), "share": round(v / len(s), 4)} for k, v in vc.items()}
    cond_cols = [c for c in inc.columns if c.startswith("cond_")]
    return {
        "rows": int(len(inc)),
        "condition_share": {c[5:]: round(float(inc[c].mean()), 4) for c in cond_cols},
        "conditions_per_record": {int(k): int(v) for k, v in inc["n_conditions"].value_counts().sort_index().items()},
        "primary_category": dist(inc["primary_category"]),
        "severity": dist(inc["severity"]),
        "priority": dist(inc["priority"]),
        "severity_x_priority": pd.crosstab(inc["severity"], inc["priority"]).to_dict(),
        "severity_score_percentiles": {f"p{int(q*100)}": float(inc["severity_score"].quantile(q))
                                       for q in (0.25, 0.5, 0.75, 0.9, 0.95, 0.99)},
    }


def main() -> pd.DataFrame:
    inc = derive(load_processed())
    INCIDENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    inc.to_parquet(INCIDENTS_PATH, index=False)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summarize(inc), indent=2), encoding="utf-8")
    logger.info("wrote %s (%d rows)", INCIDENTS_PATH, len(inc))
    return inc


def load_incidents() -> pd.DataFrame:
    if not INCIDENTS_PATH.exists():
        raise FileNotFoundError(f"{INCIDENTS_PATH} not found - run `python -m src.preprocessing.build_incidents`.")
    return pd.read_parquet(INCIDENTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    inc = main()
    print(inc[["record_id", "conditions", "primary_category", "severity", "priority"]].head())
    print()
    print(inc.loc[0, "search_text"])
    print()
    print(inc.loc[0, "embedding_text"])
