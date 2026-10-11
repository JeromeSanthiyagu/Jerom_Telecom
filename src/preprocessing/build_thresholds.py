"""Check (or report) that thresholds.yaml matches the processed data.

    python -m src.preprocessing.build_thresholds --check   # fail if any value drifted
    python -m src.preprocessing.build_thresholds           # print recomputed values + shares
"""

from __future__ import annotations

import sys

import pandas as pd

from src.preprocessing.conditions import condition_frame, load_conditions
from src.preprocessing.loader import load_processed
from src.utils.config import load_yaml


def recompute(df: pd.DataFrame) -> dict[str, dict]:
    cfg = load_yaml("thresholds.yaml")["conditions"]
    out = {}
    for name, c in cfg.items():
        s = df[c["metric"]].astype("float64")
        low_bad = c["operator"] in ("<", "<=")
        ext_q = 0.01 if low_bad else 0.99
        out[name] = {
            "threshold": round(float(s.quantile(c["percentile"] / 100)), 4) if c["method"] == "percentile" else c["threshold"],
            "extreme": round(float(s.quantile(ext_q)), 4),
            "configured_threshold": c["threshold"],
            "configured_extreme": c["extreme"],
        }
    return out


def main(check: bool) -> int:
    df = load_processed()
    values = recompute(df)
    shares = condition_frame(df, load_conditions()).mean()
    drift = []
    for name, v in values.items():
        flag = ""
        if abs(v["threshold"] - v["configured_threshold"]) > 1e-3 or abs(v["extreme"] - v["configured_extreme"]) > 1e-3:
            drift.append(name)
            flag = "  <-- DRIFT"
        print(f"{name:28s} threshold={v['configured_threshold']:>10g} extreme={v['configured_extreme']:>10g} "
              f"share={shares[name]:.4f}{flag}")
    if check and drift:
        print(f"Thresholds drifted from data: {drift}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(check="--check" in sys.argv))
