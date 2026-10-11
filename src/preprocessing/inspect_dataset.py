"""Phase 1 - inspect the raw 5G NetOps dataset.

Read-only analysis: nothing here cleans, labels or modifies the data. Every
number in docs/PHASE1_DATASET_INSPECTION.md is produced by this script and
written to docs/phase1/.

Usage:
    python -m src.preprocessing.download_dataset
    python -m src.preprocessing.inspect_dataset
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from src.preprocessing.download_dataset import RAW_PATH, download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "docs" / "phase1"
SEED = 42

TARGET = "Fault Occurrence Rate (%)"
CATEGORICAL = ["Season", "Weather", "Issue Reported", "City", "State"]
LOCATION = ["City", "State", "Zip"]

# Direction in which each KPI gets *worse*. Used only to look at tails and
# co-occurrence of adverse values; thresholds are decided in Phase 3.
ADVERSE_DIRECTION = {
    "Cell Availability (%)": "low",
    "MTTR (hours)": "high",
    "Throughput (Mbps)": "low",
    "Latency (ms)": "high",
    "Packet Loss Rate (%)": "high",
    "Call Drop Rate (%)": "high",
    "Handover Success Rate (%)": "low",
    "Alarm Count": "high",
    "Critical Alarm Count": "high",
    "Successful Configuration Changes (%)": "low",
    "User Count": "high",
    "Signal Strength (dBm)": "low",
    "Jitter (ms)": "high",
    "Connection Setup Success Rate (%)": "low",
    "Security Incidents": "high",
    "Authentication Failures": "high",
    TARGET: "high",
}

# Commonly quoted engineering rules of thumb, used here ONLY to test whether a
# fixed domain threshold would be discriminative on this dataset. They are not
# 3GPP-mandated limits; final thresholds are chosen and documented in Phase 3.
DOMAIN_REFERENCE = {
    "Cell Availability (%)": ("<", 99.0),
    "Latency (ms)": (">", 100.0),
    "Packet Loss Rate (%)": (">", 2.0),
    "Call Drop Rate (%)": (">", 2.0),
    "Handover Success Rate (%)": ("<", 98.0),
    "Signal Strength (dBm)": ("<", -110.0),
    "Jitter (ms)": (">", 30.0),
    "Connection Setup Success Rate (%)": ("<", 98.0),
}

# Plausible physical/logical ranges for the "impossible value" check.
VALID_RANGE = {
    "Cell Availability (%)": (0, 100),
    "Packet Loss Rate (%)": (0, 100),
    "Call Drop Rate (%)": (0, 100),
    "Handover Success Rate (%)": (0, 100),
    "Successful Configuration Changes (%)": (0, 100),
    "Connection Setup Success Rate (%)": (0, 100),
    "Humidity (%)": (0, 100),
    TARGET: (0, 100),
    "Signal Strength (dBm)": (-140, -44),  # LTE/NR RSRP reporting range
    "MTTR (hours)": (0, None),
    "Throughput (Mbps)": (0, None),
    "Latency (ms)": (0, None),
    "Jitter (ms)": (0, None),
    "Alarm Count": (0, None),
    "Critical Alarm Count": (0, None),
    "Parameter Changes": (0, None),
    "Data Usage (GB)": (0, None),
    "User Count": (0, None),
    "Security Incidents": (0, None),
    "Authentication Failures": (0, None),
    "Temperature (°C)": (-60, 60),
}

# Chart palette (dataviz reference palette).
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SERIES_1 = "#2a78d6"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9c2120"]
)


def r(x, nd=4):
    """Round floats for JSON output."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes("number").columns if c != "Zip"]


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric + one-hot(Season, Weather, Issue) matrix, location excluded."""
    return pd.get_dummies(
        df.drop(columns=LOCATION),
        columns=["Season", "Weather", "Issue Reported"],
        drop_first=True,
    ).astype(float)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def schema(df: pd.DataFrame) -> list[dict]:
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_unique": int(s.nunique()),
                "missing": int(s.isna().sum()),
                "examples": [v.item() if hasattr(v, "item") else v for v in s.dropna().unique()[:3]],
            }
        )
    return rows


def missing_and_text_quality(df: pd.DataFrame) -> dict:
    text_cols = df.select_dtypes("object").columns
    return {
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "total_missing_cells": int(df.isna().sum().sum()),
        "empty_or_blank_strings": {
            c: int((df[c].str.strip() == "").sum()) for c in text_cols
        },
        "leading_or_trailing_whitespace": {
            c: int((df[c] != df[c].str.strip()).sum()) for c in text_cols
        },
        "case_variants": {
            c: int(df[c].nunique() - df[c].str.lower().nunique()) for c in text_cols
        },
    }


def duplicates(df: pd.DataFrame) -> dict:
    metrics_only = df.drop(columns=LOCATION)
    return {
        "full_row_duplicates": int(df.duplicated().sum()),
        "duplicates_ignoring_location": int(metrics_only.duplicated().sum()),
        "duplicates_ignoring_location_and_categoricals": int(
            df[numeric_columns(df)].duplicated().sum()
        ),
    }


def numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    num = df[numeric_columns(df)]
    stats = num.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]).T
    stats["skew"] = num.skew()
    stats["excess_kurtosis"] = num.kurt()
    stats["n_unique"] = num.nunique()
    stats["share_integer_valued"] = (num % 1 == 0).mean()
    return stats


def categorical_distributions(df: pd.DataFrame) -> dict:
    out = {}
    for c in ["Season", "Weather", "Issue Reported"]:
        vc = df[c].value_counts()
        out[c] = {k: {"count": int(v), "share": r(v / len(df))} for k, v in vc.items()}
    states = df["State"].value_counts()
    us_50_plus_dc = set(
        "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV "
        "NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC".split()
    )
    out["State"] = {
        "n_unique": int(states.size),
        "min_count": int(states.min()),
        "max_count": int(states.max()),
        "codes_outside_50_states_plus_DC": sorted(set(states.index) - us_50_plus_dc),
    }
    out["City"] = {
        "n_unique": int(df["City"].nunique()),
        "cities_appearing_in_more_than_one_state": int((df.groupby("City")["State"].nunique() > 1).sum()),
        "top_5": {k: int(v) for k, v in df["City"].value_counts().head(5).items()},
    }
    out["Zip"] = {
        "n_unique": int(df["Zip"].nunique()),
        "zips_appearing_in_more_than_one_state": int((df.groupby("Zip")["State"].nunique() > 1).sum()),
    }
    out["Season_x_Weather"] = pd.crosstab(df["Season"], df["Weather"]).to_dict()
    return out


def correlations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    num = df[numeric_columns(df)].copy()
    num["Issue Reported (yes=1)"] = (df["Issue Reported"] == "yes").astype(int)
    pearson, spearman = num.corr(), num.corr("spearman")
    upper = np.triu(np.ones(pearson.shape, dtype=bool), 1)
    pairs = pearson.where(upper).stack()
    top = pairs[pairs.abs() >= 0.1].sort_values(key=abs, ascending=False)
    top_pairs = [
        {"a": a, "b": b, "pearson": r(v, 3), "spearman": r(spearman.loc[a, b], 3)}
        for (a, b), v in top.items()
    ]
    return pearson, spearman, top_pairs


def outliers_and_validity(df: pd.DataFrame) -> dict:
    out: dict = {"per_column": {}, "logical_checks": {}}
    for c in numeric_columns(df):
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        z = (s - s.mean()) / s.std()
        lo, hi = VALID_RANGE.get(c, (None, None))
        out["per_column"][c] = {
            "iqr_outliers_1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "abs_z_gt_3": int((z.abs() > 3).sum()),
            "share_at_min": r((s == s.min()).mean()),
            "share_at_max": r((s == s.max()).mean()),
            "below_valid_range": int((s < lo).sum()) if lo is not None else 0,
            "above_valid_range": int((s > hi).sum()) if hi is not None else 0,
        }
    out["logical_checks"] = {
        "critical_alarms_gt_alarm_count": int((df["Critical Alarm Count"] > df["Alarm Count"]).sum()),
        "critical_alarm_count_non_integer": int((df["Critical Alarm Count"] % 1 != 0).sum()),
        "snow_outside_winter": int(((df["Weather"] == "Snow") & (df["Season"] != "Winter")).sum()),
        "snow_with_temperature_above_5C": int(((df["Weather"] == "Snow") & (df["Temperature (°C)"] > 5)).sum()),
        "authentication_failure_values_by_weather": {
            k: sorted(int(x) for x in v.unique())
            for k, v in df.groupby("Weather")["Authentication Failures"]
        },
        "temperature_range_by_season": {
            k: [r(v["min"], 2), r(v["max"], 2)]
            for k, v in df.groupby("Season")["Temperature (°C)"].agg(["min", "max"]).iterrows()
        },
    }
    return out


def dependency_structure(df: pd.DataFrame) -> dict:
    """How much of each column is explained by the other columns?

    A near-1 R^2 means the column is (almost) a function of other columns -
    critical to know before choosing ML targets and features.
    """
    X = model_matrix(df)
    sample = X.sample(30_000, random_state=SEED)
    explained = {}
    for c in numeric_columns(df):
        others = [x for x in sample.columns if x != c]
        lin = cross_val_score(LinearRegression(), sample[others], sample[c], cv=3, scoring="r2").mean()
        gbm = cross_val_score(
            HistGradientBoostingRegressor(max_iter=200, random_state=SEED),
            sample[others], sample[c], cv=3, scoring="r2",
        ).mean()
        explained[c] = {"linear_r2": r(lin), "gbm_r2": r(gbm)}

    # Explicit linear fit of the target on the columns it appears to be built from.
    # Rows with target == 0 are excluded because the target looks clipped at 0.
    formula_cols = [
        "Critical Alarm Count", "Authentication Failures", "Security Incidents",
        "Call Drop Rate (%)", "Alarm Count", "Successful Configuration Changes (%)",
        "Parameter Changes",
    ]
    mask = df[TARGET] > 0.5
    lr = LinearRegression().fit(df.loc[mask, formula_cols], df.loc[mask, TARGET])
    resid = df.loc[mask, TARGET] - lr.predict(df.loc[mask, formula_cols])
    target_formula = {
        "intercept": r(lr.intercept_),
        "coefficients": {c: r(v) for c, v in zip(formula_cols, lr.coef_)},
        "residual_std": r(resid.std()),
        "r2": r(1 - resid.var() / df.loc[mask, TARGET].var()),
    }

    kpis = [
        "Throughput (Mbps)", "Latency (ms)", "Packet Loss Rate (%)", "Call Drop Rate (%)",
        "Authentication Failures", "Critical Alarm Count", TARGET,
        "Handover Success Rate (%)", "Signal Strength (dBm)", "Cell Availability (%)",
        "Jitter (ms)", "User Count",
    ]
    by_weather = df.groupby("Weather")[kpis].mean().round(3)
    by_issue = df.groupby("Issue Reported")[kpis].mean().round(3)

    # Can "Issue Reported" be predicted from everything else? (AUC 0.5 = no signal)
    y_issue = (df["Issue Reported"] == "yes").astype(int)
    X_issue = X.drop(columns=[c for c in X.columns if c.startswith("Issue Reported")])
    Xtr, Xte, ytr, yte = train_test_split(X_issue, y_issue, test_size=0.2, random_state=SEED)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=SEED).fit(Xtr, ytr)
    issue_auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    return {
        "r2_of_each_column_from_all_others": explained,
        "target_linear_formula": target_formula,
        "mean_kpis_by_weather": by_weather.to_dict(),
        "mean_kpis_by_issue_reported": by_issue.to_dict(),
        "issue_reported_auc_from_all_other_columns": r(issue_auc),
        "row_order_lag1_autocorrelation": {
            c: r(df[c].autocorr(1)) for c in [TARGET, "Throughput (Mbps)", "Latency (ms)"]
        },
    }


def target_feasibility(df: pd.DataFrame) -> dict:
    """Hold-out R^2 for candidate prediction targets under different feature sets."""
    X = model_matrix(df)
    context = [c for c in X.columns if c.startswith(("Season_", "Weather_"))] + [
        "Temperature (°C)", "Humidity (%)",
    ]
    radio_transport = [
        "Latency (ms)", "Packet Loss Rate (%)", "Jitter (ms)", "Signal Strength (dBm)",
        "Handover Success Rate (%)", "Cell Availability (%)",
        "Connection Setup Success Rate (%)", "User Count", "Data Usage (GB)",
    ]
    formula_components = [
        "Critical Alarm Count", "Authentication Failures", "Security Incidents",
        "Call Drop Rate (%)", "Alarm Count", "Successful Configuration Changes (%)",
        "Parameter Changes",
    ]
    feature_sets = {
        TARGET: {
            "A_all_columns": [c for c in X.columns if c != TARGET],
            "B_all_except_MTTR": [c for c in X.columns if c not in (TARGET, "MTTR (hours)")],
            "C_context_plus_radio_transport_plus_throughput": context + radio_transport + ["Throughput (Mbps)"],
            "D_context_plus_radio_transport": context + radio_transport,
            "E_context_only": context,
        },
        "Throughput (Mbps)": {
            "A_all_columns": [c for c in X.columns if c != "Throughput (Mbps)"],
            "D_context_plus_radio_transport": context + radio_transport,
            "E_context_only": context,
        },
        "Latency (ms)": {
            "D_context_plus_other_radio_transport": context + [c for c in radio_transport if c != "Latency (ms)"],
        },
        "Handover Success Rate (%)": {
            "A_all_columns": [c for c in X.columns if c != "Handover Success Rate (%)"],
        },
        "Call Drop Rate (%)": {
            "D_context_plus_radio_transport": context + radio_transport,
        },
    }
    out = {"feature_set_definitions": {
        "context": context, "radio_transport": radio_transport,
        "fault_formula_components": formula_components,
    }}
    for target, sets in feature_sets.items():
        y = X[target]
        res = {}
        for name, cols in sets.items():
            Xtr, Xte, ytr, yte = train_test_split(X[cols], y, test_size=0.2, random_state=SEED)
            lin = LinearRegression().fit(Xtr, ytr)
            gbm = HistGradientBoostingRegressor(max_iter=300, random_state=SEED).fit(Xtr, ytr)
            res[name] = {
                "n_features": len(cols),
                "linear_r2": r(lin.score(Xte, yte)),
                "gbm_r2": r(gbm.score(Xte, yte)),
                "gbm_mae": r(np.abs(gbm.predict(Xte) - yte).mean(), 3),
            }
        out[target] = res
    return out


def condition_candidates(df: pd.DataFrame) -> dict:
    """Tail shares under domain references vs. percentiles, plus co-occurrence lift."""
    domain = {}
    for c, (op, thr) in DOMAIN_REFERENCE.items():
        flag = df[c] < thr if op == "<" else df[c] > thr
        domain[c] = {"rule": f"{op} {thr}", "share_flagged": r(flag.mean())}

    flags = {}
    for c, direction in ADVERSE_DIRECTION.items():
        if direction == "high":
            thr = df[c].quantile(0.90)
            flags[c] = df[c] > thr
        else:
            thr = df[c].quantile(0.10)
            flags[c] = df[c] < thr
    flag_df = pd.DataFrame(flags)
    p = flag_df.mean()
    lifts = []
    for a, b in combinations(flag_df.columns, 2):
        joint = (flag_df[a] & flag_df[b]).mean()
        expected = p[a] * p[b]
        lifts.append({"a": a, "b": b, "joint_share": r(joint, 5), "lift": r(joint / expected, 2)})
    lifts.sort(key=lambda d: -d["lift"])
    n_flags = flag_df.sum(axis=1).value_counts().sort_index()
    return {
        "domain_reference_share_flagged": domain,
        "worst_decile_flag_share": {c: r(v) for c, v in p.items()},
        "worst_decile_cooccurrence_lift_top": lifts[:15],
        "worst_decile_cooccurrence_lift_bottom": lifts[-5:],
        "rows_by_number_of_worst_decile_flags": {int(k): int(v) for k, v in n_flags.items()},
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK_2, labelsize=7)


def plot_histograms(df: pd.DataFrame, path: Path) -> None:
    cols = numeric_columns(df)
    ncols = 4
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.4 * nrows), facecolor=SURFACE)
    for ax, c in zip(axes.flat, cols):
        ax.hist(df[c], bins=60, color=SERIES_1, edgecolor=SURFACE, linewidth=0.4)
        ax.set_title(c, fontsize=8.5, color=INK, loc="left")
        ax.set_yticks([])
        _style(ax)
    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    fig.suptitle("Distribution of every numeric column (100,000 rows, 60 bins)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_correlation(pearson: pd.DataFrame, path: Path) -> None:
    labels = [c.split(" (")[0] for c in pearson.columns]
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=SURFACE)
    im = ax.imshow(pearson.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(labels)), labels, fontsize=8, color=INK_2)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = pearson.values[i, j]
            if i != j and abs(v) >= 0.1:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="#ffffff" if abs(v) > 0.6 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.tick_params(labelsize=8, colors=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title("Pearson correlation (|r| >= 0.10 labelled)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_weather_effect(df: pd.DataFrame, path: Path) -> None:
    kpis = ["Throughput (Mbps)", "Latency (ms)", "Packet Loss Rate (%)", "Call Drop Rate (%)",
            "Authentication Failures", "Critical Alarm Count", TARGET,
            "Handover Success Rate (%)"]
    order = ["Clear", "Clouds", "Rain", "Snow"]
    means = df.groupby("Weather")[kpis].mean().loc[order]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), facecolor=SURFACE)
    for ax, c in zip(axes.flat, kpis):
        vals = means[c]
        ax.bar(order, vals, color=SERIES_1, width=0.6, edgecolor=SURFACE, linewidth=2)
        ax.set_ylim(0, vals.max() * 1.15)
        for x, v in zip(order, vals):
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_title(c, fontsize=9, color=INK, loc="left")
        _style(ax)
    fig.suptitle("Mean KPI by Weather - the dominant driver in this dataset",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> dict:
    download()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH)  # pandas decompresses .xz transparently

    print("Running inspection ...")
    stats = numeric_stats(df)
    pearson, spearman, top_pairs = correlations(df)

    summary = {
        "source": {
            "file": RAW_PATH.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "memory_mb": r(df.memory_usage(deep=True).sum() / 1e6, 1),
        },
        "schema": schema(df),
        "missing_and_text_quality": missing_and_text_quality(df),
        "duplicates": duplicates(df),
        "categorical": categorical_distributions(df),
        "top_correlated_pairs": top_pairs,
        "outliers_and_validity": outliers_and_validity(df),
    }
    print("  dependency structure ...")
    summary["dependency_structure"] = dependency_structure(df)
    print("  target feasibility ...")
    summary["target_feasibility"] = target_feasibility(df)
    summary["condition_candidates"] = condition_candidates(df)

    stats.round(4).to_csv(OUT_DIR / "numeric_stats.csv")
    pearson.round(4).to_csv(OUT_DIR / "correlation_pearson.csv")
    spearman.round(4).to_csv(OUT_DIR / "correlation_spearman.csv")
    with (OUT_DIR / "inspection_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)

    plot_histograms(df, OUT_DIR / "fig_histograms.png")
    plot_correlation(pearson, OUT_DIR / "fig_correlation.png")
    plot_weather_effect(df, OUT_DIR / "fig_weather_effect.png")

    print(f"Wrote outputs to {OUT_DIR}")
    return summary


if __name__ == "__main__":
    main()
