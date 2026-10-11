"""Figures for the technology primer.

Charts marked REAL are computed from the project's own data, models and
configuration; diagrams marked ILLUSTRATION explain a concept and contain no
measurements.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from docs.guide.figures import C1, C2, C3, C4, GRID, INK, INK2, SURFACE, _save, _style
from src.utils.config import PROJECT_ROOT

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
LIGHT = {C1: "#e6f0fb", C2: "#fdece4", C3: "#e3f5ee", C4: "#fdf3dc", "#4a3aa7": "#ebe8f7", "#52514e": "#f0efec"}


def _box(ax, x, y, w, h, text, color=C1, size=8.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=LIGHT.get(color, "#f0efec"), ec=color, lw=1.2))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x + w / 2, y + h * 0.64, lines[0], ha="center", va="center", fontsize=size, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.3, "\n".join(lines[1:]), ha="center", va="center", fontsize=size - 1.3,
                color=INK2, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight="bold", color=INK)


def _arrow(ax, x1, y1, x2, y2, color="#7a7975"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=color, lw=1.1))


def _canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------------- ILLUSTRATION: big picture
def pipeline(out: Path) -> Path:
    fig, ax = _canvas(10, 6.0)
    build = [("1  Raw data", "100,000 rows · pandas", C1), ("2  Rules", "conditions · category\nseverity · priority", C3),
             ("3  Search index", "embeddings → ChromaDB\nBM25 keyword index", C2),
             ("4  ML models", "fault risk · throughput\nclassifier", C4)]
    ask = [("5  Your question", "guardrails · query\nunderstanding", "#52514e"),
           ("6  Hybrid retrieval", "semantic + keyword\n+ metric similarity", C2),
           ("7  Agents", "classify · risk ·\ndiagnose · recommend", "#4a3aa7"),
           ("8  LLM (gpt-oss-20b)", "writes the explanation\nfrom evidence only", "#4a3aa7")]
    answer = [("9  Checks", "grounding gate ·\nLLM judge · provenance", C3),
              ("10  FastAPI", "REST API · Pydantic\nvalidation", C1),
              ("11  React web UI", "the engineer's screen\n(+ Streamlit)", C1),
              ("12  Feedback", "SQLite → evaluation\nreview", "#52514e")]
    xs = [2 + i * 24.5 for i in range(4)]
    ax.text(2, 58, "BUILD TIME — runs once, offline", fontsize=8, weight="bold", color=INK2)
    for i, (t, d, c) in enumerate(build):
        _box(ax, xs[i], 43, 21, 13, f"{t}\n{d}", c)
        if i:
            _arrow(ax, xs[i] - 3.4, 49.5, xs[i] - 0.2, 49.5)
    ax.text(2, 37, "QUESTION TIME — every analysis", fontsize=8, weight="bold", color=INK2)
    for i, (t, d, c) in enumerate(ask):
        _box(ax, xs[i], 22, 21, 13, f"{t}\n{d}", c)
        if i:
            _arrow(ax, xs[i] - 3.4, 28.5, xs[i] - 0.2, 28.5)
    for i, (t, d, c) in enumerate(answer):          # third row runs right to left
        x = xs[3 - i]
        _box(ax, x, 2, 21, 13, f"{t}\n{d}", c)
        if i:
            _arrow(ax, x + 21 + 3.2, 8.5, x + 21 + 0.2, 8.5)
    _arrow(ax, xs[3] + 10.5, 22, xs[3] + 10.5, 15.4)
    for i in (2, 3):                                # index and models are used at question time
        ax.add_patch(FancyArrowPatch((xs[i] + 10.5, 43), (xs[i - 1] + 10.5, 35.4), arrowstyle="-|>", mutation_scale=10,
                                     color="#a9a8a3", lw=1, ls=(0, (3, 2))))
    return _save(fig, out)


# ----------------------------------------------------------------------------- REAL: percentile threshold
def percentile(df: pd.DataFrame, out: Path) -> Path:
    from src.preprocessing.conditions import load_conditions
    c = load_conditions()["high_latency"]
    x = df["latency_ms"].dropna()
    share = float((x > c.threshold).mean())
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.hist(x, bins=80, color=C1, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(c.threshold, color=C2, lw=2)
    ax.text(c.threshold + 3, ax.get_ylim()[1] * 0.88, f"P90 threshold = {c.threshold:.2f} ms\n"
            f"{share:.1%} of records are above it\n→ condition 'high_latency'", color=C2, fontsize=8, va="top")
    ax.set_xlabel("latency_ms (all 100,000 records)")
    ax.set_ylabel("number of records")
    _style(ax, "y")
    return _save(fig, out)


# ----------------------------------------------------------------------------- REAL: embeddings
PHRASES = [
    "high latency", "long network delay", "slow response time",
    "weak signal strength", "poor radio coverage", "good signal strength",
    "many dropped calls", "handover failures between cells",
    "heavy rain storm", "snowfall in winter",
]


def embeddings(out_scatter: Path, out_heat: Path) -> tuple[Path, Path, dict]:
    from src.rag.embeddings import embed
    v = embed(PHRASES)
    sim = v @ v.T                                         # vectors are L2-normalised: dot = cosine
    centred = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    xy = centred @ vt[:2].T
    groups = [C1, C1, C1, C3, C3, C3, C2, C2, C4, C4]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.scatter(xy[:, 0], xy[:, 1], s=55, c=groups, edgecolor="white", linewidth=1.2, zorder=3)
    offsets = {"high latency": (8, 6), "long network delay": (8, -12), "slow response time": (-8, 8),
               "good signal strength": (-8, 6), "weak signal strength": (8, -12)}
    for (x, y), t in zip(xy, PHRASES):
        dx, dy = offsets.get(t, (6, 4))
        ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK,
                    ha="right" if dx < 0 else "left")
    ax.set_xlabel("direction 1 (of 384, compressed with PCA)")
    ax.set_ylabel("direction 2")
    ax.margins(0.25)
    ax.set_xlim(left=float(xy[:, 0].min()) - 0.45)
    _style(ax, "both")
    _save(fig, out_scatter)

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(sim, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(PHRASES)), PHRASES, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(PHRASES)), PHRASES, fontsize=7.5)
    for i in range(len(PHRASES)):
        for j in range(len(PHRASES)):
            ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if sim[i, j] > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04, label="cosine similarity")
    _save(fig, out_heat)
    return out_scatter, out_heat, {(a, b): float(sim[PHRASES.index(a), PHRASES.index(b)])
                                   for a in PHRASES for b in PHRASES}


# ----------------------------------------------------------------------------- REAL parameters: BM25 curves
def bm25(out: Path) -> Path:
    from src.utils.config import load_yaml
    p = load_yaml("retrieval.yaml")["bm25"]
    k1, b = p["k1"], p["b"]
    tf = np.arange(0, 11)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 2.9))
    for ratio, col, lab in ((0.5, C3, "short document (½ avg length)"), (1.0, C1, "average length"),
                            (2.0, C2, "long document (2× avg length)")):
        norm = k1 * (1 - b + b * ratio)
        a1.plot(tf, tf * (k1 + 1) / (tf + norm), marker="o", ms=3, color=col, label=lab)
    a1.set_xlabel("times the word appears in the document (tf)")
    a1.set_ylabel("term-frequency part of the score")
    a1.set_title(f"Saturation (k1 = {k1}, b = {b})", fontsize=9, loc="left", color=INK)
    a1.legend(fontsize=7, frameon=False)
    _style(a1, "y")
    n = 100_000
    dfs = np.logspace(0, 5, 200)
    a2.plot(dfs, np.log(1 + (n - dfs + 0.5) / (dfs + 0.5)), color=C1)
    a2.set_xscale("log")
    a2.set_xlabel("number of records containing the word (of 100,000)")
    a2.set_ylabel("IDF weight")
    a2.set_title("Rare words count more (IDF)", fontsize=9, loc="left", color=INK)
    _style(a2, "y")
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------------------- ILLUSTRATION: HNSW
def hnsw(out: Path) -> Path:
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.axis("off")
    layers = [(2.4, 4), (1.2, 10), (0.0, 28)]
    pts_by_layer = []
    base = rng.uniform(0, 10, size=(28, 2))
    for k, (y0, n) in enumerate(layers):
        pts = base[:n]
        pts_by_layer.append(pts)
        ax.add_patch(FancyBboxPatch((-0.3, y0 - 0.15), 10.6, 0.95, boxstyle="round,pad=0.02", fc="#f6f5f2",
                                    ec="#d9d8d3", lw=0.8))
        px, py = pts[:, 0], y0 + pts[:, 1] * 0.065
        for i in range(n):
            d = np.argsort(np.abs(pts[:, 0] - pts[i, 0]))[1:3]
            for j in d:
                ax.plot([px[i], px[j]], [py[i], py[j]], color="#c9c8c2", lw=0.7, zorder=1)
        ax.scatter(px, py, s=18, color=[C1, C3, C2][k], zorder=2)
        ax.text(10.5, y0 + 0.33, ["top layer: few points,\nlong jumps", "middle layer", "bottom layer: every\nrecord, short hops"][k],
                fontsize=7.5, color=INK2, va="center")
    path = [(base[0, 0], 2.4 + base[0, 1] * 0.065), (base[0, 0], 1.2 + base[0, 1] * 0.065),
            (base[5, 0], 1.2 + base[5, 1] * 0.065), (base[5, 0], 0 + base[5, 1] * 0.065),
            (base[17, 0], 0 + base[17, 1] * 0.065)]
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=C2, lw=1.6, zorder=4))
    ax.scatter([path[-1][0]], [path[-1][1]], s=90, facecolor="none", edgecolor=C2, lw=1.8, zorder=5)
    ax.text(path[-1][0], path[-1][1] - 0.32, "nearest neighbour found", color=C2, fontsize=7.5, ha="center")
    ax.set_xlim(-0.5, 13)
    ax.set_ylim(-0.6, 3.5)
    return _save(fig, out)


# ----------------------------------------------------------------------------- REAL: metric similarity
def metric_similarity(df: pd.DataFrame, out: Path) -> Path:
    from src.preprocessing.conditions import load_conditions
    x = df["latency_ms"].astype(float)
    rng_ = float(x.quantile(0.99) - x.quantile(0.01))
    grid = np.linspace(x.min(), x.max(), 300)
    thr = load_conditions()["high_latency"].threshold
    rank = np.searchsorted(np.sort(x.to_numpy()), grid) / len(x)
    adverse = 0.5 * rank + 0.5 * (grid > thr)
    eq = np.clip(1 - np.abs(grid - 150) / rng_, 0, 1)
    normal = 1 - 2 * np.abs(rank - 0.5)
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.plot(grid, eq, color=C1, label='"latency 150 ms" (stated value)')
    ax.plot(grid, adverse, color=C2, label='"high latency" (½ percentile rank + ½ threshold)')
    ax.plot(grid, normal, color=C3, label='"normal latency" (closeness to the median)')
    ax.axvline(thr, color=C2, ls=":", lw=1)
    ax.set_xlabel("a record's latency_ms")
    ax.set_ylabel("similarity (0–1)")
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")
    _style(ax, "y")
    return _save(fig, out)


# ----------------------------------------------------------------------------- REAL: ML figures
def _ml_frame(df: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    from src.services.models.common import cfg, encode, resolve_feature_set
    c = cfg()
    raw = resolve_feature_set(c["fault_risk"]["feature_sets"]["C_primary"], df.columns, "fault_rate_pct")
    X, y = encode(df, raw), df["fault_rate_pct"].astype(float)
    itr, ite = train_test_split(df.index, test_size=c["test_size"], random_state=c["random_state"])
    return X, y, itr, ite


def tree(df: pd.DataFrame, out: Path) -> Path:
    from sklearn.tree import DecisionTreeRegressor, plot_tree
    X, y, itr, _ = _ml_frame(df)
    t = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X.loc[itr].fillna(-999), y.loc[itr])
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    plot_tree(t, feature_names=list(X.columns), filled=True, rounded=True, impurity=False, precision=1,
              fontsize=8, ax=ax, proportion=True)
    return _save(fig, out)


def boosting(df: pd.DataFrame, out: Path) -> tuple[Path, dict]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    X, y, itr, ite = _ml_frame(df)
    iters = [1, 2, 5, 10, 20, 50, 100, 300]
    scores = []
    for n in iters:
        m = HistGradientBoostingRegressor(max_iter=n, learning_rate=0.05, random_state=42,
                                          early_stopping=False).fit(X.loc[itr], y.loc[itr])
        scores.append(r2_score(y.loc[ite], m.predict(X.loc[ite])))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.plot(iters, scores, marker="o", color=C1)
    for n, s in zip(iters, scores):
        ax.text(n, s + 0.03, f"{s:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(iters, [str(i) for i in iters])
    ax.set_xlabel("number of trees added (max_iter), learning rate 0.05")
    ax.set_ylabel("R² on the 20% test set")
    ax.set_ylim(min(scores) - 0.05, 1.0)
    _style(ax, "y")
    return _save(fig, out), dict(zip(iters, [round(s, 3) for s in scores]))


def pred_vs_actual(df: pd.DataFrame, out: Path) -> tuple[Path, float]:
    from src.services.models.common import load_model
    X, y, _, ite = _ml_frame(df)
    art = load_model("fault_risk")
    meta = art["meta"]
    pred = art["model"].predict(X.loc[ite][meta["encoded_features"]])
    actual = y.loc[ite].to_numpy()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    hb = ax.hexbin(actual, pred, gridsize=45, cmap="Blues", mincnt=1, linewidths=0)
    lim = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lim, lim, color=C2, lw=1.2, label="perfect prediction")
    ax.plot(lim, np.array(lim) + meta["residual_p90"], color=C2, lw=0.8, ls="--", label="80% interval")
    ax.plot(lim, np.array(lim) + meta["residual_p10"], color=C2, lw=0.8, ls="--")
    ax.set_xlabel("actual fault_rate_pct (test set, 20,000 records)")
    ax.set_ylabel("predicted fault_rate_pct")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.colorbar(hb, ax=ax, fraction=0.045, label="records")
    _style(ax, "both")
    inside = float(np.mean((actual - pred >= meta["residual_p10"]) & (actual - pred <= meta["residual_p90"])))
    return _save(fig, out), inside


def confusion(out: Path) -> Path:
    import json
    r = json.loads((PROJECT_ROOT / "docs" / "models" / "incident_classifier.json").read_text(encoding="utf-8"))
    labels, m = r["confusion_matrix"]["labels"], np.array(r["confusion_matrix"]["matrix"], dtype=float)
    norm = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=6.6,
                        color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xlabel("category predicted by the classifier")
    ax.set_ylabel("rule-derived category (the 'label')")
    return _save(fig, out)


# ----------------------------------------------------------------------------- ILLUSTRATIONS: methods
def split_cv(out: Path) -> Path:
    fig, ax = _canvas(9, 3.4)
    ax.text(1, 31, "Train / test split (80 / 20, random_state = 42)", fontsize=8.5, weight="bold", color=INK)
    ax.add_patch(FancyBboxPatch((1, 21), 70, 6, boxstyle="square,pad=0", fc=LIGHT[C1], ec=C1))
    ax.add_patch(FancyBboxPatch((71, 21), 17.5, 6, boxstyle="square,pad=0", fc=LIGHT[C2], ec=C2))
    ax.text(36, 24, "80,000 records — the model learns from these", ha="center", va="center", fontsize=8)
    ax.text(79.75, 24, "20,000 — exam", ha="center", va="center", fontsize=8)
    ax.text(1, 15.5, "5-fold cross-validation on the training part", fontsize=8.5, weight="bold", color=INK)
    for k in range(5):
        y = 11.5 - k * 2.4
        for f in range(5):
            col = C2 if f == k else C1
            ax.add_patch(FancyBboxPatch((1 + f * 14, y), 14, 1.8, boxstyle="square,pad=0", fc=LIGHT[col], ec=col, lw=0.8))
        ax.text(72, y + 0.9, f"round {k + 1}: train on 4 folds, score on the orange one", fontsize=7.3, va="center", color=INK2)
    return _save(fig, out)


def rag(out: Path) -> Path:
    fig, ax = _canvas(10, 2.6)
    steps = [("Question", "engineer's words", "#52514e"), ("Retrieve", "similar records\nfrom 100,000", C2),
             ("Evidence package", "records · statistics ·\nhypotheses · actions", C3),
             ("Generate", "the LLM writes the\nexplanation", "#4a3aa7"), ("Check", "grounding gate ·\nLLM judge", C1)]
    for i, (t, d, c) in enumerate(steps):
        _box(ax, 1 + i * 20, 5, 17, 15, f"{t}\n{d}", c)
        if i:
            _arrow(ax, 1 + i * 20 - 2.8, 12.5, 1 + i * 20 - 0.2, 12.5)
    return _save(fig, out)


def build_all(folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "incidents.parquet")
    f: dict = {}
    f["pipeline"] = pipeline(folder / "pipeline.png")
    f["percentile"] = percentile(df, folder / "percentile.png")
    f["emb_scatter"], f["emb_heat"], f["sim"] = embeddings(folder / "emb_scatter.png", folder / "emb_heat.png")
    f["bm25"] = bm25(folder / "bm25.png")
    f["hnsw"] = hnsw(folder / "hnsw.png")
    f["metric"] = metric_similarity(df, folder / "metric.png")
    f["tree"] = tree(df, folder / "tree.png")
    f["boosting"], f["boosting_scores"] = boosting(df, folder / "boosting.png")
    f["pred"], f["inside_interval"] = pred_vs_actual(df, folder / "pred.png")
    f["confusion"] = confusion(folder / "confusion.png")
    f["split"] = split_cv(folder / "split.png")
    f["rag"] = rag(folder / "rag.png")
    return f


if __name__ == "__main__":
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    figs = build_all(Path(__file__).parent / "figs")
    print({k: v for k, v in figs.items() if k not in ("sim",)})
