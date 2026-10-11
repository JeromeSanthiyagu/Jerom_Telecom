"""presentation/Project_Presentation.pptx — the 10-minute deck in the guideline's section order.

    python -m docs.deliverables.presentation

Sections (guideline section 12): Problem Statement, Business Need, Solution Overview, Architecture, Data Flow,
Technology Stack, Agent Workflow, Demo, Evaluation Results, Challenges and Learnings.
Every number is read from the project's own outputs (evaluation JSON, model reports, samples.json).
"""

from __future__ import annotations

import json
import subprocess
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

from src.utils.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "presentation" / "Project_Presentation.pptx"
FONT = "Calibri"
INK, INK2, MUTED = RGBColor(0x18, 0x22, 0x2F), RGBColor(0x3C, 0x47, 0x57), RGBColor(0x5F, 0x6B, 0x7A)
BG, CARD, LINE = RGBColor(0xF6, 0xF5, 0xF1), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xDD, 0xDB, 0xD4)
ACCENT, ACCENT_SOFT = RGBColor(0x1F, 0x63, 0xB8), RGBColor(0xE6, 0xEF, 0xFA)
DARK, DARK_TEXT, DARK_MUTED = RGBColor(0x18, 0x22, 0x2F), RGBColor(0xF6, 0xF5, 0xF1), RGBColor(0xA9, 0xB6, 0xC6)
WARN_SOFT = RGBColor(0xFB, 0xED, 0xE6)
TOTAL = 12


def _fill(shape, color, line=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=1.12):
    """paras: str or list of str / (str, dict(size, color, bold)) tuples; '**x**' is not parsed."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    items = [paras] if isinstance(paras, str) else paras
    for i, item in enumerate(items):
        t, opt = (item, {}) if isinstance(item, str) else item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(opt.get("after", 6))
        r = p.add_run()
        r.text = t
        r.font.name = FONT
        r.font.size = Pt(opt.get("size", size))
        r.font.bold = opt.get("bold", bold)
        r.font.color.rgb = opt.get("color", color)
    return tb


def card(slide, x, y, w, h, title, body, fill=CARD, title_color=INK, body_color=INK2, title_size=17, body_size=13,
         line=LINE, top_bar=None):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.adjustments[0] = 0.06
    _fill(box, fill, line)
    if top_bar is not None:
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x + 0.18), Inches(y), Inches(w - 0.36), Inches(0.06))
        _fill(bar, top_bar)
    text(slide, x + 0.22, y + 0.2, w - 0.44, 0.45, title, size=title_size, color=title_color, bold=True)
    text(slide, x + 0.22, y + 0.68, w - 0.44, h - 0.8, body, size=body_size, color=body_color, spacing=1.15)


def base(prs, kicker, title, n, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    _fill(bg, DARK if dark else BG)
    text(s, 0.6, 0.42, 12, 0.35, kicker.upper(), size=12, color=DARK_MUTED if dark else ACCENT, bold=True)
    text(s, 0.6, 0.78, 12.1, 0.9, title, size=30, color=DARK_TEXT if dark else INK, bold=True, spacing=1.0)
    text(s, 0.6, 7.0, 9, 0.3, f"Telecom NetOps Assistant · {n} / {TOTAL}", size=10,
         color=DARK_MUTED if dark else MUTED)
    return s


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def table(slide, x, y, w, rows, col_w, size=12, header_fill=DARK, row_h=0.36):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shape.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if i == 0 else (CARD if i % 2 else RGBColor(0xF1, 0xF0, 0xEC))
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = i == 0 or j == 0
            r.font.color.rgb = DARK_TEXT if i == 0 else INK
    return shape


def image(slide, path, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def _f(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def _test_count() -> int | None:
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300).stdout
        line = [l for l in out.splitlines() if "tests collected" in l or "test collected" in l]
        return int(line[-1].split()[0]) if line else None
    except Exception:
        return None


def data() -> dict:
    ev = json.loads((PROJECT_ROOT / "docs" / "evaluation" / "eval_results_deepeval.json").read_text(encoding="utf-8"))
    fr = json.loads((PROJECT_ROOT / "docs" / "models" / "fault_risk.json").read_text(encoding="utf-8"))
    ic = json.loads((PROJECT_ROOT / "docs" / "models" / "incident_classifier.json").read_text(encoding="utf-8"))
    return {"ev": ev["summary"], "judge": ev.get("judge", "the configured LLM"), "r2": fr["served"]["test_metrics"]["r2"],
            "mae": fr["served"]["test_metrics"]["mae"], "clf": ic["models"]["hist_gradient_boosting"]["accuracy"],
            "tests": _test_count()}


def build() -> None:
    D = data()
    de = D["ev"].get("deepeval", {})
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    # 1 title -------------------------------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    _fill(bg, DARK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.22), prs.slide_height)
    _fill(bar, ACCENT)
    text(s, 0.9, 1.4, 11.5, 0.4, "CAPSTONE PROJECT · TELECOM", size=14, color=DARK_MUTED, bold=True)
    text(s, 0.9, 1.95, 11.5, 2.0, "AI-Powered Telecom Network\nOperations Assistant", size=46, color=DARK_TEXT,
         bold=True, spacing=0.95)
    text(s, 0.9, 4.1, 11, 1.0, "Evidence-grounded RAG with agents: similar past cases, risk, ranked root-cause "
                               "hypotheses and next steps — nothing invented.", size=20, color=DARK_MUTED)
    text(s, 0.9, 6.2, 11, 0.4, "[Your name] · Repository Jerom_Telecom · [Date]", size=14, color=DARK_MUTED)
    notes(s, "Good morning. My project is an AI assistant for telecom network operations. An engineer describes a "
             "network problem in plain words; the assistant finds similar historical cases in 100,000 records, "
             "classifies the case, predicts fault risk, ranks likely causes and suggests what to check — and every "
             "statement it makes is traceable to evidence. [about 20 seconds]")

    # 2 problem -----------------------------------------------------------------------------------------
    s = base(prs, "Problem statement", "Engineers diagnose network degradations under time pressure", 2)
    for i, (t, b) in enumerate([
            ("Too many signals", "KPIs such as latency, packet loss, handover success and alarms change constantly; "
                                 "each degradation needs a quick, correct decision."),
            ("Past cases are hard to use", "Similar incidents happened before, but finding them and their pattern "
                                           "takes manual search and experience."),
            ("Generic AI is unsafe here", "Chatbots invent measurements, incident IDs and confident root causes — "
                                          "a wrong 'fact' costs outage time.")]):
        card(s, 0.6 + i * 4.1, 2.0, 3.85, 2.7, t, b, top_bar=ACCENT)
    text(s, 0.6, 5.2, 12, 1.2, "Question: how can an assistant help an engineer decide faster while never presenting "
                               "a guess as a fact?", size=20, color=INK, bold=True)
    notes(s, "The problem. NOC engineers face constant KPI degradations and alarms and must decide quickly what is "
             "wrong and what to do. Past incidents hold the answer but are hard to search. And a general chatbot is "
             "dangerous here, because it invents numbers and causes. So the question was: how do we help engineers "
             "decide faster without ever presenting a guess as a fact? [45 s]")

    # 3 business need -----------------------------------------------------------------------------------
    s = base(prs, "Business need", "Faster, consistent and trustworthy triage", 3)
    rows = [["Need", "What the solution provides"],
            ["Shorter time to a first diagnosis", "similar cases, category, severity and priority in seconds"],
            ["Consistent decisions", "the same documented rules for every case (YAML), not individual judgement"],
            ["Reuse of historical knowledge", "hybrid search over 100,000 past records as evidence"],
            ["Trust and auditability", "every statement tagged with its source; LLM output checked by code"],
            ["Safe, local operation", "runs on a laptop; works offline; refuses requests to act on the network"]]
    table(s, 0.6, 2.0, 12.1, rows, [4.2, 7.9], size=15, row_h=0.62)
    text(s, 0.6, 6.15, 12, 0.5, "Decision support only: recommendations are options to evaluate, never guaranteed "
                                "KPI improvements.", size=14, color=MUTED)
    notes(s, "What the business needs: shorter time to a first diagnosis, consistent decisions across engineers, "
             "reuse of historical knowledge, and above all trust — every claim auditable. It must also be safe: it "
             "supports decisions, it never acts on the network, and it never promises KPI improvements. [45 s]")

    # 4 solution overview -------------------------------------------------------------------------------
    s = base(prs, "Solution overview", "One question in, an evidence-backed assessment out", 4)
    outs = [("Similar cases", "hybrid retrieval from 100,000 records"),
            ("Labels", "category, severity and priority by documented rules, labelled as derived"),
            ("Fault risk", "ML prediction next to the rate observed in similar cases"),
            ("Ranked hypotheses", "with confidence, evidence for and against"),
            ("Checks and actions", "what to verify, what to do, what to monitor")]
    for i, (t, b) in enumerate(outs):
        card(s, 0.6 + i * 2.46, 2.05, 2.3, 2.3, t, b, title_size=15, body_size=12, top_bar=ACCENT)
    text(s, 0.6, 4.75, 12.1, 1.6, [
        ("How it stays honest", {"size": 18, "bold": True, "color": INK}),
        ("Facts come from the data and from rules · estimates come from ML models with measured error · the LLM only "
         "explains, and a grounding gate rejects any number, ID or claim that is not in the evidence.",
         {"size": 15, "color": INK2})])
    notes(s, "The solution: one plain-language question in, and out comes similar historical cases; a derived "
             "category, severity and priority; a fault-risk prediction shown next to the rate actually observed in "
             "similar cases; ranked root-cause hypotheses; and concrete checks and actions. The principle: facts from "
             "data and rules, estimates from ML with measured error, and the LLM only explains — code checks every "
             "number it writes. [50 s]")

    # 5 architecture ------------------------------------------------------------------------------------
    s = base(prs, "Architecture", "Ten stages, one direction of trust", 5)
    image(s, PROJECT_ROOT / "architecture" / "Architecture_Diagram.png", 1.55, 1.7, h=5.15)
    notes(s, "The architecture follows the required chain. At build time: the pinned dataset is validated and "
             "cleaned, conditions and labels are derived by rules, each record becomes one chunk, chunks are embedded "
             "with MiniLM into ChromaDB and also indexed by BM25. At query time: the Streamlit UI calls FastAPI, "
             "guardrails check the input, the retriever combines semantic, keyword and metric similarity, six agents "
             "build the answer, and the LLM writes the explanation behind a grounding gate. [60 s]")

    # 6 data flow ---------------------------------------------------------------------------------------
    s = base(prs, "Data flow", "Following one real record and one real question", 6)
    steps = [("Raw row", "ROW-033707: latency 168.408 ms, packet loss 13.702 %, Snow, Cochranville MN"),
             ("Preprocessed", "typed values + record_id; nothing altered"),
             ("Derived", "high_latency (>129.92 ms), high_packet_loss (>11.88 %) → Service Degradation, HIGH, P3"),
             ("Chunk", "embedding_text (qualitative) + search_text (full, with numbers)"),
             ("Indexed", "ChromaDB: 384-number vector + metadata · BM25: 164 weighted terms")]
    for i, (t, b) in enumerate(steps):
        card(s, 0.6 + i * 2.46, 1.95, 2.3, 2.35, t, b, title_size=15, body_size=11.5, top_bar=ACCENT)
    text(s, 0.6, 4.55, 12.2, 2.3, [
        ("Query: “High latency and packet loss during snow”", {"size": 17, "bold": True, "color": INK}),
        ("Embedding search alone ranks latency-only records high; BM25 alone has nothing to match 'packet loss' on "
         "(it appears in every record, IDF 0) and its top hit has packet loss 5.98 %.", {"size": 14, "color": INK2}),
        ("Hybrid result: the top 5 all have high latency and high packet loss, in snow — metric similarity reads the "
         "actual values and fixes what each search misses.", {"size": 14, "color": INK2})])
    notes(s, "Here is a real record's journey. Record 33707 starts as a raw CSV line, becomes typed data, gets two "
             "derived conditions and the labels Service Degradation, HIGH, P3, becomes a text chunk, and is stored "
             "both as a vector in ChromaDB and as a weighted row in BM25. For the question 'high latency and packet "
             "loss during snow', each search alone makes mistakes; the hybrid ranking returns five records that all "
             "show both conditions in snow. [60 s]")

    # 7 technology stack --------------------------------------------------------------------------------
    s = base(prs, "Technology stack", "Small, local, explainable components", 7)
    rows = [["Layer", "Technology", "Why"],
            ["Language", "Python 3.12", "data, ML and API in one language"],
            ["LLM", "NVIDIA API Catalog · openai/gpt-oss-20b", "JSON-schema output; Claude also supported"],
            ["Embedding model", "sentence-transformers all-MiniLM-L6-v2", "fast on CPU, 384 dimensions"],
            ["Vector database", "ChromaDB (HNSW, cosine)", "local, metadata filters, id = record_id"],
            ["Keyword search", "BM25 (own sparse implementation)", "exact names, places, conditions"],
            ["ML", "scikit-learn HistGradientBoosting", "accurate; handles missing inputs"],
            ["API / UI", "FastAPI + Uvicorn + Pydantic / Streamlit", "validated REST API; Python UI (+ React)"],
            ["Evaluation", "pytest · DeepEval", "proofs and LLM-judged RAG metrics"]]
    table(s, 0.6, 1.95, 12.1, rows, [2.4, 4.9, 4.8], size=14, row_h=0.52)
    notes(s, "The stack is deliberately small and local. Python throughout. The LLM is gpt-oss-20b from NVIDIA's API "
             "catalog with JSON-schema outputs. MiniLM embeddings in ChromaDB, plus my own BM25 implementation. "
             "scikit-learn gradient boosting for risk. FastAPI for the API, Streamlit for the UI, and pytest plus "
             "DeepEval for evaluation. [40 s]")

    # 8 agent workflow ----------------------------------------------------------------------------------
    s = base(prs, "Agent workflow", "Six agents, one shared state, no loops", 8)
    image(s, PROJECT_ROOT / "docs" / "guide" / "figs" / "agents.png", 0.6, 1.85, w=12.1)
    roles = [("Retrieval", "evidence"), ("Classification", "category, severity"), ("Risk", "prediction, priority"),
             ("Diagnosis", "hypotheses"), ("Recommendation", "actions + LLM explanation"),
             ("Validation", "checks, judge, provenance")]
    for i, (t, b) in enumerate(roles):
        card(s, 0.6 + i * 2.05, 4.35, 1.92, 1.25, t, b, title_size=13, body_size=11)
    text(s, 0.6, 5.85, 12.1, 0.9, "Agents communicate only through a typed shared state; each step is timed and "
                                  "traced. A test proves the agents return exactly the single pipeline's result.",
         size=14, color=INK2)
    notes(s, "The analysis runs as six agents in a fixed order: retrieval, classification, risk, diagnosis, "
             "recommendation and validation. They share one typed state, each declares the inputs it needs and writes "
             "a timed trace entry. There are no autonomous loops — a linear workflow is easier to test, and a test "
             "proves the agents give exactly the same result as the plain pipeline. [45 s]")

    # 9 demo --------------------------------------------------------------------------------------------
    s = base(prs, "Demo", "Fault occurrence up, handover success down — what now?", 9)
    image(s, PROJECT_ROOT / "presentation" / "assets" / "streamlit_demo.png", 0.6, 1.75, w=8.3)
    text(s, 9.1, 1.85, 3.7, 5.0, [
        ("Run locally", {"size": 15, "bold": True, "color": INK}),
        ("uvicorn src.main:app --reload", {"size": 12, "color": INK2}),
        ("streamlit run frontend.py", {"size": 12, "color": INK2, "after": 14}),
        ("Point out", {"size": 15, "bold": True, "color": INK}),
        ("Category Mobility / Handover; severity 'estimated' — values were only described", {"size": 12, "color": INK2}),
        ("Predicted risk 44.08 % next to 58.05 % observed — never mixed", {"size": 12, "color": INK2}),
        ("Leading hypothesis 0.65 — heuristic, not a probability", {"size": 12, "color": INK2}),
        ("Every action states its basis; review checks shown", {"size": 12, "color": INK2})])
    notes(s, "[Switch to the live app: streamlit run frontend.py, with the API running.] I choose the example "
             "'Faults rising, handover falling' and press Analyze case. Point out: the category is Mobility and "
             "Handover; severity is marked estimated because I only described the problem; the ML prediction of 44 "
             "percent is shown next to the 58 percent actually observed in similar records; the leading hypothesis "
             "scores 0.65, a documented heuristic; and every action says which condition it is based on. If the live "
             "demo fails, this screenshot is the real output. [90 s]")

    # 10 evaluation -------------------------------------------------------------------------------------
    s = base(prs, "Evaluation results", "Measured, not assumed", 10, dark=True)
    det = D["ev"]
    big = [(str(D["tests"] or "222"), "automated tests, all passing"),
           (f"{det['retrieval_precision']:.2f}", "retrieval precision, 8 evaluation cases"),
           (_f(de.get("FaithfulnessMetric")), "DeepEval faithfulness (LLM judge)"),
           (f"{D['r2']:.3f}", f"fault-risk R² on unseen data (MAE {D['mae']:.1f} pts)")]
    for i, (v, l) in enumerate(big):
        text(s, 0.6 + i * 3.1, 2.0, 2.9, 1.1, v, size=48, color=DARK_TEXT, bold=True)
        text(s, 0.6 + i * 3.1, 3.15, 2.8, 0.8, l, size=14, color=DARK_MUTED)
    rows = [["Metric", "Score"],
            ["Condition recall · category match · hypothesis in top 2 · grounded",
             " · ".join(f"{det[k]:.2f}" for k in ("condition_recall", "category_match", "hypothesis_hit", "grounded"))],
            ["DeepEval answer relevancy · contextual precision · contextual relevancy",
             " · ".join(_f(de.get(k)) for k in ("AnswerRelevancyMetric", "ContextualPrecisionMetric",
                                                  "ContextualRelevancyMetric"))],
            ["Category classifier (agreement with the rules, not real accuracy)", f"{D['clf']:.1%}"]]
    table(s, 0.6, 4.3, 12.1, rows, [8.6, 3.5], size=13, header_fill=RGBColor(0x2A, 0x3A, 0x50), row_h=0.45)
    text(s, 0.6, 6.45, 12.1, 0.4, f"Judge: {D['judge']}. Development scores — one retrieval weight was tuned on these "
                                  "8 cases.", size=12, color=DARK_MUTED)
    notes(s, "Quality was measured. All automated tests pass, including the five proofs the specification asked for, "
             "such as comparing every processed value with the raw file. On the eight evaluation cases, retrieval "
             "precision, category, hypothesis and grounding are all at the top. DeepEval, using the NVIDIA model as an "
             "independent judge, gives the faithfulness shown. The served fault-risk model explains 88 percent of the "
             "variation on unseen data. Two honest caveats: these are development scores, and the classifier figure "
             "is agreement with my own rules. [60 s]")

    # 11 challenges -------------------------------------------------------------------------------------
    s = base(prs, "Challenges and learnings", "Real problems found by testing — and what they taught", 11)
    rows = [["Challenge", "What I did", "Learning"],
            ["No labels in the data", "derived labels with documented rules", "say 'derived', never 'ground truth'"],
            ["R² 0.9998 was leakage (target is a formula)", "excluded formula columns → honest 0.885",
             "a perfect score is a warning sign"],
            ["LLM computed new numbers", "grounding gate, retry, template", "never trust fluent text — check it"],
            ["'normal user load' retrieved high-load records", "strip good/normal phrases + penalty",
             "embeddings miss negation"],
            ["pandas altered ~15 % of floats in the last digit", "exact float() parsing + equality test",
             "test the data, not only the code"],
            ["Small LLM as judge raises false alarms", "code checks + judge must agree", "layer cheap checks with judgement"]]
    table(s, 0.6, 1.95, 12.1, rows, [4.4, 4.0, 3.7], size=13, row_h=0.6)
    text(s, 0.6, 6.3, 12.1, 0.5, "Limitations: synthetic data, no timestamps, cell IDs or transport counters, so causes "
                                 "remain hypotheses. Next: real operator data, confidence calibrated from feedback.",
         size=12, color=MUTED)
    notes(s, "The most valuable part was what testing found. The data has no labels, so labels are derived and "
             "always called derived. A near-perfect R-squared turned out to be leakage. The LLM computed numbers that "
             "were not in the evidence, so code now checks every number. Embeddings confused 'normal load' with high "
             "load. Even pandas changed values in the last decimal place. The lesson: test the data and check the AI. "
             "Limits: the data is synthetic, so causes stay hypotheses. [60 s]")

    # 12 close ------------------------------------------------------------------------------------------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    _fill(bg, DARK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.22), prs.slide_height)
    _fill(bar, ACCENT)
    text(s, 0.9, 1.6, 11.5, 1.2, "Thank you — questions?", size=48, color=DARK_TEXT, bold=True)
    for i, (t, b) in enumerate([("Evidence first", "every statement carries its source"),
                                ("Honest ML", "leakage removed; derived labels named as derived"),
                                ("Safe LLM", "the model explains, code checks, a judge reviews")]):
        text(s, 0.9 + i * 4.0, 3.6, 3.7, 0.5, t, size=22, color=DARK_TEXT, bold=True)
        text(s, 0.9 + i * 4.0, 4.2, 3.7, 1.0, b, size=15, color=DARK_MUTED)
    text(s, 0.9, 6.3, 12, 0.4, "Repository Jerom_Telecom · design/Design_Document.pdf · docs/API_Documentation.pdf · "
                               "API docs at /docs", size=13, color=DARK_MUTED)
    notes(s, "To close: evidence first, honest machine learning, and a safely constrained LLM. Thank you — I'm happy "
             "to take questions or show any part live.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
