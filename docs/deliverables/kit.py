"""Shared layout for the submission PDFs (styling shared with the project guide)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reportlab.platypus import CondPageBreak

from docs.guide.build_guide import glue_headings
from docs.guide.pdf_kit import (CONTENT_W, MARGIN, MUTED, PAGE_H, PAGE_W, RULE, GuideDoc, NextPageTemplate, P,
                                PageBreak, Spacer, mm, table, toc)
from src.utils.config import PROJECT_ROOT

SAMPLES_PATH = Path(__file__).with_name("samples.json")
PROJECT = "AI-Powered Telecom Network Operations Assistant"
REPO = "Jerom_Telecom"


def samples() -> dict:
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


class DeliverableDoc(GuideDoc):
    def __init__(self, filename: str, name: str, **kw):
        super().__init__(filename, **kw)
        self.doc_name = name
        self.title = f"{name} - {PROJECT}"

    def _decorate(self, canv, doc):
        canv.saveState()
        canv.setFont("Sans", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN, PAGE_H - 12 * mm, f"{PROJECT} — {self.doc_name}")
        if self.current_chapter:
            ch = self.current_chapter
            canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm, ch if len(ch) <= 48 else ch[:46] + "…")
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(MARGIN, PAGE_H - 13.5 * mm, PAGE_W - MARGIN, PAGE_H - 13.5 * mm)
        canv.drawCentredString(PAGE_W / 2, 10 * mm, str(doc.page))
        canv.restoreState()


def cover(name: str, subtitle: str, facts: list[list[str]], with_toc: bool = True) -> list:
    s = [Spacer(1, 18 * mm),
         P(f'<font color="#ffffff">{name}</font>', "title"),
         P(f'<font color="#cfd8e3">{PROJECT}<br/>{subtitle}</font>', "subtitle"),
         Spacer(1, 40 * mm),
         table([["Item", "Value"]] + facts, widths=[1.3, 4.7], bold_first_col=True),
         NextPageTemplate("normal"), PageBreak()]
    if with_toc:
        s += [P("Contents", "h1"), toc(), PageBreak()]
    return s


def section(title: str) -> list:
    return [CondPageBreak(60 * mm), P(title, "h1")]


def mono(text: str) -> str:
    return f"<font name='Mono'>{text}</font>"


def abridge(obj: Any, max_list: int = 3, max_str: int = 160, depth: int = 0, max_depth: int = 6) -> Any:
    """Shorten a real JSON response for printing: long lists keep their first items plus a count marker."""
    if depth > max_depth:
        return "…"
    if isinstance(obj, dict):
        return {k: abridge(v, max_list, max_str, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        out = [abridge(v, max_list, max_str, depth + 1, max_depth) for v in obj[:max_list]]
        if len(obj) > max_list:
            out.append(f"… {len(obj) - max_list} more")
        return out
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + "…"
    if isinstance(obj, float):
        return round(obj, 4)
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False)


def build(filename: Path, name: str, story: list) -> Path:
    import os
    os.chdir(PROJECT_ROOT)
    filename.parent.mkdir(parents=True, exist_ok=True)
    doc = DeliverableDoc(str(filename), name)
    doc.multiBuild(glue_headings(story))
    print(f"wrote {filename}")
    return filename


__all__ = ["DeliverableDoc", "cover", "section", "mono", "abridge", "dumps", "build", "samples", "PROJECT", "REPO",
           "CONTENT_W", "P", "PageBreak", "Spacer", "mm", "table"]
