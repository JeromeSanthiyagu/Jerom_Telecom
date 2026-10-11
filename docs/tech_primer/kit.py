"""Page frame and the per-topic layout for the technology primer (styling shared with the project guide)."""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.platypus import CondPageBreak

from docs.guide.pdf_kit import (CONTENT_W, MARGIN, MUTED, PAGE_H, PAGE_W, RULE, GuideDoc, KeepTogether, P,
                                Spacer, bullets, callout, image, mm)


class PrimerDoc(GuideDoc):
    """The guide's page templates with the primer's own title and running header."""

    def __init__(self, filename: str, **kw):
        super().__init__(filename, **kw)
        self.title = "Technology Primer - AI-Powered Telecom Network Operations Assistant"

    def _decorate(self, canv, doc):
        canv.saveState()
        canv.setFont("Sans", 7.5)
        canv.setFillColor(MUTED)
        canv.drawString(MARGIN, PAGE_H - 12 * mm, "Telecom NetOps Assistant — Technology Primer")
        if self.current_chapter:
            ch = self.current_chapter
            ch = ch if len(ch) <= 62 else ch[:60].rstrip() + "…"
            canv.drawRightString(PAGE_W - MARGIN, PAGE_H - 12 * mm, ch)
        canv.setStrokeColor(RULE)
        canv.setLineWidth(0.4)
        canv.line(MARGIN, PAGE_H - 13.5 * mm, PAGE_W - MARGIN, PAGE_H - 13.5 * mm)
        canv.drawCentredString(PAGE_W / 2, 10 * mm, str(doc.page))
        canv.restoreState()


def _items(x) -> list:
    """Strings become body paragraphs; lists of strings become bullets; flowables pass through."""
    if x is None:
        return []
    if isinstance(x, str):
        return [P(x)]
    if not isinstance(x, (list, tuple)):
        return [x]
    out = []
    for item in x:
        if isinstance(item, str):
            out.append(P(item))
        elif isinstance(item, list) and all(isinstance(i, str) for i in item):
            out += bullets(item)
        elif isinstance(item, list):
            out += _items(item)
        else:
            out.append(item)
    return out


def topic(title: str, *, what: str, analogy: str | None = None, how=None, project=None, say: str | None = None,
          after_how=None, after_project=None) -> list:
    """One technology, always in the same order:
    what it is -> everyday analogy -> how it works -> in your project -> what to tell the panel."""
    s = [CondPageBreak(85 * mm), P(title, "h1"), P(f"<b>In one sentence.</b> {what}")]
    if analogy:
        s.append(callout(analogy, "example", "Everyday analogy"))
    if how:
        s += [P("How it works", "h2")] + _items(how)
    s += _items(after_how)
    if project:
        s += [P("How your project uses it", "h2")] + _items(project)
    s += _items(after_project)
    if say:
        s.append(callout(say, "key", "If the panel asks"))
    return s


def fig(path, caption: str, width: float = 1.0, max_h: float = 120) -> list:
    return image(str(path), width=CONTENT_W * width, caption=caption, max_h=max_h * mm)


def mono(text: str) -> str:
    return f"<font name='Mono'>{text}</font>"


__all__ = ["PrimerDoc", "topic", "fig", "mono", "KeepTogether", "Spacer", "colors"]
