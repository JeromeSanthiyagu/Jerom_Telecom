"""Build docs/Technology_Primer.pdf — every model, method and tool in the project, explained for a newcomer.

    python -m docs.tech_primer.build_primer          # figures are recomputed from the real data and models
"""

from __future__ import annotations

import os
from pathlib import Path

from docs.guide.build_guide import glue_headings
from docs.tech_primer import content_a, content_b, content_c
from docs.tech_primer.figures import build_all
from docs.tech_primer.kit import PrimerDoc
from src.utils.config import PROJECT_ROOT

HERE = Path(__file__).parent
OUT = PROJECT_ROOT / "docs" / "Technology_Primer.pdf"


def story(F: dict) -> list:
    s: list = []
    s += content_a.cover()
    s += content_a.how_to_read()
    s += content_a.part_a()
    s += content_a.part_b(F)
    s += content_a.part_c(F)
    s += content_b.part_d(F)
    s += content_b.part_e(F)
    s += content_c.part_f()
    s += content_c.part_g()
    s += content_c.cheat_sheet()
    s += content_c.glossary()
    return s


def main() -> Path:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    F = build_all(HERE / "figs")
    os.chdir(PROJECT_ROOT)                       # figure paths in the content are project-relative
    doc = PrimerDoc(str(OUT))
    doc.multiBuild(glue_headings(story(F)))
    print(f"wrote {OUT}")
    return OUT


if __name__ == "__main__":
    main()
