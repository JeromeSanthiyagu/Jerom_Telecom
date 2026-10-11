"""Rebuild every submission deliverable from the project's real outputs.

    uvicorn src.main:app --port 8000                     # running API (for the live samples)
    python -m docs.deliverables.collect_samples          # live request/response samples + data/sample_data
    python -m tests.evaluation.run_eval --deepeval       # evaluation results (uses LLM API calls)
    python -m docs.deliverables.build_all                # everything below

Outputs: architecture/Architecture_Diagram.pdf, design/Design_Document.pdf, docs/API_Documentation.pdf,
docs/Dataset_Details.pdf, docs/Evaluation_Report.pdf, presentation/Project_Presentation.pptx.
"""

from __future__ import annotations

import runpy

from docs.deliverables import api_doc, dataset_doc, design_doc, eval_doc, presentation
from src.utils.config import PROJECT_ROOT


def main() -> None:
    runpy.run_path(str(PROJECT_ROOT / "docs" / "make_architecture_diagram.py"), run_name="__main__")
    dataset_doc.main()
    api_doc.main()
    design_doc.main()
    eval_doc.main()
    presentation.build()


if __name__ == "__main__":
    main()
