"""Feedback loop (Phase 26).

Feedback is stored in SQLite (data/feedback/feedback.db). It is NOT used to
retrain anything automatically: a single rating is weak evidence. Instead,
`export_evaluation_candidates` writes rated cases to a JSONL file that a
person reviews before adding them to the evaluation set
(tests/evaluation/eval_cases.json).

    python -m src.services.feedback export
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.config import PROJECT_ROOT

FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
DB_PATH = FEEDBACK_DIR / "feedback.db"
EXPORT_PATH = FEEDBACK_DIR / "evaluation_candidates.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    analysis_id TEXT,
    query TEXT NOT NULL,
    retrieved_record_ids TEXT NOT NULL,
    diagnosis TEXT,
    recommendations TEXT,
    risk_prediction TEXT,
    rating TEXT NOT NULL CHECK (rating IN ('useful', 'not_useful')),
    comment TEXT
)
"""


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(SCHEMA)
    return con


def save_feedback(item: dict[str, Any], db_path: Path = DB_PATH) -> dict:
    fid = str(uuid.uuid4())
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect(db_path) as con:
        con.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?,?)",
            (fid, created, item.get("analysis_id"), item["query"], json.dumps(item.get("retrieved_record_ids", [])),
             json.dumps(item.get("diagnosis")), json.dumps(item.get("recommendations")),
             json.dumps(item.get("risk_prediction")), item["rating"], item.get("comment")),
        )
    return {"feedback_id": fid, "created_at": created, "stored": True,
            "note": "Stored for evaluation review; no model is retrained automatically."}


def list_feedback(db_path: Path = DB_PATH) -> list[dict]:
    with _connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM feedback ORDER BY created_at").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        for k in ("retrieved_record_ids", "diagnosis", "recommendations", "risk_prediction"):
            d[k] = json.loads(d[k]) if d[k] else None
        out.append(d)
    return out


def export_evaluation_candidates(db_path: Path = DB_PATH, out_path: Path = EXPORT_PATH) -> int:
    rows = list_feedback(db_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"query": r["query"], "rating": r["rating"], "comment": r["comment"],
                                 "retrieved_record_ids": r["retrieved_record_ids"],
                                 "diagnosis": r["diagnosis"], "needs_review": True}) + "\n")
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        print(f"exported {export_evaluation_candidates()} feedback rows to {EXPORT_PATH}")
    else:
        for row in list_feedback():
            print(row["created_at"], row["rating"], row["query"][:80])
