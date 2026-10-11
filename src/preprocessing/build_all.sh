#!/bin/sh
# Rebuild everything from the raw dataset, in order.
set -e
python -m src.preprocessing.download_dataset        # Phase 1: pinned download + checksum
python -m src.preprocessing.inspect_dataset         # Phase 1: inspection report (docs/phase1)
python -m src.preprocessing.processor               # Phase 2: validated, cleaned records
python -m src.preprocessing.build_thresholds --check
python -m src.preprocessing.build_incidents         # Phases 3-8: conditions, labels, search text
python -m src.preprocessing.build_index             # Phases 9-10: BM25 + Chroma (~8 min on CPU)
python -m src.services.models.train             # Phases 19-22: models
python docs/make_architecture_diagram.py
