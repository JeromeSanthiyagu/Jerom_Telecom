# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Checksum mismatch for …csv.xz` | Partial download or upstream change | Delete `data/raw/*.xz` and run `python -m src.preprocessing.download_dataset --force`. If the upstream file truly changed, review it before updating `SHA256`. |
| `FileNotFoundError: … incidents.parquet` / `Vector index not built` | Build steps not run | Run `sh src/preprocessing/build_all.sh` (or each step in the README order) |
| `BM25 index is out of sync with incidents` | `build_incidents` re-run without re-indexing | `python -m src.preprocessing.build_index` |
| `Index built with X, config uses Y` | Embedding model changed in `retrieval.yaml` | Rebuild the index |
| `Thresholds drifted from data` | Processed data changed | Inspect, then update `thresholds.yaml` (values printed by `python -m src.preprocessing.build_thresholds`) |
| First `/analyze` takes ~40 s | Chroma loads the HNSW index and the embedding model loads | The API warms up at start-up; wait for `warm-up done` in the log. Set `HF_HUB_OFFLINE=1` to skip Hugging Face network checks |
| `narrative.source = "template"` with `llm_rejected: true` | The LLM's text failed the grounding check twice | Working as designed. See `narrative.attempts[*].grounding_issues` and `rejected_output` |
| `llm_judge_info.status = "error"` | Bad key, model name or network | Check `NVIDIA_API_KEY` (or `ANTHROPIC_API_KEY`) / `LLM_MODEL` in `.env`; `/health` shows the active provider |
| Want no LLM at all | — | `LLM_PROVIDER=offline` in `.env`, or `"use_llm": false` per request |
| `Port … is not available` / `error while attempting to bind` | Another server uses 8000/8501 | `uvicorn src.main:app --port 8010` and `API_URL=http://localhost:8010 streamlit run frontend.py --server.port 8511` |
| Streamlit shows `connection_error` | API not running, or wrong `API_URL` | Start the API first; check `API_URL` |
| `/ui` returns 404 | The React UI has not been built | `cd frontend && npm install && npm run build`, then restart the API |
| Web UI header says "API unreachable" | API stopped, or (dev server) wrong `API_TARGET` | Start the API; for `npm run dev`, set `API_TARGET=http://localhost:8010` in `frontend/.env.local` if not on 8000 |
| Web UI shows "LLM explanation unavailable" | The provider returned an error (for example, no API credit) | The notice quotes the provider's message; add credit or switch the LLM toggle off |
| `Failed to send telemetry event` lines | ChromaDB telemetry client version mismatch | Harmless; telemetry is disabled (`ANONYMIZED_TELEMETRY=False`) |
| Tests skip ("not built") | Full-data tests need the built artifacts | Build the data, then rerun `pytest` |
| `422 invalid_filter` "Unknown city" | City names must match exactly (case-insensitive) | Use `/api/v1/records/{id}` to see stored names |
| `numpy` 2.x import errors | A package upgraded numpy | `pip install "numpy<2"` |
