# One image for both services (API and UI); docker-compose picks the command.

# Stage 1: build the React web UI (served by the API at /ui)
FROM node:20-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface
WORKDIR /app

# CPU-only torch keeps the image small
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
COPY --from=web /web/dist web/dist
# Cache the embedding model inside the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000 8501
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
