# Serving image: only the runtime dependencies (no ZenML / BigQuery / dev tools).
FROM python:3.12-slim AS base
RUN pip install --no-cache-dir uv==0.6.8
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["/app/.venv/bin/uvicorn", "reco.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
