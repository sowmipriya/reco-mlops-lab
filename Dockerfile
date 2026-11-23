# Two images from one file:
#   pipelines  data prep, BigQuery load, dbt, ZenML pipelines, demo website, tests and lint
#   serving    real-time API and Kafka consumer, runtime dependencies only (default / last stage)
FROM python:3.12-slim AS base
RUN pip install --no-cache-dir uv==0.6.8
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1
WORKDIR /app
# uid 1000, the same as the zenml user of the ZenML server image (they share the artifact volume)
RUN useradd --create-home --uid 1000 app


FROM base AS pipelines
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --group pipelines --group dev --no-install-project
COPY src ./src
RUN uv sync --frozen --group pipelines --group dev
# dbt in its own virtualenv so its dependencies never clash with the project's
RUN uv venv /opt/dbt && uv pip install --python /opt/dbt/bin/python dbt-core==1.12.7 dbt-bigquery==1.12.4
ENV PATH="/app/.venv/bin:/opt/dbt/bin:$PATH" \
    ZENML_CONFIG_PATH=/home/app/.zenconfig \
    ZENML_ANALYTICS_OPT_IN=false \
    RUFF_CACHE_DIR=/tmp/ruff_cache
COPY scripts/bootstrap.sh /usr/local/bin/bootstrap
RUN chmod +x /usr/local/bin/bootstrap && chown -R app /app
USER app


FROM base AS serving
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups
ENV PATH="/app/.venv/bin:$PATH"
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "reco.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
