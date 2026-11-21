# End-to-end workflow. Requires: uv, dbt (dbt-bigquery), terraform, docker, gcloud ADC.
include .env
export

DBT = cd dbt && DBT_PROFILES_DIR=. dbt
export ZENML_ANALYTICS_OPT_IN=false

.PHONY: help infra data load dbt-deps dbt-build services mlflow train materialize serve test lint all

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

infra:  ## Provision BigQuery datasets, GCS bucket, service account (Terraform)
	cd infra/terraform && terraform init -input=false && terraform apply -var project_id=$(GCP_PROJECT)

data:  ## Generate the synthetic DIY retail dataset
	uv run python -m reco.data.generate --out data/raw

load:  ## Land raw data into BigQuery (Extract/Load)
	uv run python -m reco.data.load_raw --src data/raw

dbt-deps:
	$(DBT) deps

dbt-build: dbt-deps  ## Transform + test (staging -> marts -> features), Elementary anomalies
	$(DBT) build

services:  ## Start Redis (online store)
	docker run -d --name reco-redis -p 6380:6379 redis:7.4-alpine || docker start reco-redis

mlflow:  ## Start the MLflow tracking server / model registry on :5001
	mkdir -p mlflow && uv run mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db \
		--artifacts-destination mlflow/artifacts --host 127.0.0.1 --port 5001 \
		--allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*"

train:  ## ZenML training pipeline: train, evaluate vs baseline, register, promote
	uv run python -m reco.pipelines.run train

materialize:  ## ZenML pipeline: BigQuery features -> Redis online store
	uv run python -m reco.pipelines.run materialize

serve:  ## Run the real-time API on :8010
	uv run uvicorn reco.serving.app:app --host 127.0.0.1 --port 8010

test:  ## Unit + API tests
	uv run pytest -q

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

all: data load dbt-build train materialize  ## Full batch flow
