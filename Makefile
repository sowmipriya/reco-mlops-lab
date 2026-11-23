# End-to-end workflow, everything in Docker (docker compose). Host requirements: docker and gcloud ADC
# (~/.config/gcloud/application_default_credentials.json, mounted read-only into the containers).
include .env
export

DC = docker compose
RUN = $(DC) run --rm pipelines
# One month of the public REES46 dataset (Kaggle), see src/reco/data/prepare_rees46.py.
# realpath resolves the data/source symlink so the real file can be mounted into the container.
REES46_ZIP ?= data/source/2019-Nov.csv.zip

.PHONY: help infra build up down logs data load dbt-deps dbt-build train materialize shell test lint all

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

infra:  ## Provision BigQuery datasets, GCS bucket, service account (Terraform)
	$(DC) run --rm terraform init -input=false
	$(DC) run --rm terraform apply -var project_id=$(GCP_PROJECT)

build:  ## Build the serving and pipelines images
	$(DC) --profile jobs build

up:  ## Start Redis, Redpanda+console, MLflow, ZenML server, API, consumer, website
	$(DC) up -d --build

down:  ## Stop everything (volumes are kept)
	$(DC) --profile jobs down

logs:  ## Follow logs of all services
	$(DC) logs -f

data:  ## Prepare the public REES46 month as a DIY store (filter + reshape)
	@test -n "$(realpath $(REES46_ZIP))" || { echo "$(REES46_ZIP) not found (broken symlink?). Pass REES46_ZIP=/path/to/2019-Nov.csv.zip"; exit 1; }
	$(DC) run --rm -v "$(realpath $(REES46_ZIP))":/app/source/rees46.zip:ro pipelines \
		python -m reco.data.prepare_rees46 --src /app/source/rees46.zip --out data/raw

load:  ## Land raw data into BigQuery (Extract/Load)
	$(RUN) python -m reco.data.load_raw --src data/raw

dbt-deps:
	$(RUN) sh -c "cd dbt && dbt deps --profiles-dir ."

dbt-build: dbt-deps  ## Transform + test (staging -> marts -> features), Elementary anomalies
	$(RUN) sh -c "cd dbt && dbt build --profiles-dir ."

train:  ## ZenML training pipeline: train, evaluate vs baseline, register, promote
	$(RUN) python -m reco.pipelines.run train

materialize:  ## ZenML pipeline: BigQuery features -> Redis online store
	$(RUN) python -m reco.pipelines.run materialize

shell:  ## Interactive shell in the pipelines container
	$(RUN) bash

test:  ## Unit + API tests
	$(DC) run --rm --no-deps dev pytest -q -p no:cacheprovider

lint:
	$(DC) run --rm --no-deps dev sh -c "ruff check src tests && ruff format --check src tests"

all: data load dbt-build train materialize  ## Full batch flow
