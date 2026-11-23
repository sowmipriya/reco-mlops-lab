#!/bin/sh
# One-shot job run by `docker compose up` before the API starts:
#   1. (REFRESH_DATA=1) land data/raw/*.parquet in BigQuery and rebuild the dbt models
#   2. train if there is no champion yet (or REFRESH_DATA=1 / RETRAIN=1)
#   3. always materialize features into Redis (Redis is not persisted)
set -eu

if [ "${REFRESH_DATA:-0}" = "1" ]; then
    echo "== load raw data into BigQuery"
    python -m reco.data.load_raw --src data/raw
    echo "== dbt build"
    (cd dbt && dbt deps --profiles-dir . && dbt build --profiles-dir .)
fi

has_champion() {
    python - <<'EOF'
from mlflow import MlflowClient
from reco.config import get_settings

s = get_settings()
MlflowClient(tracking_uri=s.mlflow_tracking_uri).get_model_version_by_alias(s.model_name, s.champion_alias)
EOF
}

if [ "${REFRESH_DATA:-0}" = "1" ] || [ "${RETRAIN:-0}" = "1" ] || ! has_champion 2>/dev/null; then
    echo "== training pipeline"
    python -m reco.pipelines.run train
else
    echo "== champion model already registered, skipping training (RETRAIN=1 to force)"
fi

echo "== feature materialization -> Redis"
python -m reco.pipelines.run materialize
echo "== bootstrap done"
