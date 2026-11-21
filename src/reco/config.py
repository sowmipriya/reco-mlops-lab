"""Central configuration, read from environment variables (or a local .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GCP / BigQuery
    gcp_project: str = "reco-mlops-lab"
    bq_location: str = "EU"
    raw_dataset: str = "raw"
    marts_dataset: str = "marts"
    features_dataset: str = "features"

    # MLflow
    mlflow_tracking_uri: str = "http://127.0.0.1:5001"
    mlflow_experiment: str = "reco-embeddings"
    model_name: str = "reco-embeddings"
    champion_alias: str = "champion"

    # Online feature store
    redis_url: str = "redis://localhost:6380/0"
    recent_events_max: int = 20

    # Modelling
    top_k: int = 10
    test_days: int = 14


@lru_cache
def get_settings() -> Settings:
    return Settings()
