"""Read helpers for the dbt-built BigQuery tables."""

from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from reco.config import get_settings


def client() -> bigquery.Client:
    s = get_settings()
    return bigquery.Client(project=s.gcp_project, location=s.bq_location)


def read_table(dataset: str, table: str, columns: list[str] | None = None) -> pd.DataFrame:
    s = get_settings()
    cols = ", ".join(columns) if columns else "*"
    sql = f"select {cols} from `{s.gcp_project}.{dataset}.{table}`"
    return client().query(sql).to_dataframe(create_bqstorage_client=False)
