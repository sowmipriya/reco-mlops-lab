"""Land the prepared parquet files into the BigQuery `raw` dataset.

In production this is the Extract/Load step fed by other domains (Kafka topics,
domain APIs). Here it is a batch load of the public REES46 month (see prepare_rees46.py).

Usage:
    uv run python -m reco.data.load_raw --src data/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from reco.config import get_settings

# table -> (time partitioning column, clustering columns)
LAYOUT: dict[str, tuple[str | None, list[str] | None]] = {
    "events": ("event_ts", ["customer_id", "event_type"]),
    "products": (None, ["category"]),
    "customers": (None, None),
}


def load_table(client: bigquery.Client, df: pd.DataFrame, table_id: str, name: str) -> int:
    partition_col, cluster_cols = LAYOUT[name]
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        time_partitioning=(
            bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field=partition_col)
            if partition_col
            else None
        ),
        clustering_fields=cluster_cols,
    )
    df = df.assign(_loaded_at=pd.Timestamp.now(tz="UTC"))
    client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
    return client.get_table(table_id).num_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/raw", type=Path)
    args = parser.parse_args()

    settings = get_settings()
    client = bigquery.Client(project=settings.gcp_project, location=settings.bq_location)
    for name in LAYOUT:
        df = pd.read_parquet(args.src / f"{name}.parquet")
        table_id = f"{settings.gcp_project}.{settings.raw_dataset}.{name}"
        rows = load_table(client, df, table_id, name)
        print(f"{table_id:<50} {rows:>8,} rows")


if __name__ == "__main__":
    main()
