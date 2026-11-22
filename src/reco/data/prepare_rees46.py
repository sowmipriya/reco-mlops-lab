"""Prepare the public REES46 e-commerce dataset as a home-improvement (DIY) store.

Source: "eCommerce behavior data from multi category store" (REES46 Marketing Platform),
https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store
One month (e.g. 2019-Nov.csv.zip, ~67M events) is streamed without unzipping, filtered to the
categories a DIY retailer sells, and reshaped into the raw tables the pipeline expects:

    events     one row per view / add_to_cart / purchase
    products   one row per product (department, sub-category, brand, median price)
    customers  one row per customer (first time seen); segments are derived later in dbt

Usage:
    uv run python -m reco.data.prepare_rees46 --src data/source/2019-Nov.csv.zip --out data/raw
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv

# category_code prefix -> department of a DIY store. First match wins, so order matters.
DEPARTMENTS: list[tuple[str, str]] = [
    ("construction.components.faucet", "plumbing_bathroom"),
    ("furniture.bathroom.", "plumbing_bathroom"),
    ("construction.tools.", "tools"),
    ("furniture.kitchen.", "kitchen"),
    ("appliances.kitchen.hood", "kitchen"),
    ("appliances.kitchen.hob", "kitchen"),
    ("appliances.kitchen.oven", "kitchen"),
    ("appliances.kitchen.dishwasher", "kitchen"),
    ("appliances.environment.air_heater", "heating_climate"),
    ("appliances.environment.water_heater", "heating_climate"),
    ("appliances.environment.air_conditioner", "heating_climate"),
    ("appliances.environment.fan", "heating_climate"),
    ("furniture.living_room.cabinet", "storage_lighting"),
    ("furniture.universal.light", "storage_lighting"),
    ("country_yard.", "garden"),
]
EVENT_TYPES = {"view": "view", "cart": "add_to_cart", "purchase": "purchase"}


def department(category_code: str) -> str | None:
    for prefix, dept in DEPARTMENTS:
        if category_code.startswith(prefix):
            return dept
    return None


def read_filtered(src: Path) -> pd.DataFrame:
    """Stream the zipped CSV and keep only DIY categories (the full month never sits in memory)."""
    prefixes = sorted({p for p, _ in DEPARTMENTS})
    tables = []
    with zipfile.ZipFile(src) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            reader = pv.open_csv(
                f,
                read_options=pv.ReadOptions(block_size=64 << 20),
                convert_options=pv.ConvertOptions(
                    column_types={"product_id": pa.string(), "user_id": pa.string(), "price": pa.float64()},
                    include_columns=[
                        "event_time",
                        "event_type",
                        "product_id",
                        "category_code",
                        "brand",
                        "price",
                        "user_id",
                        "user_session",
                    ],
                ),
            )
            for batch in reader:
                code = pc.fill_null(batch["category_code"], "")
                mask = pc.starts_with(code, prefixes[0])
                for p in prefixes[1:]:
                    mask = pc.or_(mask, pc.starts_with(code, p))
                tables.append(pa.Table.from_batches([batch]).filter(mask))
    return pa.concat_tables(tables).to_pandas()


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    df = df[df["event_type"].isin(EVENT_TYPES) & df["user_session"].notna()].copy()
    df["event_ts"] = pd.to_datetime(df["event_time"].str.replace(" UTC", "", regex=False), utc=True)
    df["brand"] = df["brand"].replace("", None)  # the CSV encodes a missing brand as an empty string
    df["department"] = df["category_code"].map({c: department(c) for c in df["category_code"].unique()})

    # Identical rows are duplicates sent twice by the source: same hash -> same event_id,
    # and dbt staging keeps one (the same at-least-once handling as for a Kafka topic).
    key = df[["event_time", "event_type", "product_id", "user_id", "user_session"]]
    events = pd.DataFrame(
        {
            "event_id": "E" + pd.util.hash_pandas_object(key, index=False).map("{:016x}".format),
            "customer_id": "U" + df["user_id"],
            "session_id": df["user_session"],
            "product_id": "P" + df["product_id"],
            "event_type": df["event_type"].map(EVENT_TYPES),
            "event_ts": df["event_ts"],
            "quantity": 1,
            "channel": "web",
        }
    ).sort_values("event_ts", kind="stable")

    by_product = df.groupby("product_id")
    products = pd.DataFrame(
        {
            "category_code": by_product["category_code"].agg(lambda s: s.mode().iat[0]),
            "brand": by_product["brand"].agg(lambda s: s.mode().iat[0] if s.notna().any() else "unbranded"),
            "price": by_product["price"].median().round(2),
        }
    ).reset_index()
    products["category"] = products["category_code"].map(department)
    products["subcategory"] = products["category_code"].str.rsplit(".", n=1).str[-1]
    products["product_name"] = (
        products["brand"].str.title()
        + " "
        + products["subcategory"].str.replace("_", " ")
        + " "
        + products["product_id"]
    )
    products["product_id"] = "P" + products["product_id"]
    products["is_active"] = True
    products = products[
        ["product_id", "product_name", "category", "subcategory", "category_code", "brand", "price", "is_active"]
    ]

    customers = events.groupby("customer_id")["event_ts"].min().rename("first_seen_ts").reset_index()
    return {"events": events.reset_index(drop=True), "products": products, "customers": customers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="data/source/2019-Nov.csv.zip", type=Path)
    parser.add_argument("--out", default="data/raw", type=Path)
    args = parser.parse_args()

    t0 = time.time()
    raw = read_filtered(args.src)
    print(f"kept {len(raw):,} DIY rows from {args.src.name} in {time.time() - t0:.0f}s")
    args.out.mkdir(parents=True, exist_ok=True)
    for old in args.out.glob("*.parquet"):
        old.unlink()
    for name, table in build_tables(raw).items():
        table.to_parquet(args.out / f"{name}.parquet", index=False)
        print(f"{name:<10} {len(table):>10,} rows")


if __name__ == "__main__":
    main()
