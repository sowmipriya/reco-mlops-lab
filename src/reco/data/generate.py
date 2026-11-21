"""Generate a synthetic home-improvement (DIY) retail dataset.

The data has learnable structure so recommendation quality is measurable:
- customers belong to segments with preferred product categories,
- shopping sessions are "projects" focused on one category (paint + brushes + tape),
- popularity within a category is long-tailed (Zipf),
- promotions temporarily boost some products.

Usage:
    uv run python -m reco.data.generate --out data/raw
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CATALOG: dict[str, list[str]] = {
    "paint": ["wall_paint", "brushes_rollers", "masking_tape", "primer"],
    "tools": ["power_drills", "hand_tools", "saws", "tool_storage"],
    "garden": ["lawn_mowers", "plants_seeds", "garden_furniture", "watering"],
    "bathroom": ["showers", "taps", "bathroom_furniture", "toilets"],
    "kitchen": ["kitchen_units", "worktops", "sinks", "kitchen_appliances"],
    "electrical": ["cables", "switches_sockets", "circuit_breakers", "smart_home"],
    "plumbing": ["pipes", "fittings", "water_heaters", "pumps"],
    "flooring": ["laminate", "tiles", "carpet", "underlay"],
    "woodworking": ["timber", "panels", "wood_finish", "fixings"],
    "lighting": ["ceiling_lights", "lamps", "bulbs", "outdoor_lighting"],
}
CATEGORY_PRICE = {
    "paint": 25,
    "tools": 80,
    "garden": 60,
    "bathroom": 150,
    "kitchen": 200,
    "electrical": 30,
    "plumbing": 40,
    "flooring": 35,
    "woodworking": 20,
    "lighting": 45,
}
BRANDS = ["Atelier", "ProLine", "HomeCraft", "Verdi", "Brico+", "Solida", "Lumen", "AquaFix"]
REGIONS = [
    "ile_de_france",
    "hauts_de_france",
    "auvergne_rhone_alpes",
    "occitanie",
    "nouvelle_aquitaine",
    "grand_est",
    "bretagne",
    "paca",
]

# segment -> (share of customers, mean sessions over the period, preferred categories)
SEGMENTS: dict[str, tuple[float, float, list[str]]] = {
    "diy_beginner": (0.40, 5, ["paint", "lighting", "tools", "flooring"]),
    "diy_expert": (0.25, 10, ["tools", "woodworking", "electrical", "plumbing"]),
    "professional": (0.15, 18, ["plumbing", "electrical", "bathroom", "kitchen"]),
    "gardener": (0.20, 7, ["garden", "tools", "lighting"]),
}
EVENT_TYPES = ("view", "add_to_cart", "purchase")


@dataclass
class GeneratorConfig:
    n_products_per_subcategory: int = 50
    n_customers: int = 5000
    n_days: int = 120
    n_promotions: int = 80
    end_date: datetime = datetime(2026, 11, 1, tzinfo=UTC)
    seed: int = 42


def _products(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    pid = 0
    for category, subcategories in CATALOG.items():
        for sub in subcategories:
            for i in range(cfg.n_products_per_subcategory):
                pid += 1
                price = rng.lognormal(np.log(CATEGORY_PRICE[category]), 0.6)
                rows.append(
                    {
                        "product_id": f"P{pid:05d}",
                        "product_name": f"{sub.replace('_', ' ').title()} {i + 1:03d}",
                        "category": category,
                        "subcategory": sub,
                        "brand": rng.choice(BRANDS),
                        "price_eur": round(float(price), 2),
                        "is_active": bool(rng.random() > 0.02),
                    }
                )
    return pd.DataFrame(rows)


def _customers(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    names = list(SEGMENTS)
    shares = np.array([SEGMENTS[s][0] for s in names])
    segments = rng.choice(names, size=cfg.n_customers, p=shares / shares.sum())
    signup = [cfg.end_date - timedelta(days=int(d)) for d in rng.integers(cfg.n_days, 3 * 365, size=cfg.n_customers)]
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(1, cfg.n_customers + 1)],
            "segment": segments,
            "region": rng.choice(REGIONS, size=cfg.n_customers),
            "loyalty_member": rng.random(cfg.n_customers) < 0.45,
            "signup_ts": signup,
        }
    )


def _promotions(cfg: GeneratorConfig, products: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    start = cfg.end_date - timedelta(days=cfg.n_days)
    chosen = rng.choice(products["product_id"].to_numpy(), size=cfg.n_promotions, replace=False)
    starts = [start + timedelta(days=int(d)) for d in rng.integers(0, cfg.n_days - 7, size=cfg.n_promotions)]
    return pd.DataFrame(
        {
            "promotion_id": [f"PR{i:04d}" for i in range(1, cfg.n_promotions + 1)],
            "product_id": chosen,
            "discount_pct": rng.choice([10, 15, 20, 30, 40], size=cfg.n_promotions),
            "start_ts": starts,
            "end_ts": [s + timedelta(days=int(rng.integers(5, 15))) for s in starts],
        }
    )


def _events(
    cfg: GeneratorConfig,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    promotions: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    start = cfg.end_date - timedelta(days=cfg.n_days)
    active = products[products["is_active"]]
    by_category = {c: g.reset_index(drop=True) for c, g in active.groupby("category")}
    # Long-tail popularity inside each category (shuffled so ids don't encode popularity).
    base_weight = {}
    for c, g in by_category.items():
        w = 1.0 / np.arange(1, len(g) + 1) ** 1.1
        base_weight[c] = rng.permutation(w)
    promo_windows = {r.product_id: (r.start_ts, r.end_ts) for r in promotions.itertuples()}
    categories = list(CATALOG)

    rows = []
    for cust in customers.itertuples():
        _, mean_sessions, prefs = SEGMENTS[cust.segment]
        # Each customer has a personal mix of the segment's preferred categories.
        personal = rng.choice(prefs, size=min(2, len(prefs)), replace=False)
        n_sessions = rng.poisson(mean_sessions)
        for _ in range(n_sessions):
            ts = start + timedelta(seconds=float(rng.uniform(0, cfg.n_days * 86400)))
            category = rng.choice(personal) if rng.random() < 0.8 else rng.choice(categories)
            catalog = by_category[category]
            weights = base_weight[category].copy()
            # Promotions boost selection probability while active.
            for idx, pid in enumerate(catalog["product_id"]):
                window = promo_windows.get(pid)
                if window and window[0] <= ts <= window[1]:
                    weights[idx] *= 4
            # A session is a "project": concentrate on 1-2 subcategories of the category.
            subs = rng.choice(CATALOG[category], size=2, replace=False)
            mask = catalog["subcategory"].isin(subs).to_numpy()
            weights = np.where(mask, weights * 3, weights)
            n_views = int(rng.integers(2, 9))
            picks = rng.choice(len(catalog), size=n_views, replace=True, p=weights / weights.sum())
            session_id = f"S{int(rng.integers(0, 2**63)):016x}"
            for step, idx in enumerate(picks):
                event_ts = ts + timedelta(seconds=30 * step + float(rng.uniform(0, 20)))
                product_id = catalog.at[idx, "product_id"]
                rows.append((cust.customer_id, session_id, product_id, "view", event_ts, 1))
                if rng.random() < 0.3:
                    qty = int(rng.integers(1, 6)) if cust.segment == "professional" else 1
                    rows.append(
                        (cust.customer_id, session_id, product_id, "add_to_cart", event_ts + timedelta(seconds=5), qty)
                    )
                    if rng.random() < 0.55:
                        rows.append(
                            (
                                cust.customer_id,
                                session_id,
                                product_id,
                                "purchase",
                                event_ts + timedelta(minutes=10),
                                qty,
                            )
                        )

    events = pd.DataFrame(
        rows, columns=["customer_id", "session_id", "product_id", "event_type", "event_ts", "quantity"]
    )
    events.insert(0, "event_id", [f"E{int(x):016x}" for x in rng.integers(0, 2**63, len(events))])
    events["channel"] = rng.choice(["web", "app", "store"], size=len(events), p=[0.6, 0.3, 0.1])
    return events.sort_values("event_ts").reset_index(drop=True)


def generate(cfg: GeneratorConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    products = _products(cfg, rng)
    customers = _customers(cfg, rng)
    promotions = _promotions(cfg, products, rng)
    events = _events(cfg, products, customers, promotions, rng)
    return {"products": products, "customers": customers, "promotions": promotions, "events": events}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw", type=Path)
    parser.add_argument("--customers", default=5000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tables = generate(GeneratorConfig(n_customers=args.customers, seed=args.seed))
    for name, df in tables.items():
        path = args.out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"{name:<11} {len(df):>8,} rows -> {path}")


if __name__ == "__main__":
    main()
