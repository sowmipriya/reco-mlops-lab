"""Click event contract shared by the producer (website) and the consumer (online features)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ClickEvent(BaseModel):
    """One customer action, as published to the `reco.events` topic (key = client_id)."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    client_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    event_type: Literal["view", "add_to_cart", "purchase"] = "view"
    event_ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = "website"


def ensure_topics(bootstrap: str, topics: dict[str, int]) -> list[str]:
    """Create missing topics ({name: partitions}). Returns the names that were created."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": bootstrap})
    existing = set(admin.list_topics(timeout=10).topics)
    new = [NewTopic(name, num_partitions=n, replication_factor=1) for name, n in topics.items() if name not in existing]
    if new:
        for future in admin.create_topics(new).values():
            future.result()
    return [t.topic for t in new]
