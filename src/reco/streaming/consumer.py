"""Kafka consumer that keeps the real-time features in Redis fresh.

    reco.events --(consumer group reco-online-features)--> Redis fs:recent:{client_id}
                                                       \\-> reco.events.dlq (malformed messages)

Delivery is at-least-once: the offset is committed only AFTER Redis is updated, so a crash
replays messages instead of losing them. Replays are made harmless by an idempotency marker
per event_id (and the API de-duplicates product ids in the history anyway).

    uv run python -m reco.streaming.consumer
"""

from __future__ import annotations

import json
import logging
import signal
import time
from collections import Counter
from typing import Literal

from pydantic import ValidationError

from reco.config import get_settings
from reco.feature_store.online import OnlineFeatureStore
from reco.streaming.events import ClickEvent, ensure_topics

logger = logging.getLogger("reco.consumer")
Outcome = Literal["applied", "duplicate", "invalid"]


def process(store: OnlineFeatureStore, value: bytes | str | None) -> Outcome:
    """Apply one message to the online store. Pure apart from Redis, so it is unit-tested."""
    try:
        event = ClickEvent.model_validate(json.loads(value or b""))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return "invalid"
    if store.is_event_seen(event.event_id):
        return "duplicate"
    # Order matters: update first, mark second. A crash in between replays the event
    # (a harmless duplicate product id), it never loses it.
    store.push_event(event.client_id, event.product_id)
    store.mark_event_seen(event.event_id)
    return "applied"


def run() -> None:
    from confluent_kafka import Consumer, KafkaException, Producer

    s = get_settings()
    created = ensure_topics(s.kafka_bootstrap, {s.events_topic: 3, s.events_dlq_topic: 1})
    if created:
        logger.info("created topics %s", created)
    store = OnlineFeatureStore.from_url(s.redis_url, s.recent_events_max)
    consumer = Consumer(
        {
            "bootstrap.servers": s.kafka_bootstrap,
            "group.id": s.events_consumer_group,
            "enable.auto.commit": False,  # commit only after Redis is updated
            "auto.offset.reset": "earliest",
        }
    )
    dlq = Producer({"bootstrap.servers": s.kafka_bootstrap, "acks": "all"})
    consumer.subscribe([s.events_topic])

    running = True

    def stop(*_) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    stats: Counter = Counter()
    last_log = time.monotonic()
    logger.info("consuming %s as group %s", s.events_topic, s.events_consumer_group)
    try:
        while running:
            msg = consumer.poll(1.0)
            if msg is not None:
                if msg.error():
                    raise KafkaException(msg.error())
                outcome = process(store, msg.value())
                if outcome == "invalid":
                    dlq.produce(s.events_dlq_topic, key=msg.key(), value=msg.value())
                    dlq.flush(5)
                stats[outcome] += 1
                consumer.commit(message=msg, asynchronous=False)
                logger.info(
                    "%s partition=%s offset=%s key=%s",
                    outcome,
                    msg.partition(),
                    msg.offset(),
                    (msg.key() or b"").decode(),
                )
            if time.monotonic() - last_log > 30 and stats:
                logger.info("totals %s", dict(stats))
                last_log = time.monotonic()
    finally:
        consumer.close()  # leaves the group cleanly so partitions are reassigned at once
        logger.info("stopped, totals %s", dict(stats))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()
