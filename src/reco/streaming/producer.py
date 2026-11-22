"""Publishes click events to Kafka. Keyed by client_id: one customer's events stay ordered
in one partition, which the consumer relies on to keep "most recent first"."""

from __future__ import annotations

import time

from confluent_kafka import Producer

from reco.streaming.events import ClickEvent


class EventProducer:
    def __init__(self, bootstrap: str, topic: str) -> None:
        self.topic = topic
        self.producer = Producer(
            {
                "bootstrap.servers": bootstrap,
                "acks": "all",  # wait for the broker to persist the event
                "enable.idempotence": True,  # producer retries never create duplicates
                "linger.ms": 5,
            }
        )

    def publish(self, event: ClickEvent, timeout_s: float = 5.0) -> dict:
        """Send and wait for the broker's acknowledgement (simple and visible for the demo;
        a high-traffic producer would use delivery callbacks instead of waiting)."""
        start = time.perf_counter()
        result: dict = {}

        def on_delivery(err, msg) -> None:
            result.update(error=str(err) if err else None, partition=msg.partition(), offset=msg.offset())

        self.producer.produce(
            self.topic,
            key=event.client_id,
            value=event.model_dump_json(),
            on_delivery=on_delivery,
        )
        self.producer.flush(timeout_s)
        if not result or result.get("error"):
            raise RuntimeError(f"Kafka delivery failed: {result.get('error', 'timeout')}")
        return result | {
            "topic": self.topic,
            "event_id": event.event_id,
            "ms": round((time.perf_counter() - start) * 1000, 1),
        }
