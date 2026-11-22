import json

from reco.streaming.consumer import process
from reco.streaming.events import ClickEvent


def test_valid_event_updates_realtime_features(store):
    event = ClickEvent(client_id="C1", product_id="P9")
    assert process(store, event.model_dump_json()) == "applied"
    _, recent = store.get_customer("C1")
    assert recent == ["P9"]


def test_redelivered_event_is_applied_once(store):
    message = ClickEvent(client_id="C2", product_id="P1").model_dump_json()
    assert process(store, message) == "applied"
    assert process(store, message) == "duplicate"  # at-least-once redelivery
    _, recent = store.get_customer("C2")
    assert recent == ["P1"]


def test_malformed_messages_are_rejected(store):
    assert process(store, b"not json") == "invalid"
    assert process(store, json.dumps({"client_id": "C3"})) == "invalid"  # missing product_id
    assert process(store, json.dumps({"client_id": "C3", "product_id": "P1", "event_type": "click"})) == "invalid"
    assert process(store, None) == "invalid"
    _, recent = store.get_customer("C3")
    assert recent == []
