import json
import pytest
from intelligent_trading_bot.common.mappedbus.registry import MappedBusRegistry

CONFIG_JSON = {
    "buses": [
        {
            "name": "price-feed-BTC",
            "slot_size": 256,
            "num_slots": 1024,
            "max_consumers": 2
        },
        {
            "name": "order-book-ETH",
            "slot_size": 512,
            "num_slots": 512,
            "max_consumers": 1
        }
    ]
}


def test_registry_lifecycle():
    registry = MappedBusRegistry()
    bus = registry.create("test-bus", 256, 512, 1)
    assert registry.exists("test-bus")
    assert registry.get("test-bus") is bus
    registry.remove("test-bus")
    assert not registry.exists("test-bus")

def test_registry_from_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CONFIG_JSON))

    registry = MappedBusRegistry()
    registry.load_from_file(str(path))

    assert registry.exists("price-feed-BTC")
    assert registry.exists("order-book-ETH")
    assert registry.get("order-book-ETH").slot_size == 512

def test_registry_from_env(monkeypatch):
    monkeypatch.setenv("MAPPEDBUS_CONFIG", json.dumps(CONFIG_JSON))

    registry = MappedBusRegistry()
    registry.load_from_env()

    assert registry.exists("price-feed-BTC")
    assert registry.get("price-feed-BTC").max_consumers == 2
