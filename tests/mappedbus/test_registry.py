import pytest
import json
import os
from intelligent_trading_bot.asyncbus.registry import ChannelRegistry

def test_channel_create_and_get():
    registry = ChannelRegistry()
    ch = registry.create("test", 16, 128, 2)
    assert registry.get("test") is ch
    assert registry.exists("test")

def test_channel_registry_duplicate_create_raises():
    registry = ChannelRegistry()
    registry.create("test", 16, 128, 2)
    with pytest.raises(ValueError):
        registry.create("test", 16, 128, 2)

def test_channel_registry_load_from_config():
    config = [
        {"name": "one", "slot_size": 32, "num_slots": 64, "max_consumers": 1},
        {"name": "two", "slot_size": 64, "num_slots": 128, "max_consumers": 2},
    ]
    registry = ChannelRegistry()
    registry.load_config(config)
    assert registry.exists("one")
    assert registry.exists("two")

def test_channel_registry_load_from_env(monkeypatch):
    cfg = {
        "channels": [
            {"name": "env-bus", "slot_size": 32, "num_slots": 64, "max_consumers": 1}
        ]
    }
    monkeypatch.setenv("CHANNEL_CONFIG", json.dumps(cfg))
    registry = ChannelRegistry()
    registry.load_from_env()
    assert registry.exists("env-bus")
