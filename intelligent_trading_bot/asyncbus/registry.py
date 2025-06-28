import os
import json
from typing import Dict, Optional, List
from intelligent_trading_bot.asyncbus.channel import Channel


class ChannelRegistry:
    """
    Registry for managing named Channel instances.
    Allows centralized creation, retrieval, and destruction of communication channels.
    """
    _global_instance: Optional["ChannelRegistry"] = None

    def __init__(self):
        self._channels: Dict[str, Channel] = {}

    def create(self, name: str, slot_size: int, num_slots: int, max_consumers: int) -> Channel:
        if name in self._channels:
            raise ValueError(f"Channel with name '{name}' already exists.")
        channel = Channel(name, slot_size, num_slots, max_consumers)
        self._channels[name] = channel
        return channel

    def get(self, name: str) -> Channel:
        if name not in self._channels:
            raise KeyError(f"No channel found with name '{name}'.")
        return self._channels[name]

    def exists(self, name: str) -> bool:
        return name in self._channels

    def all_channels(self) -> Dict[str, Channel]:
        return dict(self._channels)

    def remove(self, name: str):
        if name in self._channels:
            del self._channels[name]

    def close_all(self):
        for name in list(self._channels.keys()):
            del self._channels[name]  # Triggers __dealloc__
        self._channels.clear()

    def load_config(self, config: List[Dict]):
        for entry in config:
            try:
                name = entry["name"]
                slot_size = entry["slot_size"]
                num_slots = entry["num_slots"]
                max_consumers = entry["max_consumers"]
            except KeyError as e:
                raise ValueError(f"Missing required config key: {e}")
            if not self.exists(name):
                self.create(name, slot_size, num_slots, max_consumers)

    def load_from_file(self, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        self.load_config(config.get("channels", []))

    def load_from_env(self):
        raw = os.getenv("CHANNEL_CONFIG", "")
        if raw:
            try:
                config = json.loads(raw)
                self.load_config(config.get("channels", []))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid CHANNEL_CONFIG env JSON: {e}")

    @classmethod
    def global_instance(cls) -> "ChannelRegistry":
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    def __del__(self):
        self.close_all()
