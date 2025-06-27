import os
import json
from typing import Dict, Optional, List
from intelligent_trading_bot.common.mappedbus.bus import MappedBus


class MappedBusRegistry:
    _global_instance: Optional["MappedBusRegistry"] = None

    def __init__(self):
        self._buses: Dict[str, MappedBus] = {}

    def create(self, name: str, slot_size: int, num_slots: int, max_consumers: int) -> MappedBus:
        if name in self._buses:
            raise ValueError(f"Bus with name '{name}' already exists.")
        bus = MappedBus(name, slot_size, num_slots, max_consumers)
        self._buses[name] = bus
        return bus

    def get(self, name: str) -> MappedBus:
        if name not in self._buses:
            raise KeyError(f"No bus found with name '{name}'")
        return self._buses[name]

    def exists(self, name: str) -> bool:
        return name in self._buses

    def all_buses(self) -> Dict[str, MappedBus]:
        return dict(self._buses)

    def remove(self, name: str):
        if name in self._buses:
            del self._buses[name]

    def close_all(self):
        for name, bus in self._buses.items():
            del bus  # trigger __dealloc__
        self._buses.clear()

    def load_config(self, config: List[Dict]):
        for entry in config:
            name = entry["name"]
            slot_size = entry["slot_size"]
            num_slots = entry["num_slots"]
            max_consumers = entry["max_consumers"]
            if not self.exists(name):
                self.create(name, slot_size, num_slots, max_consumers)

    def load_from_file(self, path: str):
        with open(path, "r") as f:
            config = json.load(f)
        self.load_config(config.get("buses", []))

    def load_from_env(self):
        raw = os.getenv("MAPPEDBUS_CONFIG", "")
        if raw:
            try:
                config = json.loads(raw)
                self.load_config(config.get("buses", []))
            except json.JSONDecodeError:
                pass

    @classmethod
    def global_instance(cls) -> "MappedBusRegistry":
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    def __del__(self):
        self.close_all()

