import asyncio
import numpy as np
from typing import Dict, Tuple
from intelligent_trading_bot.common.mappedbus.bus import MappedBus
from intelligent_trading_bot.common.mappedbus.registry import MappedBusRegistry

# TODO: Extend the router to support filtering/topic-based dispatching
class MappedBusRouter:
    def __init__(self):
        self._routes: Dict[Tuple[str, int], asyncio.Queue] = {}
        self._tasks: Dict[Tuple[str, int], asyncio.Task] = {}

    async def _route_loop(self, bus: MappedBus, queue: asyncio.Queue, consumer_id: int, dim: int, poll_delay: float):
        buf = np.empty(dim, dtype=np.float32)
        while True:
            if bus.recv(buf, consumer_id):
                await queue.put(buf.copy())
            else:
                await asyncio.sleep(poll_delay)

    def attach(
        self,
        bus: MappedBus,
        consumer_id: int,
        dim: int,
        poll_delay: float = 1e-5,
        max_queue_size: int = 1000,
    ) -> asyncio.Queue:
        route_id = (bus.name, consumer_id)
        if route_id in self._routes:
            raise ValueError(f"Route for {route_id} already exists.")

        queue = asyncio.Queue(maxsize=max_queue_size)
        self._routes[route_id] = queue

        task = asyncio.create_task(self._route_loop(bus, queue, consumer_id, dim, poll_delay))
        self._tasks[route_id] = task

        return queue

    def attach_all_from_registry(self, dim: int, poll_delay: float = 1e-5):
        registry = MappedBusRegistry.global_instance()
        for name, bus in registry.all_buses().items():
            for consumer_id in range(bus.max_consumers):
                route_id = (name, consumer_id)
                if route_id not in self._routes:
                    self.attach(bus, consumer_id, dim, poll_delay)

    def get_queue(self, bus_name: str, consumer_id: int) -> asyncio.Queue:
        return self._routes[(bus_name, consumer_id)]

    async def stop_all(self):
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._routes.clear()
        self._tasks.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_all()
