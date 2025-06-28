import asyncio
import numpy as np
from typing import Callable, Dict, Tuple, Optional, List
from intelligent_trading_bot.asyncbus.channel import Channel


class ChannelRouter:
    def __init__(self):
        self._routes: Dict[Tuple[str, int], asyncio.Queue] = {}
        self._tasks: Dict[Tuple[str, int], asyncio.Task] = {}
        self._topic_routes: Dict[str, List[Tuple[int, Callable[[np.ndarray], bool]]]] = {}

    async def _route_loop(self, channel: Channel, topic: str, consumer_id: int, queue: asyncio.Queue,
                          dim: int, poll_delay: float, filter_fn: Optional[Callable[[np.ndarray], bool]] = None):
        buf = np.empty(dim, dtype=np.float32)
        while True:
            if channel.recv(buf, consumer_id):
                if filter_fn is None or filter_fn(buf):
                    await queue.put(buf.copy())
            else:
                await asyncio.sleep(poll_delay)

    def attach(
        self,
        channel: Channel,
        consumer_id: int,
        dim: int,
        poll_delay: float = 1e-5,
        max_queue_size: int = 1000,
        topic: Optional[str] = None,
        filter_fn: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> asyncio.Queue:
        route_id = (channel.name, consumer_id)
        if route_id in self._routes:
            raise ValueError(f"Route for {route_id} already exists.")

        queue = asyncio.Queue(maxsize=max_queue_size)
        self._routes[route_id] = queue

        _topic = topic or channel.name
        if _topic not in self._topic_routes:
            self._topic_routes[_topic] = []
        self._topic_routes[_topic].append((consumer_id, filter_fn))

        task = asyncio.create_task(
            self._route_loop(channel, _topic, consumer_id, queue, dim, poll_delay, filter_fn)
        )
        self._tasks[route_id] = task

        return queue

    def attach_all_from_registry(self, dim: int, poll_delay: float = 1e-5):
        from intelligent_trading_bot.asyncbus.registry import ChannelRegistry
        registry = ChannelRegistry.global_instance()
        for name, channel in registry.all_channels().items():
            for consumer_id in range(channel.max_consumers):
                route_id = (name, consumer_id)
                if route_id not in self._routes:
                    self.attach(channel, consumer_id, dim, poll_delay)

    def get_queue(self, channel_name: str, consumer_id: int) -> asyncio.Queue:
        return self._routes[(channel_name, consumer_id)]

    async def unattach(self, channel: Channel, consumer_id: int):
        """
        Remove a specific consumer route and clean up task and queue.
        """
        route_id = (channel.name, consumer_id)
        if route_id not in self._routes:
            raise KeyError(f"Route {route_id} not found.")

        # Cancel background task
        task = self._tasks.pop(route_id)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Remove route and queue
        self._routes.pop(route_id)

        # Clean from _topic_routes
        for topic, consumers in self._topic_routes.items():
            self._topic_routes[topic] = [
                (cid, f) for cid, f in consumers if cid != consumer_id
            ]
        # Clean empty topics
        self._topic_routes = {
            k: v for k, v in self._topic_routes.items() if v
        }
        
    async def stop_all(self):
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._routes.clear()
        self._tasks.clear()
        self._topic_routes.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_all()

"""
router = ChannelRouter()

# Filter for high volume trades
def high_volume_filter(vec: np.ndarray) -> bool:
    return vec[1] > 10_000

queue = router.attach(
    channel=my_channel,
    consumer_id=0,
    dim=64,
    topic="price-feed-BTC",
    filter_fn=high_volume_filter,
)

"""