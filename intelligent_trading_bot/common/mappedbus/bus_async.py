# common/mappedbus/bus_async.py
import asyncio
import numpy as np
from intelligent_trading_bot.common.mappedbus.bus import MappedBus

class AsyncMappedBus:
    def __init__(self, name: str, slot_size: int, num_slots: int, max_consumers: int):
        self.bus = MappedBus(name, slot_size, num_slots, max_consumers)

    async def send(self, arr: np.ndarray, max_retries: int = 100, base_delay: float = 0.00001) -> bool:
        """Try to send, using exponential backoff if full."""
        for i in range(max_retries):
            if self.bus.send(arr):
                return True
            await asyncio.sleep(base_delay * (2 ** i))
        return False

    async def recv(self, arr_out: np.ndarray, consumer_id: int, max_retries: int = 100, base_delay: float = 0.00001) -> bool:
        """Try to receive, using exponential backoff if empty."""
        for i in range(max_retries):
            if self.bus.recv(arr_out, consumer_id):
                return True
            await asyncio.sleep(base_delay * (2 ** i))
        return False

    def send_nowait(self, arr: np.ndarray) -> bool:
        """Send without waiting (non-blocking)."""
        return self.bus.send(arr)

    def recv_nowait(self, arr_out: np.ndarray, consumer_id: int) -> bool:
        """Receive without waiting (non-blocking)."""
        return self.bus.recv(arr_out, consumer_id)

    async def send_batch(self, arrs: list[np.ndarray], max_retries: int = 100, base_delay: float = 0.00001) -> int:
        """Send a batch of messages asynchronously."""
        count = 0
        for arr in arrs:
            if await self.send(arr, max_retries, base_delay):
                count += 1
        return count

    async def recv_batch(self, out_list: list[np.ndarray], consumer_id: int, max_retries: int = 100, base_delay: float = 0.00001) -> int:
        """Receive a batch of messages asynchronously."""
        count = 0
        for arr_out in out_list:
            if await self.recv(arr_out, consumer_id, max_retries, base_delay):
                count += 1
        return count

    def raw(self):
        return self.bus
