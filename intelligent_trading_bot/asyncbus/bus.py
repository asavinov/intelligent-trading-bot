import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Union

from intelligent_trading_bot.asyncbus.channel import Channel
from intelligent_trading_bot.asyncbus.router import ChannelRouter
from intelligent_trading_bot.asyncbus.registry import ChannelRegistry


class AsyncMappedBus:
    """
    A topic-based pub/sub interface for high-performance in-memory communication via channels.

    Responsibilities:
    - Register topics (channels)
    - Publish messages (async/backoff supported)
    - Subscribe consumers to topics (auto ID allocation)
    - Memory and consumer tracking per topic
    """

    def __init__(self):
        self._registry = ChannelRegistry.global_instance()
        self._router = ChannelRouter()

        self._topics: Dict[str, Channel] = {}
        self._consumer_ids: Dict[str, Dict[str, int]] = {}
        self._free_ids: Dict[str, List[int]] = {}

    def create_topic(
        self,
        topic: str,
        slot_size: int,
        num_slots: int,
        max_consumers: int,
    ) -> Channel:
        """
        Create a new topic and its backing channel.
        """
        if topic in self._topics:
            raise ValueError(f"Topic '{topic}' already exists.")
        channel = self._registry.create(topic, slot_size, num_slots, max_consumers)
        self._topics[topic] = channel
        self._consumer_ids[topic] = {}
        self._free_ids[topic] = list(reversed(range(max_consumers)))  # Stack
        return channel

    def topic_memory_bytes(self, topic: str) -> int:
        """
        Estimate memory usage in bytes for a topic's backing channel.
        """
        ch = self._get_channel(topic)
        aligned_slot = ((1 + ch.slot_size + 64 - 1) // 64) * 64  # 64-byte cache alignment
        return aligned_slot * ch.num_slots

    def subscribe(
        self,
        topic: str,
        consumer_name: str,
        dim: int,
        *,
        poll_delay: float = 1e-5,
        max_queue_size: int = 1000,
    ) -> asyncio.Queue:
        """
        Subscribe a consumer to a topic and return an asyncio.Queue of messages.
        """
        ch = self._get_channel(topic)

        if consumer_name in self._consumer_ids[topic]:
            raise ValueError(f"Consumer '{consumer_name}' already subscribed to topic '{topic}'.")

        if not self._free_ids[topic]:
            raise RuntimeError(f"No free consumer slots for topic '{topic}'.")

        consumer_id = self._free_ids[topic].pop()
        self._consumer_ids[topic][consumer_name] = consumer_id

        return self._router.attach(
            ch,
            consumer_id=consumer_id,
            dim=dim,
            poll_delay=poll_delay,
            max_queue_size=max_queue_size,
        )

    def unsubscribe(self, topic: str, consumer_name: str):
        """
        Unsubscribe a consumer and reclaim their slot.
        """
        if topic not in self._consumer_ids or consumer_name not in self._consumer_ids[topic]:
            raise KeyError(f"Consumer '{consumer_name}' not subscribed to topic '{topic}'.")

        consumer_id = self._consumer_ids[topic].pop(consumer_name)
        self._free_ids[topic].append(consumer_id)
        
        ch = self._get_channel(topic)
        self._router.unattach(ch, consumer_id)

    def get_queue(self, topic: str, consumer_name: str) -> asyncio.Queue:
        """
        Access the underlying asyncio queue for a consumer.
        """
        consumer_id = self._consumer_ids[topic][consumer_name]
        return self._router.get_queue(topic, consumer_id)

    def publish_nowait(self, topic: str, arr: np.ndarray) -> bool:
        """
        Publish to topic immediately without waiting.
        """
        if self._get_channel(topic).send(arr):
            return True
        return False

    async def publish_async(
        self,
        topic: str,
        arr: np.ndarray,
        max_retries: int = 100,
        base_delay: float = 1e-5,
    ) -> bool:
        """
        Publish to topic with async retry and exponential backoff.
        """
        channel = self._get_channel(topic)
        for i in range(max_retries):
            if channel.send(arr):
                return True
            await asyncio.sleep(base_delay * (2 ** i))
        return False

    def summary(self, topic: Optional[str] = None, as_dict: bool = False) -> Union[str, List[Dict[str, Any]]]:
        """
        Show a human-readable or machine-friendly summary of topic stats.
        """
        def describe(ch: Channel) -> Dict[str, Any]:
            aligned_slot = ((1 + ch.slot_size + 64 - 1) // 64) * 64
            memory = aligned_slot * ch.num_slots
            return {
                "topic": ch.name,
                "slot_size": ch.slot_size,
                "num_slots": ch.num_slots,
                "max_consumers": ch.max_consumers,
                "memory_bytes": memory,
                "write_idx": ch.write_idx,
                "read_indices": list(ch.read_indices),
            }

        if topic:
            ch = self._get_channel(topic)
            info = describe(ch)
            return info if as_dict else self._format_summary([info])

        all_info = [describe(ch) for ch in self._topics.values()]
        return all_info if as_dict else self._format_summary(all_info)

    def _format_summary(self, data: List[Dict[str, Any]]) -> str:
        lines = []
        for d in data:
            lines.append(
                f"[Topic: {d['topic']}] Slots={d['num_slots']} Size={d['slot_size']} Bytes "
                f"Consumers={d['max_consumers']} Mem={round(d['memory_bytes']/1024, 2)}KB\n"
                f"Write Index={d['write_idx']} Reads={d['read_indices']}"
            )
        return "\n\n".join(lines)

    async def stop(self):
        """
        Stop all routers and clean up state.
        """
        await self._router.stop_all()
        self._topics.clear()
        self._consumer_ids.clear()
        self._free_ids.clear()

    def _get_channel(self, topic: str) -> Channel:
        if topic not in self._topics:
            raise KeyError(f"Topic '{topic}' does not exist.")
        return self._topics[topic]
