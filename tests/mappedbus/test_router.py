import asyncio
import numpy as np
import pytest

from intelligent_trading_bot.asyncbus.channel import Channel
from intelligent_trading_bot.asyncbus.router import ChannelRouter
from intelligent_trading_bot.asyncbus.registry import ChannelRegistry


@pytest.mark.asyncio
async def test_router_with_topic_and_filter():
    DIM = 4
    TOTAL = 50

    channel = Channel(name="price-feed-BTC", slot_size=DIM * 4, num_slots=128, max_consumers=1)
    test_data = np.random.rand(DIM).astype(np.float32)
    test_data[1] = 50_000  # simulate high volume

    for _ in range(TOTAL):
        while not channel.send(test_data):
            await asyncio.sleep(1e-5)

    def high_volume_filter(vec: np.ndarray) -> bool:
        return vec[1] > 10_000

    async with ChannelRouter() as router:
        queue = router.attach(
            channel=channel,
            consumer_id=0,
            dim=DIM,
            topic="price-feed-BTC",
            filter_fn=high_volume_filter,
        )

        collected = []
        while len(collected) < TOTAL:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.5)
                collected.append(item)
            except asyncio.TimeoutError:
                break

        assert len(collected) == TOTAL
        for vec in collected:
            assert vec[1] > 10_000


@pytest.mark.asyncio
async def test_router_multiple_topics():
    DIM = 4

    btc_channel = Channel(name="price-feed-BTC", slot_size=DIM * 4, num_slots=64, max_consumers=1)
    eth_channel = Channel(name="price-feed-ETH", slot_size=DIM * 4, num_slots=64, max_consumers=1)

    vec_btc = np.ones(DIM, dtype=np.float32)
    vec_eth = np.zeros(DIM, dtype=np.float32)

    async with ChannelRouter() as router:
        btc_q = router.attach(btc_channel, consumer_id=0, dim=DIM, topic="BTC")
        eth_q = router.attach(eth_channel, consumer_id=0, dim=DIM, topic="ETH")

        while not btc_channel.send(vec_btc):
            await asyncio.sleep(1e-5)
        while not eth_channel.send(vec_eth):
            await asyncio.sleep(1e-5)

        out_btc = await asyncio.wait_for(btc_q.get(), timeout=0.2)
        out_eth = await asyncio.wait_for(eth_q.get(), timeout=0.2)

        assert np.allclose(out_btc, vec_btc)
        assert np.allclose(out_eth, vec_eth)


@pytest.mark.asyncio
async def test_mapped_bus_router_produces_and_consumes():
    DIM = 64
    registry = ChannelRegistry.global_instance()
    bus = registry.create("price-feed-BTC", slot_size=DIM * 4, num_slots=256, max_consumers=1)

    async def ml_model_infer(vec):
        await asyncio.sleep(0.001)
        return np.sum(vec, dtype=np.float32)

    async def consumer_worker(queue):
        results = []
        while True:
            try:
                vec = await asyncio.wait_for(queue.get(), timeout=0.1)
                result = await ml_model_infer(vec)
                results.append(result)
            except asyncio.TimeoutError:
                break
        return results

    async def producer(bus, dim, total=100):
        vec = np.random.rand(dim).astype(np.float32)
        for _ in range(total):
            while not bus.send(vec):
                await asyncio.sleep(1e-5)

    from intelligent_trading_bot.asyncbus.router import ChannelRouter as MappedBusRouter

    async with MappedBusRouter() as router:
        queue = router.attach(bus, consumer_id=0, dim=DIM)
        producer_task = asyncio.create_task(producer(bus, DIM, total=100))
        consumer_task = asyncio.create_task(consumer_worker(queue))

        await asyncio.gather(producer_task, consumer_task)

        results = consumer_task.result()
        assert len(results) > 0
        assert all(isinstance(r, np.float32) for r in results)
