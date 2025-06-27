import asyncio
import numpy as np
import pytest

from intelligent_trading_bot.common.mappedbus.registry import MappedBusRegistry
from intelligent_trading_bot.common.mappedbus.router import MappedBusRouter


async def ml_model_infer(vec):
    await asyncio.sleep(0.001)  # Simulate model latency
    return np.sum(vec)


async def consumer_worker(queue, label):
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
            await asyncio.sleep(0.00001)


@pytest.mark.asyncio
async def test_mapped_bus_router_produces_and_consumes():
    DIM = 64
    registry = MappedBusRegistry.global_instance()
    bus = registry.create("price-feed-BTC", slot_size=DIM * 4, num_slots=256, max_consumers=1)

    async with MappedBusRouter() as router:
        queue = router.attach(bus, consumer_id=0, dim=DIM)
        producer_task = asyncio.create_task(producer(bus, DIM, total=100))
        consumer_task = asyncio.create_task(consumer_worker(queue, "BTC-Model"))

        await asyncio.gather(producer_task, consumer_task)

        results = consumer_task.result()

        assert len(results) > 0
        assert all(isinstance(r, np.float32) for r in results)
        # assert abs(results[0] - DIM / 2) < DIM  # Just a basic sanity check
