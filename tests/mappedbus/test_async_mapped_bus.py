import asyncio
import numpy as np
import pytest

from intelligent_trading_bot.asyncbus.bus import AsyncMappedBus


@pytest.mark.asyncio
async def test_topic_creation_and_summary():
    bus = AsyncMappedBus()
    topic = "test-topic"
    slot_size = 32
    num_slots = 64
    max_consumers = 2

    channel = bus.create_topic(topic, slot_size, num_slots, max_consumers)
    assert channel.name == topic

    mem = bus.topic_memory_bytes(topic)
    assert isinstance(mem, int)
    assert mem > 0

    summary = bus.summary(topic, as_dict=True)
    assert summary["topic"] == topic
    assert summary["slot_size"] == slot_size


@pytest.mark.asyncio
async def test_publish_nowait_and_async_publish():
    bus = AsyncMappedBus()
    topic = "pub-test"
    dim = 8
    bus.create_topic(topic, slot_size=dim * 4, num_slots=16, max_consumers=1)

    arr = np.random.rand(dim).astype(np.float32)
    success = bus.publish_nowait(topic, arr)
    assert success is True

    success = await bus.publish_async(topic, arr)
    assert success is True


@pytest.mark.asyncio
async def test_subscribe_and_receive():
    bus = AsyncMappedBus()
    topic = "sub-test"
    dim = 16
    bus.create_topic(topic, slot_size=dim * 4, num_slots=32, max_consumers=1)

    queue = bus.subscribe(topic, consumer_name="test-consumer", dim=dim)

    arr = np.ones(dim, dtype=np.float32)
    await bus.publish_async(topic, arr)

    received = await asyncio.wait_for(queue.get(), timeout=0.1)
    assert np.allclose(received, arr)


@pytest.mark.asyncio
async def test_unsubscribe_and_slot_reuse():
    bus = AsyncMappedBus()
    topic = "reuse-test"
    dim = 4
    bus.create_topic(topic, slot_size=dim * 4, num_slots=16, max_consumers=1)

    q1 = bus.subscribe(topic, "consumer1", dim)
    bus.unsubscribe(topic, "consumer1")

    # After unsubscribe, slot should be reusable
    q2 = bus.subscribe(topic, "consumer2", dim)
    assert isinstance(q2, asyncio.Queue)


@pytest.mark.asyncio
async def test_multiple_consumers():
    bus = AsyncMappedBus()
    topic = "multi-test"
    dim = 4
    max_consumers = 2
    bus.create_topic(topic, slot_size=dim * 4, num_slots=16, max_consumers=max_consumers)

    q1 = bus.subscribe(topic, "c1", dim)
    q2 = bus.subscribe(topic, "c2", dim)

    data = np.arange(dim, dtype=np.float32)
    await bus.publish_async(topic, data)

    r1 = await asyncio.wait_for(q1.get(), timeout=0.1)
    r2 = await asyncio.wait_for(q2.get(), timeout=0.1)

    assert np.allclose(r1, data)
    assert np.allclose(r2, data)


@pytest.mark.asyncio
async def test_summary_format_and_stop():
    bus = AsyncMappedBus()
    topic = "summary-test"
    bus.create_topic(topic, slot_size=16, num_slots=8, max_consumers=1)
    text = bus.summary()
    assert topic in text

    await bus.stop()
