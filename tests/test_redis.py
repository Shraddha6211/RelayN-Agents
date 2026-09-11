def test_module_constants_present():
    from db.redis import (
        INDEX_PREFIX,
        KEEP_LATEST,
        MAX_MEMORY_PER_THREAD,
        PruningAsyncRedisSaver,
        THREAD_TTL_SECONDS,
        init_semantic_cache,
    )

    assert KEEP_LATEST >= 1
    assert MAX_MEMORY_PER_THREAD > 0
    assert THREAD_TTL_SECONDS == 60 * 60 * 24 * 30
    assert INDEX_PREFIX
    assert callable(init_semantic_cache)


def test_prune_keeps_at_least_keep_latest():
    """The pruning loop must never drop below KEEP_LATEST entries even if the
    running total is still over budget."""
    import asyncio

    from db.redis import KEEP_LATEST, MAX_MEMORY_PER_THREAD, PruningAsyncRedisSaver

    entries = [f"checkpoint:t:{i}:{MAX_MEMORY_PER_THREAD}" for i in range(KEEP_LATEST + 3)]
    deleted = []

    class FakeRedis:
        async def lrange(self, *_a):
            return list(entries)

        async def delete(self, key):
            deleted.append(key)

        async def lpop(self, *_a):
            entries.pop(0)

    saver = PruningAsyncRedisSaver.__new__(PruningAsyncRedisSaver)
    saver._my_redis = FakeRedis()
    asyncio.run(saver._prune_checkpoints("t", "idx"))

    assert len(entries) >= KEEP_LATEST
