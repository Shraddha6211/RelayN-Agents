# db/redis.py
import logging
import asyncio
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.runnables import RunnableConfig
from config import settings

logger = logging.getLogger("relayn_agents")

MAX_MEMORY_PER_THREAD = 5 * 1024 * 1024  # 5 MB
KEEP_LATEST = 5
INDEX_PREFIX = "checkpoint_index:"
THREAD_TTL_SECONDS = 60 * 60 * 24 * 30   # 30 days inactivity expiry
MAX_CONCURRENT_REDIS_OPS = 50

_redis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REDIS_OPS)


def init_semantic_cache():
    from redis import Redis
    print(f"🧠 Initializing Standard Redis Cache at {settings.REDIS_HOST}")
    sync_client = Redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    set_llm_cache(RedisCache(redis_=sync_client))


class PruningAsyncRedisSaver(AsyncRedisSaver):
    def __init__(self, redis_client, *args, **kwargs):
        super().__init__(redis_client=redis_client, *args, **kwargs)
        self._my_redis = redis_client

    async def aput(self, config: RunnableConfig, checkpoint: dict, metadata: dict, new_versions: dict) -> RunnableConfig:
        async with _redis_semaphore:
            updated_config = await super().aput(config, checkpoint, metadata, new_versions)

            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = updated_config["configurable"]["checkpoint_id"]

            checkpoint_key = f"checkpoint:{thread_id}:{checkpoint_id}"
            index_key = f"{INDEX_PREFIX}{thread_id}"

            type_str, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
            size = len(serialized_checkpoint)

            await self._my_redis.rpush(index_key, f"{checkpoint_key}:{size}")

            await self._my_redis.expire(index_key, THREAD_TTL_SECONDS)
            await self._my_redis.expire(checkpoint_key, THREAD_TTL_SECONDS)

            await self._prune_checkpoints(thread_id, index_key)

            return updated_config

    async def aget(self, config: RunnableConfig, *args, **kwargs):
        async with _redis_semaphore:
            try:
                return await super().aget(config, *args, **kwargs)
            except Exception as e:
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                logger.warning(
                    f"⚠️ Checkpoint read failed for {thread_id} — "
                    f"likely TTL expiry mid-conversation. Starting fresh. Error: {e}"
                )
                return None

    async def aget_tuple(self, config: RunnableConfig):
        async with _redis_semaphore:
            try:
                return await super().aget_tuple(config)
            except Exception as e:
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                logger.warning(
                    f"⚠️ Checkpoint tuple read failed for {thread_id} — "
                    f"starting fresh thread. Error: {e}"
                )
                return None

    async def _prune_checkpoints(self, thread_id: str, index_key: str):
        try:
            entries = await self._my_redis.lrange(index_key, 0, -1)
            if not entries:
                return

            entries = [e.decode("utf-8") if isinstance(e, bytes) else e for e in entries]
            total_size = sum(int(entry.rsplit(":", 1)[1]) for entry in entries)

            while total_size > MAX_MEMORY_PER_THREAD and len(entries) > KEEP_LATEST:
                oldest = entries.pop(0)
                old_key, old_size_str = oldest.rsplit(":", 1)

                await self._my_redis.delete(old_key)
                await self._my_redis.lpop(index_key)

                total_size -= int(old_size_str)
                logger.info(f"🧹 Pruned checkpoint for {thread_id} ({old_size_str} bytes)")

        except Exception as e:
            logger.error(f"⚠️ Checkpoint Pruning Error: {e}")
