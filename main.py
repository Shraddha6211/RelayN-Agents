import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException

from agent.graph import build_workflow
from config import settings
from db.redis import PruningAsyncRedisSaver, init_semantic_cache
from schemas import GenerateReplyRequest, GenerateReplyResponse
from service import generate_reply

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("relayn_agents")


def _redis_client():
    return aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        health_check_interval=30,
        retry_on_timeout=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = _redis_client()
    try:
        init_semantic_cache()
        checkpointer = PruningAsyncRedisSaver(redis_client=redis_client)
        try:
            await checkpointer.setup()
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
        app.state.agent_app = build_workflow().compile(checkpointer=checkpointer)
        app.state.is_ready = True
        logger.info("✅ relayn_agents online")
        yield
    except Exception as e:
        logger.error(f"❌ Startup error: {e}", exc_info=True)
        app.state.is_ready = False
        yield
    finally:
        await redis_client.aclose()
        logger.info("🛑 relayn_agents shutdown")


app = FastAPI(title="relayn_agents", lifespan=lifespan)
app.state.agent_app = None
app.state.is_ready = False


@app.get("/health")
async def health():
    if not app.state.is_ready:
        raise HTTPException(status_code=503, detail="Initializing")
    return {"status": "ok"}


@app.post("/generate-reply", response_model=GenerateReplyResponse)
async def generate_reply_endpoint(payload: GenerateReplyRequest):
    if app.state.agent_app is None:
        raise HTTPException(status_code=503, detail="Graph initializing")
    return await generate_reply(payload, app.state.agent_app)
