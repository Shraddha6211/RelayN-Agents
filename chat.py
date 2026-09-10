"""Interactive CLI to chat with the RelayN agent.

    python chat.py                # in-memory checkpointer (no Redis needed)
    python chat.py --redis        # use the real PruningAsyncRedisSaver
    python chat.py --thread demo1 # set the starting thread id

Runs the compiled LangGraph directly (same path as tests/test_graph.py), so
the router and generator make real OpenAI calls and PRODUCT_QA / PRICING hit
Supabase's match_workflow_kb_chunks. Wizard state (demo_step, sales_step, ...)
persists across turns within the session.

Requires a real OPENAI_API_KEY in .env. For grounded PRODUCT_QA/PRICING answers
also set real SUPABASE_URL / SUPABASE_SERVICE_KEY and a RELAYN_WORKFLOW_ID that
actually has ingested chunks; otherwise retrieval just comes back empty and the
generator says it doesn't have that detail.

Commands: /help  /state  /reset  /quit
"""
import argparse
import asyncio
import json
import os

# The service pins these; the CLI only needs them to exist so config.py can
# construct. Real OPENAI_API_KEY / SUPABASE_* still load from .env.
os.environ.setdefault("RELAYN_ORG_ID", "cli-local-org")
os.environ.setdefault("RELAYN_WORKFLOW_ID", "cli-local-workflow")
os.environ.setdefault("RELAYN_SERVICES_URL", "http://localhost:8001")

from langchain_core.messages import HumanMessage  # noqa: E402

from agent.graph import build_workflow  # noqa: E402
from config import settings  # noqa: E402

BANNER = "relayn_agents CLI — type a message, or /help. Ctrl+D or /quit to exit."


def _status_line(state: dict) -> str:
    bits = [f"intent={state.get('intent') or '—'}"]
    if state.get("search_query"):
        bits.append(f"topic={state['search_query']}")
    if state.get("demo_step"):
        bits.append(f"demo_step={state['demo_step']}")
    if state.get("demo_completed"):
        bits.append("demo_completed")
    if state.get("sales_step"):
        bits.append(f"sales_step={state['sales_step']}")
    if state.get("sales_completed"):
        bits.append("sales_completed")
    return "      · " + "  ".join(bits)


def _dump_state(state: dict) -> str:
    view = {k: v for k, v in state.items() if k != "messages"}
    view["messages"] = [f"{getattr(m, 'type', '?')}: {m.content}" for m in state.get("messages", [])]
    return json.dumps(view, indent=2, ensure_ascii=False, default=str)


async def _make_app(use_redis: bool):
    if not use_redis:
        from langgraph.checkpoint.memory import MemorySaver

        return build_workflow().compile(checkpointer=MemorySaver()), None

    import redis.asyncio as aioredis

    from db.redis import PruningAsyncRedisSaver, init_semantic_cache

    init_semantic_cache()
    client = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    checkpointer = PruningAsyncRedisSaver(redis_client=client)
    try:
        await checkpointer.setup()
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise
    return build_workflow().compile(checkpointer=checkpointer), client


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the RelayN agent from the CLI.")
    parser.add_argument("--redis", action="store_true", help="use the real Redis checkpointer")
    parser.add_argument("--thread", default="cli-1", help="starting thread id (default: cli-1)")
    args = parser.parse_args()

    app, redis_client = await _make_app(args.redis)
    thread = args.thread
    reset_count = 0
    last_state: dict = {}

    print(BANNER)
    print(f"thread: {thread}  |  checkpointer: {'redis' if args.redis else 'memory'}\n")

    try:
        while True:
            try:
                line = (await asyncio.to_thread(input, "you> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                continue

            if line in ("/quit", "/exit"):
                break
            if line == "/help":
                print("  /state  dump the last graph state")
                print("  /reset  start a fresh conversation thread")
                print("  /quit   exit\n")
                continue
            if line == "/state":
                print(_dump_state(last_state) if last_state else "(no turns yet)", "\n")
                continue
            if line == "/reset":
                reset_count += 1
                thread = f"{args.thread}-{reset_count}"
                last_state = {}
                print(f"(new thread: {thread})\n")
                continue

            cfg = {"configurable": {"thread_id": thread}}
            state_in = {
                "messages": [HumanMessage(content=line)],
                "user_data": {
                    "user_name": "CLI",
                    "conversation_id": thread,
                    "workflow_id": settings.RELAYN_WORKFLOW_ID,
                    "org_id": settings.RELAYN_ORG_ID,
                },
            }
            try:
                last_state = await app.ainvoke(state_in, cfg)
            except Exception as e:  # keep the REPL alive on an API/network error
                print(f"!! {type(e).__name__}: {e}\n")
                continue

            reply = last_state["messages"][-1].content if last_state.get("messages") else "(no reply)"
            print(f"bot> {reply}")
            print(_status_line(last_state), "\n")
    finally:
        if redis_client is not None:
            await redis_client.aclose()
        print("bye")


if __name__ == "__main__":
    asyncio.run(main())
