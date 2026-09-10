"""Interactive CLI to chat with the RelayN agent.

    python chat.py                                   # in-memory checkpointer
    python chat.py --redis                           # real Redis checkpointer
    python chat.py --org <uuid> --workflow <uuid>    # scope RAG to a real workflow
    python chat.py --thread demo1                    # set the starting thread id

Drives the compiled LangGraph directly (same path as tests/test_graph.py), so
the router and generator make real OpenAI calls and PRODUCT_QA / PRICING hit
Supabase's match_workflow_kb_chunks. Wizard state (demo_step, sales_step, ...)
persists across turns within the session.

Env resolution for the two scope ids, highest priority first:
  1. --org / --workflow flags
  2. RELAYN_ORG_ID / RELAYN_WORKFLOW_ID already in the environment or .env
  3. a non-UUID placeholder — the conversation flows still work, but any
     PRODUCT_QA / PRICING turn will error because the RPC needs real UUIDs.

Commands: /help  /state  /reset  /quit
"""
import argparse
import asyncio
import json
import os

from langchain_core.messages import HumanMessage


def _bootstrap_env(org: str | None, workflow: str | None) -> bool:
    """Populate the env config.py needs. Returns True if RAG is usable."""
    if org:
        os.environ["RELAYN_ORG_ID"] = org
    if workflow:
        os.environ["RELAYN_WORKFLOW_ID"] = workflow

    # Pull anything still missing from a local .env before we placeholder it.
    try:
        from dotenv import dotenv_values

        file_vals = dotenv_values(".env")
    except Exception:
        file_vals = {}

    def ensure(key: str, placeholder: str) -> str:
        if os.environ.get(key):
            return os.environ[key]
        if file_vals.get(key):
            os.environ[key] = file_vals[key]
            return file_vals[key]
        os.environ[key] = placeholder
        return placeholder

    org_val = ensure("RELAYN_ORG_ID", "cli-local-org")
    wf_val = ensure("RELAYN_WORKFLOW_ID", "cli-local-workflow")
    ensure("RELAYN_SERVICES_URL", "http://localhost:8001")

    return "cli-local" not in org_val and "cli-local" not in wf_val


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
    from agent.graph import build_workflow

    if not use_redis:
        from langgraph.checkpoint.memory import MemorySaver

        return build_workflow().compile(checkpointer=MemorySaver()), None

    import redis.asyncio as aioredis

    from config import settings
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
    parser.add_argument("--org", help="RELAYN_ORG_ID (uuid) to scope retrieval")
    parser.add_argument("--workflow", help="RELAYN_WORKFLOW_ID (uuid) to scope retrieval")
    args = parser.parse_args()

    rag_ready = _bootstrap_env(args.org, args.workflow)

    from config import settings

    app, redis_client = await _make_app(args.redis)
    thread = args.thread
    reset_count = 0
    last_state: dict = {}

    print(BANNER)
    print(f"thread: {thread}  |  checkpointer: {'redis' if args.redis else 'memory'}")
    if not rag_ready:
        print("note: no real RELAYN_ORG_ID / RELAYN_WORKFLOW_ID — PRODUCT_QA / PRICING")
        print("      turns will error. Pass --org <uuid> --workflow <uuid> to enable RAG.")
    print()

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
                print(f"!! {type(e).__name__}: {e}")
                if "uuid" in str(e).lower():
                    print("   (retrieval needs real UUIDs — restart with --org / --workflow)")
                print()
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
