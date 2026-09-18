"""Streamlit UI for the RelayN agent — visualizes the same graph chat.py drives.

    streamlit run streamlit_app.py

Uses the real Redis checkpointer (same setup as `python chat.py --redis`), so
conversation state persists across turns and survives an app restart. Each
browser session gets its own thread id.
"""
import asyncio
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from config import settings

st.set_page_config(page_title="RelayN Agent", page_icon="💬")


@st.cache_resource
def _init_cache() -> None:
    from db.redis import init_semantic_cache

    init_semantic_cache()


async def _run_turn(user_msg: str, thread_id: str) -> dict:
    import redis.asyncio as aioredis

    from agent.graph import build_workflow
    from db.redis import PruningAsyncRedisSaver

    client = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    checkpointer = PruningAsyncRedisSaver(redis_client=client)
    try:
        await checkpointer.setup()
    except Exception as exc:
        if "already exists" not in str(exc).lower():
            raise
    app = build_workflow().compile(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": thread_id}}
    state_in = {
        "messages": [HumanMessage(content=user_msg)],
        "user_data": {
            "user_name": "Streamlit",
            "conversation_id": thread_id,
            "workflow_id": settings.RELAYN_WORKFLOW_ID,
            "org_id": settings.RELAYN_ORG_ID,
        },
    }
    try:
        return await app.ainvoke(state_in, cfg)
    finally:
        await client.aclose()


def _status_lines(state: dict) -> list[str]:
    lines = [f"**Intent:** {state.get('intent') or '—'}"]
    if state.get("demo_step"):
        lines.append(f"**Demo step:** {state['demo_step']}/6")
    if state.get("demo_booking_status"):
        lines.append(f"**Booking status:** {state['demo_booking_status']}")
    if state.get("demo_available_slots"):
        lines.append(f"**Slots offered:** {len(state['demo_available_slots'])}")
    if state.get("demo_completed"):
        lines.append("✅ **Demo booked**")
    if state.get("sales_step"):
        lines.append(f"**Sales step:** {state['sales_step']}/3")
    if state.get("sales_completed"):
        lines.append("✅ **Sales request sent**")
    return lines


_init_cache()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
if "history" not in st.session_state:
    st.session_state.history: list[tuple[str, str]] = []
if "last_state" not in st.session_state:
    st.session_state.last_state: dict = {}

st.title("RelayN Agent")
st.caption(f"thread: `{st.session_state.thread_id}` · Redis checkpointer")

with st.sidebar:
    st.subheader("Live state")
    if st.session_state.last_state:
        for line in _status_lines(st.session_state.last_state):
            st.markdown(line)
    else:
        st.write("(no turns yet)")

    st.divider()
    if st.button("🔄 Reset conversation"):
        st.session_state.thread_id = f"streamlit-{uuid.uuid4().hex[:8]}"
        st.session_state.history = []
        st.session_state.last_state = {}
        st.rerun()

    with st.expander("Raw state (debug)"):
        st.json({k: v for k, v in st.session_state.last_state.items() if k != "messages"})

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Message RelayN..."):
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(_run_turn(prompt, st.session_state.thread_id))
                reply = result["messages"][-1].content if result.get("messages") else "(no reply)"
                st.session_state.last_state = result
            except Exception as exc:  # keep the app alive on an API/network error
                reply = f"⚠️ {type(exc).__name__}: {exc}"
        st.markdown(reply)
    st.session_state.history.append(("assistant", reply))
