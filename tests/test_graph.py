import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


def test_route_decision_maps_every_intent():
    from agent.graph import route_decision

    cases = {
        "BOOK_DEMO": "demo", "STOP_DEMO": "demo_end",
        "CONTACT_SALES": "sales", "STOP_SALES": "sales_end",
        "HANDOFF": "handoff", "GREETING": "greeting",
        "PRODUCT_QA": "executor", "PRICING": "executor",
        "GENERAL_CHAT": "generator", None: "generator",
    }
    for intent, node in cases.items():
        assert route_decision({"intent": intent}) == node


_FULL_DEMO = {"contact_name": "Sam", "contact_info": "sam@acme.com"}


def _demo_turn(reply_intent, **slots):
    out = {key: None for key in _FULL_DEMO}
    out.update(slots)
    out.update({"reply_intent": reply_intent, "side_question_query": None, "language": "English"})
    return out


def _patch_demo_llms(monkeypatch, turns):
    import agent.nodes as nodes

    async def extract(message, _data, _phase):
        return turns[message.splitlines()[-1]]  # start sends every recent message; the last is new

    async def compose(text, *_args, **_kwargs):
        return text

    monkeypatch.setattr(nodes, "_extract_demo_turn", extract)
    monkeypatch.setattr(nodes, "_compose", compose)


def test_demo_flow_persists_phase_across_invocations(monkeypatch):
    """Router locks the flow; checkpointer carries demo_phase turn to turn."""
    import agent.graph as graph_mod
    import agent.nodes as nodes

    import agent.scheduling as sched
    from datetime import datetime, time, timedelta
    from zoneinfo import ZoneInfo

    npt = ZoneInfo("Asia/Kathmandu")
    bookings = []

    async def day_slots(day):
        if day.weekday() >= 5:
            return []
        start = datetime.combine(day, time(9, 0), npt)
        return [(start + timedelta(minutes=15 * i)).isoformat() for i in range(32)]

    async def create_booking(**kwargs):
        bookings.append(kwargs)
        return {"uid": "b-1"}

    _patch_demo_llms(monkeypatch, {
        "book demo": _demo_turn("OTHER"),
        "sam@acme": _demo_turn("ANSWER", contact_info="sam@acme"),
        "sam@acme.com": _demo_turn("ANSWER", contact_info="sam@acme.com"),
        "11am sunday?": _demo_turn("ANSWER", requested_date="2026-10-04", requested_time="11:00"),
        "11:15 works": _demo_turn("ANSWER", requested_time="11:15"),
    })
    monkeypatch.setattr(nodes, "get_day_slots", day_slots)
    monkeypatch.setattr(nodes, "create_booking", create_booking)
    monkeypatch.setattr(sched, "now", lambda: datetime(2026, 9, 30, 9, 0, tzinfo=npt))

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    s1 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="book demo")],
         "user_data": {"user_name": "Sam"}}, cfg))
    assert s1["demo_phase"] == "collecting"
    assert "email" in s1["messages"][-1].content.lower()

    s2 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="sam@acme")]}, cfg))
    assert s2["demo_data"]["contact_info"] is None
    assert s2["demo_phase"] == "collecting"

    s3 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="sam@acme.com")]}, cfg))
    assert s3["demo_phase"] == "scheduling"
    assert "Monday to Friday" in s3["messages"][-1].content

    s4 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="11am sunday?")]}, cfg))
    assert s4["demo_schedule"]["day"] == "2026-10-05"
    assert bookings == []

    s5 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="11:15 works")]}, cfg))
    assert s5["demo_completed"] is True
    assert s5["demo_phase"] is None
    assert bookings[0]["start_iso"] == "2026-10-05T11:15:00+05:45"
    assert "[DEMO_BOOKED]" in s5["messages"][-1].content
    assert "http" not in s5["messages"][-1].content


def test_demo_hard_stop_word_resets_flow(monkeypatch):
    import agent.graph as graph_mod

    _patch_demo_llms(monkeypatch, {"book demo": _demo_turn("OTHER")})

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t-stop"}}

    asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="book demo")],
         "user_data": {"user_name": "Sam"}}, cfg))
    s2 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="cancel")]}, cfg))
    assert s2["demo_phase"] is None
    assert s2["intent"] == "STOP_DEMO"
    assert s2["messages"][-1].content.startswith("No worries Sam")


def test_sales_flow_persists_automated_slots_across_invocations(monkeypatch):
    import agent.graph as graph_mod
    import agent.nodes as nodes

    async def extract(message, _data):
        if "Acme" in message:
            return {"need": "rollout", "company": "Acme"}
        return {"contact_info": message}

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "sales-t1"}}

    s1 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="contact sales")],
         "user_data": {"user_name": "Sam"}}, cfg))
    assert s1["sales_step"] == 1
    assert "*1/3*" in s1["messages"][-1].content

    s2 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="We need a rollout for Acme")]}, cfg))
    assert s2["sales_data"]["need"] == "rollout"
    assert s2["sales_data"]["company"] == "Acme"
    assert s2["sales_step"] == 3

    s3 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="sam@acme.com")]}, cfg))
    assert s3["sales_completed"] is True
    assert s3["sales_step"] == 0
    assert "[SALES_REQUEST]" in s3["messages"][-1].content


import pytest


@pytest.mark.parametrize("text, reply", [
    ("hi", "Hi! How can I help you?"),
    ("Hello!", "Hi! How can I help you?"),
    ("namaste", "Namaste! Ma RelayN barey kasari maddat garna sakchu?"),
    ("नमस्ते", "नमस्ते! म RelayN बारे कसरी मद्दत गर्न सक्छु?"),
])
def test_greeting_gets_a_fixed_reply_in_the_same_language(text, reply):
    import agent.graph as graph_mod
    from langchain_core.messages import AIMessage

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    out = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content=text)], "user_data": {"user_name": "Sam"}},
        {"configurable": {"thread_id": f"greet-{text}"}}))
    assert out["messages"][-1].content == reply
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["intent"] == "GREETING"


def test_handoff_path_runs():
    import agent.graph as graph_mod
    from agent.prompts import HANDOFF_MESSAGE

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    out = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="talk to a human")],
         "user_data": {"user_name": "Sam"}},
        {"configurable": {"thread_id": "t2"}}))
    assert out["messages"][-1].content == HANDOFF_MESSAGE
    assert out["intent"] == "HANDOFF"
