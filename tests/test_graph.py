import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


def test_route_decision_maps_every_intent():
    from agent.graph import route_decision

    cases = {
        "BOOK_DEMO": "demo", "STOP_DEMO": "demo_end",
        "CONTACT_SALES": "sales", "STOP_SALES": "sales_end",
        "HANDOFF": "handoff",
        "PRODUCT_QA": "executor", "PRICING": "executor",
        "GENERAL_CHAT": "generator", None: "generator",
    }
    for intent, node in cases.items():
        assert route_decision({"intent": intent}) == node


_FULL_DEMO = {
    "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
    "contact_name": "Sam", "contact_info": "sam@acme.com", "contact_phone": "9800000000",
}


def _demo_turn(reply_intent, **slots):
    out = {key: None for key in _FULL_DEMO}
    out.update(slots)
    out.update({"reply_intent": reply_intent, "side_question_query": None, "language": "English"})
    return out


def _patch_demo_llms(monkeypatch, turns):
    import agent.nodes as nodes

    async def extract(message, _data, _phase):
        return turns[message]

    async def compose(text, *_args, **_kwargs):
        return text

    monkeypatch.setattr(nodes, "_extract_demo_turn", extract)
    monkeypatch.setattr(nodes, "_compose", compose)


def test_demo_flow_persists_phase_across_invocations(monkeypatch):
    """Router locks the flow; checkpointer carries demo_phase turn to turn."""
    import agent.graph as graph_mod
    import agent.nodes as nodes

    _patch_demo_llms(monkeypatch, {
        "book demo": _demo_turn("OTHER"),
        "Acme, WhatsApp": _demo_turn("ANSWER", business_name="Acme", channels="WhatsApp"),
        "rest": _demo_turn("ANSWER", **{k: v for k, v in _FULL_DEMO.items()
                                        if k not in ("business_name", "channels")}),
        "yes": _demo_turn("CONFIRM"),
    })
    monkeypatch.setattr(nodes, "build_booking_link", lambda **_kw: "https://cal.com/relayn/demo")

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    s1 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="book demo")],
         "user_data": {"user_name": "Sam"}}, cfg))
    assert s1["demo_phase"] == "collecting"
    assert "5. " in s1["messages"][-1].content

    s2 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="Acme, WhatsApp")]}, cfg))
    assert s2["demo_data"]["business_name"] == "Acme"
    assert s2["demo_phase"] == "collecting"

    s3 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="rest")]}, cfg))
    assert s3["demo_phase"] == "confirming"

    s4 = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="yes")]}, cfg))
    assert s4["demo_completed"] is True
    assert s4["demo_phase"] is None
    assert "[DEMO_BOOKED]" in s4["messages"][-1].content


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
