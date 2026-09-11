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


def test_demo_flow_persists_step_across_invocations(monkeypatch):
    """Router locks the flow; checkpointer carries demo_step turn to turn.

    No LLM is mocked because no LLM is reached: the keyword fast path starts the
    flow, then the active-demo lock holds every following turn.
    """
    import agent.graph as graph_mod
    import agent.nodes as nodes

    async def extract(message, _data):
        return {"business_name": message}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    s1 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="book demo")],
         "user_data": {"user_name": "Sam"}}, cfg))
    assert s1["demo_step"] == 1
    assert "*1/6*" in s1["messages"][-1].content

    s2 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="Acme Corp")]}, cfg))
    assert s2["demo_data"]["business_name"] == "Acme Corp"
    assert s2["demo_step"] == 2

    s3 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="cancel")]}, cfg))
    assert s3["demo_step"] == 0
    assert s3["intent"] == "STOP_DEMO"


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
