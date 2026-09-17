import asyncio
import types

from langchain_core.messages import HumanMessage


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {}}
    base.update(extra)
    return base


def _run(state):
    import agent.router as router
    return asyncio.run(router.router_node(state))


def test_active_demo_lock_holds_plain_message_in_flow():
    assert _run(_state("acme corp", demo_step=1, demo_completed=False))["intent"] == "BOOK_DEMO"


def test_demo_cal_booking_phase_remains_locked_in_demo_flow():
    assert _run(_state(
        "1",
        demo_step=0,
        demo_completed=False,
        demo_booking_status="selecting",
    ))["intent"] == "BOOK_DEMO"


def test_active_demo_lock_detects_stop_word():
    assert _run(_state("cancel", demo_step=2, demo_completed=False))["intent"] == "STOP_DEMO"


def test_completed_demo_does_not_lock(monkeypatch):
    # demo_step 0 -> no lock; a plain greeting falls through to the LLM path.
    import agent.router as router

    async def fake_ainvoke(_payload):
        return router.RouteDecision(intent="GENERAL_CHAT", search_query=None)

    monkeypatch.setattr(router, "structured_router",
                        types.SimpleNamespace(ainvoke=fake_ainvoke))

    out = asyncio.run(router.router_node(_state("hello", demo_step=0, demo_completed=True)))
    assert out["intent"] == "GENERAL_CHAT"


def test_active_sales_lock():
    assert _run(_state("we are 20 people", sales_step=1, sales_completed=False))["intent"] == "CONTACT_SALES"
    assert _run(_state("stop", sales_step=1, sales_completed=False))["intent"] == "STOP_SALES"


def test_keyword_fast_path_book_demo_sets_start_marker():
    out = _run(_state("book demo"))
    assert out["intent"] == "BOOK_DEMO"
    assert out["tool_data"] == "START_DEMO"


def test_keyword_fast_path_contact_sales_and_handoff():
    assert _run(_state("contact sales"))["tool_data"] == "START_SALES"
    assert _run(_state("talk to a human"))["intent"] == "HANDOFF"


def test_llm_fallback_used_for_freeform_message(monkeypatch):
    import agent.router as router

    async def fake_ainvoke(_payload):
        return router.RouteDecision(intent="PRICING", search_query="pricing plans")

    monkeypatch.setattr(router, "structured_router",
                        types.SimpleNamespace(ainvoke=fake_ainvoke))

    out = asyncio.run(router.router_node(_state("what do your plans cost?")))
    assert out["intent"] == "PRICING"
    assert out["search_query"] == "pricing plans"


def test_llm_fallback_book_demo_gets_start_marker(monkeypatch):
    import agent.router as router

    async def fake_ainvoke(_payload):
        return router.RouteDecision(intent="BOOK_DEMO", search_query=None)

    monkeypatch.setattr(router, "structured_router",
                        types.SimpleNamespace(ainvoke=fake_ainvoke))

    out = asyncio.run(router.router_node(_state("can you show me how it works live")))
    assert out["intent"] == "BOOK_DEMO"
    assert out["tool_data"] == "START_DEMO"
