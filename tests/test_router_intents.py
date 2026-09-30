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
    assert _run(_state("acme corp", demo_phase="collecting"))["intent"] == "BOOK_DEMO"


def test_demo_flow_remains_locked_while_confirming():
    assert _run(_state("yes", demo_phase="confirming"))["intent"] == "BOOK_DEMO"


def test_active_demo_lock_detects_whole_message_stop_word():
    assert _run(_state("Cancel ", demo_phase="collecting"))["intent"] == "STOP_DEMO"


def test_stop_word_inside_an_answer_does_not_cancel_demo():
    # "Back Office Ltd" is a business name, not a request to leave.
    assert _run(_state("Back Office Ltd", demo_phase="collecting"))["intent"] == "BOOK_DEMO"


def test_completed_demo_does_not_lock(monkeypatch):
    # demo_phase None -> no lock; small talk falls through to the LLM path.
    import agent.router as router

    async def fake_ainvoke(_payload):
        return router.RouteDecision(intent="GENERAL_CHAT", search_query=None)

    monkeypatch.setattr(router, "structured_router",
                        types.SimpleNamespace(ainvoke=fake_ainvoke))

    out = asyncio.run(router.router_node(_state("thanks, that was quick", demo_phase=None,
                                                 demo_completed=True)))
    assert out["intent"] == "GENERAL_CHAT"


import pytest


@pytest.mark.parametrize("text", [
    "hi", "Hi!", "hello", "Hello!!", "hey 👋", "hii", "hi there",
    "namaste", "Namaste ji", "namaskar", "नमस्ते", "नमस्कार 🙏",
])
def test_bare_greeting_takes_the_greeting_fast_path(text):
    assert _run(_state(text))["intent"] == "GREETING"


def test_greeting_with_a_question_goes_to_the_llm(monkeypatch):
    import agent.router as router

    async def fake_ainvoke(_payload):
        return router.RouteDecision(intent="PRODUCT_QA", search_query="features")

    monkeypatch.setattr(router, "structured_router",
                        types.SimpleNamespace(ainvoke=fake_ainvoke))
    assert _run(_state("hi, what does relayn do?"))["intent"] == "PRODUCT_QA"


def test_greeting_inside_active_demo_stays_in_the_demo():
    assert _run(_state("hello", demo_phase="collecting"))["intent"] == "BOOK_DEMO"


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
