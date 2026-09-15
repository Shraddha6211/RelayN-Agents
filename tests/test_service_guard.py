import asyncio

from schemas import GenerateReplyRequest


class _SpyApp:
    def __init__(self, final_state):
        self.final_state = final_state
        self.calls = []

    async def ainvoke(self, state, config):
        self.calls.append((state, config))
        return self.final_state


def _payload(**over):
    base = dict(org_id="org-relayn-test", asset_id="a1", asset_type="whatsapp",
                workflow_id="wf-relayn-test", conversation_id="c1",
                message={"body": "hi"}, user_name="Sam")
    base.update(over)
    return GenerateReplyRequest(**base)


def _patch_workflow(monkeypatch, row):
    import service
    monkeypatch.setattr(service, "_fetch_workflow", lambda wid: row)


def test_wrong_org_id_returns_empty_and_skips_graph(monkeypatch):
    import service
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(org_id="someone-else"), app))
    assert out.reply is None and out.intent is None
    assert app.calls == []


def test_wrong_workflow_id_returns_empty_and_skips_graph(monkeypatch):
    import service
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(workflow_id="wf-other"), app))
    assert out.intent is None
    assert app.calls == []


def test_inactive_workflow_returns_empty(monkeypatch):
    import service
    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": False, "workflow_type": "ai_chatbot"})
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(), app))
    assert out.intent is None
    assert app.calls == []


def test_org_mismatch_on_row_returns_empty(monkeypatch):
    import service
    _patch_workflow(monkeypatch, {"organization_id": "different-org",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(), app))
    assert out.intent is None
    assert app.calls == []


def test_happy_path_invokes_graph_with_thread_id_and_collapses_intent(monkeypatch):
    import service
    from langchain_core.messages import AIMessage

    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    final = {"messages": [AIMessage(content="RelayN unifies your channels.")],
             "intent": "PRODUCT_QA", "search_query": "channels"}
    app = _SpyApp(final)

    out = asyncio.run(service.generate_reply(_payload(), app))

    state, config = app.calls[0]
    assert config["configurable"]["thread_id"] == "wf-relayn-test:c1"
    assert state["user_data"]["org_id"] == "org-relayn-test"
    assert out.reply == "RelayN unifies your channels."
    assert out.intent == "RAG"
    assert out.topic == "channels"
    assert out.capture is None
    assert out.handoff_requested is False


def test_demo_completion_produces_capture_and_strips_sentinel(monkeypatch):
    import service
    from langchain_core.messages import SystemMessage

    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    final = {"messages": [SystemMessage(content="All set. Thanks! [DEMO_BOOKED]")],
             "intent": "BOOK_DEMO", "demo_data": {"business_name": "Acme"}}
    out = asyncio.run(service.generate_reply(_payload(), _SpyApp(final)))

    assert "[DEMO_BOOKED]" not in out.reply
    assert out.reply == "All set. Thanks!"
    assert out.intent == "ORDER"
    assert out.capture.type == "demo"
    assert out.capture.data == {"business_name": "Acme"}


def test_sales_completion_produces_capture_and_strips_sentinel(monkeypatch):
    import service
    from langchain_core.messages import SystemMessage

    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    sales_data = {"need": "rollout", "company": "Acme, 40 people", "contact_info": "sam@acme.com"}
    final = {"messages": [SystemMessage(content="Thanks! [SALES_REQUEST]")],
             "intent": "CONTACT_SALES", "sales_data": sales_data}
    out = asyncio.run(service.generate_reply(_payload(), _SpyApp(final)))

    assert "[SALES_REQUEST]" not in out.reply
    assert out.reply == "Thanks!"
    assert out.intent == "ORDER"
    assert out.capture.type == "sales"
    assert out.capture.data == sales_data


def test_handoff_intent_sets_flag(monkeypatch):
    import service
    from langchain_core.messages import SystemMessage

    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    final = {"messages": [SystemMessage(content="Bringing in a human.")], "intent": "HANDOFF"}
    out = asyncio.run(service.generate_reply(_payload(), _SpyApp(final)))
    assert out.handoff_requested is True
    assert out.intent == "CHAT"
