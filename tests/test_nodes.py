import asyncio

from langchain_core.messages import AIMessage, HumanMessage


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)],
            "user_data": {"user_name": "Sam", "org_id": "o1", "workflow_id": "w1"}}
    base.update(extra)
    return base


def test_executor_uses_router_query_when_present(monkeypatch):
    import agent.nodes as nodes

    seen = {}

    def fake_search(org, wf, query, k=5):
        seen["args"] = (org, wf, query)
        return ["RelayN has a shared inbox", "and automation"]

    monkeypatch.setattr(nodes, "search_knowledge_base", fake_search)

    out = asyncio.run(nodes.executor_node(_state("x", intent="PRODUCT_QA", search_query="shared inbox")))
    assert seen["args"] == ("o1", "w1", "shared inbox")
    assert out["tool_data"] == "- RelayN has a shared inbox\n\n- and automation"
    assert out["search_query"] == "shared inbox"


def test_executor_falls_back_to_intent_default_query(monkeypatch):
    import agent.nodes as nodes
    monkeypatch.setattr(nodes, "search_knowledge_base", lambda o, w, q, k=5: [])

    out = asyncio.run(nodes.executor_node(_state("x", intent="PRICING", search_query=None)))
    assert out["search_query"] == "RelayN pricing and plans"
    assert out["tool_data"] is None


def test_executor_prefixes_pricing_query(monkeypatch):
    import agent.nodes as nodes
    captured = {}
    monkeypatch.setattr(nodes, "search_knowledge_base",
                        lambda o, w, q, k=5: captured.setdefault("q", q) or [])

    asyncio.run(nodes.executor_node(_state("x", intent="PRICING", search_query="team plan")))
    assert captured["q"] == "pricing plans team plan"


def test_generator_builds_prompt_and_returns_ai_message(monkeypatch):
    import agent.nodes as nodes

    captured = {}

    class FakeLLM:
        async def ainvoke(self, msgs):
            captured["system"] = msgs[0].content
            return AIMessage(content="Sure — RelayN unifies your channels.")

    monkeypatch.setattr(nodes, "gen_llm", FakeLLM())

    out = asyncio.run(nodes.generator_node(_state("what is relayn?", tool_data="- RelayN unifies channels")))
    assert isinstance(out["messages"][0], AIMessage)
    assert "RelayN" in captured["system"]
    assert "what is relayn?" in captured["system"]
    assert "- RelayN unifies channels" in captured["system"]


def test_generator_handles_missing_tool_data(monkeypatch):
    import agent.nodes as nodes

    class FakeLLM:
        async def ainvoke(self, msgs):
            return AIMessage(content="ok")

    monkeypatch.setattr(nodes, "gen_llm", FakeLLM())
    out = asyncio.run(nodes.generator_node(_state("hi")))
    assert out["messages"][0].content == "ok"
