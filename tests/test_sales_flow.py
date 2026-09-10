import asyncio

from langchain_core.messages import HumanMessage


def _run(fn_name, state):
    import agent.nodes as nodes
    return asyncio.run(getattr(nodes, fn_name)(state))


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {"user_name": "Sam"}}
    base.update(extra)
    return base


def test_start_sales_emits_first_question():
    out = _run("sales_node", _state("contact sales", tool_data="START_SALES"))
    assert out["sales_step"] == 1
    assert out["sales_completed"] is False
    assert out["intent"] == "CONTACT_SALES"
    assert "*1/3*" in out["messages"][0].content


def test_sales_answers_accumulate():
    out = _run("sales_node", _state("rollout for support team", sales_step=1, sales_data={}))
    assert out["sales_data"]["need"] == "rollout for support team"
    assert out["sales_step"] == 2


def test_sales_completes_with_sentinel():
    data = {"need": "rollout", "company": "Acme, 40 people"}
    out = _run("sales_node", _state("sam@acme.com", sales_step=3, sales_data=data))
    assert out["sales_completed"] is True
    assert out["sales_step"] == 0
    assert out["intent"] == "CONTACT_SALES"
    assert "[SALES_REQUEST]" in out["messages"][0].content
    assert out["sales_data"]["contact_info"] == "sam@acme.com"


def test_sales_end_resets():
    out = _run("sales_end_node", _state("stop", sales_step=2, sales_data={"need": "x"}))
    assert out["sales_step"] == 0
    assert out["sales_data"] == {}
    assert out["sales_completed"] is False
    assert out["intent"] == "STOP_SALES"
