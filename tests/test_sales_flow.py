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
    assert set(out["sales_data"]) == {"need", "company", "contact_info"}
    assert "*1/3*" in out["messages"][0].content


def test_sales_answers_accumulate(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"need": "rollout for support team"}

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)
    out = _run("sales_node", _state("rollout for support team", sales_step=1, sales_data={}))
    assert out["sales_data"]["need"] == "rollout for support team"
    assert out["sales_data"]["company"] is None
    assert out["sales_step"] == 2


def test_sales_completes_with_sentinel(monkeypatch):
    import agent.nodes as nodes

    data = {"need": "rollout", "company": "Acme, 40 people"}

    async def extract(_message, _data):
        return {"contact_info": "sam@acme.com"}

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)
    out = _run("sales_node", _state("sam@acme.com", sales_step=3, sales_data=data))
    assert out["sales_completed"] is True
    assert out["sales_step"] == 0
    assert out["intent"] == "CONTACT_SALES"
    assert "[SALES_REQUEST]" in out["messages"][0].content
    assert out["sales_data"]["contact_info"] == "sam@acme.com"


def test_message_can_fill_all_sales_slots_and_complete(monkeypatch):
    import agent.nodes as nodes

    data = {
        "need": "rollout for support team",
        "company": "Acme, 40 people",
        "contact_info": "sam@acme.com",
    }

    async def extract(_message, _data):
        return data

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)
    out = _run("sales_node", _state("All details", sales_step=1, sales_data={}))

    assert out["sales_completed"] is True
    assert out["sales_step"] == 0
    assert out["sales_data"] == data
    assert "[SALES_REQUEST]" in out["messages"][0].content


def test_partial_sales_message_fills_multiple_slots_and_asks_first_missing(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"need": "rollout", "contact_info": "sam@acme.com"}

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)
    out = _run("sales_node", _state("rollout, sam@acme.com", sales_step=1, sales_data={}))

    assert out["sales_data"]["need"] == "rollout"
    assert out["sales_data"]["contact_info"] == "sam@acme.com"
    assert out["sales_step"] == 2
    assert "company" in out["messages"][0].content.lower()


def test_prefilled_later_sales_slot_is_skipped(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"need": "rollout"}

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)
    out = _run(
        "sales_node",
        _state(
            "rollout",
            sales_step=1,
            sales_data={"company": "Acme, 40 people", "contact_info": "sam@acme.com"},
        ),
    )

    assert out["sales_data"]["company"] == "Acme, 40 people"
    assert out["sales_data"]["contact_info"] == "sam@acme.com"
    assert out["sales_completed"] is True


def test_sales_completion_is_based_on_slots_not_step(monkeypatch):
    import agent.nodes as nodes

    data = {"need": "rollout", "company": "Acme", "contact_info": "sam@acme.com"}

    async def extract(_message, _data):
        return {}

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)
    out = _run("sales_node", _state("No changes", sales_step=1, sales_data=data))

    assert out["sales_completed"] is True
    assert out["sales_data"] == data


def test_empty_sales_extraction_does_not_overwrite_existing_values(monkeypatch):
    import agent.nodes as nodes

    data = {"need": "rollout", "company": "Acme", "contact_info": None}

    async def extract(_message, _data):
        return {"need": "", "company": None, "contact_info": ""}

    monkeypatch.setattr(nodes, "_extract_sales_slots", extract)
    out = _run("sales_node", _state("unclear", sales_step=1, sales_data=data))

    assert out["sales_data"] == data
    assert out["sales_step"] == 3


def test_sales_end_resets():
    out = _run("sales_end_node", _state("stop", sales_step=2, sales_data={
        "need": "x", "company": None, "contact_info": None,
    }))
    assert out["sales_step"] == 0
    assert out["sales_data"] == {}
    assert out["sales_completed"] is False
    assert out["intent"] == "STOP_SALES"
