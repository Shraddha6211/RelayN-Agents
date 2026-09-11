import asyncio

from langchain_core.messages import HumanMessage


def _run(fn_name, state):
    import agent.nodes as nodes
    return asyncio.run(getattr(nodes, fn_name)(state))


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {"user_name": "Sam"}}
    base.update(extra)
    return base


def test_start_demo_emits_first_question():
    out = _run("demo_node", _state("book demo", tool_data="START_DEMO"))
    assert out["demo_step"] == 1
    assert out["demo_completed"] is False
    assert out["intent"] == "BOOK_DEMO"
    assert set(out["demo_data"]) == {
        "business_name", "channels", "monthly_volume",
        "contact_name", "contact_info", "preferred_time",
    }
    assert "*1/6*" in out["messages"][0].content


def test_open_answer_is_stored_and_flow_advances(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"business_name": "Acme Corp"}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state("Acme Corp", demo_step=1, demo_data={}))
    assert out["demo_data"]["business_name"] == "Acme Corp"
    assert out["demo_data"]["channels"] is None
    assert out["demo_step"] == 2
    assert "*2/6*" in out["messages"][0].content


def test_unknown_button_value_is_ignored_and_re_asks(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"channels": "banana"}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state("banana", demo_step=2, demo_data={"business_name": "Acme"}))
    assert out["demo_step"] == 2                            # not advanced
    assert out["demo_data"]["business_name"] == "Acme"      # nothing stored, nothing lost
    assert out["demo_data"]["channels"] is None
    text = out["messages"][0].content
    assert "*2/6*" in text
    assert "ch_all) All of them" in text


def test_button_step_accepts_option_id_and_stores_title(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"channels": "ch_all"}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state("ch_all", demo_step=2, demo_data={"business_name": "Acme"}))
    assert out["demo_data"]["channels"] == "All of them"
    assert out["demo_step"] == 3


def test_message_can_fill_all_slots_and_complete(monkeypatch):
    import agent.nodes as nodes

    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
        "preferred_time": "Tuesday afternoon",
    }

    async def extract(_message, _data):
        return data

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state("Everything is above", demo_step=1, demo_data={}))
    assert out["demo_completed"] is True
    assert out["demo_step"] == 0
    assert out["demo_data"] == data
    assert "[DEMO_BOOKED]" in out["messages"][0].content


def test_partial_message_fills_multiple_slots_and_asks_first_missing(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {
            "business_name": "Acme",
            "contact_name": "Sam",
            "preferred_time": "Tuesday afternoon",
        }

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state("Acme, Sam, Tuesday afternoon", demo_step=1, demo_data={}))
    assert out["demo_data"]["business_name"] == "Acme"
    assert out["demo_data"]["contact_name"] == "Sam"
    assert out["demo_data"]["preferred_time"] == "Tuesday afternoon"
    assert out["demo_step"] == 2
    assert "channels" in out["messages"][0].content.lower()


def test_prefilled_later_slots_are_skipped(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"business_name": "Acme", "contact_info": "sam@acme.com"}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run(
        "demo_node",
        _state(
            "Acme, sam@acme.com",
            demo_step=1,
            demo_data={"contact_info": "old@example.com", "preferred_time": "Friday"},
        ),
    )
    assert out["demo_data"]["contact_info"] == "sam@acme.com"
    assert out["demo_data"]["preferred_time"] == "Friday"
    assert out["demo_step"] == 2
    assert "channels" in out["messages"][0].content.lower()


def test_completion_is_based_on_slots_not_step(monkeypatch):
    import agent.nodes as nodes

    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
        "preferred_time": "Tuesday afternoon",
    }

    async def extract(_message, _data):
        return {}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state("No changes", demo_step=2, demo_data=data))
    assert out["demo_completed"] is True
    assert out["demo_step"] == 0
    assert out["intent"] == "BOOK_DEMO"
    assert "[DEMO_BOOKED]" in out["messages"][0].content
    assert out["demo_data"] == data


def test_demo_end_resets_and_reports_count():
    out = _run("demo_end_node", _state("stop", demo_step=3, demo_data={
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": None,
        "contact_name": None, "contact_info": None, "preferred_time": None,
    }))
    assert out["demo_step"] == 0
    assert out["demo_data"] == {}
    assert out["demo_completed"] is False
    assert out["intent"] == "STOP_DEMO"
    assert "2" in out["messages"][0].content
