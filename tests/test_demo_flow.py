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
        "contact_name", "contact_info", "contact_phone",
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


def test_message_can_fill_all_slots_and_sends_booking_link(monkeypatch):
    import agent.nodes as nodes

    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
        "contact_phone": "9800000000",
    }

    async def extract(_message, _data):
        return data

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    monkeypatch.setattr(nodes, "build_booking_link", lambda **kwargs: "https://cal.com/relayn/demo?name=Sam")

    out = _run("demo_node", _state("Everything is above", demo_step=1, demo_data={}))
    assert out["demo_completed"] is True
    assert out["demo_step"] == 0
    assert out["demo_data"]["cal_booking_link"] == "https://cal.com/relayn/demo?name=Sam"
    assert "https://cal.com/relayn/demo?name=Sam" in out["messages"][0].content
    assert "[DEMO_BOOKED]" in out["messages"][0].content


def test_partial_message_fills_multiple_slots_and_asks_first_missing(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {
            "business_name": "Acme",
            "contact_name": "Sam",
            "contact_phone": "9800000000",
        }

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state("Acme, Sam, 9800000000", demo_step=1, demo_data={}))
    assert out["demo_data"]["business_name"] == "Acme"
    assert out["demo_data"]["contact_name"] == "Sam"
    assert out["demo_data"]["contact_phone"] == "9800000000"
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
            demo_data={"contact_info": "old@example.com", "contact_phone": "9800000000"},
        ),
    )
    assert out["demo_data"]["contact_info"] == "sam@acme.com"
    assert out["demo_data"]["contact_phone"] == "9800000000"
    assert out["demo_step"] == 2
    assert "channels" in out["messages"][0].content.lower()


def test_completion_is_based_on_data_not_step(monkeypatch):
    import agent.nodes as nodes

    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
        "contact_phone": "9800000000",
    }

    async def extract(_message, _data):
        return {}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    monkeypatch.setattr(nodes, "build_booking_link", lambda **kwargs: "https://cal.com/relayn/demo?name=Sam")

    out = _run("demo_node", _state("No changes", demo_step=2, demo_data=data))
    assert out["demo_completed"] is True
    assert out["demo_step"] == 0
    assert out["intent"] == "BOOK_DEMO"
    assert out["demo_data"]["cal_booking_link"] == "https://cal.com/relayn/demo?name=Sam"


def test_booking_link_includes_notes_with_business_details(monkeypatch):
    import agent.nodes as nodes

    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
        "contact_phone": "9800000000",
    }

    async def extract(_message, _data):
        return {}

    captured = {}

    def fake_build_link(**kwargs):
        captured.update(kwargs)
        return "https://cal.com/relayn/demo?name=Sam"

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    monkeypatch.setattr(nodes, "build_booking_link", fake_build_link)

    _run("demo_node", _state("No changes", demo_step=2, demo_data=data))
    assert captured["name"] == "Sam"
    assert captured["email"] == "sam@acme.com"
    assert "Acme" in captured["notes"]
    assert "WhatsApp" in captured["notes"]
    assert "500 to 2k" in captured["notes"]
    assert "9800000000" in captured["notes"]


def test_unconfigured_booking_link_shows_retry_message(monkeypatch):
    import agent.nodes as nodes
    from calcom_client import CalComError

    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
        "contact_phone": "9800000000",
    }

    async def extract(_message, _data):
        return {}

    def fake_build_link(**kwargs):
        raise CalComError("Cal.com booking link is not configured")

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    monkeypatch.setattr(nodes, "build_booking_link", fake_build_link)

    out = _run("demo_node", _state("No changes", demo_step=2, demo_data=data))
    assert out["demo_completed"] is False
    assert "try again" in out["messages"][0].content.lower()


def test_invalid_mobile_number_is_rejected_with_specific_message(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"contact_phone": "+1-555-0100"}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state(
        "+1-555-0100", demo_step=6,
        demo_data={
            "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
            "contact_name": "Sam", "contact_info": "sam@acme.com", "contact_phone": None,
        },
    ))
    assert out["demo_data"]["contact_phone"] is None       # rejected, not stored
    assert out["demo_step"] == 6                            # stays on the same step
    assert out["demo_completed"] is False
    assert "start with 9" in out["messages"][0].content.lower()


def test_valid_mobile_number_is_accepted(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"contact_phone": "9812345678"}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    monkeypatch.setattr(nodes, "build_booking_link", lambda **kwargs: "https://cal.com/relayn/demo")

    out = _run("demo_node", _state(
        "9812345678", demo_step=6,
        demo_data={
            "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
            "contact_name": "Sam", "contact_info": "sam@acme.com", "contact_phone": None,
        },
    ))
    assert out["demo_completed"] is True
    assert out["demo_data"]["contact_phone"] == "9812345678"


def test_invalid_email_is_rejected_with_specific_message(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"contact_info": "not-an-email"}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state(
        "not-an-email", demo_step=5,
        demo_data={
            "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
            "contact_name": "Sam", "contact_info": None, "contact_phone": "9800000000",
        },
    ))
    assert out["demo_data"]["contact_info"] is None
    assert out["demo_step"] == 5
    assert out["demo_completed"] is False
    assert "valid email" in out["messages"][0].content.lower()


def test_explicit_cancel_intent_mid_flow_clears_data(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"wants_to_cancel": True}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state(
        "actually never mind, forget the whole thing", demo_step=3,
        demo_data={"business_name": "Acme", "channels": "WhatsApp", "monthly_volume": None,
                   "contact_name": None, "contact_info": None, "contact_phone": None},
    ))
    assert out["demo_completed"] is False
    assert out["demo_step"] == 0
    assert out["demo_data"] == {}
    assert out["intent"] == "STOP_DEMO"
    assert "2" in out["messages"][0].content


def test_nuisance_message_without_cancel_intent_just_re_asks(monkeypatch):
    import agent.nodes as nodes

    async def extract(_message, _data):
        return {"wants_to_cancel": False}

    monkeypatch.setattr(nodes, "_extract_demo_slots", extract)
    out = _run("demo_node", _state(
        "lol what are you, a robot? tell me a joke", demo_step=1,
        demo_data={},
    ))
    assert out["demo_data"]["business_name"] is None       # nothing stored
    assert out["demo_step"] == 1                            # same question re-asked
    assert out["intent"] == "BOOK_DEMO"
    assert "*1/6*" in out["messages"][0].content


def test_demo_end_resets_and_reports_count():
    out = _run("demo_end_node", _state("stop", demo_step=3, demo_data={
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": None,
        "contact_name": None, "contact_info": None, "contact_phone": None,
    }))
    assert out["demo_step"] == 0
    assert out["demo_data"] == {}
    assert out["demo_completed"] is False
    assert out["intent"] == "STOP_DEMO"
    assert "2" in out["messages"][0].content
