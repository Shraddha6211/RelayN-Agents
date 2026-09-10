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
    assert "*1/6*" in out["messages"][0].content


def test_open_answer_is_stored_and_flow_advances():
    out = _run("demo_node", _state("Acme Corp", demo_step=1, demo_data={}))
    assert out["demo_data"]["business_name"] == "Acme Corp"
    assert out["demo_step"] == 2
    assert "*2/6*" in out["messages"][0].content


def test_button_step_rejects_unknown_option_and_re_asks():
    out = _run("demo_node", _state("banana", demo_step=2, demo_data={"business_name": "Acme"}))
    assert out["demo_step"] == 2                            # not advanced
    assert out["demo_data"] == {"business_name": "Acme"}    # nothing stored, nothing lost
    text = out["messages"][0].content
    assert "Please tap one" in text
    assert "ch_all) All of them" in text                   # options shown again


def test_button_step_accepts_option_id_and_stores_title():
    out = _run("demo_node", _state("ch_all", demo_step=2, demo_data={"business_name": "Acme"}))
    assert out["demo_data"]["channels"] == "All of them"
    assert out["demo_step"] == 3


def test_final_answer_completes_and_emits_sentinel():
    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
    }
    out = _run("demo_node", _state("Tuesday afternoon", demo_step=6, demo_data=data))
    assert out["demo_completed"] is True
    assert out["demo_step"] == 0
    assert out["intent"] == "BOOK_DEMO"
    assert "[DEMO_BOOKED]" in out["messages"][0].content
    assert out["demo_data"]["preferred_time"] == "Tuesday afternoon"
    assert set(out["demo_data"]) == set(data) | {"preferred_time"}


def test_demo_end_resets_and_reports_count():
    out = _run("demo_end_node", _state("stop", demo_step=3, demo_data={"business_name": "Acme", "channels": "WhatsApp"}))
    assert out["demo_step"] == 0
    assert out["demo_data"] == {}
    assert out["demo_completed"] is False
    assert out["intent"] == "STOP_DEMO"
    assert "2" in out["messages"][0].content
