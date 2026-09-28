import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

FULL = {
    "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
    "contact_name": "Sam", "contact_info": "sam@acme.com",
    "contact_phone": "9800000000",
}
EMPTY = {key: None for key in FULL}


def _run(fn_name, state):
    import agent.nodes as nodes
    return asyncio.run(getattr(nodes, fn_name)(state))


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {"user_name": "Sam"}}
    base.update(extra)
    return base


def _turn(reply_intent="ANSWER", **slots):
    out = {key: None for key in FULL}
    out.update(slots)
    out.update({"reply_intent": reply_intent, "side_question_query": None, "language": "English"})
    return out


@pytest.fixture(autouse=True)
def identity_compose(monkeypatch):
    """Composer passes the English template through unchanged."""
    import agent.nodes as nodes

    async def compose(text, *_args, **_kwargs):
        return text

    monkeypatch.setattr(nodes, "_compose", compose)


def _extract_returns(monkeypatch, turn, calls=None):
    import agent.nodes as nodes

    async def extract(message, data, phase):
        if calls is not None:
            calls.append((message, dict(data), phase))
        return turn

    monkeypatch.setattr(nodes, "_extract_demo_turn", extract)


# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------

def test_start_lists_all_five_questions_at_once(monkeypatch):
    _extract_returns(monkeypatch, _turn("OTHER"))
    out = _run("demo_node", _state("book demo", tool_data="START_DEMO"))

    assert out["demo_phase"] == "collecting"
    assert out["demo_completed"] is False
    assert out["intent"] == "BOOK_DEMO"
    assert set(out["demo_data"]) == set(FULL)

    text = out["messages"][0].content
    assert "Great, Sam!" in text
    assert "tailored demo experience" in text
    for n in range(1, 6):
        assert f"{n}. " in text
    assert "6. " not in text
    for option in ("WhatsApp", "Instagram", "Facebook", "All of them",
                   "Under 500", "500 to 2k", "2k to 10k", "10k or more"):
        assert option in text
    assert "email" in text.lower() and "mobile" in text.lower()


def test_start_prefills_answers_from_earlier_chat(monkeypatch):
    calls = []
    _extract_returns(monkeypatch, _turn("OTHER", business_name="Acme Corp"), calls)
    state = _state("demo", tool_data="START_DEMO")
    state["messages"] = [
        HumanMessage(content="Hi, I run Acme Corp, a bakery"),
        AIMessage(content="Nice to meet you!"),
        HumanMessage(content="demo"),
    ]
    out = _run("demo_node", state)

    assert "Hi, I run Acme Corp, a bakery" in calls[0][0]
    assert out["demo_data"]["business_name"] == "Acme Corp"
    text = out["messages"][0].content
    assert "already noted" in text
    assert "Acme Corp" in text
    assert "name of your business" not in text
    assert "4. " in text and "5. " not in text


def test_start_ignores_invalid_prefilled_contact_details(monkeypatch):
    _extract_returns(monkeypatch, _turn("OTHER", contact_phone="12345"))
    out = _run("demo_node", _state("demo", tool_data="START_DEMO"))
    assert out["demo_data"]["contact_phone"] is None


# ---------------------------------------------------------------------------
# COLLECTING
# ---------------------------------------------------------------------------

def test_complete_answer_moves_to_confirmation_with_summary(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", **FULL))
    out = _run("demo_node", _state("everything", demo_phase="collecting", demo_data=EMPTY))

    assert out["demo_phase"] == "confirming"
    assert out["demo_completed"] is False
    text = out["messages"][0].content
    for value in FULL.values():
        assert value in text
    assert "name of your business" in text           # question shown next to its answer
    assert "correct" in text.lower()


def test_partial_answer_asks_only_for_unanswered_questions(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", business_name="Acme", channels="ch_all"))
    out = _run("demo_node", _state("Acme, all of them", demo_phase="collecting", demo_data=EMPTY))

    assert out["demo_phase"] == "collecting"
    assert out["demo_data"]["business_name"] == "Acme"
    assert out["demo_data"]["channels"] == "All of them"
    text = out["messages"][0].content
    assert text.startswith("Thanks for the information.")
    assert "so that I could book an appointment" in text
    assert "name of your business" not in text
    assert "Which channel" not in text
    assert "messages" in text.lower()
    assert "1. " in text and "3. " in text and "4. " not in text


def test_half_answered_contact_question_asks_only_for_the_missing_part(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="sam@acme.com"))
    data = {**FULL, "contact_info": None, "contact_phone": None}
    out = _run("demo_node", _state("sam@acme.com", demo_phase="collecting", demo_data=data))

    text = out["messages"][0].content
    assert out["demo_data"]["contact_info"] == "sam@acme.com"
    assert "mobile" in text.lower()
    assert "email" not in text.lower()


def test_unknown_button_value_is_not_stored(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", channels="banana"))
    out = _run("demo_node", _state("banana", demo_phase="collecting", demo_data=EMPTY))
    assert out["demo_data"]["channels"] is None
    assert out["demo_phase"] == "collecting"


def test_invalid_email_is_rejected_before_confirmation(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="not-an-email"))
    data = {**FULL, "contact_info": None}
    out = _run("demo_node", _state("not-an-email", demo_phase="collecting", demo_data=data))

    assert out["demo_data"]["contact_info"] is None
    assert out["demo_phase"] == "collecting"
    assert "valid email" in out["messages"][0].content.lower()


def test_invalid_mobile_is_rejected_before_confirmation(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_phone="+1-555-0100"))
    data = {**FULL, "contact_phone": None}
    out = _run("demo_node", _state("+1-555-0100", demo_phase="collecting", demo_data=data))

    assert out["demo_data"]["contact_phone"] is None
    assert out["demo_phase"] == "collecting"
    assert "start with 9" in out["messages"][0].content.lower()


@pytest.mark.parametrize("intent", ["SKIP", "DENY"])
def test_skip_returns_to_general_chat(monkeypatch, intent):
    _extract_returns(monkeypatch, _turn(intent))
    out = _run("demo_node", _state("not now", demo_phase="collecting",
                                   demo_data={**EMPTY, "business_name": "Acme"}))

    assert out["messages"][0].content == (
        "No worries Sam, we can book a demo later. How may I help you with RelayN today?"
    )
    assert out["demo_phase"] is None
    assert out["demo_data"] == {}
    assert out["demo_completed"] is False
    assert out["intent"] == "STOP_DEMO"


def test_side_question_is_answered_then_brings_user_back(monkeypatch):
    import agent.nodes as nodes

    turn = _turn("QUESTION", business_name="Acme")
    turn["side_question_query"] = "pricing plans"
    _extract_returns(monkeypatch, turn)

    retrieved = {}

    async def retrieve(user_data, intent, query):
        retrieved["query"] = query
        return "- Starter: NPR 2,000/month", query

    async def kb_answer(state, tool_data, extra_instruction=None):
        assert tool_data == "- Starter: NPR 2,000/month"
        return AIMessage(content="Our Starter plan is NPR 2,000/month.")

    monkeypatch.setattr(nodes, "_retrieve", retrieve)
    monkeypatch.setattr(nodes, "_kb_answer", kb_answer)

    out = _run("demo_node", _state("Acme here. How much does it cost?",
                                   demo_phase="collecting", demo_data=EMPTY))

    text = out["messages"][0].content
    assert text.startswith("Our Starter plan is NPR 2,000/month.")
    assert text.endswith("Should we continue to schedule a demo appointment with the RelayN team?")
    assert retrieved["query"] == "pricing plans"
    assert out["search_query"] == "pricing plans"
    assert out["demo_phase"] == "collecting"
    assert out["demo_data"]["business_name"] == "Acme"
    assert out["intent"] == "BOOK_DEMO"


def test_yes_after_side_question_reshows_remaining_questions(monkeypatch):
    _extract_returns(monkeypatch, _turn("CONFIRM"))
    data = {**EMPTY, "business_name": "Acme"}
    out = _run("demo_node", _state("yes", demo_phase="collecting", demo_data=data))

    text = out["messages"][0].content
    assert out["demo_phase"] == "collecting"
    assert "Thanks for the information" not in text      # nothing new was given
    assert "Which channel" in text
    assert "name of your business" not in text


# ---------------------------------------------------------------------------
# CONFIRMING
# ---------------------------------------------------------------------------

def test_confirmation_sends_booking_link(monkeypatch):
    import agent.nodes as nodes

    captured = {}

    def fake_build_link(**kwargs):
        captured.update(kwargs)
        return "https://cal.com/relayn/demo?name=Sam"

    _extract_returns(monkeypatch, _turn("CONFIRM"))
    monkeypatch.setattr(nodes, "build_booking_link", fake_build_link)

    out = _run("demo_node", _state("yes", demo_phase="confirming", demo_data=FULL))

    assert out["demo_completed"] is True
    assert out["demo_phase"] is None
    assert out["intent"] == "BOOK_DEMO"
    assert out["demo_data"]["cal_booking_link"] == "https://cal.com/relayn/demo?name=Sam"
    text = out["messages"][0].content
    assert "https://cal.com/relayn/demo?name=Sam" in text
    assert text.endswith("[DEMO_BOOKED]")
    assert captured["name"] == "Sam"
    assert captured["email"] == "sam@acme.com"
    for value in ("Acme", "WhatsApp", "500 to 2k", "9800000000"):
        assert value in captured["notes"]


def test_booking_error_keeps_answers_for_retry(monkeypatch):
    import agent.nodes as nodes
    from calcom_client import CalComError

    def fake_build_link(**kwargs):
        raise CalComError("Cal.com booking link is not configured")

    _extract_returns(monkeypatch, _turn("CONFIRM"))
    monkeypatch.setattr(nodes, "build_booking_link", fake_build_link)

    out = _run("demo_node", _state("yes", demo_phase="confirming", demo_data=FULL))
    assert out["demo_completed"] is False
    assert out["demo_phase"] == "confirming"
    assert out["demo_data"] == FULL
    assert "try again" in out["messages"][0].content.lower()


def test_correction_updates_and_reconfirms(monkeypatch):
    _extract_returns(monkeypatch, _turn("DENY", contact_info="new@acme.com"))
    out = _run("demo_node", _state("email is new@acme.com",
                                   demo_phase="confirming", demo_data=FULL))

    assert out["demo_phase"] == "confirming"
    assert out["demo_completed"] is False
    assert out["demo_data"]["contact_info"] == "new@acme.com"
    assert "new@acme.com" in out["messages"][0].content


def test_invalid_correction_shows_error_and_keeps_old_value(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_phone="12345"))
    out = _run("demo_node", _state("phone is 12345", demo_phase="confirming", demo_data=FULL))

    assert out["demo_phase"] == "confirming"
    assert out["demo_data"]["contact_phone"] == "9800000000"
    assert "start with 9" in out["messages"][0].content.lower()


def test_deny_without_details_asks_what_to_change(monkeypatch):
    _extract_returns(monkeypatch, _turn("DENY"))
    out = _run("demo_node", _state("no, that's wrong", demo_phase="confirming", demo_data=FULL))

    assert out["demo_phase"] == "confirming"
    assert "which detail" in out["messages"][0].content.lower()


def test_skip_while_confirming_returns_to_general_chat(monkeypatch):
    _extract_returns(monkeypatch, _turn("SKIP"))
    out = _run("demo_node", _state("forget it", demo_phase="confirming", demo_data=FULL))
    assert out["demo_phase"] is None
    assert out["intent"] == "STOP_DEMO"
    assert out["messages"][0].content.startswith("No worries Sam")


def test_side_question_while_confirming_asks_to_go_ahead(monkeypatch):
    import agent.nodes as nodes

    _extract_returns(monkeypatch, _turn("QUESTION"))

    async def retrieve(user_data, intent, query):
        return None, "RelayN features and capabilities"

    async def kb_answer(state, tool_data, extra_instruction=None):
        return AIMessage(content="Yes, Instagram is supported.")

    monkeypatch.setattr(nodes, "_retrieve", retrieve)
    monkeypatch.setattr(nodes, "_kb_answer", kb_answer)

    out = _run("demo_node", _state("does it do instagram?", demo_phase="confirming", demo_data=FULL))
    text = out["messages"][0].content
    assert text.startswith("Yes, Instagram is supported.")
    assert "go ahead and book" in text
    assert out["demo_phase"] == "confirming"


# ---------------------------------------------------------------------------
# HARD STOP + COMPOSER
# ---------------------------------------------------------------------------

def test_demo_end_uses_skip_message_and_resets():
    out = _run("demo_end_node", _state("stop", demo_phase="collecting",
                                       demo_data={**EMPTY, "business_name": "Acme"}))
    assert out["messages"][0].content.startswith("No worries Sam")
    assert out["demo_phase"] is None
    assert out["demo_data"] == {}
    assert out["demo_completed"] is False
    assert out["intent"] == "STOP_DEMO"


def test_compose_falls_back_to_template_on_llm_error(monkeypatch):
    import types
    import agent.nodes as nodes

    async def boom(_messages):
        raise RuntimeError("openai down")

    monkeypatch.undo()  # restore the real _compose (the autouse fixture patched it)
    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=boom))

    out = asyncio.run(nodes._compose("Hello there", "Sam", "English"))
    assert out == "Hello there"


def test_compose_falls_back_when_llm_drops_a_required_value(monkeypatch):
    import types
    import agent.nodes as nodes

    async def lossy(_messages):
        return AIMessage(content="Namaste! Tapai ko business ko naam?")

    monkeypatch.undo()
    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=lossy))

    text = "Which channel?\n   • WhatsApp\n   • Instagram"
    out = asyncio.run(nodes._compose(text, "Sam", "Nepali", keep=["WhatsApp", "Instagram"]))
    assert out == text


def test_compose_uses_llm_output_when_it_keeps_required_values(monkeypatch):
    import types
    import agent.nodes as nodes

    async def good(_messages):
        return AIMessage(content="Kun channel? • WhatsApp • Instagram")

    monkeypatch.undo()
    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=good))

    out = asyncio.run(nodes._compose("Which channel?", "Sam", "Nepali", keep=["WhatsApp", "Instagram"]))
    assert out == "Kun channel? • WhatsApp • Instagram"
