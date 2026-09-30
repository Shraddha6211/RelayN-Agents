import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

FULL = {"contact_name": "Sam", "contact_info": "sam@acme.com"}
EMPTY = {key: None for key in FULL}
LINK = "https://cal.com/relayn/demo?name=Sam"


def _run(fn_name, state):
    import agent.nodes as nodes
    return asyncio.run(getattr(nodes, fn_name)(state))


def _state(text, user_name="Sam", **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {"user_name": user_name}}
    base.update(extra)
    return base


def _turn(reply_intent="ANSWER", **slots):
    out = {key: None for key in FULL}
    out.update(slots)
    out.update({"reply_intent": reply_intent, "side_question_query": None, "language": "English"})
    return out


@pytest.fixture(autouse=True)
def identity_compose(monkeypatch):
    """Reply writer passes the English template through unchanged."""
    import agent.nodes as nodes

    async def compose(text, *_args, **_kwargs):
        return text

    monkeypatch.setattr(nodes, "_compose", compose)


@pytest.fixture
def booking_link(monkeypatch):
    import agent.nodes as nodes

    captured = {}

    def fake_build_link(**kwargs):
        captured.update(kwargs)
        return LINK

    monkeypatch.setattr(nodes, "build_booking_link", fake_build_link)
    return captured


def _extract_returns(monkeypatch, turn, calls=None):
    import agent.nodes as nodes

    async def extract(message, data, phase):
        if calls is not None:
            calls.append((message, dict(data), phase))
        return turn

    monkeypatch.setattr(nodes, "_extract_demo_turn", extract)


def _text(out):
    return out["messages"][0].content


def _assert_not_a_form(text):
    assert "1. " not in text and "2. " not in text
    assert "following" not in text.lower()


# ---------------------------------------------------------------------------
# START
# ---------------------------------------------------------------------------

def test_start_with_profile_name_asks_only_for_email(monkeypatch):
    _extract_returns(monkeypatch, _turn("OTHER"))
    out = _run("demo_node", _state("book demo", tool_data="START_DEMO"))

    assert out["demo_phase"] == "collecting"
    assert out["demo_completed"] is False
    assert out["intent"] == "BOOK_DEMO"
    assert out["demo_data"] == {"contact_name": "Sam", "contact_info": None}

    text = _text(out)
    assert "Sam" in text
    assert "email" in text.lower()
    assert "name" not in text.lower().replace("sam", "")
    _assert_not_a_form(text)


def test_start_without_profile_name_asks_name_and_email_in_one_line(monkeypatch):
    _extract_returns(monkeypatch, _turn("OTHER"))
    out = _run("demo_node", _state("book demo", user_name="there", tool_data="START_DEMO"))

    assert out["demo_data"] == EMPTY
    text = _text(out)
    assert "name and email" in text.lower()
    _assert_not_a_form(text)


def test_start_reads_earlier_chat_and_books_when_everything_is_known(monkeypatch, booking_link):
    calls = []
    _extract_returns(monkeypatch, _turn("OTHER", contact_info="sam@acme.com"), calls)
    state = _state("demo", tool_data="START_DEMO")
    state["messages"] = [
        HumanMessage(content="Hi, I'm Sam from Acme, sam@acme.com"),
        AIMessage(content="Nice to meet you!"),
        HumanMessage(content="demo"),
    ]
    out = _run("demo_node", state)

    assert "sam@acme.com" in calls[0][0]
    assert out["demo_completed"] is True
    assert LINK in _text(out)


def test_start_ignores_invalid_prefilled_email(monkeypatch):
    _extract_returns(monkeypatch, _turn("OTHER", contact_info="sam@acme"))
    out = _run("demo_node", _state("demo", tool_data="START_DEMO"))

    assert out["demo_data"]["contact_info"] is None
    assert out["demo_phase"] == "collecting"


# ---------------------------------------------------------------------------
# COLLECTING — one thing at a time
# ---------------------------------------------------------------------------

def test_name_only_reply_asks_just_for_email(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_name="Sita"))
    out = _run("demo_node", _state("Sita", user_name="there",
                                   demo_phase="collecting", demo_data=EMPTY))

    assert out["demo_data"] == {"contact_name": "Sita", "contact_info": None}
    text = _text(out)
    assert "email" in text.lower()
    assert "name" not in text.lower()


def test_email_only_reply_asks_just_for_name(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="sita@acme.com"))
    out = _run("demo_node", _state("sita@acme.com", user_name="there",
                                   demo_phase="collecting", demo_data=EMPTY))

    assert out["demo_data"] == {"contact_name": None, "contact_info": "sita@acme.com"}
    text = _text(out)
    assert "name" in text.lower()
    assert "email" not in text.lower()


def test_valid_email_books_straight_away_without_a_confirm_step(monkeypatch, booking_link):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="sam@acme.com"))
    out = _run("demo_node", _state("sam@acme.com", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    assert out["demo_completed"] is True
    assert out["demo_phase"] is None
    assert out["intent"] == "BOOK_DEMO"
    assert out["demo_data"]["cal_booking_link"] == LINK
    text = _text(out)
    assert LINK in text
    assert text.endswith("[DEMO_BOOKED]")
    assert booking_link["name"] == "Sam"
    assert booking_link["email"] == "sam@acme.com"


def test_invalid_email_is_rejected_and_asked_again(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="sam@acme"))
    out = _run("demo_node", _state("sam@acme", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    assert out["demo_data"]["contact_info"] is None
    assert out["demo_phase"] == "collecting"
    text = _text(out)
    assert "sam@acme" in text
    assert "double-check" in text


def test_invalid_email_with_no_name_also_asks_for_name(monkeypatch):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="sam@acme"))
    out = _run("demo_node", _state("sam@acme", user_name="there",
                                   demo_phase="collecting", demo_data=EMPTY))

    text = _text(out)
    assert "double-check" in text
    assert "name" in text.lower()


def test_user_can_override_profile_name(monkeypatch, booking_link):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_name="Hari", contact_info="hari@acme.com"))
    out = _run("demo_node", _state("send it to Hari, hari@acme.com", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    assert out["demo_completed"] is True
    assert booking_link["name"] == "Hari"


@pytest.mark.parametrize("intent", ["SKIP", "DENY"])
def test_skip_returns_to_general_chat(monkeypatch, intent):
    _extract_returns(monkeypatch, _turn(intent))
    out = _run("demo_node", _state("not now", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    assert _text(out) == (
        "No worries Sam, we can book a demo later. How may I help you with RelayN today?"
    )
    assert out["demo_phase"] is None
    assert out["demo_data"] == {}
    assert out["demo_completed"] is False
    assert out["intent"] == "STOP_DEMO"


def test_side_question_is_answered_then_gently_nudges(monkeypatch):
    import agent.nodes as nodes

    turn = _turn("QUESTION")
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

    out = _run("demo_node", _state("How much does it cost?", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    text = _text(out)
    assert text.startswith("Our Starter plan is NPR 2,000/month.")
    assert "your email" in text
    assert retrieved["query"] == "pricing plans"
    assert out["search_query"] == "pricing plans"
    assert out["demo_phase"] == "collecting"
    assert out["intent"] == "BOOK_DEMO"


def test_ok_after_side_question_asks_for_what_is_missing(monkeypatch):
    _extract_returns(monkeypatch, _turn("CONFIRM"))
    out = _run("demo_node", _state("ok", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    assert out["demo_phase"] == "collecting"
    assert "email" in _text(out).lower()


def test_booking_error_keeps_details_and_yes_retries(monkeypatch):
    import agent.nodes as nodes
    from calcom_client import CalComError

    def fake_build_link(**kwargs):
        raise CalComError("Cal.com booking link is not configured")

    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="sam@acme.com"))
    monkeypatch.setattr(nodes, "build_booking_link", fake_build_link)

    out = _run("demo_node", _state("sam@acme.com", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))
    assert out["demo_completed"] is False
    assert out["demo_phase"] == "collecting"
    assert out["demo_data"] == FULL
    assert "try again" in _text(out).lower()

    _extract_returns(monkeypatch, _turn("CONFIRM"))
    monkeypatch.setattr(nodes, "build_booking_link", lambda **_kw: LINK)
    retry = _run("demo_node", _state("yes", demo_phase="collecting", demo_data=FULL))
    assert retry["demo_completed"] is True


def test_stale_confirming_thread_with_old_fields_still_books(monkeypatch, booking_link):
    """Threads checkpointed by the old 6-question flow keep working."""
    _extract_returns(monkeypatch, _turn("CONFIRM"))
    old = {"business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
           "contact_name": "Sam", "contact_info": "sam@acme.com", "contact_phone": "9800000000"}
    out = _run("demo_node", _state("yes", demo_phase="confirming", demo_data=old))

    assert out["demo_completed"] is True
    assert booking_link["email"] == "sam@acme.com"


def test_wordless_reply_keeps_the_conversation_language(monkeypatch):
    """A bare email has no language of its own — keep replying in the user's script."""
    import agent.nodes as nodes

    seen = {}

    async def compose(text, name, language, *_args, **_kwargs):
        seen["language"] = language
        return text

    monkeypatch.setattr(nodes, "_compose", compose)
    turn = _turn("ANSWER", contact_name="Hari")
    turn["language"] = "unknown"
    _extract_returns(monkeypatch, turn)

    out = _run("demo_node", _state("Hari", user_name="there", demo_phase="collecting",
                                   demo_data=EMPTY, demo_language="Nepali (romanized)"))

    assert seen["language"] == "Nepali (romanized)"
    assert out["demo_language"] == "Nepali (romanized)"


@pytest.mark.parametrize("message", ["hari@gmail.com", "ok", "Hari Sharma", "ok hari@gmail.com"])
def test_too_few_words_keep_the_language_even_if_the_llm_says_english(monkeypatch, message):
    turn = _turn("ANSWER")
    turn["language"] = "English"
    _extract_returns(monkeypatch, turn)

    out = _run("demo_node", _state(message, user_name="there", demo_phase="collecting",
                                   demo_data=EMPTY, demo_language="Nepali (romanized)"))
    assert out["demo_language"] == "Nepali (romanized)"


def test_start_judges_language_by_the_latest_message_with_words(monkeypatch):
    calls = []
    _extract_returns(monkeypatch, _turn("OTHER"), calls)
    state = _state("demo", tool_data="START_DEMO")
    state["messages"] = [
        HumanMessage(content="does it support instagram?"),
        AIMessage(content="Yes!"),
        HumanMessage(content="ok malai demo chaincha"),
        AIMessage(content="..."),
        HumanMessage(content="demo"),
    ]
    _run("demo_node", state)

    sent = calls[0][0]
    assert sent.startswith("LATEST MESSAGE")
    assert sent.split("\n")[1] == "ok malai demo chaincha"
    assert "does it support instagram?" in sent


def test_new_language_replaces_the_previous_one(monkeypatch):
    turn = _turn("ANSWER", contact_name="Hari")
    turn["language"] = "Nepali (Devanagari)"
    _extract_returns(monkeypatch, turn)

    out = _run("demo_node", _state("मेरो नाम Hari", user_name="there", demo_phase="collecting",
                                   demo_data=EMPTY, demo_language="Nepali (romanized)"))
    assert out["demo_language"] == "Nepali (Devanagari)"


def test_unknown_language_with_no_history_defaults_to_english(monkeypatch):
    turn = _turn("OTHER")
    turn["language"] = "unknown"
    _extract_returns(monkeypatch, turn)

    out = _run("demo_node", _state("demo", tool_data="START_DEMO"))
    assert out["demo_language"] == "English"


def test_demo_replies_are_stored_as_ai_messages(monkeypatch):
    _extract_returns(monkeypatch, _turn("OTHER"))
    out = _run("demo_node", _state("book demo", tool_data="START_DEMO"))
    assert isinstance(out["messages"][0], AIMessage)


# ---------------------------------------------------------------------------
# HARD STOP + REPLY WRITER
# ---------------------------------------------------------------------------

def test_demo_end_uses_skip_message_and_resets():
    out = _run("demo_end_node", _state("stop", demo_phase="collecting",
                                       demo_data={"contact_name": "Sam", "contact_info": None}))
    assert _text(out).startswith("No worries Sam")
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
        return AIMessage(content="Yo! Here's your link.")

    monkeypatch.undo()
    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=lossy))

    text = f"Pick a time here: {LINK}"
    out = asyncio.run(nodes._compose(text, "Sam", "English", keep=[LINK]))
    assert out == text


def test_compose_shows_the_writer_the_recent_chat(monkeypatch):
    import types
    import agent.nodes as nodes

    seen = []

    async def good(messages):
        seen.extend(messages)
        return AIMessage(content="Sure Sam! What's the best email for the invite?")

    monkeypatch.undo()
    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=good))

    history = [
        HumanMessage(content="does it work with instagram?"),
        AIMessage(content="Yes, Instagram DMs land in the same inbox."),
        HumanMessage(content="cool, can I see it?"),
    ]
    out = asyncio.run(nodes._compose("What's the best email?", "Sam", "English", history=history))

    assert out == "Sure Sam! What's the best email for the invite?"
    prompt = seen[-1].content
    assert "Customer: cool, can I see it?" in prompt
    assert "You: Yes, Instagram DMs land in the same inbox." in prompt
    assert "What's the best email?" in prompt
