import asyncio
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from langchain_core.messages import AIMessage, HumanMessage

FULL = {"contact_name": "Sam", "contact_info": "sam@acme.com"}
EMPTY = {key: None for key in FULL}
NPT = ZoneInfo("Asia/Kathmandu")
NOW = datetime(2026, 9, 30, 9, 0, tzinfo=NPT)  # Wednesday, 9 am Nepal time
THU = date(2026, 10, 1)
MON = date(2026, 10, 5)


def _run(fn_name, state):
    import agent.nodes as nodes
    return asyncio.run(getattr(nodes, fn_name)(state))


def _state(text, user_name="Sam", **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {"user_name": user_name}}
    base.update(extra)
    return base


def _turn(reply_intent="ANSWER", **slots):
    out = {key: None for key in FULL}
    out.update({"requested_date": None, "requested_time": None, "wants_more_slots": False})
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


class FakeCalendar:
    """Cal.com stand-in: 09:00-16:45 every 15 minutes on weekdays, minus `taken`."""

    def __init__(self):
        self.taken: set[datetime] = set()
        self.full_days: set[date] = set()
        self.bookings: list[dict] = []
        self.fetched: list[date] = []
        self.down = False
        self.book_error = False

    def slots(self, day):
        if day.weekday() >= 5 or day in self.full_days:
            return []
        out, t = [], datetime.combine(day, time(9, 0), NPT)
        while t.time() <= time(16, 45):
            if t not in self.taken:
                out.append(t.isoformat(timespec="milliseconds"))
            t += timedelta(minutes=15)
        return out

    async def get_day_slots(self, day):
        from calcom_client import CalComError

        self.fetched.append(day)
        if self.down:
            raise CalComError("slots fetch failed: 503")
        return self.slots(day)

    async def create_booking(self, **kwargs):
        from calcom_client import CalComError

        if self.book_error:
            raise CalComError("booking failed: 500")
        start = datetime.fromisoformat(kwargs["start_iso"]).astimezone(NPT)
        if start in self.taken:
            raise CalComError("booking failed: 400 slot no longer available")
        self.bookings.append(kwargs)
        self.taken.add(start)
        return {"uid": "booking-1"}


@pytest.fixture
def calendar(monkeypatch):
    import agent.nodes as nodes
    import agent.scheduling as sched

    cal = FakeCalendar()
    monkeypatch.setattr(nodes, "get_day_slots", cal.get_day_slots)
    monkeypatch.setattr(nodes, "create_booking", cal.create_booking)
    monkeypatch.setattr(sched, "now", lambda: NOW)
    return cal


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


def _at(day, hh, mm):
    return datetime.combine(day, time(hh, mm), NPT)


def _scheduling(**schedule):
    return {"demo_phase": "scheduling", "demo_data": dict(FULL), "demo_schedule": schedule}


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


def test_start_reads_earlier_chat_and_asks_for_a_time_when_details_are_known(monkeypatch, calendar):
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
    assert out["demo_phase"] == "scheduling"
    assert "Monday to Friday" in _text(out)
    assert calendar.bookings == []


def test_start_uses_a_time_already_given_in_the_chat(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("OTHER", contact_info="sam@acme.com",
                                        requested_date="2026-10-01", requested_time="11:00"))
    out = _run("demo_node", _state("demo on thursday at 11? sam@acme.com", tool_data="START_DEMO"))

    assert out["demo_completed"] is True
    assert calendar.bookings[0]["start_iso"] == _at(THU, 11, 0).isoformat()


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


def test_complete_details_ask_for_a_day_and_time_not_a_link(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_info="sam@acme.com"))
    out = _run("demo_node", _state("sam@acme.com", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    assert out["demo_phase"] == "scheduling"
    assert out["demo_completed"] is False
    assert out["demo_data"] == FULL
    text = _text(out)
    assert "Monday to Friday" in text
    assert "10:30 am" in text and "5 pm" in text
    assert "http" not in text


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


def test_user_can_override_profile_name(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("ANSWER", contact_name="Hari", contact_info="hari@acme.com"))
    out = _run("demo_node", _state("send it to Hari, hari@acme.com", demo_phase="collecting",
                                   demo_data={"contact_name": "Sam", "contact_info": None}))

    assert out["demo_data"] == {"contact_name": "Hari", "contact_info": "hari@acme.com"}


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


def _patch_side_answer(monkeypatch, answer="Our Starter plan is NPR 2,000/month."):
    import agent.nodes as nodes

    retrieved = {}

    async def retrieve(user_data, intent, query):
        retrieved["query"] = query
        return "- Starter: NPR 2,000/month", query

    async def kb_answer(state, tool_data, extra_instruction=None):
        return AIMessage(content=answer)

    monkeypatch.setattr(nodes, "_retrieve", retrieve)
    monkeypatch.setattr(nodes, "_kb_answer", kb_answer)
    return retrieved


def test_side_question_is_answered_then_gently_nudges(monkeypatch):
    turn = _turn("QUESTION")
    turn["side_question_query"] = "pricing plans"
    _extract_returns(monkeypatch, turn)
    retrieved = _patch_side_answer(monkeypatch)

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


def test_stale_confirming_thread_with_old_fields_moves_on_to_scheduling(monkeypatch, calendar):
    """Threads checkpointed by the old 6-question flow keep working."""
    _extract_returns(monkeypatch, _turn("CONFIRM"))
    old = {"business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
           "contact_name": "Sam", "contact_info": "sam@acme.com", "contact_phone": "9800000000"}
    out = _run("demo_node", _state("yes", demo_phase="confirming", demo_data=old))

    assert out["demo_phase"] == "scheduling"
    assert out["demo_data"] == FULL


# ---------------------------------------------------------------------------
# SCHEDULING — slots booked in the chat, never a link
# ---------------------------------------------------------------------------

def test_free_slot_is_booked_and_confirmed_by_email(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01", requested_time="11:00"))
    out = _run("demo_node", _state("can I see the demo at 11am thursday?", **_scheduling()))

    assert calendar.bookings == [{
        "name": "Sam", "email": "sam@acme.com",
        "start_iso": _at(THU, 11, 0).isoformat(), "notes": "Booked from the RelayN chat",
    }]
    assert out["demo_completed"] is True
    assert out["demo_phase"] is None
    assert out["demo_data"]["booked_start"] == _at(THU, 11, 0).isoformat()
    assert out["demo_data"]["booking_uid"] == "booking-1"
    text = _text(out)
    assert "sent an email with the meeting details" in text
    assert "Thursday, 1 October" in text and "11:00 am" in text
    assert text.endswith("[DEMO_BOOKED]")
    assert "http" not in text


def test_taken_slot_shows_five_closest_and_mentions_more(monkeypatch, calendar):
    calendar.taken.add(_at(THU, 11, 0))
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01", requested_time="11:00"))
    out = _run("demo_node", _state("11am thursday?", **_scheduling()))

    assert calendar.bookings == []
    assert out["demo_phase"] == "scheduling"
    text = _text(out)
    assert "already booked" in text
    for t in ("10:30 am", "10:45 am", "11:15 am", "11:30 am", "11:45 am"):
        assert t in text
    assert "12:00 pm" not in text
    assert "more slots" in text
    schedule = out["demo_schedule"]
    assert schedule["day"] == "2026-10-01"
    assert schedule["target"] == "11:00"
    assert len(schedule["shown"]) == 5


def test_asking_for_more_shows_ten_new_slots_up_to_5pm(monkeypatch, calendar):
    calendar.taken.add(_at(THU, 11, 0))
    shown = [_at(THU, h, m).isoformat() for h, m in ((10, 30), (10, 45), (11, 15), (11, 30), (11, 45))]
    _extract_returns(monkeypatch, _turn("OTHER", wants_more_slots=True))
    out = _run("demo_node", _state("none of these work",
                                   **_scheduling(day="2026-10-01", target="11:00", shown=shown)))

    text = _text(out)
    assert "up to 5 pm" in text
    for t in ("10:30 am", "11:15 am"):
        assert t not in text
    assert "12:00 pm" in text and "1:15 pm" in text
    assert len(out["demo_schedule"]["shown"]) == 15
    assert calendar.bookings == []


def test_more_when_the_day_is_used_up_suggests_another_day(monkeypatch, calendar):
    all_open = [s for s in calendar.slots(THU) if "T09" not in s and "T10:0" not in s
                and "T10:15" not in s]
    _extract_returns(monkeypatch, _turn("OTHER", wants_more_slots=True))
    out = _run("demo_node", _state("any other time?",
                                   **_scheduling(day="2026-10-01", target="11:00", shown=all_open)))

    assert "another day" in _text(out)
    assert out["demo_phase"] == "scheduling"


def test_weekend_request_offers_monday_instead_without_booking(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-04", requested_time="11:00"))
    out = _run("demo_node", _state("can I see the demo at 11am sunday?", **_scheduling()))

    assert calendar.bookings == []
    text = _text(out)
    assert "Monday to Friday" in text
    assert "Monday, 5 October" in text
    assert "11:00 am" in text
    assert out["demo_schedule"]["day"] == "2026-10-05"


def test_out_of_hours_request_is_not_booked_even_if_cal_has_it(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01", requested_time="09:00"))
    out = _run("demo_node", _state("9am thursday", **_scheduling()))

    assert calendar.bookings == []
    text = _text(out)
    assert "10:30 am and 5 pm" in text
    assert "9:00 am" not in text
    assert "10:30 am" in text


def test_picking_a_shown_time_books_it_on_the_offered_day(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("ANSWER", requested_time="11:30"))
    out = _run("demo_node", _state("11:30 works",
                                   **_scheduling(day="2026-10-01", target="11:00", shown=[])))

    assert out["demo_completed"] is True
    assert calendar.bookings[0]["start_iso"] == _at(THU, 11, 30).isoformat()


def test_slot_taken_at_booking_time_shows_alternatives(monkeypatch, calendar):
    import agent.nodes as nodes

    real_create = calendar.create_booking

    async def snatched(**kwargs):
        calendar.taken.add(datetime.fromisoformat(kwargs["start_iso"]).astimezone(NPT))
        return await real_create(**kwargs)

    monkeypatch.setattr(nodes, "create_booking", snatched)
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01", requested_time="11:00"))
    out = _run("demo_node", _state("11am thursday", **_scheduling()))

    assert out["demo_completed"] is False
    assert "already booked" in _text(out)
    assert "11:15 am" in _text(out)


def test_fully_booked_day_offers_the_next_weekday(monkeypatch, calendar):
    calendar.full_days.add(THU)
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01", requested_time="11:00"))
    out = _run("demo_node", _state("11am thursday", **_scheduling()))

    text = _text(out)
    assert "Thursday, 1 October is fully booked" in text
    assert "Friday, 2 October" in text
    assert out["demo_schedule"]["day"] == "2026-10-02"
    assert calendar.bookings == []


def test_day_without_time_lists_the_earliest_slots(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01"))
    out = _run("demo_node", _state("thursday?", **_scheduling()))

    text = _text(out)
    for t in ("10:30 am", "10:45 am", "11:00 am", "11:15 am", "11:30 am"):
        assert t in text
    assert calendar.bookings == []


def test_no_time_given_asks_when(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("OTHER"))
    out = _run("demo_node", _state("hmm", **_scheduling()))

    assert "10:30 am" in _text(out)
    assert out["demo_phase"] == "scheduling"


def test_no_time_given_after_slots_were_shown_asks_to_pick(monkeypatch, calendar):
    _extract_returns(monkeypatch, _turn("OTHER"))
    out = _run("demo_node", _state("hmm", **_scheduling(day="2026-10-01", target="11:00",
                                                         shown=[])))

    assert "which of those times" in _text(out)


def test_calendar_down_keeps_everything_for_a_retry(monkeypatch, calendar):
    calendar.down = True
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01", requested_time="11:00"))
    out = _run("demo_node", _state("11am thursday", **_scheduling()))

    assert out["demo_phase"] == "scheduling"
    assert out["demo_data"] == FULL
    assert "couldn't reach our calendar" in _text(out)


def test_booking_error_keeps_everything_for_a_retry(monkeypatch, calendar):
    calendar.book_error = True
    _extract_returns(monkeypatch, _turn("ANSWER", requested_date="2026-10-01", requested_time="11:00"))
    out = _run("demo_node", _state("11am thursday", **_scheduling()))

    assert out["demo_completed"] is False
    assert out["demo_phase"] == "scheduling"
    assert "couldn't reach our calendar" in _text(out)


def test_side_question_while_scheduling_nudges_for_a_time(monkeypatch, calendar):
    turn = _turn("QUESTION")
    turn["side_question_query"] = "pricing plans"
    _extract_returns(monkeypatch, turn)
    _patch_side_answer(monkeypatch)

    out = _run("demo_node", _state("how much is it?", **_scheduling()))
    assert "day and time" in _text(out)
    assert out["demo_phase"] == "scheduling"


# ---------------------------------------------------------------------------
# LANGUAGE
# ---------------------------------------------------------------------------

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
    out = _run("demo_end_node", _state("stop", demo_phase="scheduling",
                                       demo_data=dict(FULL), demo_schedule={"day": "2026-10-01"}))
    assert _text(out).startswith("No worries Sam")
    assert out["demo_phase"] is None
    assert out["demo_data"] == {}
    assert out["demo_schedule"] == {}
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
        return AIMessage(content="Here are some times.")

    monkeypatch.undo()
    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=lossy))

    text = "Open times:\n• 10:30 am\n• 10:45 am"
    out = asyncio.run(nodes._compose(text, "Sam", "English", keep=["10:30 am", "10:45 am"]))
    assert out == text


def test_compose_retries_once_when_a_required_value_is_dropped(monkeypatch):
    import types
    import agent.nodes as nodes

    calls = []

    async def second_time_right(messages):
        calls.append(messages)
        if len(calls) == 1:
            return AIMessage(content="Friday 11 baje book ho gaya.")
        return AIMessage(content="Friday 11:00 am par book ho gaya.")

    monkeypatch.undo()
    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=second_time_right))

    out = asyncio.run(nodes._compose("Booked for 11:00 am.", "Amit", "Hindi (romanized)",
                                     keep=["11:00 am"]))
    assert out == "Friday 11:00 am par book ho gaya."
    assert "11:00 am" in calls[1][-1].content  # the retry names what was left out


def test_translate_allows_times_that_are_in_the_original_line(monkeypatch):
    import types
    import agent.nodes as nodes

    async def writer(_messages):
        return AIMessage(content="Thursday, 1 October ko 11:00 am book bhaisakyo. Najikaka khali samaya:")

    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=writer))
    brief = "11:00 am on Thursday, 1 October is already booked. Here are the closest open times:"
    out = asyncio.run(nodes._translate(brief, "Nepali (romanized)"))
    assert out.startswith("Thursday, 1 October ko 11:00 am")


def test_question_about_times_while_scheduling_is_not_a_side_question(monkeypatch, calendar):
    import agent.nodes as nodes

    async def no_side_answer(*_args, **_kwargs):
        raise AssertionError("asking for more times must not go to the knowledge base")

    monkeypatch.setattr(nodes, "_side_answer", no_side_answer)
    _extract_returns(monkeypatch, _turn("QUESTION", wants_more_slots=True))
    out = _run("demo_node", _state("aru samaya cha?",
                                   **_scheduling(day="2026-10-01", target="11:00", shown=[])))
    assert "up to 5 pm" in _text(out)


def test_translate_falls_back_when_times_keep_being_invented(monkeypatch):
    import types
    import agent.nodes as nodes

    async def writer(_messages):
        return AIMessage(content="5 pm samma slots: 3:30 pm, 4:15 pm")

    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=writer))

    brief = "Slots are available up to 5 pm that day."
    assert asyncio.run(nodes._translate(brief, "Nepali (romanized)", keep=["5 pm"])) == brief


def test_translate_passes_english_through_without_an_llm_call(monkeypatch):
    import types
    import agent.nodes as nodes

    async def never(_messages):
        raise AssertionError("no LLM call for English")

    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=never))
    brief = "Here are more open times on Monday, 5 October:"
    assert asyncio.run(nodes._translate(brief, "English")) == brief


def test_translate_uses_the_llm_and_rejects_invented_times(monkeypatch):
    import types
    import agent.nodes as nodes

    replies = iter([
        AIMessage(content="Monday ko thap samaya:\n• 2:00 pm"),
        AIMessage(content="Monday, 5 October ma yi thap samaya khali chan:"),
    ])

    async def writer(_messages):
        return next(replies)

    monkeypatch.setattr(nodes, "composer_llm", types.SimpleNamespace(ainvoke=writer))
    out = asyncio.run(nodes._translate("Here are more open times on Monday, 5 October:",
                                       "Nepali (romanized)"))
    assert out == "Monday, 5 October ma yi thap samaya khali chan:"


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
