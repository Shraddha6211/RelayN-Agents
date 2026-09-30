# agent/nodes.py
import asyncio
import json
import logging
import re
from datetime import date, datetime, time, timedelta
from typing import Iterable, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent.prompts import (
    DEMO_ASK_EMAIL,
    DEMO_ASK_NAME,
    DEMO_ASK_NAME_EMAIL,
    DEMO_ASK_TIME,
    DEMO_BAD_EMAIL,
    DEMO_BOOKED,
    DEMO_CALENDAR_ERROR,
    DEMO_COMPOSER_SYSTEM_PROMPT,
    DEMO_DAY_FULL,
    DEMO_DAY_USED_UP,
    DEMO_MORE_HINT,
    DEMO_MORE_SLOTS,
    DEMO_NUDGE,
    DEMO_NUDGE_TIME,
    DEMO_OPEN_TIMES,
    DEMO_OUT_OF_HOURS,
    DEMO_PICK_SLOT,
    DEMO_SIDE_ANSWER_INSTRUCTION,
    DEMO_SKIP,
    DEMO_SLOT_TAKEN,
    DEMO_START,
    DEMO_TRANSLATE_SYSTEM_PROMPT,
    DEMO_TURN_SYSTEM_PROMPT,
    DEMO_UP_TO_5PM,
    DEMO_WEEKDAYS_ONLY,
    GENERATOR_SYSTEM_PROMPT,
    HANDOFF_MESSAGE,
    RELAYN_BUSINESS,
    RELAYN_CONTACT,
    SALES_EXTRACTION_SYSTEM_PROMPT,
    SALES_QUESTIONS,
    format_question,
    greeting_reply,
    render,
)
from agent import scheduling as sched
from agent.state import AgentState
from agent.tools import search_knowledge_base
from config import settings
from calcom_client import CalComError, create_booking, get_day_slots

logger = logging.getLogger("relayn_agents.nodes")


# ---------------------------------------------------------------------------
# SHARED
# ---------------------------------------------------------------------------

def _wizard_summary(data: dict) -> str:
    return "\n".join(f"• {k.replace('_', ' ').title()}: {v}" for k, v in data.items())


class DemoTurn(BaseModel):
    contact_name: Optional[str] = Field(default=None)
    contact_info: Optional[str] = Field(default=None)
    requested_date: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    requested_time: Optional[str] = Field(default=None, description="HH:MM, 24-hour")
    wants_more_slots: bool = Field(default=False)
    reply_intent: Literal["ANSWER", "CONFIRM", "DENY", "QUESTION", "SKIP", "OTHER"] = Field(
        default="OTHER"
    )
    side_question_query: Optional[str] = Field(default=None)
    language: str = Field(default="English")


class SalesSlotExtraction(BaseModel):
    need: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    contact_info: Optional[str] = Field(default=None)


demo_extractor = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY,
).with_structured_output(DemoTurn)

sales_extractor = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY,
).with_structured_output(SalesSlotExtraction)

composer_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3,
    openai_api_key=settings.OPENAI_API_KEY,
)

_DEMO_KEYS = ("contact_name", "contact_info")
_SALES_KEYS = tuple(q["key"] for q in SALES_QUESTIONS)

_DEMO_PHASE_CONTEXT = {
    "start": "the user just asked for a demo; USER MESSAGE holds their recent chat messages, "
             "one per line — extract a name or email they already gave, and judge language "
             "only by the LATEST MESSAGE",
    "collecting": "asked the user for the name and/or email to put on the demo invite "
                  "(CURRENT STATE shows which one is still null)",
    "scheduling": "asked which day and time suits the user for the demo (Monday to Friday, "
                  "10:30 am to 5 pm), or listed open times on OFFERED DAY for them to pick from",
}


async def _extract_demo_turn(user_msg: str, current_data: dict, phase: str) -> dict:
    """`current_data` holds the name/email so far, plus `offered_day` while scheduling."""
    context = json.dumps(
        {key: current_data.get(key) for key in _DEMO_KEYS},
        ensure_ascii=True,
    )
    result = await demo_extractor.ainvoke([
        SystemMessage(content=DEMO_TURN_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"TODAY: {sched.now():%A %Y-%m-%d}\n"
            f"OFFERED DAY: {current_data.get('offered_day') or 'none'}\n"
            f"BOT IS CURRENTLY: {_DEMO_PHASE_CONTEXT[phase]}\n\n"
            f"CURRENT STATE:\n{context}\n\nUSER MESSAGE:\n{user_msg}"
        )),
    ])
    values = result.model_dump() if isinstance(result, DemoTurn) else dict(result)
    out = {key: values.get(key) for key in _DEMO_KEYS}
    out["requested_date"] = values.get("requested_date")
    out["requested_time"] = values.get("requested_time")
    out["wants_more_slots"] = bool(values.get("wants_more_slots"))
    out["reply_intent"] = values.get("reply_intent") or "OTHER"
    out["side_question_query"] = values.get("side_question_query")
    out["language"] = values.get("language") or "unknown"
    return out


_EMAIL_OR_URL = re.compile(r"\S+@\S+|https?://\S+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _has_own_words(text: str) -> bool:
    """Whether a message says enough to tell its language by. A bare email, "ok" or a
    name doesn't — the LLM calls those English, which would flip a Nepali chat."""
    if _DEVANAGARI.search(text):
        return True
    return len(re.findall(r"[^\W\d_]+", _EMAIL_OR_URL.sub(" ", text))) >= 3


def _demo_language(detected: str | None, previous: str | None) -> str:
    """The language + script to reply in; "unknown" keeps the conversation's earlier one."""
    if detected and detected.strip().lower() != "unknown":
        return detected
    return previous or "English"


def _empty_demo_data(data: dict) -> dict:
    return {key: data.get(key) for key in _DEMO_KEYS}


def _is_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value))


def _merge_demo_slots(data: dict, extracted: dict) -> tuple[dict, str | None]:
    """Merge the extracted name and email. An invalid email is dropped and returned."""
    merged = _empty_demo_data(data)
    bad_email = None
    for key in _DEMO_KEYS:
        value = str(extracted.get(key) or "").strip()
        if not value:
            continue
        if key == "contact_info" and not _is_email(value):
            bad_email = value
            continue
        merged[key] = value
    return merged, bad_email


async def _extract_sales_slots(user_msg: str, current_data: dict) -> dict:
    context = json.dumps(
        {key: current_data.get(key) for key in _SALES_KEYS},
        ensure_ascii=True,
    )
    result = await sales_extractor.ainvoke([
        SystemMessage(content=SALES_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"CURRENT STATE:\n{context}\n\nUSER MESSAGE:\n{user_msg}"),
    ])
    values = result.model_dump() if isinstance(result, SalesSlotExtraction) else dict(result)
    return {key: values.get(key) for key in _SALES_KEYS}


def _empty_sales_data(data: dict) -> dict:
    return {key: data.get(key) for key in _SALES_KEYS}


def _merge_sales_slots(data: dict, extracted: dict) -> dict:
    merged = _empty_sales_data(data)
    for key in _SALES_KEYS:
        value = extracted.get(key)
        if value is not None and str(value).strip():
            merged[key] = str(value).strip()
    return merged


# ---------------------------------------------------------------------------
# DEMO FLOW
# Collects just a name and an email, one ask at a time, like a support rep
# would, then books a slot in the chat (DEMO SCHEDULING below). Code decides
# what the next message must do; the reply writer says it naturally with the
# recent chat in view.
# ---------------------------------------------------------------------------

def _chat_transcript(messages: Iterable, limit: int = 6) -> str:
    lines = []
    for m in list(messages)[-limit:]:
        kind = getattr(m, "type", None)
        if kind == "human":
            lines.append(f"Customer: {m.content}")
        elif kind in ("ai", "system"):  # older threads stored bot replies as system messages
            lines.append(f"You: {m.content}")
    return "\n".join(lines)


_CLOCK_TIME = re.compile(r"\d{1,2}:\d{2}")
_BULLET = re.compile(r"^\s*[•*-]\s", re.MULTILINE)


def _invents_times(out: str, source: str) -> bool:
    """A clock time the source line doesn't have, or a list of its own."""
    return bool(set(_CLOCK_TIME.findall(out)) - set(_CLOCK_TIME.findall(source))
                or _BULLET.search(out))


async def _compose(
    text: str,
    name: str | None,
    language: str,
    keep: Iterable[str] = (),
    history: Iterable = (),
) -> str:
    """Write the next demo message from a brief (an English template), in the
    user's language, as a natural reply to the recent chat in `history`."""
    return await _write(
        DEMO_COMPOSER_SYSTEM_PROMPT,
        f"CUSTOMER NAME: {name or 'unknown'}\n\n"
        f"RECENT CHAT:\n{_chat_transcript(history) or '(none)'}\n\nBRIEF:\n{text}\n\n"
        f"WRITE IN: {language}",
        text, keep=keep,
    )


async def _translate(text: str, language: str, keep: Iterable[str] = ()) -> str:
    """A faithful translation of a fixed line, for the text around a slot list.
    English goes out exactly as written."""
    if language.strip().lower().startswith("english"):
        return text
    return await _write(
        DEMO_TRANSLATE_SYSTEM_PROMPT, f"WRITE IN: {language}\n\nMESSAGE:\n{text}",
        text, keep=keep, no_times=True,
    )


async def _write(
    system: str, prompt: str, text: str, keep: Iterable[str] = (), no_times: bool = False
) -> str:
    """Run the reply writer and check its output.

    Every value in `keep` must appear verbatim, and with `no_times` the message
    may not add a clock time or a list of its own (the lines around a slot list,
    where the writer would otherwise invent slots). A miss gets one retry that names
    it; after that, or if the LLM fails, the English template `text` goes out —
    a less polished message beats a wrong one.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
    dropped: list[str] = []
    invented = False
    for _attempt in range(2):
        try:
            response = await composer_llm.ainvoke(messages)
        except Exception:
            logger.exception("demo composer failed; sending the template")
            return text
        out = (getattr(response, "content", "") or "").strip()
        dropped = [k for k in keep if k and k not in out]
        invented = no_times and _invents_times(out, text)
        if out and not dropped and not invented:
            return out
        problems = []
        if dropped or not out:
            problems.append("You left out " + ", ".join(f'"{k}"' for k in dropped or ["the message"])
                            + " — include each exactly as written.")
        if invented:
            problems.append("Do not mention any times or write a list — only what the brief says.")
        messages = [*messages, AIMessage(content=out), HumanMessage(content=" ".join(problems))]
    logger.warning("demo composer missed the brief (dropped %s, invented times: %s); "
                   "sending the template", dropped, invented)
    return text


def _user_name(state: AgentState) -> str | None:
    name = (state.get("user_data") or {}).get("user_name")
    return name if name and name != "there" else None


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def _is_demo_complete(data: dict) -> bool:
    return all(data.get(key) for key in _DEMO_KEYS)


def _missing_label(data: dict) -> str:
    """What is still needed, for DEMO_NUDGE."""
    return {
        (True, True): "your name and email",
        (True, False): "your name",
        (False, True): "your email",
    }[(data["contact_name"] is None, data["contact_info"] is None)]


def _demo_reply(
    text: str,
    *,
    phase: str | None,
    data: dict,
    language: str,
    schedule: dict | None = None,
    intent: str = "BOOK_DEMO",
    completed: bool = False,
    search_query: str | None = None,
) -> dict:
    out = {
        "messages": [AIMessage(content=text)],
        "demo_phase": phase,
        "demo_data": data,
        "demo_schedule": schedule or {},
        "demo_completed": completed,
        "demo_language": language,
        "intent": intent,
        "tool_data": None,
    }
    if search_query:
        out["search_query"] = search_query
    return out


async def _say(
    state: AgentState,
    text: str,
    language: str,
    name: str | None = None,
    keep: Iterable[str] = (),
    after: str | None = None,
) -> str:
    """Compose `text` against the recent chat; `after` is a reply already written this turn."""
    history = [*state["messages"], *([AIMessage(content=after)] if after else [])]
    return await _compose(text, name or _user_name(state), language, keep=keep, history=history)


async def _demo_skip(state: AgentState, language: str) -> dict:
    name = _user_name(state)
    text = DEMO_SKIP.replace("{{NAME}}", f" {name}" if name else "")
    return _demo_reply(
        await _say(state, text, language),
        phase=None, data={}, language=language, intent="STOP_DEMO",
    )


async def _ask_missing(
    state: AgentState, data: dict, bad_email: str | None, language: str, opener: str = ""
) -> dict:
    """One short message asking only for what is still missing."""
    if bad_email:
        ask = DEMO_BAD_EMAIL.replace("{{VALUE}}", bad_email)
        if data["contact_name"] is None:
            ask = f"{ask} {DEMO_ASK_NAME}"
    elif data["contact_name"] is None and data["contact_info"] is None:
        ask = DEMO_ASK_NAME_EMAIL
    elif data["contact_info"] is None:
        ask = DEMO_ASK_EMAIL
    else:
        ask = DEMO_ASK_NAME
    text = f"{opener} {ask}".strip()
    return _demo_reply(
        await _say(state, text, language, name=data["contact_name"]),
        phase="collecting", data=data, language=language,
    )


async def _side_answer(
    state: AgentState, turn: dict, data: dict, schedule: dict, language: str
) -> dict:
    tool_data, query = await _retrieve(
        state.get("user_data") or {}, None, turn.get("side_question_query")
    )
    answer = await _kb_answer(state, tool_data, extra_instruction=DEMO_SIDE_ANSWER_INSTRUCTION)
    answer_text = str(answer.content).strip()
    complete = _is_demo_complete(data)
    nudge = DEMO_NUDGE_TIME if complete else DEMO_NUDGE.replace("{{MISSING}}", _missing_label(data))
    follow = await _say(state, nudge, language, name=data["contact_name"], after=answer_text)
    return _demo_reply(
        _join(answer_text, follow),
        phase="scheduling" if complete else "collecting",
        data=data, schedule=schedule, language=language, search_query=query,
    )


# ---------------------------------------------------------------------------
# DEMO SCHEDULING
# The bot books the slot itself — no links. The extractor only reads the day
# and time the user asked for; code decides the day, the slots on offer
# (agent/scheduling.py) and whether to book. demo_schedule remembers the day
# on offer, the time it was centred on, and the slots already shown.
# ---------------------------------------------------------------------------

_BOOKING_NOTES = "Booked from the RelayN chat"
_FIRST_OFFER = 5
_MORE_OFFER = 10
_MAX_DAYS_AHEAD = 10


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def _parse_time(value: object) -> time | None:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value or "").strip())
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        return None
    return time(int(match[1]), int(match[2]))


async def _open_slots(day: date) -> list[datetime]:
    return sched.open_slots(await get_day_slots(day))


def _slot_lines(slots: list[datetime]) -> str:
    return "\n".join(f"• {sched.fmt_time(s)}" for s in slots)


async def _scheduling_reply(
    state: AgentState, text: str, data: dict, schedule: dict, language: str,
    keep: Iterable[str] = (),
) -> dict:
    return _demo_reply(
        await _say(state, text, language, name=data["contact_name"], keep=keep),
        phase="scheduling", data=data, schedule=schedule, language=language,
    )


async def _slot_message(
    state: AgentState, data: dict, language: str, *,
    lead: str, picks: list[datetime], note: str = "", note_keep: Iterable[str] = (),
) -> str:
    """Lead, the slot list, then a note. The list is added by code so it is always
    there; the lead and note are translated, not rewritten, so they say exactly
    what they should and no invented times slip in."""
    body = _join(await _translate(lead, language), _slot_lines(picks))
    if not note:
        return body
    return _join(body, await _translate(note, language, keep=note_keep))


async def _offer_slots(
    state: AgentState, data: dict, language: str, *,
    lead: str, day: date, target: time, slots: list[datetime],
) -> dict:
    """The five open slots closest to `target`, with a note that the day has more."""
    picks = sched.closest(slots, target, exclude=(), count=_FIRST_OFFER)
    text = await _slot_message(
        state, data, language,
        lead=lead.replace("{{DAY}}", sched.fmt_day(day)), picks=picks,
        note=DEMO_MORE_HINT if len(slots) > len(picks) else "",
    )
    schedule = {
        "day": day.isoformat(),
        "target": f"{target:%H:%M}",
        "shown": [p.isoformat() for p in picks],
    }
    return _demo_reply(text, phase="scheduling", data=data, schedule=schedule, language=language)


async def _more_slots(state: AgentState, data: dict, schedule: dict, language: str) -> dict:
    day = _parse_date(schedule["day"])
    target = _parse_time(schedule.get("target")) or sched.FIRST_START
    shown = [datetime.fromisoformat(s) for s in schedule.get("shown", [])]
    picks = sched.closest(await _open_slots(day), target, exclude=shown, count=_MORE_OFFER)
    if not picks:
        text = DEMO_DAY_USED_UP.replace("{{DAY}}", sched.fmt_day(day))
        return await _scheduling_reply(state, text, data, schedule, language)
    text = await _slot_message(
        state, data, language,
        lead=DEMO_MORE_SLOTS.replace("{{DAY}}", sched.fmt_day(day)), picks=picks,
        note=DEMO_UP_TO_5PM, note_keep=["5 pm"],
    )
    schedule = {**schedule, "shown": [*schedule.get("shown", []), *(p.isoformat() for p in picks)]}
    return _demo_reply(text, phase="scheduling", data=data, schedule=schedule, language=language)


async def _booked(
    state: AgentState, data: dict, slot: datetime, booking: dict, language: str
) -> dict:
    text = (DEMO_BOOKED.replace("{{DAY}}", sched.fmt_day(slot.date()))
            .replace("{{TIME}}", sched.fmt_time(slot)))
    reply = await _say(state, text, language, name=data["contact_name"],
                       keep=[sched.fmt_time(slot)])
    return _demo_reply(
        f"{reply} [DEMO_BOOKED]",
        phase=None,
        data={**data, "booked_start": slot.isoformat(), "booking_uid": booking.get("uid")},
        language=language, completed=True,
    )


async def _find_slot(
    state: AgentState, data: dict, requested: date | None, at: time | None, language: str
) -> dict:
    """Book the requested slot if it is free; otherwise offer the closest open ones."""
    day, weekend = sched.plan_day(requested, at, sched.now())
    slots = await _open_slots(day)
    full_day = None
    for _ in range(_MAX_DAYS_AHEAD):
        if slots:
            break
        full_day = full_day or day
        day = sched.next_weekday(day + timedelta(days=1))
        slots = await _open_slots(day)
    if not slots:
        raise CalComError(f"no open slots within {_MAX_DAYS_AHEAD} days of {day}")

    wanted = datetime.combine(day, at, sched.tz()) if at else None
    if wanted and not weekend and not full_day and wanted in slots:
        try:
            booking = await create_booking(
                name=data["contact_name"], email=data["contact_info"],
                start_iso=wanted.isoformat(), notes=_BOOKING_NOTES,
            )
            return await _booked(state, data, wanted, booking, language)
        except CalComError:
            slots = await _open_slots(day)
            if wanted in slots:
                raise  # the slot is still free, so this was a real failure
            logger.info("slot %s was taken while booking; offering others", wanted)

    if weekend:
        lead = DEMO_WEEKDAYS_ONLY
    elif full_day:
        lead = DEMO_DAY_FULL.replace("{{FULL_DAY}}", sched.fmt_day(full_day))
    elif at and not sched.within_hours(at):
        lead = DEMO_OUT_OF_HOURS
    elif wanted:
        lead = DEMO_SLOT_TAKEN.replace("{{TIME}}", sched.fmt_time(wanted))
    else:
        lead = DEMO_OPEN_TIMES
    return await _offer_slots(
        state, data, language,
        lead=lead, day=day, target=at or sched.FIRST_START, slots=slots,
    )


async def _schedule(
    state: AgentState, data: dict, schedule: dict, turn: dict, language: str
) -> dict:
    offered = _parse_date(schedule.get("day"))
    requested = _parse_date(turn.get("requested_date"))
    at = _parse_time(turn.get("requested_time"))
    wants_more = turn.get("wants_more_slots") or turn.get("reply_intent") == "DENY"
    try:
        if not (requested or at):
            if offered and wants_more:
                return await _more_slots(state, data, schedule, language)
            brief = DEMO_PICK_SLOT if offered else DEMO_ASK_TIME
            return await _scheduling_reply(state, brief, data, schedule, language)
        # A bare time ("11:30 works") is for the day already on offer.
        return await _find_slot(state, data, requested or offered, at, language)
    except CalComError:
        logger.exception("Cal.com scheduling failed")
        return await _scheduling_reply(state, DEMO_CALENDAR_ERROR, data, schedule, language)


async def _start_demo(state: AgentState) -> dict:
    # Prefill from the chat profile and from what the user already said.
    recent = [m.content for m in state["messages"] if getattr(m, "type", None) == "human"][-10:]
    latest = next((text for text in reversed(recent) if _has_own_words(text)), recent[-1])
    profile = _empty_demo_data({"contact_name": _user_name(state)})
    turn = await _extract_demo_turn(
        f"LATEST MESSAGE\n{latest}\n\nALL RECENT MESSAGES, oldest first\n" + "\n".join(recent),
        profile, "start",
    )
    language = _demo_language(turn["language"], None)
    data, _bad = _merge_demo_slots(profile, turn)  # a bad prefilled email just gets asked for

    name = data["contact_name"]
    opener = DEMO_START.replace("{{NAME}}", f", {name}" if name else "")
    if not _is_demo_complete(data):
        return await _ask_missing(state, data, None, language, opener=opener)
    if turn.get("requested_date") or turn.get("requested_time"):
        return await _schedule(state, data, {}, turn, language)
    return await _scheduling_reply(state, f"{opener} {DEMO_ASK_TIME}", data, {}, language)


async def demo_node(state: AgentState) -> dict:
    if state.get("tool_data") == "START_DEMO":
        return await _start_demo(state)

    # Threads from the old flow may carry extra fields or a "confirming" phase;
    # only the name and email matter, and the phase follows from them.
    data = _empty_demo_data(dict(state.get("demo_data") or {}))
    schedule = dict(state.get("demo_schedule") or {})
    user_msg = state["messages"][-1].content.strip()
    phase = "scheduling" if _is_demo_complete(data) else "collecting"

    turn = await _extract_demo_turn(
        user_msg, {**data, "offered_day": schedule.get("day")}, phase
    )
    detected = turn["language"] if _has_own_words(user_msg) else "unknown"
    language = _demo_language(detected, state.get("demo_language"))
    intent = turn["reply_intent"]
    merged, bad_email = _merge_demo_slots(data, turn)

    # While collecting, a plain "no" to carrying on counts as skipping; while
    # scheduling, "no" is about the times offered (_schedule shows more).
    if intent == "SKIP" or (phase == "collecting" and intent == "DENY" and merged == data):
        return await _demo_skip(state, language)
    # "Is 3pm free?" / "any other times?" are about the booking, not side questions.
    about_times = bool(turn.get("requested_date") or turn.get("requested_time")
                       or turn.get("wants_more_slots"))
    if intent == "QUESTION" and not (phase == "scheduling" and about_times):
        return await _side_answer(state, turn, merged, schedule, language)
    if not _is_demo_complete(merged) or bad_email:
        return await _ask_missing(state, merged, bad_email, language)
    return await _schedule(state, merged, schedule, turn, language)


async def demo_end_node(state: AgentState) -> dict:
    return await _demo_skip(state, state.get("demo_language") or "English")


# ---------------------------------------------------------------------------
# SALES FLOW
# ---------------------------------------------------------------------------

async def sales_node(state: AgentState) -> dict:
    step = state.get("sales_step", 0)
    data = _empty_sales_data(dict(state.get("sales_data") or {}))
    tool_data = state.get("tool_data")
    user_msg = state["messages"][-1].content.strip()
    total = len(SALES_QUESTIONS)

    # 1. INIT
    if tool_data == "START_SALES":
        return {
            "messages": [SystemMessage(content=format_question(SALES_QUESTIONS[0], 1, total))],
            "sales_step": 1,
            "sales_data": _empty_sales_data({}),
            "sales_completed": False,
            "intent": "CONTACT_SALES",
            "tool_data": None,
        }

    # 2. EXTRACT AND MERGE EVERY SALES SLOT PRESENT IN THE LATEST MESSAGE
    if step > 0:
        data = _merge_sales_slots(data, await _extract_sales_slots(user_msg, data))

    # 3. COMPLETE WHEN ALL REQUIRED SALES SLOTS ARE FILLED
    next_index = next((i for i, q in enumerate(SALES_QUESTIONS) if data[q["key"]] is None), None)
    if next_index is None:
        msg = (
            f"Thanks — I've passed this to our sales team:\n\n{_wizard_summary(data)}\n\n"
            "Someone will be in touch shortly. [SALES_REQUEST]"
        )
        return {
            "messages": [SystemMessage(content=msg)],
            "sales_step": 0,
            "sales_data": data,
            "sales_completed": True,
            "intent": "CONTACT_SALES",
            "tool_data": None,
        }

    # 4. ASK THE FIRST REMAINING SALES QUESTION
    q = SALES_QUESTIONS[next_index]
    return {
        "messages": [SystemMessage(content=format_question(q, next_index + 1, total))],
        "sales_step": next_index + 1,
        "sales_data": data,
        "intent": "CONTACT_SALES",
        "tool_data": None,
    }


async def sales_end_node(state: AgentState) -> dict:
    answered = sum(value is not None for value in (state.get("sales_data") or {}).values())
    name = state.get("user_data", {}).get("user_name", "there")
    if answered == 0:
        msg = f"No problem, {name} — cancelled. What else can I help with?"
    else:
        msg = (
            f"Got it, {name} — I've stopped that. You'd answered {answered} "
            "question(s); nothing was saved. Say 'contact sales' to start again."
        )
    return {
        "messages": [SystemMessage(content=msg)],
        "sales_step": 0,
        "sales_data": {},
        "sales_completed": False,
        "intent": "STOP_SALES",
        "tool_data": None,
    }


# ---------------------------------------------------------------------------
# RAG PATH: EXECUTOR + GENERATOR
# ---------------------------------------------------------------------------

gen_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    openai_api_key=settings.OPENAI_API_KEY,
)

_DEFAULT_QUERY = {
    "PRICING": "RelayN pricing and plans",
    "PRODUCT_QA": "RelayN features and capabilities",
}


async def _retrieve(user_data: dict, intent: str | None, query: str | None) -> tuple[str | None, str]:
    """KB search for one query. Returns (bulleted chunks or None, the query used)."""
    if not query:
        query = _DEFAULT_QUERY.get(intent, "RelayN features and capabilities")
    elif intent == "PRICING":
        query = f"pricing plans {query}"

    chunks = await asyncio.to_thread(
        search_knowledge_base, user_data["org_id"], user_data["workflow_id"], query
    )
    return ("\n\n".join(f"- {c}" for c in chunks) if chunks else None), query


async def _kb_answer(
    state: AgentState, tool_data: str | None, extra_instruction: str | None = None
) -> AIMessage:
    """Answer the latest user message, grounded in tool_data, in the RelayN voice."""
    latest_user_query = ""
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human":
            latest_user_query = m.content
            break

    system = render(
        GENERATOR_SYSTEM_PROMPT,
        business_name=RELAYN_BUSINESS["name"],
        persona=RELAYN_BUSINESS["persona"],
        tone=RELAYN_BUSINESS["tone"],
        contact_info=RELAYN_CONTACT,
        tool_data=tool_data or "(nothing retrieved)",
        user_query=latest_user_query,
    )
    history = state["messages"][-20:]
    extra = [SystemMessage(content=extra_instruction)] if extra_instruction else []
    return await gen_llm.ainvoke([SystemMessage(content=system), *history, *extra])


async def executor_node(state: AgentState) -> dict:
    tool_data, query = await _retrieve(
        state.get("user_data", {}), state.get("intent"), state.get("search_query")
    )
    return {"tool_data": tool_data, "search_query": query}


async def generator_node(state: AgentState) -> dict:
    return {"messages": [await _kb_answer(state, state.get("tool_data"))]}


# ---------------------------------------------------------------------------
# HANDOFF
# ---------------------------------------------------------------------------

async def greeting_node(state: AgentState) -> dict:
    return {
        "messages": [AIMessage(content=greeting_reply(state["messages"][-1].content))],
        "intent": "GREETING",
        "tool_data": None,
    }


async def handoff_node(state: AgentState) -> dict:
    return {
        "messages": [SystemMessage(content=HANDOFF_MESSAGE)],
        "intent": "HANDOFF",
        "tool_data": None,
    }
