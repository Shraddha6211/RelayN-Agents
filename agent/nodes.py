# agent/nodes.py
import asyncio
import json
import logging
import re
from typing import Iterable, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent.prompts import (
    DEMO_ASK_EMAIL,
    DEMO_ASK_NAME,
    DEMO_ASK_NAME_EMAIL,
    DEMO_BAD_EMAIL,
    DEMO_BOOKED,
    DEMO_BOOKING_ERROR,
    DEMO_COMPOSER_SYSTEM_PROMPT,
    DEMO_NUDGE,
    DEMO_SIDE_ANSWER_INSTRUCTION,
    DEMO_SKIP,
    DEMO_START,
    DEMO_TURN_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    HANDOFF_MESSAGE,
    RELAYN_BUSINESS,
    RELAYN_CONTACT,
    SALES_EXTRACTION_SYSTEM_PROMPT,
    SALES_QUESTIONS,
    format_question,
    render,
)
from agent.state import AgentState
from agent.tools import search_knowledge_base
from config import settings
from calcom_client import CalComError, build_booking_link

logger = logging.getLogger("relayn_agents.nodes")


# ---------------------------------------------------------------------------
# SHARED
# ---------------------------------------------------------------------------

def _wizard_summary(data: dict) -> str:
    return "\n".join(f"• {k.replace('_', ' ').title()}: {v}" for k, v in data.items())


class DemoTurn(BaseModel):
    contact_name: Optional[str] = Field(default=None)
    contact_info: Optional[str] = Field(default=None)
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
}


async def _extract_demo_turn(user_msg: str, current_data: dict, phase: str) -> dict:
    context = json.dumps(
        {key: current_data.get(key) for key in _DEMO_KEYS},
        ensure_ascii=True,
    )
    result = await demo_extractor.ainvoke([
        SystemMessage(content=DEMO_TURN_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"BOT IS CURRENTLY: {_DEMO_PHASE_CONTEXT[phase]}\n\n"
            f"CURRENT STATE:\n{context}\n\nUSER MESSAGE:\n{user_msg}"
        )),
    ])
    values = result.model_dump() if isinstance(result, DemoTurn) else dict(result)
    out = {key: values.get(key) for key in _DEMO_KEYS}
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
# would. Code decides what the next message must do; the reply writer says it
# naturally with the recent chat in view. Once both are valid, the Cal.com
# link goes out — its page shows them prefilled, so there is no confirm step.
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


async def _compose(
    text: str,
    name: str | None,
    language: str,
    keep: Iterable[str] = (),
    history: Iterable = (),
) -> str:
    """Write the next demo message from a brief (an English template), in the
    user's language, as a natural reply to the recent chat in `history`.

    Falls back to the template itself if the LLM fails or drops any value in
    `keep` (e.g. the booking link) — a less polished message beats a wrong one.
    """
    try:
        response = await composer_llm.ainvoke([
            SystemMessage(content=DEMO_COMPOSER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"CUSTOMER NAME: {name or 'unknown'}\n\n"
                f"RECENT CHAT:\n{_chat_transcript(history) or '(none)'}\n\nBRIEF:\n{text}\n\n"
                f"WRITE IN: {language}"
            )),
        ])
    except Exception:
        logger.exception("demo composer failed; sending the template")
        return text
    out = (getattr(response, "content", "") or "").strip()
    dropped = [k for k in keep if k and k not in out]
    if not out or dropped:
        logger.warning("demo composer dropped %s; sending the template", dropped or "everything")
        return text
    return out


def _user_name(state: AgentState) -> str | None:
    name = (state.get("user_data") or {}).get("user_name")
    return name if name and name != "there" else None


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def _is_demo_complete(data: dict) -> bool:
    return all(data.get(key) for key in _DEMO_KEYS)


def _missing_label(data: dict) -> str:
    """What is still needed, for DEMO_NUDGE; a quick yes when nothing is."""
    return {
        (True, True): "your name and email",
        (True, False): "your name",
        (False, True): "your email",
        (False, False): "a quick *yes*",
    }[(data["contact_name"] is None, data["contact_info"] is None)]


def _demo_reply(
    text: str,
    *,
    phase: str | None,
    data: dict,
    language: str,
    intent: str = "BOOK_DEMO",
    completed: bool = False,
    search_query: str | None = None,
) -> dict:
    out = {
        "messages": [AIMessage(content=text)],
        "demo_phase": phase,
        "demo_data": data,
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


async def _side_answer(state: AgentState, turn: dict, data: dict, language: str) -> dict:
    tool_data, query = await _retrieve(
        state.get("user_data") or {}, None, turn.get("side_question_query")
    )
    answer = await _kb_answer(state, tool_data, extra_instruction=DEMO_SIDE_ANSWER_INSTRUCTION)
    answer_text = str(answer.content).strip()
    nudge = DEMO_NUDGE.replace("{{MISSING}}", _missing_label(data))
    follow = await _say(state, nudge, language, name=data["contact_name"], after=answer_text)
    return _demo_reply(
        _join(answer_text, follow),
        phase="collecting", data=data, language=language, search_query=query,
    )


async def _book_demo(state: AgentState, data: dict, language: str) -> dict:
    name = data["contact_name"]
    try:
        link = build_booking_link(
            name=name, email=data["contact_info"], notes="Booked from the RelayN chat"
        )
    except CalComError:
        logger.exception("could not build the Cal.com booking link")
        # Stay in the flow with the details kept, so a later "yes" retries.
        return _demo_reply(
            await _say(state, DEMO_BOOKING_ERROR, language, name=name),
            phase="collecting", data=data, language=language,
        )
    text = await _say(state, DEMO_BOOKED.replace("{{LINK}}", link), language, name=name, keep=[link])
    return _demo_reply(
        f"{text} [DEMO_BOOKED]",
        phase=None, data={**data, "cal_booking_link": link},
        language=language, completed=True,
    )


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

    if _is_demo_complete(data):
        return await _book_demo(state, data, language)
    name = data["contact_name"]
    opener = DEMO_START.replace("{{NAME}}", f", {name}" if name else "")
    return await _ask_missing(state, data, None, language, opener=opener)


async def demo_node(state: AgentState) -> dict:
    if state.get("tool_data") == "START_DEMO":
        return await _start_demo(state)

    # Threads from the old flow may carry extra fields or a "confirming" phase;
    # only the name and email matter now, and every active phase is handled alike.
    data = _empty_demo_data(dict(state.get("demo_data") or {}))
    user_msg = state["messages"][-1].content.strip()

    turn = await _extract_demo_turn(user_msg, data, "collecting")
    detected = turn["language"] if _has_own_words(user_msg) else "unknown"
    language = _demo_language(detected, state.get("demo_language"))
    intent = turn["reply_intent"]
    merged, bad_email = _merge_demo_slots(data, turn)

    # A plain "no" to carrying on with the booking counts as skipping it.
    if intent == "SKIP" or (intent == "DENY" and merged == data):
        return await _demo_skip(state, language)
    if intent == "QUESTION":
        return await _side_answer(state, turn, merged, language)
    if _is_demo_complete(merged) and not bad_email:
        return await _book_demo(state, merged, language)
    return await _ask_missing(state, merged, bad_email, language)


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

async def handoff_node(state: AgentState) -> dict:
    return {
        "messages": [SystemMessage(content=HANDOFF_MESSAGE)],
        "intent": "HANDOFF",
        "tool_data": None,
    }
