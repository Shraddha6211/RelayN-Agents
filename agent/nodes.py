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
    DEMO_BOOKED,
    DEMO_BOOKING_ERROR,
    DEMO_COMPOSER_SYSTEM_PROMPT,
    DEMO_CONFIRM_ASK,
    DEMO_CONFIRM_LEAD,
    DEMO_CONTINUE_COLLECTING,
    DEMO_CONTINUE_CONFIRMING,
    DEMO_FIX_LEAD,
    DEMO_INTRO,
    DEMO_MISSING,
    DEMO_PARTIAL_QUESTIONS,
    DEMO_PREFILLED_NOTE,
    DEMO_QUESTIONS,
    DEMO_REASK,
    DEMO_SIDE_ANSWER_INSTRUCTION,
    DEMO_SKIP,
    DEMO_SLOT_LABELS,
    DEMO_TURN_SYSTEM_PROMPT,
    DEMO_WHICH_FIX,
    GENERATOR_SYSTEM_PROMPT,
    HANDOFF_MESSAGE,
    RELAYN_BUSINESS,
    RELAYN_CONTACT,
    SALES_EXTRACTION_SYSTEM_PROMPT,
    SALES_QUESTIONS,
    demo_question_text,
    format_question,
    numbered,
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

def _match_button(user_msg: str, options: list[dict]) -> dict | None:
    probe = user_msg.strip().lower()
    for o in options:
        if probe == o["id"].lower() or probe == o["title"].lower():
            return o
    return None


def _wizard_summary(data: dict) -> str:
    return "\n".join(f"• {k.replace('_', ' ').title()}: {v}" for k, v in data.items())


class DemoTurn(BaseModel):
    business_name: Optional[str] = Field(default=None)
    channels: Optional[str] = Field(default=None)
    monthly_volume: Optional[str] = Field(default=None)
    contact_name: Optional[str] = Field(default=None)
    contact_info: Optional[str] = Field(default=None)
    contact_phone: Optional[str] = Field(default=None)
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

_DEMO_KEYS = tuple(key for q in DEMO_QUESTIONS for key in q["keys"])
_SALES_KEYS = tuple(q["key"] for q in SALES_QUESTIONS)

_DEMO_PHASE_CONTEXT = {
    "start": "the user just asked for a demo; USER MESSAGE holds their recent chat messages, "
             "one per line — extract any demo details they already gave",
    "collecting": "waiting for the user's answers to the demo questions",
    "confirming": "showed the user a summary of their details and asked them to confirm it",
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
    out["language"] = values.get("language") or "English"
    return out


def _empty_demo_data(data: dict) -> dict:
    return {key: data.get(key) for key in _DEMO_KEYS}


def _canonical_demo_value(key: str, value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    question = next(q for q in DEMO_QUESTIONS if key in q["keys"])
    if question["type"] != "button":
        return text

    probe = text.lower()
    matches = [o["title"] for o in question["options"] if o["title"].lower() in probe]
    if len(matches) == 1:
        return matches[0]
    match = _match_button(text, question["options"])
    return match["title"] if match else None


def _is_email(value: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value))


def _is_valid_mobile(value: str) -> bool:
    digits = re.sub(r"[\s-]", "", value)
    return bool(re.fullmatch(r"9\d{9}", digits))


# Landline/telephone formats vary too much to validate reliably, so we only
# accept and check mobile numbers — the error message steers the user there.
_VALIDATORS: dict[str, tuple] = {
    "contact_info": (
        _is_email,
        '"{value}" doesn\'t look like a valid email address — could you double-check it?',
    ),
    "contact_phone": (
        _is_valid_mobile,
        '"{value}" doesn\'t look like a mobile number — mobile numbers need to start '
        "with 9 and have exactly 10 digits.",
    ),
}


def _merge_demo_slots(data: dict, extracted: dict) -> tuple[dict, list[str]]:
    """Merge every valid extracted value; invalid ones are dropped and reported."""
    merged = _empty_demo_data(data)
    errors = []
    for key in _DEMO_KEYS:
        value = _canonical_demo_value(key, extracted.get(key))
        if value is None:
            continue
        validator = _VALIDATORS.get(key)
        if validator and not validator[0](value):
            errors.append(validator[1].format(value=value))
            continue
        merged[key] = value
    return merged, errors


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
# One message lists every question; each reply is classified (answer, confirm,
# deny, side question, skip) and the flow moves collecting -> confirming -> booked.
# ---------------------------------------------------------------------------

async def _compose(text: str, name: str | None, language: str, keep: Iterable[str] = ()) -> str:
    """Rewrite an English template politely, in the user's language.

    Falls back to the template itself if the LLM fails or drops any value in
    `keep` (option labels, the user's data, a booking link) — a less polished
    message beats a wrong one.
    """
    try:
        response = await composer_llm.ainvoke([
            SystemMessage(content=DEMO_COMPOSER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"LANGUAGE: {language}\nCUSTOMER NAME: {name or 'unknown'}\n\nDRAFT:\n{text}"
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


def _pending_questions(data: dict) -> list[dict]:
    """The questions (or halves of the combined contact question) still unanswered."""
    pending = []
    for q in DEMO_QUESTIONS:
        missing = [key for key in q["keys"] if data.get(key) is None]
        if len(missing) == len(q["keys"]):
            pending.append(q)
        else:
            pending.extend({"type": "open", "question": DEMO_PARTIAL_QUESTIONS[key]}
                           for key in missing)
    return pending


def _option_titles(questions: list[dict]) -> list[str]:
    return [o["title"] for q in questions for o in q.get("options", [])]


def _errors_block(errors: list[str]) -> str:
    return "\n".join(f"⚠️ {e}" for e in errors)


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
        "messages": [SystemMessage(content=text)],
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


async def _demo_skip(state: AgentState, language: str) -> dict:
    name = _user_name(state)
    text = DEMO_SKIP.replace("{{NAME}}", f" {name}" if name else "")
    return _demo_reply(
        await _compose(text, name, language),
        phase=None, data={}, language=language, intent="STOP_DEMO",
    )


async def _ask_missing(
    state: AgentState, data: dict, errors: list[str], got_new: bool, language: str
) -> dict:
    pending = _pending_questions(data)
    lead = DEMO_MISSING if got_new else (DEMO_FIX_LEAD if errors else DEMO_REASK)
    text = _join(lead, _errors_block(errors), numbered([demo_question_text(q) for q in pending]))
    text = await _compose(text, _user_name(state), language, keep=_option_titles(pending))
    return _demo_reply(text, phase="collecting", data=data, language=language)


async def _ask_confirm(state: AgentState, data: dict, errors: list[str], language: str) -> dict:
    summary = numbered([
        f"{q['question']}\n   ➜ {', '.join(data[key] for key in q['keys'])}"
        for q in DEMO_QUESTIONS
    ])
    text = _join(_errors_block(errors), DEMO_CONFIRM_LEAD, summary, DEMO_CONFIRM_ASK)
    text = await _compose(text, _user_name(state), language, keep=[data[k] for k in _DEMO_KEYS])
    return _demo_reply(text, phase="confirming", data=data, language=language)


async def _side_answer(
    state: AgentState, turn: dict, data: dict, phase: str, follow_up: str, language: str
) -> dict:
    tool_data, query = await _retrieve(
        state.get("user_data") or {}, None, turn.get("side_question_query")
    )
    answer = await _kb_answer(state, tool_data, extra_instruction=DEMO_SIDE_ANSWER_INSTRUCTION)
    follow = await _compose(follow_up, _user_name(state), language)
    return _demo_reply(
        _join(str(answer.content).strip(), follow),
        phase=phase, data=data, language=language, search_query=query,
    )


async def _book_demo(state: AgentState, data: dict, language: str) -> dict:
    notes = (
        f"Business: {data['business_name']} | Channels: {data['channels']} | "
        f"Size: {data['monthly_volume']} | Phone: {data['contact_phone']}"
    )
    name = _user_name(state)
    try:
        link = build_booking_link(name=data["contact_name"], email=data["contact_info"], notes=notes)
    except CalComError:
        logger.exception("could not build the Cal.com booking link")
        # Stay in confirming with the answers kept, so a later "yes" retries.
        return _demo_reply(
            await _compose(DEMO_BOOKING_ERROR, name, language),
            phase="confirming", data=data, language=language,
        )
    text = await _compose(DEMO_BOOKED.replace("{{LINK}}", link), name, language, keep=[link])
    return _demo_reply(
        f"{text} [DEMO_BOOKED]",
        phase=None, data={**data, "cal_booking_link": link},
        language=language, completed=True,
    )


async def _start_demo(state: AgentState) -> dict:
    # Prefill from what the user already said in this conversation.
    recent = [m.content for m in state["messages"] if getattr(m, "type", None) == "human"][-10:]
    turn = await _extract_demo_turn("\n".join(recent), _empty_demo_data({}), "start")
    language = turn["language"]
    data, _invalid = _merge_demo_slots(_empty_demo_data({}), turn)  # bad prefills just get asked

    pending = _pending_questions(data)
    if not pending:
        return await _ask_confirm(state, data, [], language)

    name = _user_name(state)
    prefilled = [key for key in _DEMO_KEYS if data[key] is not None]
    noted = (
        DEMO_PREFILLED_NOTE + "\n"
        + "\n".join(f"• {DEMO_SLOT_LABELS[key]}: {data[key]}" for key in prefilled)
        if prefilled else ""
    )
    text = _join(
        DEMO_INTRO.replace("{{NAME}}", f", {name}" if name else ""),
        noted,
        numbered([demo_question_text(q) for q in pending]),
    )
    text = await _compose(
        text, name, language,
        keep=_option_titles(pending) + [data[key] for key in prefilled],
    )
    return _demo_reply(text, phase="collecting", data=data, language=language)


async def demo_node(state: AgentState) -> dict:
    if state.get("tool_data") == "START_DEMO":
        return await _start_demo(state)

    phase = state.get("demo_phase") or "collecting"
    data = _empty_demo_data(dict(state.get("demo_data") or {}))
    user_msg = state["messages"][-1].content.strip()

    turn = await _extract_demo_turn(user_msg, data, phase)
    language = turn["language"]
    intent = turn["reply_intent"]
    merged, errors = _merge_demo_slots(data, turn)
    changed = merged != data

    # A plain "no" while collecting answers "should we continue?" -> skip too.
    if intent == "SKIP" or (phase == "collecting" and intent == "DENY" and not changed):
        return await _demo_skip(state, language)

    if phase == "confirming" and not _pending_questions(merged):
        if changed or errors:
            return await _ask_confirm(state, merged, errors, language)
        if intent == "CONFIRM":
            return await _book_demo(state, merged, language)
        if intent == "QUESTION":
            return await _side_answer(
                state, turn, merged, "confirming", DEMO_CONTINUE_CONFIRMING, language
            )
        if intent == "DENY":
            return _demo_reply(
                await _compose(DEMO_WHICH_FIX, _user_name(state), language),
                phase="confirming", data=merged, language=language,
            )
        return await _ask_confirm(state, merged, [], language)

    # COLLECTING
    if intent == "QUESTION":
        return await _side_answer(
            state, turn, merged, "collecting", DEMO_CONTINUE_COLLECTING, language
        )
    if _pending_questions(merged):
        return await _ask_missing(state, merged, errors, changed, language)
    return await _ask_confirm(state, merged, errors, language)


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
