# agent/nodes.py
import asyncio
import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent.prompts import (
    DEMO_QUESTIONS,
    DEMO_EXTRACTION_SYSTEM_PROMPT,
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


class DemoSlotExtraction(BaseModel):
    business_name: Optional[str] = Field(default=None)
    channels: Optional[str] = Field(default=None)
    monthly_volume: Optional[str] = Field(default=None)
    contact_name: Optional[str] = Field(default=None)
    contact_info: Optional[str] = Field(default=None)
    preferred_time: Optional[str] = Field(default=None)


class SalesSlotExtraction(BaseModel):
    need: Optional[str] = Field(default=None)
    company: Optional[str] = Field(default=None)
    contact_info: Optional[str] = Field(default=None)


demo_extractor = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY,
).with_structured_output(DemoSlotExtraction)

sales_extractor = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY,
).with_structured_output(SalesSlotExtraction)

_DEMO_KEYS = tuple(q["key"] for q in DEMO_QUESTIONS)
_SALES_KEYS = tuple(q["key"] for q in SALES_QUESTIONS)


async def _extract_demo_slots(user_msg: str, current_data: dict) -> dict:
    context = json.dumps(
        {key: current_data.get(key) for key in _DEMO_KEYS},
        ensure_ascii=True,
    )
    result = await demo_extractor.ainvoke([
        SystemMessage(content=DEMO_EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(content=f"CURRENT STATE:\n{context}\n\nUSER MESSAGE:\n{user_msg}"),
    ])
    values = result.model_dump() if isinstance(result, DemoSlotExtraction) else dict(result)
    return {key: values.get(key) for key in _DEMO_KEYS}


def _empty_demo_data(data: dict) -> dict:
    return {key: data.get(key) for key in _DEMO_KEYS}


def _canonical_demo_value(key: str, value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    question = next(q for q in DEMO_QUESTIONS if q["key"] == key)
    if question["type"] != "button":
        return text

    probe = text.lower()
    matches = [o["title"] for o in question["options"] if o["title"].lower() in probe]
    if len(matches) == 1:
        return matches[0]
    match = _match_button(text, question["options"])
    return match["title"] if match else None


def _merge_demo_slots(data: dict, extracted: dict) -> dict:
    merged = _empty_demo_data(data)
    for key in _DEMO_KEYS:
        value = _canonical_demo_value(key, extracted.get(key))
        if value is not None:
            merged[key] = value
    return merged


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
# ---------------------------------------------------------------------------

async def demo_node(state: AgentState) -> dict:
    step = state.get("demo_step", 0)
    data = _empty_demo_data(dict(state.get("demo_data") or {}))
    tool_data = state.get("tool_data")
    user_msg = state["messages"][-1].content.strip()
    total = len(DEMO_QUESTIONS)

    # 1. INIT
    if tool_data == "START_DEMO":
        return {
            "messages": [SystemMessage(content=format_question(DEMO_QUESTIONS[0], 1, total))],
            "demo_step": 1,
            "demo_data": _empty_demo_data({}),
            "demo_completed": False,
            "intent": "BOOK_DEMO",
            "tool_data": None,
        }

    # 2. EXTRACT AND MERGE EVERY SLOT PRESENT IN THE LATEST MESSAGE
    if step > 0:
        data = _merge_demo_slots(data, await _extract_demo_slots(user_msg, data))

    # 3. COMPLETE WHEN ALL REQUIRED SLOTS ARE FILLED
    next_index = next((i for i, q in enumerate(DEMO_QUESTIONS) if data[q["key"]] is None), None)
    if next_index is None:
        msg = (
            f"Perfect — here's what I've got:\n\n{_wizard_summary(data)}\n\n"
            "Our team will reach out to lock in a time. Thanks! [DEMO_BOOKED]"
        )
        return {
            "messages": [SystemMessage(content=msg)],
            "demo_step": 0,
            "demo_data": data,
            "demo_completed": True,
            "intent": "BOOK_DEMO",
            "tool_data": None,
        }

    # 4. ASK THE FIRST REMAINING QUESTION
    q = DEMO_QUESTIONS[next_index]
    return {
        "messages": [SystemMessage(content=format_question(q, next_index + 1, total))],
        "demo_step": next_index + 1,
        "demo_data": data,
        "intent": "BOOK_DEMO",
        "tool_data": None,
    }


async def demo_end_node(state: AgentState) -> dict:
    answered = sum(value is not None for value in (state.get("demo_data") or {}).values())
    name = state.get("user_data", {}).get("user_name", "there")
    if answered == 0:
        msg = (
            f"No problem, {name} — I've cancelled the demo booking. "
            "What else can I help with?"
        )
    else:
        msg = (
            f"Got it, {name} — I've stopped the demo booking. You'd answered "
            f"{answered} question(s); nothing was saved. Just say 'book demo' "
            "whenever you want to start again."
        )
    return {
        "messages": [SystemMessage(content=msg)],
        "demo_step": 0,
        "demo_data": {},
        "demo_completed": False,
        "intent": "STOP_DEMO",
        "tool_data": None,
    }


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


async def executor_node(state: AgentState) -> dict:
    intent = state.get("intent")
    query = state.get("search_query")
    if not query:
        query = _DEFAULT_QUERY.get(intent, "RelayN features and capabilities")
    elif intent == "PRICING":
        query = f"pricing plans {query}"

    ud = state.get("user_data", {})
    chunks = await asyncio.to_thread(
        search_knowledge_base, ud["org_id"], ud["workflow_id"], query
    )
    tool_data = "\n\n".join(f"- {c}" for c in chunks) if chunks else None
    return {"tool_data": tool_data, "search_query": query}


async def generator_node(state: AgentState) -> dict:
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
        tool_data=state.get("tool_data") or "(nothing retrieved)",
        user_query=latest_user_query,
    )
    history = state["messages"][-20:]
    response = await gen_llm.ainvoke([SystemMessage(content=system), *history])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# HANDOFF
# ---------------------------------------------------------------------------

async def handoff_node(state: AgentState) -> dict:
    return {
        "messages": [SystemMessage(content=HANDOFF_MESSAGE)],
        "intent": "HANDOFF",
        "tool_data": None,
    }
