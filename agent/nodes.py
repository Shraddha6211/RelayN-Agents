# agent/nodes.py
import asyncio

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts import (
    DEMO_QUESTIONS,
    GENERATOR_SYSTEM_PROMPT,
    HANDOFF_MESSAGE,
    RELAYN_BUSINESS,
    RELAYN_CONTACT,
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


# ---------------------------------------------------------------------------
# DEMO FLOW
# ---------------------------------------------------------------------------

async def demo_node(state: AgentState) -> dict:
    step = state.get("demo_step", 0)
    data = dict(state.get("demo_data") or {})
    tool_data = state.get("tool_data")
    user_msg = state["messages"][-1].content.strip()
    total = len(DEMO_QUESTIONS)

    # 1. INIT
    if tool_data == "START_DEMO":
        return {
            "messages": [SystemMessage(content=format_question(DEMO_QUESTIONS[0], 1, total))],
            "demo_step": 1,
            "demo_data": {},
            "demo_completed": False,
            "intent": "BOOK_DEMO",
            "tool_data": None,
        }

    # 2. VALIDATE + STORE PREVIOUS ANSWER
    if step > 0:
        prev = DEMO_QUESTIONS[step - 1]
        if prev["type"] == "button":
            match = _match_button(user_msg, prev["options"])
            if match is None:
                nudge = "Please tap one of the options below:\n" + format_question(
                    prev, step, total
                )
                return {
                    "messages": [SystemMessage(content=nudge)],
                    "demo_step": step,
                    "demo_data": data,
                    "demo_completed": False,
                    "intent": "BOOK_DEMO",
                    "tool_data": None,
                }
            data[prev["key"]] = match["title"]
        else:
            data[prev["key"]] = user_msg

    # 3. COMPLETE
    if step >= total:
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

    # 4. ASK NEXT
    q = DEMO_QUESTIONS[step]
    return {
        "messages": [SystemMessage(content=format_question(q, step + 1, total))],
        "demo_step": step + 1,
        "demo_data": data,
        "intent": "BOOK_DEMO",
        "tool_data": None,
    }


async def demo_end_node(state: AgentState) -> dict:
    answered = len(state.get("demo_data") or {})
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
    data = dict(state.get("sales_data") or {})
    tool_data = state.get("tool_data")
    user_msg = state["messages"][-1].content.strip()
    total = len(SALES_QUESTIONS)

    # 1. INIT
    if tool_data == "START_SALES":
        return {
            "messages": [SystemMessage(content=format_question(SALES_QUESTIONS[0], 1, total))],
            "sales_step": 1,
            "sales_data": {},
            "sales_completed": False,
            "intent": "CONTACT_SALES",
            "tool_data": None,
        }

    # 2. STORE PREVIOUS ANSWER (all sales questions are open)
    if step > 0:
        data[SALES_QUESTIONS[step - 1]["key"]] = user_msg

    # 3. COMPLETE
    if step >= total:
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

    # 4. ASK NEXT
    q = SALES_QUESTIONS[step]
    return {
        "messages": [SystemMessage(content=format_question(q, step + 1, total))],
        "sales_step": step + 1,
        "sales_data": data,
        "intent": "CONTACT_SALES",
        "tool_data": None,
    }


async def sales_end_node(state: AgentState) -> dict:
    answered = len(state.get("sales_data") or {})
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
