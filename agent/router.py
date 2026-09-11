# agent/router.py
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent.prompts import ROUTER_SYSTEM_PROMPT
from agent.state import AgentState
from config import settings

STOP_WORDS = {"stop", "quit", "exit", "cancel", "back", "menu"}

_DEMO_TRIGGERS = {"book demo", "book a demo", "schedule demo", "schedule a demo",
                  "get a demo", "demo", "see a demo"}
_SALES_TRIGGERS = {"talk to sales", "contact sales", "speak to sales", "sales team"}
_HANDOFF_TRIGGERS = {"human", "agent", "real person", "talk to a human",
                     "talk to a person", "support", "help me"}


class RouteDecision(BaseModel):
    intent: Literal[
        "GENERAL_CHAT", "PRODUCT_QA", "PRICING",
        "BOOK_DEMO", "CONTACT_SALES", "HANDOFF",
    ]
    search_query: Optional[str] = Field(default=None)


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY,
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

structured_router = _prompt | llm.with_structured_output(RouteDecision)


def _norm(text: str) -> str:
    return text.strip().lower()


def _has_stop_word(text: str) -> bool:
    return bool(STOP_WORDS.intersection(_norm(text).split()))


async def router_node(state: AgentState) -> dict:
    messages = state["messages"]
    raw = messages[-1].content
    msg = _norm(raw)

    demo_step = state.get("demo_step", 0)
    demo_completed = state.get("demo_completed", False)
    sales_step = state.get("sales_step", 0)
    sales_completed = state.get("sales_completed", False)

    # 1. ACTIVE DEMO LOCK
    if demo_step > 0 and not demo_completed:
        return {"intent": "STOP_DEMO" if _has_stop_word(raw) else "BOOK_DEMO"}

    # 2. ACTIVE SALES LOCK
    if sales_step > 0 and not sales_completed:
        return {"intent": "STOP_SALES" if _has_stop_word(raw) else "CONTACT_SALES"}

    # 3. KEYWORD FAST PATHS
    if msg in _DEMO_TRIGGERS:
        return {"intent": "BOOK_DEMO", "tool_data": "START_DEMO"}
    if msg in _SALES_TRIGGERS:
        return {"intent": "CONTACT_SALES", "tool_data": "START_SALES"}
    if msg in _HANDOFF_TRIGGERS:
        return {"intent": "HANDOFF"}

    # 4. LLM FALLBACK
    result = await structured_router.ainvoke(
        {"history": messages[-6:-1], "input": raw}
    )
    out: dict = {"intent": result.intent, "search_query": result.search_query, "tool_data": None}
    if result.intent == "BOOK_DEMO":
        out["tool_data"] = "START_DEMO"
    elif result.intent == "CONTACT_SALES":
        out["tool_data"] = "START_SALES"
    return out
