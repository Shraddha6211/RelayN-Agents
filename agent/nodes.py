# agent/nodes.py
from langchain_core.messages import SystemMessage

from agent.prompts import DEMO_QUESTIONS, format_question
from agent.state import AgentState


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
