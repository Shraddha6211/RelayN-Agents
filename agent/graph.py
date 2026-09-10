# agent/graph.py
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    demo_end_node,
    demo_node,
    executor_node,
    generator_node,
    handoff_node,
    sales_end_node,
    sales_node,
)
from agent.router import router_node
from agent.state import AgentState

_INTENT_TO_NODE = {
    "BOOK_DEMO": "demo",
    "STOP_DEMO": "demo_end",
    "CONTACT_SALES": "sales",
    "STOP_SALES": "sales_end",
    "HANDOFF": "handoff",
    "PRODUCT_QA": "executor",
    "PRICING": "executor",
}


def route_decision(state: AgentState) -> str:
    return _INTENT_TO_NODE.get(state.get("intent"), "generator")


def build_workflow() -> StateGraph:
    """Uncompiled graph. main.py compiles it with the Redis checkpointer."""
    wf = StateGraph(AgentState)

    wf.add_node("router", router_node)
    wf.add_node("executor", executor_node)
    wf.add_node("generator", generator_node)
    wf.add_node("demo", demo_node)
    wf.add_node("demo_end", demo_end_node)
    wf.add_node("sales", sales_node)
    wf.add_node("sales_end", sales_end_node)
    wf.add_node("handoff", handoff_node)

    wf.add_edge(START, "router")
    wf.add_conditional_edges(
        "router",
        route_decision,
        {
            "demo": "demo",
            "demo_end": "demo_end",
            "sales": "sales",
            "sales_end": "sales_end",
            "handoff": "handoff",
            "executor": "executor",
            "generator": "generator",
        },
    )

    wf.add_edge("executor", "generator")
    wf.add_edge("generator", END)
    wf.add_edge("demo", END)
    wf.add_edge("demo_end", END)
    wf.add_edge("sales", END)
    wf.add_edge("sales_end", END)
    wf.add_edge("handoff", END)

    return wf
