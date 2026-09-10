from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_data: dict          # {user_name, conversation_id, workflow_id, org_id}
    intent: str              # internal fine intent set by the router / sub-flows
    search_query: str | None
    tool_data: Any           # retrieved-context string, or a sub-flow marker

    # --- DEMO FLOW ---
    demo_step: int           # 0 = inactive; 1..len(DEMO_QUESTIONS) = awaiting that answer
    demo_data: dict
    demo_completed: bool

    # --- SALES FLOW ---
    sales_step: int
    sales_data: dict
    sales_completed: bool
