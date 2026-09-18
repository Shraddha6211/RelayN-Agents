from typing import Literal

from pydantic import BaseModel


class MessageIn(BaseModel):
    body: str
    type: str = "text"


class GenerateReplyRequest(BaseModel):
    org_id: str
    asset_id: str
    asset_type: str
    workflow_id: str
    conversation_id: str
    message: MessageIn
    user_name: str | None = None


# --- relayn_gateway contract -------------------------------------------
# The gateway resolves the workflow, enforces limits and sends the history it
# holds; this service still answers from its own Redis checkpointer, so the
# history is accepted and ignored.


class HistoryEntryIn(BaseModel):
    role: str
    text: str
    at: str | None = None


class WorkflowIn(BaseModel):
    id: str
    type: str
    config: dict = {}


class ReplyV1Request(BaseModel):
    org_id: str
    workflow: WorkflowIn
    asset_id: str
    conversation_id: str
    channel: str
    message: MessageIn
    history: list[HistoryEntryIn] = []


class TextMessageOut(BaseModel):
    type: Literal["text"] = "text"
    text: str


class UsageOut(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int


class ReplyV1Response(BaseModel):
    messages: list[TextMessageOut] = []
    intent: str | None = None
    topic: str | None = None
    # This service's own fields. relayn_gateway forwards anything it does not
    # recognise untouched, so they reach the dashboard without the gateway
    # knowing what a demo or a handoff is.
    capture: "FlowCapture | None" = None
    handoff_requested: bool = False
    # This service does not meter itself yet, so the gateway records no usage
    # for it and its replies do not count against the org's token budget.
    usage: UsageOut | None = None


class FlowCapture(BaseModel):
    type: Literal["demo", "sales"]
    data: dict


class GenerateReplyResponse(BaseModel):
    reply: str | None
    # Collapsed classification (RAG | ORDER | CHAT) for the dashboard to stamp
    # onto its chat_turns row. Lead scoring only ever sees these three values.
    intent: str | None = None
    topic: str | None = None
    # Set only on a demo/sales completion turn; the dashboard writes it into
    # chat_turns.flow_capture.
    capture: FlowCapture | None = None
    handoff_requested: bool = False
