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
