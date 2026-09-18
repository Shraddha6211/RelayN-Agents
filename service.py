import logging

from langchain_core.messages import HumanMessage

from clients import supabase_client
from config import settings
from schemas import (
    FlowCapture,
    GenerateReplyRequest,
    GenerateReplyResponse,
    ReplyV1Request,
    ReplyV1Response,
    TextMessageOut,
)

logger = logging.getLogger("relayn_agents")

_EMPTY = GenerateReplyResponse(reply=None, intent=None, topic=None)

# Internal fine intent -> the value logged on chat_turns.intent. These three are
# the only ones the shared lead scorer recognises.
COLLAPSE = {
    "BOOK_DEMO": "ORDER",
    "CONTACT_SALES": "ORDER",
    "PRODUCT_QA": "RAG",
    "PRICING": "RAG",
    "GENERAL_CHAT": "CHAT",
    "HANDOFF": "CHAT",
    "STOP_DEMO": "CHAT",
    "STOP_SALES": "CHAT",
}


def _fetch_workflow(workflow_id: str) -> dict | None:
    res = (
        supabase_client.table("workflows")
        .select("id, organization_id, workflow_type, is_active")
        .eq("id", workflow_id)
        .maybe_single()
        .execute()
    )
    return res.data


async def generate_reply(payload: GenerateReplyRequest, agent_app) -> GenerateReplyResponse:
    # 1. PINNED-IDENTITY GUARD — this deployment serves exactly one workflow.
    if payload.org_id != settings.RELAYN_ORG_ID or payload.workflow_id != settings.RELAYN_WORKFLOW_ID:
        logger.warning(
            "rejected request for org=%s workflow=%s (pinned to %s / %s)",
            payload.org_id, payload.workflow_id,
            settings.RELAYN_ORG_ID, settings.RELAYN_WORKFLOW_ID,
        )
        return _EMPTY

    # 2. WORKFLOW ROW GUARD — active, right type, and the row's org matches.
    workflow = _fetch_workflow(payload.workflow_id)
    if (
        not workflow
        or not workflow.get("is_active")
        or workflow.get("workflow_type") != "ai_chatbot"
        or workflow.get("organization_id") != payload.org_id
    ):
        return _EMPTY

    # 3. RUN THE GRAPH. History + wizard state come from the checkpointer.
    state_in = {
        "messages": [HumanMessage(content=payload.message.body)],
        "user_data": {
            "user_name": payload.user_name or "there",
            "conversation_id": payload.conversation_id,
            "workflow_id": payload.workflow_id,
            "org_id": payload.org_id,
        },
    }
    config = {"configurable": {"thread_id": f"{payload.workflow_id}:{payload.conversation_id}"}}
    final = await agent_app.ainvoke(state_in, config)

    reply = final["messages"][-1].content if final.get("messages") else None
    internal_intent = final.get("intent") or "GENERAL_CHAT"

    # 4. EXTRACT A COMPLETION CAPTURE, STRIP THE SENTINEL.
    capture = None
    if reply and "[DEMO_BOOKED]" in reply:
        capture = FlowCapture(type="demo", data=final.get("demo_data", {}))
        reply = reply.replace("[DEMO_BOOKED]", "").strip()
    elif reply and "[SALES_REQUEST]" in reply:
        capture = FlowCapture(type="sales", data=final.get("sales_data", {}))
        reply = reply.replace("[SALES_REQUEST]", "").strip()

    return GenerateReplyResponse(
        reply=reply or None,
        intent=COLLAPSE.get(internal_intent, "CHAT"),
        topic=final.get("search_query"),
        capture=capture,
        handoff_requested=(internal_intent == "HANDOFF"),
    )


# The channel the gateway names -> the asset_type the request model carries.
# Nothing in the graph reads it; it exists so one request model serves both
# callers.
_ASSET_TYPE_BY_CHANNEL = {
    "whatsapp": "phone_number",
    "instagram": "instagram_account",
    "messenger": "facebook_page",
    "facebook": "facebook_page",
}


async def reply_v1(payload: ReplyV1Request, agent_app) -> ReplyV1Response:
    """The relayn_gateway contract for one inbound message.

    The gateway has already verified the caller and the org and resolved the
    workflow. Its history is accepted and ignored: this service replays the
    conversation from its own Redis checkpointer, which also holds the demo
    and sales wizard state that a history list cannot carry.

    capture is this service's own wizard state: the gateway forwards whatever
    it is given without reading it, and nothing downstream stores it, so it is
    logged here and kept in Redis with the rest of the flow. handoff_requested
    travels the same way -- see the note where it is returned.
    """
    result = await generate_reply(
        GenerateReplyRequest(
            org_id=payload.org_id,
            asset_id=payload.asset_id,
            asset_type=_ASSET_TYPE_BY_CHANNEL.get(payload.channel, payload.channel),
            workflow_id=payload.workflow.id,
            conversation_id=payload.conversation_id,
            message=payload.message,
        ),
        agent_app,
    )

    if result.capture or result.handoff_requested:
        logger.info(
            "conversation %s: capture=%s handoff=%s (not carried by the gateway contract)",
            payload.conversation_id,
            result.capture.type if result.capture else None,
            result.handoff_requested,
        )

    # Returned as sent. The gateway passes fields it does not know straight
    # through to the dashboard, so nothing here needs a gateway change; what
    # the dashboard does with them is a separate decision.
    return ReplyV1Response(
        messages=[TextMessageOut(text=result.reply)] if result.reply else [],
        intent=result.intent,
        topic=result.topic,
        usage=None,
        capture=result.capture,
        handoff_requested=result.handoff_requested,
    )
