import pytest
from pydantic import ValidationError


def test_request_parses_nested_message():
    from schemas import GenerateReplyRequest

    req = GenerateReplyRequest(
        org_id="o", asset_id="a", asset_type="whatsapp",
        workflow_id="w", conversation_id="c",
        message={"body": "hi"},
    )
    assert req.message.body == "hi"
    assert req.message.type == "text"
    assert req.user_name is None


def test_response_defaults_are_safe():
    from schemas import GenerateReplyResponse

    resp = GenerateReplyResponse(reply=None)
    assert resp.intent is None
    assert resp.capture is None
    assert resp.handoff_requested is False


def test_flow_capture_rejects_unknown_type():
    from schemas import FlowCapture

    FlowCapture(type="demo", data={})
    with pytest.raises(ValidationError):
        FlowCapture(type="order", data={})
