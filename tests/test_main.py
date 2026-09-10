from fastapi.testclient import TestClient


def _client(monkeypatch):
    import main

    async def fake_generate_reply(payload, agent_app):
        from schemas import GenerateReplyResponse
        return GenerateReplyResponse(reply="pong", intent="CHAT", topic=None)

    monkeypatch.setattr(main, "generate_reply", fake_generate_reply)
    # Skip the real lifespan (no Redis in tests): mark ready, stub the app.
    main.app.state.agent_app = object()
    main.app.state.is_ready = True
    return TestClient(main.app)


def test_health_ok(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "healthy")


def test_generate_reply_routes_to_service(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/generate-reply", json={
        "org_id": "o", "asset_id": "a", "asset_type": "whatsapp",
        "workflow_id": "w", "conversation_id": "c",
        "message": {"body": "ping"},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "pong"
    assert body["intent"] == "CHAT"
    assert body["handoff_requested"] is False
