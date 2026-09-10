def test_settings_expose_pinned_ids_and_redis_url():
    from config import settings

    assert settings.RELAYN_ORG_ID == "org-relayn-test"
    assert settings.RELAYN_WORKFLOW_ID == "wf-relayn-test"
    assert settings.REDIS_URL == "redis://localhost:6379/0"


def test_redis_url_includes_password_when_set(monkeypatch):
    monkeypatch.setenv("REDIS_PASSWORD", "hunter2")
    from config import Settings

    s = Settings()
    assert s.REDIS_URL == "redis://:hunter2@localhost:6379/0"
