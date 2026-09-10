import types


class _FakeRpc:
    def __init__(self, store):
        self.store = store

    def execute(self):
        return types.SimpleNamespace(data=self.store["data"])


class _FakeSupabase:
    def __init__(self, data):
        self.data = data
        self.last_call = None

    def rpc(self, name, params):
        self.last_call = (name, params)
        return _FakeRpc({"data": self.data})


def test_search_calls_the_workflow_rpc_with_both_scope_keys(monkeypatch):
    import agent.tools as tools

    fake = _FakeSupabase(data=[{"content": "RelayN supports WhatsApp"}, {"content": "and Instagram"}])
    monkeypatch.setattr(tools, "supabase_client", fake)
    monkeypatch.setattr(tools, "embeddings",
                        types.SimpleNamespace(embed_query=lambda q: [0.1, 0.2, 0.3]))

    out = tools.search_knowledge_base("org-1", "wf-1", "channels", k=3)

    name, params = fake.last_call
    assert name == "match_workflow_kb_chunks"
    assert params["p_organization_id"] == "org-1"
    assert params["p_workflow_id"] == "wf-1"
    assert params["query_embedding"] == [0.1, 0.2, 0.3]
    assert params["match_count"] == 3
    assert out == ["RelayN supports WhatsApp", "and Instagram"]


def test_search_returns_empty_list_when_rpc_has_no_data(monkeypatch):
    import agent.tools as tools

    fake = _FakeSupabase(data=None)
    monkeypatch.setattr(tools, "supabase_client", fake)
    monkeypatch.setattr(tools, "embeddings",
                        types.SimpleNamespace(embed_query=lambda q: [0.0]))

    assert tools.search_knowledge_base("o", "w", "x") == []
