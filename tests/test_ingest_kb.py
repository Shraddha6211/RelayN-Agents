import types

import pytest


def test_chunk_text_splits_and_trims():
    from scripts.ingest_kb import chunk_text

    text = "para one.\n\n" + ("word " * 300)
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert all(c == c.strip() and c for c in chunks)


def test_extract_text_strips_chrome_and_scripts():
    from scripts.ingest_kb import extract_text

    html = """
    <html><head><style>.x{color:red}</style></head>
    <body>
      <nav>menu home pricing</nav>
      <main><h1>Unified inbox</h1><p>WhatsApp, Instagram and Facebook in one place.</p></main>
      <script>tracking()</script>
      <footer>© RelayN</footer>
    </body></html>
    """
    text = extract_text(html)
    assert "Unified inbox" in text
    assert "WhatsApp, Instagram and Facebook in one place." in text
    assert "tracking()" not in text
    assert "menu home pricing" not in text
    assert "© RelayN" not in text


def test_load_sources_reads_files(tmp_path):
    from scripts.ingest_kb import load_sources

    f = tmp_path / "a.md"
    f.write_text("hello relayn", encoding="utf-8")
    out = load_sources([str(f)], [], None)
    assert out == [(str(f), "hello relayn")]


class _FakeTable:
    def __init__(self, store, name):
        self.store = store
        self.name = name
        self._filters = {}
        self._select = False

    def select(self, *a, **k):
        self._select = True
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def maybe_single(self):
        return self

    def limit(self, *a, **k):
        return self

    def in_(self, col, vals):
        self._filters[col] = ("in", vals)
        return self

    def insert(self, payload):
        rows = payload if isinstance(payload, list) else [payload]
        for r in rows:
            r = dict(r)
            r.setdefault("id", f"id-{len(self.store[self.name]) + 1}")
            self.store[self.name].append(r)
        self._last_insert = rows
        return self

    def update(self, payload):
        self._update = payload
        return self

    def _apply_update(self):
        for r in self.store[self.name]:
            if all(r.get(k) == v for k, v in self._filters.items() if not isinstance(v, tuple)):
                r.update(self._update)

    def delete(self):
        self._delete = True
        return self

    def execute(self):
        if getattr(self, "_update", None) is not None:
            self._apply_update()
            return types.SimpleNamespace(data=[])
        if getattr(self, "_delete", False):
            return types.SimpleNamespace(data=[])
        if getattr(self, "_last_insert", None) is not None:
            data = [self.store[self.name][-len(self._last_insert)], ] if len(self._last_insert) == 1 \
                else self.store[self.name][-len(self._last_insert):]
            self._last_insert = None
            return types.SimpleNamespace(data=data)
        if self._select:
            data = [r for r in self.store[self.name]
                    if all(r.get(k) == v for k, v in self._filters.items()
                           if not isinstance(v, tuple))]
            return types.SimpleNamespace(data=data)
        return types.SimpleNamespace(data=[])


class _FakeSupabase:
    def __init__(self, seed=None):
        self.store = {"workflows": [], "workflow_knowledge_base": [], "workflow_kb_chunks": []}
        for row in seed or []:
            self.store["workflows"].append(row)

    def table(self, name):
        self.store.setdefault(name, [])
        return _FakeTable(self.store, name)


def test_verify_workflow_errors_when_missing_and_not_creating():
    from scripts.ingest_kb import verify_or_create_workflow

    sb = _FakeSupabase()
    with pytest.raises(SystemExit):
        verify_or_create_workflow(sb, "wf-1", "org-1", create=False, name="X")


def test_verify_workflow_creates_row_with_setup():
    from scripts.ingest_kb import verify_or_create_workflow

    sb = _FakeSupabase()
    verify_or_create_workflow(sb, "wf-1", "org-1", create=True, name="RelayN Assistant")
    created = sb.store["workflows"][0]
    assert created["id"] == "wf-1"
    assert created["organization_id"] == "org-1"
    assert created["workflow_type"] == "ai_chatbot"
    assert created["is_active"] is True


def test_ingest_source_writes_kb_row_and_chunks():
    from scripts.ingest_kb import ingest_source

    sb = _FakeSupabase()
    fake_embeddings = types.SimpleNamespace(
        embed_documents=lambda chunks: [[0.1, 0.2, 0.3] for _ in chunks]
    )

    n = ingest_source(sb, fake_embeddings, "wf-1", "org-1", "kb/relayn.md", "word " * 400)

    assert n >= 1
    kb = sb.store["workflow_knowledge_base"][0]
    assert kb["workflow_id"] == "wf-1"
    assert kb["organization_id"] == "org-1"
    assert kb["source_type"] == "text"
    assert "source_url" not in kb                       # column does not exist
    assert kb["ingest_status"] == "indexed"

    chunks = sb.store["workflow_kb_chunks"]
    assert len(chunks) == n
    assert all(c["knowledge_base_id"] == kb["id"] for c in chunks)
    assert all(c["embedding"] == [0.1, 0.2, 0.3] for c in chunks)
    assert all(c["workflow_id"] == "wf-1" and c["organization_id"] == "org-1" for c in chunks)
