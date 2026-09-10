import types


def test_resolve_urls_falls_back_to_seed_list_on_sitemap_error():
    from scripts.scrape_relayn_site import SEED_URLS, resolve_urls

    class FakeClient:
        def get(self, url, **kw):
            raise RuntimeError("no network")

    assert resolve_urls(FakeClient()) == SEED_URLS


def test_resolve_urls_parses_sitemap_locs():
    from scripts.scrape_relayn_site import resolve_urls

    xml = """<?xml version="1.0"?>
    <urlset><url><loc>https://relayn.com/</loc></url>
    <url><loc>https://relayn.com/pricing</loc></url></urlset>"""

    class FakeResp:
        status_code = 200
        text = xml

        def raise_for_status(self):
            pass

    class FakeClient:
        def get(self, url, **kw):
            return FakeResp()

    assert resolve_urls(FakeClient()) == ["https://relayn.com/", "https://relayn.com/pricing"]


def test_extract_text_strips_chrome_and_scripts():
    from scripts.scrape_relayn_site import extract_text

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


def test_upsert_kb_row_targets_pinned_workflow(monkeypatch):
    import scripts.scrape_relayn_site as s

    captured = {}

    class FakeTable:
        def delete(self):
            captured["deleted"] = True
            return self

        def eq(self, *a):
            captured.setdefault("eq", []).append(a)
            return self

        def execute(self):
            return types.SimpleNamespace(data=[])

        def insert(self, row):
            captured["row"] = row
            return self

    class FakeSupabase:
        def table(self, name):
            captured["table"] = name
            return FakeTable()

    monkeypatch.setattr(s, "supabase_client", FakeSupabase())
    monkeypatch.setattr(s.settings, "RELAYN_WORKFLOW_ID", "wf-x", raising=False)
    monkeypatch.setattr(s.settings, "RELAYN_ORG_ID", "org-x", raising=False)

    s.upsert_kb_row("https://relayn.com/pricing", "Pricing text here")

    assert captured["table"] == "workflow_knowledge_base"
    assert captured["row"]["workflow_id"] == "wf-x"
    assert captured["row"]["organization_id"] == "org-x"
    assert captured["row"]["source_type"] == "text"
    assert captured["row"]["raw_text"] == "Pricing text here"
    assert captured["row"]["source_url"] == "https://relayn.com/pricing"


def test_trigger_ingest_posts_workflow_id():
    from scripts.scrape_relayn_site import trigger_ingest

    calls = []

    class FakeClient:
        def post(self, url, json=None, **kw):
            calls.append((url, json))

            class R:
                status_code = 202

                def raise_for_status(self):
                    pass

            return R()

    trigger_ingest(FakeClient())
    url, body = calls[0]
    assert url.endswith("/ingest-knowledge-base")
    assert "workflow_id" in body
