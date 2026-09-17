# relayn_agents

RelayN's own WhatsApp sales + support agent. A standalone custom-tier service
built on the `angan_services` blueprint: a LangGraph router gating a RAG
executor→generator path plus `demo`, `sales`, and `handoff` sub-flows,
checkpointed in Redis.

The dashboard sends WhatsApp messages and writes `chat_turns`; this service
returns `{reply, intent, topic, capture?, handoff_requested?}`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET`  | `/health` | readiness (503 until the graph is compiled) |
| `POST` | `/generate-reply` | run the agent for one inbound message |

## Design

`docs/superpowers/specs/2026-09-09-relayn-agent-design.md` and
`docs/superpowers/plans/2026-09-09-relayn-agent.md`.

## Run

```bash
cp .env.example .env      # fill in OpenAI, Supabase, Redis, the pinned RelayN ids
pip install -r requirements.txt
uvicorn main:app --reload --port 8002
pytest -q
```

Demo booking uses Cal.com after the six demo details are collected. Add
`CAL_API_KEY`, `CAL_EVENT_TYPE_ID`, and optionally `CAL_TIMEZONE` to `.env`.
The CLI presents available Cal.com slots, accepts a slot number, asks for
confirmation, and only reports the demo as booked after Cal.com confirms it.

Or `docker compose up --build` (published on `:8002`).

## Chat with it from the CLI

```bash
python chat.py            # in-memory checkpointer, no Redis needed
python chat.py --redis    # use the real Redis checkpointer
```

Drives the compiled graph directly. Needs a real `OPENAI_API_KEY`; for grounded
`PRODUCT_QA` / `PRICING` answers also set real Supabase creds and a
`RELAYN_WORKFLOW_ID` that has ingested chunks. Commands: `/help` `/state`
`/reset` `/quit`.

## Knowledge base

`scripts/ingest_kb.py` chunks + embeds text and writes it straight into Supabase
(`workflow_knowledge_base` + `workflow_kb_chunks`) for the pinned workflow. It is
self-contained — no `relayn_services` deployment needed.

```bash
# one-time: create the RelayN workflow row, print the .env lines to add
python -m scripts.ingest_kb --setup --org <org-uuid>

# ingest every kb/*.md and kb/*.txt (the default source)
python -m scripts.ingest_kb

# or point it at specific files / pages
python -m scripts.ingest_kb --file kb/relayn.md --url https://example.com/docs
```

Re-running replaces the workflow's previously ingested `text` rows and their
chunks. Edit `kb/relayn.md` (and add more `kb/*.md` files) with real product,
pricing, and FAQ content — that text is what `PRODUCT_QA` / `PRICING` answers
are grounded on. `relayn.com` itself is a client-rendered SPA and yields no
usable text, so scraping it is not wired up.

## Before first deploy

- Apply `docs/DB_MIGRATION.md` (adds `chat_turns.flow_capture`).
- Confirm the one remaining spec open item (§12 OI-1): whether the shared
  `chat_turns` scoring trigger covers the RelayN org.

### Current setup (dev)

- Org: `b0c278a3-1401-4da4-b54e-57606eb809aa` (the only org in the shared project)
- Workflow: `RelayN Assistant` (`workflow_type = ai_chatbot`), created by
  `ingest_kb --setup`. Its id and the org id are in `.env` as
  `RELAYN_WORKFLOW_ID` / `RELAYN_ORG_ID`.
- KB: `kb/relayn.md` ingested → chunks in `workflow_kb_chunks`.
