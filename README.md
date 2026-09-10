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

Or `docker compose up --build` (published on `:8002`).

## Knowledge base

```bash
python -m scripts.scrape_relayn_site
```

Scrapes relayn.com into the pinned workflow's `workflow_knowledge_base` rows and
triggers the shared `relayn_services` `/ingest-knowledge-base` to chunk + embed
into `workflow_kb_chunks`.

## Before first deploy

- Apply `docs/DB_MIGRATION.md` (adds `chat_turns.flow_capture`).
- Confirm the open items in the spec (§12): lead-scoring coverage for the RelayN
  org, `workflow_knowledge_base` column names, relayn.com sitemap.
