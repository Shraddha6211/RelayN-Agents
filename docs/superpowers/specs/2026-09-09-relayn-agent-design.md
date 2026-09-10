# RelayN Agent — Design

**Date:** 2026-09-09
**Status:** Approved for implementation planning
**Repo:** `relayn_agents/` (new, standalone)

---

## 1. Purpose and context

RelayN (Aakash Labs' unified AI-powered customer-communication platform) needs its
own customer-facing sales + support agent on WhatsApp. Businesses message RelayN's
number; the agent answers product and pricing questions from a knowledge base,
runs a demo-booking flow, runs a lighter contact-sales flow, and hands off to a
human on request.

This is a **custom-tier service** built on the `angan_services` blueprint, not a
workflow row on the shared `relayn_services`. It is single-purpose and
single-tenant (the RelayN org itself).

### Existing landscape it plugs into

| Service | Role | Relevance here |
| --- | --- | --- |
| `relayn_services` | Shared tier-2 AI chatbot; one deploy serves every org; adaptive-retrieval loop, no LangGraph | Source of the **read-side KB contract only** (`match_workflow_kb_chunks` RPC) and the `/ingest-knowledge-base` endpoint |
| `angan_services` | First custom-tier service; LangGraph router → executor/generator → multi-turn sub-flows; own Redis checkpointer | **The blueprint.** Technique and structure are copied from here |
| Shared Supabase project | Holds `chat_turns`, `leads`, `messages`, `workflows`, `workflow_knowledge_base`, `workflow_kb_chunks`; row separation by `organization_id` + `workflow_id` + `asset_id` | This service reads/writes it with a `service_role` key |
| Lead scoring | Postgres trigger on `chat_turns` insert → Supabase edge function that "owns the whole model" | This service ships **no** lead-scoring code and must not alter this path |

### Reference paths (read-only, not modified)

- `D:\Personal\Learnings\Internship\Angan - Copy\angan_services` — blueprint
- `D:\Personal\Learnings\Aakash Group\relayn_services` — KB contract source

---

## 2. Scope

### In scope (v1)

- `POST /generate-reply` HTTP endpoint returning `{reply, intent, topic, capture?, handoff_requested?}`
- LangGraph state machine: router → executor/generator + `demo`, `sales`, `handoff` sub-flows
- Redis checkpointer for resumable wizard state
- RAG over the RelayN workflow's `workflow_kb_chunks` via the shared RPC
- Hardcoded RelayN persona and prompts
- Intent collapse to `RAG` / `ORDER` / `CHAT` for logging
- `scripts/scrape_relayn_site.py` to populate the knowledge base
- One additive nullable `jsonb` column on `chat_turns` for demo/sales capture

### Out of scope (v1)

- Instagram / Messenger channels
- Calendar or CRM integration for demo booking
- Support ticketing (support = human handoff only)
- Free-trial / self-serve signup flow
- Outbound broadcasting
- This service's own lead scoring / `/lead/recompute`
- Reading persona/config from `workflows.config`
- This service sending WhatsApp messages or writing `chat_turns` (the dashboard does both)

---

## 3. Architecture

### 3.1 Integration boundary

```
Customer WhatsApp msg
      │
      ▼
Dashboard Meta webhook / runWorkflow
      │  POST /generate-reply
      │  { org_id, asset_id, asset_type, workflow_id, conversation_id,
      │    message: { body, type }, user_name? }
      ▼
relayn_agents  ──►  match_workflow_kb_chunks (Supabase RPC)   [read]
      │
      │  { reply, intent, topic, capture?, handoff_requested? }
      ▼
Dashboard
   • sends `reply` over WhatsApp with its own per-asset Meta token
   • writes the `chat_turns` row, stamping `intent`, `topic`,
     and `flow_capture` (from `capture`)
   • pauses automation for the conversation if `handoff_requested`
```

The dashboard owns message send, the `chat_turns` write, and automation state.
This service owns classification, retrieval, reply generation, and wizard state.

### 3.2 Why LangGraph + checkpointer (not the `relayn_services` loop)

The demo and sales flows are resumable multi-turn state machines with a step
counter and a partial-data accumulator. `relayn_services` explicitly defers this
case ("earns its place back if a vertical needs a resumable multi-turn booking
flow"). Conversation history and wizard state both come from the checkpointer,
keyed `thread_id = f"{workflow_id}:{conversation_id}"`. `public.messages` is not
read; the checkpointer is the agent's source of truth (same cold-start property
as Angan — a conversation with prior history in `public.messages` starts with an
empty checkpoint).

### 3.3 Components

| Component | File | Ported from | Changes |
| --- | --- | --- | --- |
| FastAPI app | `main.py` | `angan_services/main.py` | Strip all webhooks, Meta send, quiz/review/order handling, `/lead/recompute`, `/reviews`, `/classify-leads`. Keep `/health`. Add `/generate-reply`. |
| Reply orchestration | `service.py` | `relayn_services/service.py` (`generate_reply` skeleton) + `angan_services` graph invoke | Verify workflow row (org match, `is_active`, `workflow_type`), pin-check `org_id`/`workflow_id`, invoke graph with checkpointer config, shape the response. |
| Graph blueprint | `agent/graph.py` | `angan_services/agent/graph.py` | New node set and routing (Section 4). |
| State | `agent/state.py` | `angan_services/agent/state.py` | New keys (Section 4.1). |
| Router | `agent/router.py` | `angan_services/agent/router.py` | New intent enum, RelayN keyword fast paths, demo/sales active-flow locks. |
| Nodes | `agent/nodes.py` | `angan_services/agent/nodes.py` | `executor`, `generator`, `demo`, `demo_end`, `sales`, `sales_end`, `handoff`. |
| KB retrieval | `agent/tools.py` | `relayn_services/agent/tools.py` | Copied near-verbatim (`search_knowledge_base`). |
| Prompts | `agent/prompts.py` | `angan_services/agent/prompts.py` | RelayN router prompt + generator persona. Wizards are deterministic (no prompt). |
| Checkpointer | `db/redis.py` | `angan_services/db/redis.py` | Ported unchanged (`PruningAsyncRedisSaver`, semantic cache). |
| Clients | `clients.py` | `relayn_services/clients.py` | `supabase_client` (service key) + `embeddings` (`text-embedding-3-small`, 1536). |
| Config | `config.py` | `relayn_services/config.py` | Add `RELAYN_ORG_ID`, `RELAYN_WORKFLOW_ID`, `RELAYN_SERVICES_URL`, `REDIS_*`. |
| Schemas | `schemas.py` | `relayn_services/schemas.py` | Request/response models (Section 3.4). |
| KB scraper | `scripts/scrape_relayn_site.py` | new | Section 6. |

### 3.4 Request / response schemas (`schemas.py`)

```python
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
    intent: str | None = None          # collapsed: RAG | ORDER | CHAT
    topic: str | None = None           # search query used, if any
    capture: FlowCapture | None = None # set only on a completion turn
    handoff_requested: bool = False
```

---

## 4. Graph

### 4.1 `AgentState` (`agent/state.py`)

Start from Angan's `AgentState`. **Remove** `quiz_*`, `review_*`, `order_data`.
**Keep** `messages` (`add_messages`), `user_data`, `intent`, `search_query`,
`tool_data`. **Add**:

```python
demo_step: int
demo_data: Dict[str, Any]
demo_completed: bool
sales_step: int
sales_data: Dict[str, Any]
sales_completed: bool
```

`user_data` carries `{user_name, conversation_id, workflow_id, org_id}` (set by
`service.py` from the verified request; no `phone_number`/`platform`/`has_played`).

Every key a node returns must be declared here (LangGraph drops undeclared keys —
this bit Angan repeatedly; see its `state.py` FIX comments).

### 4.2 Nodes and routing (`agent/graph.py`)

Nodes: `router`, `executor`, `generator`, `demo`, `demo_end`, `sales`,
`sales_end`, `handoff`.

```
START → router → route_decision:
    BOOK_DEMO      → demo
    STOP_DEMO      → demo_end
    CONTACT_SALES  → sales
    STOP_SALES     → sales_end
    HANDOFF        → handoff
    PRODUCT_QA     → executor
    PRICING        → executor
    GENERAL_CHAT   → generator

executor → generator → END
generator → END
demo_end / sales_end / handoff → END

demo  → demo_exit:   tool_data == "DEMO_INVALID_INPUT"  → generator ; else END
sales → sales_exit:  tool_data == "SALES_INVALID_INPUT" → generator ; else END
```

(Mirrors Angan's `review` two-exit pattern.)

### 4.3 Router (`agent/router.py`)

Structured-output enum (internal, fine-grained):

```
GENERAL_CHAT, PRODUCT_QA, PRICING, BOOK_DEMO, STOP_DEMO,
CONTACT_SALES, STOP_SALES, HANDOFF
```

Layered logic, in order (Angan's structure):

1. **Active demo lock** — `demo_step > 0 and not demo_completed`:
   stop-words (`stop|quit|exit|cancel|back|menu`) → `STOP_DEMO`; else `BOOK_DEMO`.
2. **Active sales lock** — `sales_step > 0 and not sales_completed`:
   stop-words → `STOP_SALES`; else `CONTACT_SALES`.
3. **Keyword fast paths** (normalized exact / prefix match):
   - `book demo`, `schedule demo`, `get a demo`, `demo` → `BOOK_DEMO` + `tool_data="START_DEMO"`
   - `talk to sales`, `contact sales`, `sales team` → `CONTACT_SALES` + `tool_data="START_SALES"`
   - `human`, `agent`, `real person`, `support`, `help me` → `HANDOFF`
4. **LLM fallback** — `ChatOpenAI("gpt-4.1-mini", temperature=0).with_structured_output(RouteDecision)`
   over the last 5 turns; `search_query` extracted for `PRODUCT_QA` / `PRICING`.
   Model: `RouteDecision { intent: <enum>, search_query: str | None }`.

Router returns `{intent, search_query?, tool_data?}` — it does **not** collapse
the intent. Collapse happens once, in `service.py`, on the way out.

### 4.4 `executor_node`

Handles `PRODUCT_QA` and `PRICING`.

```python
query = state["search_query"] or (
    "RelayN pricing and plans" if intent == "PRICING"
    else "RelayN features and capabilities"
)
chunks = search_knowledge_base(org_id, workflow_id, query)   # agent/tools.py
return {"tool_data": "\n\n".join(f"- {c}" for c in chunks) or None,
        "search_query": query}
```

For `PRICING`, prepend `"pricing plans "` to a model-supplied query.

### 4.5 `generator_node`

Angan's generator, RelayN persona. `GENERATOR_SYSTEM_PROMPT.format(business_name,
persona, tone, contact_info, tool_data, user_query)` + last 20 messages →
`llm.ainvoke`. Temperature 0.7 (separate call, Angan style). Rules baked into the
prompt: answer only from `tool_data`; if it is empty, say so plainly and give the
contact path; never invent pricing; when a buying signal appears, offer a demo.

### 4.6 `demo_node` (deterministic wizard)

Structurally identical to Angan's `review_node`: init on `tool_data ==
"START_DEMO"`, otherwise validate the previous answer, store it, advance or
terminate. Six steps:

| # | `demo_data` key | Question | Input |
| --- | --- | --- | --- |
| 1 | `business_name` | "What's the name of your business?" | open |
| 2 | `channels` | "Which channels do you want to manage with RelayN?" | buttons: WhatsApp / Instagram / Facebook / All |
| 3 | `monthly_volume` | "Roughly how many customer messages a month?" | buttons: `<500` / `500–2k` / `2k–10k` / `10k+` |
| 4 | `contact_name` | "Who should we address the demo to?" | open |
| 5 | `contact_info` | "Best email or phone to reach you?" | open, light format check |
| 6 | `preferred_time` | "Any preferred day/time for the demo?" | open |

- Button steps validate against the option id set; a miss returns
  `{tool_data: "DEMO_INVALID_INPUT", messages: <unchanged>}` → `demo_exit` routes
  to `generator` for a nudge, `demo_step` unchanged.
- On step 6 answered: build the payload, emit a final `SystemMessage` containing
  `[DEMO_BOOKED]`, set `demo_step = 0`, `demo_completed = True`, `intent =
  "BOOK_DEMO"`.
- `service.py` detects `[DEMO_BOOKED]`, strips it, and sets
  `capture = {type: "demo", data: demo_data}` on the response.

### 4.7 `sales_node` (deterministic wizard)

Same structure, three steps:

| # | `sales_data` key | Question | Input |
|---| ---              | --- | --- |
| 1 | `need`           | "What are you looking for help with?" | open |
| 2 | `company`        | "What's your company name and rough team size?" | open |
| 3 | `contact_info`   | "Best email or phone to reach you?" | open, light format check |

Completion → `[SALES_REQUEST]` sentinel, `sales_completed = True`, `intent =
"CONTACT_SALES"`; `service.py` → `capture = {type: "sales", data: sales_data}`.

### 4.8 `demo_end_node` / `sales_end_node`

Mirror Angan's `review_end_node`: a cancellation message that acknowledges how
many fields were collected, reset `*_step = 0` and `*_data = {}`, and leave
`intent` as `STOP_DEMO` / `STOP_SALES` (the collapse table maps both to `CHAT`).
Partial captures are **not** returned in `capture`.

### 4.9 `handoff_node`

No wizard. Returns a canned message ("Connecting you with someone from our
team — one moment.") and leaves `intent = "HANDOFF"` (the collapse table maps it
to `CHAT`). `service.py` keys `handoff_requested = True` off that intent so the
dashboard pauses automation. Covers both explicit "talk to a human" and v1
support requests.

---

## 5. Intent collapse and lead scoring

### 5.1 Collapse (in `service.py`, once, on egress)

| Internal intent | Logged `intent` |
| --- | --- |
| `BOOK_DEMO`, `CONTACT_SALES` | `ORDER` |
| `PRODUCT_QA`, `PRICING` | `RAG` |
| `GENERAL_CHAT`, `HANDOFF`, `STOP_DEMO`, `STOP_SALES` | `CHAT` |

`topic` = `state["search_query"]` (nullable). These are the only values the
shared scorer's table recognises (`ORDER` 95, `RAG` 65, `CHAT` 15); any other
string scores as unknown (10).

### 5.2 No lead scoring in this service

This service ships no `lead_engine.py` and no `/lead/recompute`. It does not
create, alter, or call the `chat_turns` trigger or the scoring edge function.

**Open, non-blocking:** it is not yet confirmed whether the shared edge function
already scores the RelayN org. The collapse in 5.1 makes this safe either way —
if scoring runs, it sees valid intents; if it does not, adding it later is an
additive change with no impact on this service. Nothing in this design blocks or
depends on it, and nothing here can regress `angan_services`' scoring.

---

## 6. Knowledge base

### 6.1 Ingestion 

Get Markdown format file with the information about RelayN from the deverloper. It will be stored locally inside the main directory, the chunks and embedding generated thereafter should be stored in the supabase database, workflow_knowledge_base, workflow_kb_chunks tables and embeddings also in similar way. 

Run manually or on a cron. Not triggered by conversation traffic.

### 6.2 Retrieval (request path)

`agent/tools.py::search_knowledge_base(organization_id, workflow_id, query, k=5)`
— copied from `relayn_services/agent/tools.py`, calls
`match_workflow_kb_chunks(p_organization_id, p_workflow_id, query_embedding,
match_count)` and returns `list[str]` of chunk contents. `org_id` and
`workflow_id` are passed positionally from the verified request, never from model
output.

---

## 7. Data model changes

### 7.1 `chat_turns` — one additive nullable column

```sql
ALTER TABLE public.chat_turns
  ADD COLUMN IF NOT EXISTS flow_capture jsonb;
```

- Written by the **dashboard**, from `GenerateReplyResponse.capture`, only on a
  demo/sales completion turn: `{ "type": "demo"|"sales", "data": { ... } }`.
- Nullable and additive. `angan_services` inserts explicit column lists and its
  lead engine reads specific keys from `select("*")`, so an unread extra column
  is inert there. This is the only shared-schema change and it cannot regress
  Angan.

### 7.2 No other schema changes

No new tables. `leads` is untouched (per-`(organization_id, conversation_id)`,
not read or written here). `workflow_knowledge_base` / `workflow_kb_chunks` are
written only through the existing shared ingest endpoint.

---

## 8. Configuration (`config.py`, `.env.example`)

| Var | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | router + generator + embeddings (via scraper) |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | shared project, `service_role` |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_DB` | checkpointer + semantic cache; `REDIS_URL` computed like Angan |
| `RELAYN_ORG_ID` | pinned; requests with a different `org_id` are rejected |
| `RELAYN_WORKFLOW_ID` | pinned; requests with a different `workflow_id` are rejected |
| `RELAYN_SERVICES_URL` | base URL of the shared service, for the scraper's ingest call |

No Meta tokens (this service never calls the Graph API).

---

## 9. `service.py` control flow

```
generate_reply(payload):
    if payload.org_id != RELAYN_ORG_ID:            return empty response
    if payload.workflow_id != RELAYN_WORKFLOW_ID:  return empty response
    workflow = _fetch_workflow(payload.workflow_id)
    if not workflow or not is_active or workflow_type != "ai_chatbot":
        return GenerateReplyResponse(reply=None, intent=None, topic=None)
    if workflow.organization_id != payload.org_id: return empty response  # forgery guard

    state_in = {
        "messages": [HumanMessage(payload.message.body)],
        "user_data": { "user_name": payload.user_name or "there",
                       "conversation_id": payload.conversation_id,
                       "workflow_id": payload.workflow_id,
                       "org_id": payload.org_id },
    }
    cfg = {"configurable": {"thread_id": f"{payload.workflow_id}:{payload.conversation_id}"}}
    final = await agent_app.ainvoke(state_in, cfg)

    reply = final["messages"][-1].content
    internal_intent = final.get("intent") or "GENERAL_CHAT"

    capture = None
    if "[DEMO_BOOKED]" in reply:
        capture = FlowCapture(type="demo", data=final.get("demo_data", {}))
        reply = reply.replace("[DEMO_BOOKED]", "").strip()
    elif "[SALES_REQUEST]" in reply:
        capture = FlowCapture(type="sales", data=final.get("sales_data", {}))
        reply = reply.replace("[SALES_REQUEST]", "").strip()

    return GenerateReplyResponse(
        reply=reply or None,
        intent=COLLAPSE[internal_intent],
        topic=final.get("search_query"),
        capture=capture,
        handoff_requested=(internal_intent == "HANDOFF"),
    )
```

The graph is compiled once at startup (`lifespan`) with the
`PruningAsyncRedisSaver`, exactly as Angan compiles its workflow.

---

## 10. Testing

| File | Covers |
| --- | --- |
| `tests/test_router_intents.py` | Representative messages → expected internal intent; demo/sales active-flow lock behaviour; keyword fast paths; stop-word detection |
| `tests/test_demo_flow.py` | Step progression 1→6, button validation + `DEMO_INVALID_INPUT` path, `[DEMO_BOOKED]` sentinel, state reset, `demo_end` cancellation |
| `tests/test_sales_flow.py` | Same shape, 3 steps, `[SALES_REQUEST]` |
| `tests/test_handoff.py` | `HANDOFF` intent → canned reply + `handoff_requested=True` |
| `tests/test_intent_collapse.py` | Every internal intent maps to `RAG`/`ORDER`/`CHAT`; `topic` passthrough |
| `tests/test_service_guard.py` | Wrong `org_id` / wrong `workflow_id` / inactive workflow / org mismatch → empty response, graph not invoked |

`conftest.py` seeds fake env (like `relayn_services/tests/conftest.py`) so client
construction does not fail at import. Model and Supabase calls are mocked; no live
credentials in the default suite.

A command line testing file should also be made under `tests/test_system`. for this user should be able to send msg in CLI like they would've in whatsapp and get reply including intent and steps for each reply.

---

## 11. Assumptions about the dashboard integration (confirm at review)

1. The dashboard calls `POST /generate-reply` from its Meta webhook / `runWorkflow`
   path with the field set in 3.4, and RelayN's WhatsApp asset is wired to a
   `workflows` row of `workflow_type = "ai_chatbot"` for the RelayN org.
2. The dashboard writes the `chat_turns` row for every exchange and will stamp
   `intent`, `topic`, and the new `flow_capture` column from the response.
3. The dashboard has (or will add) automation-pause behaviour keyed off
   `handoff_requested`.
4. Adding a nullable `chat_turns.flow_capture` column is acceptable in the shared
   schema and requires no coordination beyond this doc.
5. The shared `/ingest-knowledge-base` endpoint accepts a plain
   `{ workflow_id }` POST and is reachable from wherever the scraper runs.

---

## 12. Open items (non-blocking)

- **OI-1** Whether the shared `chat_turns` scoring edge function already covers
  the RelayN org, or a custom `/lead/recompute` must be added later. Design is
  safe either way (Section 5.2).
- **OI-2** Exact column/field names on `workflow_knowledge_base`
  (`source_url`? `ingest_status` values) — confirm against the live table before
  writing the scraper's upsert.
- **OI-3** Whether `relayn.com` publishes `sitemap.xml`; if not, the seed URL
  list is the fallback.
- **OI-4** Final name for the `chat_turns` capture column (`flow_capture`
  proposed).

  # Development Workflow
- Always build incrementally. Never generate an entire feature or file at once.
- Follow a strict dependency order: Routes/Interfaces -> Validation/Types -> Core Logic -> Database/External Services.
- Write code skeleton/stubs first, verify structural soundness, then fill in the logic.
- After finishing a micro-step, pause and ask the user for a review or a test execution.

