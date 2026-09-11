# RelayN Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `relayn_agents`, a standalone FastAPI + LangGraph service that answers RelayN's WhatsApp prospects — product/pricing Q&A from a knowledge base, a demo-booking flow, a contact-sales flow, and human handoff — returning `{reply, intent, topic, capture?, handoff_requested?}` to the dashboard.

**Architecture:** Port of `angan_services`' technique: a LangGraph `StateGraph` with a deterministic router gating an executor→generator RAG path and three multi-turn sub-flow nodes (`demo`, `sales`, `handoff`), persisted with a pruning Redis checkpointer. The dashboard owns WhatsApp send and the `chat_turns` write; this service owns classification, retrieval, reply text, and wizard state. The only piece taken from `relayn_services` is the read-side KB call (`match_workflow_kb_chunks`).

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, `langgraph-checkpoint-redis`, LangChain (`langchain-openai`, `langchain-core`), OpenAI `gpt-4.1-mini` + `text-embedding-3-small`, Supabase (`service_role`), Redis, `httpx` + `beautifulsoup4` (scraper), `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-09-relayn-agent-design.md`

## Global Constraints

- **Blueprint:** structure and technique come from `angan_services` (`D:\Personal\Learnings\Internship\Angan - Copy\angan_services`). KB read contract comes from `relayn_services` (`D:\Personal\Learnings\Aakash Group\relayn_services`). Both are read-only references — never modify them.
- **No shared-schema change except one:** a single additive **nullable** `jsonb` column `chat_turns.flow_capture`. Documented, not executed by this service. Nothing else in the shared Supabase project (`chat_turns`, `leads`, `messages`, `workflows`, `workflow_knowledge_base`, `workflow_kb_chunks`, triggers, the scoring edge function) may be altered.
- **This service ships no lead-scoring code** — no `lead_engine.py`, no `/lead/recompute`.
- **This service holds no Meta credentials** and exposes no webhooks. Only `/health` and `/generate-reply`.
- **Every key a graph node returns must be declared in `AgentState`** — LangGraph silently drops undeclared keys.
- **Logged `intent` is always one of `RAG` / `ORDER` / `CHAT`.** The collapse happens once, in `service.py`.
- **Model IDs:** router + generator `gpt-4.1-mini`; embeddings `text-embedding-3-small` (1536 dims — must match `vector(1536)` on `workflow_kb_chunks`).
- **Scoping identifiers** (`org_id`, `workflow_id`) are passed positionally from the verified request into retrieval; never model-fillable.
- **Prompt slots use `{{TOKEN}}` + a `render()` replace helper**, never `str.format` / `ChatPromptTemplate` on text that can contain `{` or `}` (KB chunks, user queries).
- **Commit message trailer** on every commit:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv
  ```
- Tests use `asyncio.run(...)` for async nodes (no `pytest-asyncio` dependency), matching `angan_services/tests`.

---

## Deviations from the spec

**Invalid button input in a wizard (spec §4.2, §4.6).** The spec routes a demo
button miss through `demo_exit` → `generator` with `tool_data =
"DEMO_INVALID_INPUT"`. That marker would then reach `generator_node` as
retrieved context, and the RAG-grounding prompt is the wrong tool to ask a user
to re-pick an option. This plan instead has `demo_node` re-emit the same
question itself with a "Please tap one of the options" prefix, then go straight
to `END`. `demo_exit` / `sales_exit` and the `demo`/`sales` conditional edges
are dropped; `sales` never had an invalid path anyway (all its questions are
open text). Same user experience, no bogus `tool_data`, fewer edges.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `requirements.txt` | Pinned dependency floors |
| `.env.example` | Required environment variables |
| `.gitignore` | Standard Python ignores + `.env` |
| `config.py` | `Settings` (pydantic-settings); `REDIS_URL` computed property |
| `clients.py` | Module-level `supabase_client` (service key) and `embeddings` |
| `schemas.py` | `MessageIn`, `GenerateReplyRequest`, `FlowCapture`, `GenerateReplyResponse` |
| `agent/__init__.py` | package marker |
| `agent/prompts.py` | RelayN business constants, canned strings, `DEMO_QUESTIONS`, `SALES_QUESTIONS`, `ROUTER_SYSTEM_PROMPT`, `GENERATOR_SYSTEM_PROMPT`, `render()`, `format_question()` |
| `agent/state.py` | `AgentState` TypedDict |
| `agent/tools.py` | `search_knowledge_base()` → `match_workflow_kb_chunks` RPC |
| `agent/router.py` | `RouteDecision` model, `router_node()` |
| `agent/nodes.py` | `executor_node`, `generator_node`, `demo_node`, `demo_end_node`, `sales_node`, `sales_end_node`, `handoff_node` |
| `agent/graph.py` | `build_workflow()` → uncompiled `StateGraph`; `route_decision`, `demo_exit`, `sales_exit` |
| `db/__init__.py` | package marker |
| `db/redis.py` | `PruningAsyncRedisSaver`, `init_semantic_cache()` (ported from Angan) |
| `service.py` | `generate_reply(payload, agent_app)`, `_fetch_workflow`, `COLLAPSE`, sentinel extraction |
| `main.py` | FastAPI app, `lifespan` (compile graph + checkpointer), `/health`, `/generate-reply` |
| `scripts/scrape_relayn_site.py` | Scrape relayn.com → upsert `workflow_knowledge_base` → trigger shared ingest |
| `docs/DB_MIGRATION.md` | The one `ALTER TABLE` statement + rollout notes |
| `Dockerfile`, `docker-compose.yml` | Container build (published on `:8002`) |
| `tests/conftest.py` | Seed fake env before imports |
| `tests/test_config.py` … `tests/test_scrape.py` | One test module per component |

---

## Task 1: Project skeleton, dependencies, config, clients

**Files:**
- Create: `requirements.txt`, `.env.example`, `.gitignore`, `config.py`, `clients.py`
- Create: `agent/__init__.py`, `db/__init__.py`, `tests/__init__.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.settings` (instance of `Settings`) with attributes `OPENAI_API_KEY: str`, `SUPABASE_URL: str`, `SUPABASE_SERVICE_KEY: str`, `REDIS_HOST: str`, `REDIS_PORT: int`, `REDIS_PASSWORD: str`, `REDIS_DB: int`, `RELAYN_ORG_ID: str`, `RELAYN_WORKFLOW_ID: str`, `RELAYN_SERVICES_URL: str`; property `REDIS_URL: str`.
- Produces: `clients.supabase_client` (`supabase.Client`), `clients.embeddings` (`OpenAIEmbeddings`).

- [ ] **Step 1: Initialise the repo**

```bash
cd C:/Users/Asus/OneDrive/Desktop/relayn_agents
git init
```

- [ ] **Step 2: Write `requirements.txt`**

```
# --- API layer ---
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
gunicorn

# --- Orchestration ---
langgraph>=0.2.50
langgraph-checkpoint-redis>=0.2.0

# --- LangChain ecosystem ---
langchain>=0.3.7
langchain-core>=0.3.21
langchain-community>=0.3.7
langchain-openai>=0.2.9
langchain-text-splitters>=0.3.2

# --- Infrastructure ---
supabase>=2.10.0
redis>=5.2.0
pydantic-settings>=2.6.0
python-dotenv>=1.0.1

# --- Scraper ---
httpx>=0.27.0
beautifulsoup4>=4.12.0
lxml>=5.0.0

# --- Testing ---
pytest>=8.3.0
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.py[cod]
.env
.venv/
venv/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Write `.env.example`**

```
OPENAI_API_KEY=
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
RELAYN_ORG_ID=
RELAYN_WORKFLOW_ID=
RELAYN_SERVICES_URL=http://localhost:8001
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
"""Give config.py values to read before anything imports it.

clients.py builds Supabase and embeddings clients at import time; without
these the suite fails on a missing-env ValidationError before any test runs.
Neither client makes a network call while being constructed.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_PASSWORD", "")
os.environ.setdefault("REDIS_DB", "0")
os.environ.setdefault("RELAYN_ORG_ID", "org-relayn-test")
os.environ.setdefault("RELAYN_WORKFLOW_ID", "wf-relayn-test")
os.environ.setdefault("RELAYN_SERVICES_URL", "http://localhost:8001")
```

- [ ] **Step 6: Create empty package markers**

Create `agent/__init__.py`, `db/__init__.py`, `tests/__init__.py`, each containing a single newline.

- [ ] **Step 7: Write the failing test — `tests/test_config.py`**

```python
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
```

- [ ] **Step 8: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 9: Write `config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str

    SUPABASE_URL: str
    # service_role: every table has RLS on and this service has no end-user
    # session, so the anon key would silently return zero rows.
    SUPABASE_SERVICE_KEY: str

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # Pinned. service.py rejects any request whose org_id / workflow_id does
    # not match these — this deployment serves exactly the RelayN workflow.
    RELAYN_ORG_ID: str
    RELAYN_WORKFLOW_ID: str

    # Base URL of the shared relayn_services deployment; used only by the
    # offline KB scraper to trigger chunk+embed.
    RELAYN_SERVICES_URL: str

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
```

- [ ] **Step 10: Write `clients.py`**

```python
from langchain_openai import OpenAIEmbeddings
from supabase import Client, create_client
from supabase.client import ClientOptions

from config import settings

# 1536 dims — must match vector(1536) on public.workflow_kb_chunks.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY,
)

supabase_client: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_KEY,
    options=ClientOptions(auto_refresh_token=False, persist_session=False),
)
```

- [ ] **Step 11: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "chore: project skeleton, config, clients

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 2: Request / response schemas

**Files:**
- Create: `schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces:
  - `MessageIn(body: str, type: str = "text")`
  - `GenerateReplyRequest(org_id: str, asset_id: str, asset_type: str, workflow_id: str, conversation_id: str, message: MessageIn, user_name: str | None = None)`
  - `FlowCapture(type: Literal["demo", "sales"], data: dict)`
  - `GenerateReplyResponse(reply: str | None, intent: str | None = None, topic: str | None = None, capture: FlowCapture | None = None, handoff_requested: bool = False)`

- [ ] **Step 1: Write the failing test — `tests/test_schemas.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'schemas'`

- [ ] **Step 3: Write `schemas.py`**

```python
from typing import Literal

from pydantic import BaseModel


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
    # Collapsed classification (RAG | ORDER | CHAT) for the dashboard to stamp
    # onto its chat_turns row. Lead scoring only ever sees these three values.
    intent: str | None = None
    topic: str | None = None
    # Set only on a demo/sales completion turn; the dashboard writes it into
    # chat_turns.flow_capture.
    capture: FlowCapture | None = None
    handoff_requested: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add schemas.py tests/test_schemas.py
git commit -m "feat: request/response schemas for /generate-reply

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 3: State and prompt/content module

**Files:**
- Create: `agent/state.py`, `agent/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Produces (`agent/state.py`): `AgentState` — a `TypedDict(total=False)` with keys `messages` (annotated `add_messages`), `user_data: dict`, `intent: str`, `search_query: str | None`, `tool_data: Any`, `demo_step: int`, `demo_data: dict`, `demo_completed: bool`, `sales_step: int`, `sales_data: dict`, `sales_completed: bool`.
- Produces (`agent/prompts.py`):
  - `RELAYN_BUSINESS: dict` with keys `name`, `persona`, `tone`
  - `RELAYN_CONTACT: str`
  - `HANDOFF_MESSAGE: str`
  - `DEMO_QUESTIONS: list[dict]` (6 items) and `SALES_QUESTIONS: list[dict]` (3 items); each item `{"key": str, "type": "open"|"button", "question": str, "options"?: list[{"id": str, "title": str}]}`
  - `ROUTER_SYSTEM_PROMPT: str`, `GENERATOR_SYSTEM_PROMPT: str`
  - `render(template: str, **slots: str) -> str`
  - `format_question(q: dict, n: int, total: int) -> str`

- [ ] **Step 1: Write the failing test — `tests/test_prompts.py`**

```python
def test_demo_questions_shape():
    from agent.prompts import DEMO_QUESTIONS

    assert [q["key"] for q in DEMO_QUESTIONS] == [
        "business_name", "channels", "monthly_volume",
        "contact_name", "contact_info", "preferred_time",
    ]
    button_steps = [q for q in DEMO_QUESTIONS if q["type"] == "button"]
    assert {q["key"] for q in button_steps} == {"channels", "monthly_volume"}
    for q in button_steps:
        assert len(q["options"]) >= 2
        assert all("id" in o and "title" in o for o in q["options"])


def test_sales_questions_shape():
    from agent.prompts import SALES_QUESTIONS

    assert [q["key"] for q in SALES_QUESTIONS] == ["need", "company", "contact_info"]
    assert all(q["type"] == "open" for q in SALES_QUESTIONS)


def test_render_replaces_tokens_and_drops_leftovers():
    from agent.prompts import render

    out = render("Hi {{NAME}} — {{MISSING}} done", name="Sam")
    assert out == "Hi Sam —  done"


def test_render_survives_braces_in_values():
    from agent.prompts import render

    out = render("data: {{TOOL_DATA}}", tool_data='{"price": "NPR 5,000"}')
    assert '{"price": "NPR 5,000"}' in out


def test_format_question_open_and_button():
    from agent.prompts import DEMO_QUESTIONS, format_question

    open_q = format_question(DEMO_QUESTIONS[0], 1, 6)
    assert open_q.startswith("*1/6*")

    button_q = format_question(DEMO_QUESTIONS[1], 2, 6)
    assert "*2/6*" in button_q
    for opt in DEMO_QUESTIONS[1]["options"]:
        assert f"{opt['id']}) {opt['title']}" in button_q


def test_generator_prompt_has_expected_slots():
    from agent.prompts import GENERATOR_SYSTEM_PROMPT

    for token in ("{{BUSINESS_NAME}}", "{{PERSONA}}", "{{TONE}}",
                  "{{CONTACT_INFO}}", "{{TOOL_DATA}}", "{{USER_QUERY}}"):
        assert token in GENERATOR_SYSTEM_PROMPT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.prompts'`

- [ ] **Step 3: Write `agent/state.py`**

```python
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_data: dict          # {user_name, conversation_id, workflow_id, org_id}
    intent: str              # internal fine intent set by the router / sub-flows
    search_query: str | None
    tool_data: Any           # retrieved-context string, or a sub-flow marker

    # --- DEMO FLOW ---
    demo_step: int           # 0 = inactive; 1..len(DEMO_QUESTIONS) = awaiting that answer
    demo_data: dict
    demo_completed: bool

    # --- SALES FLOW ---
    sales_step: int
    sales_data: dict
    sales_completed: bool
```

- [ ] **Step 4: Write `agent/prompts.py`**

```python
import re

# --------------------------------------------------------------------------
# BUSINESS CONSTANTS (hardcoded — this is a single-tenant custom-tier service)
# --------------------------------------------------------------------------

RELAYN_BUSINESS = {
    "name": "RelayN",
    "persona": (
        "You work on the RelayN team. RelayN is a unified, AI-powered customer "
        "communication platform: one shared inbox for WhatsApp, Instagram and "
        "Facebook, with AI-assisted replies, automation and broadcasting, plus "
        "product catalogs, team collaboration, analytics and integrations. You "
        "help businesses understand whether RelayN fits them and get them to a "
        "demo when they are interested."
    ),
    "tone": "friendly, concise, straight-talking — never pushy",
}

# Editable. Shown when the user asks for a human or the KB has no answer.
RELAYN_CONTACT = "our team at hello@relayn.com or through the contact form at https://relayn.com"

HANDOFF_MESSAGE = (
    "Let me bring in someone from our team — one moment and a human will pick "
    "this up with you here. 👋"
)

# --------------------------------------------------------------------------
# SUB-FLOW QUESTION BANKS (deterministic wizards — no LLM)
# --------------------------------------------------------------------------

DEMO_QUESTIONS = [
    {
        "key": "business_name",
        "type": "open",
        "question": "Happy to set that up. First — what's the name of your business?",
    },
    {
        "key": "channels",
        "type": "button",
        "question": "Which channels do you want to manage with RelayN?",
        "options": [
            {"id": "ch_wa", "title": "WhatsApp"},
            {"id": "ch_ig", "title": "Instagram"},
            {"id": "ch_fb", "title": "Facebook"},
            {"id": "ch_all", "title": "All of them"},
        ],
    },
    {
        "key": "monthly_volume",
        "type": "button",
        "question": "Roughly how many customer messages do you handle a month?",
        "options": [
            {"id": "vol_s", "title": "Under 500"},
            {"id": "vol_m", "title": "500 to 2k"},
            {"id": "vol_l", "title": "2k to 10k"},
            {"id": "vol_xl", "title": "10k or more"},
        ],
    },
    {
        "key": "contact_name",
        "type": "open",
        "question": "Who should we address the demo invite to?",
    },
    {
        "key": "contact_info",
        "type": "open",
        "question": "What's the best email or phone number to reach you on?",
    },
    {
        "key": "preferred_time",
        "type": "open",
        "question": "Any preferred day or time for the demo? (for example, 'Tuesday afternoon')",
    },
]

SALES_QUESTIONS = [
    {
        "key": "need",
        "type": "open",
        "question": "I can connect you with our sales team. What are you looking for help with?",
    },
    {
        "key": "company",
        "type": "open",
        "question": "What's your company name and rough team size?",
    },
    {
        "key": "contact_info",
        "type": "open",
        "question": "And the best email or phone number to reach you on?",
    },
]

# --------------------------------------------------------------------------
# ROUTER PROMPT (LLM fallback only — deterministic paths never reach it)
# --------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You classify one inbound WhatsApp message for RelayN, a unified
AI-powered customer communication platform for businesses.

Return one intent:

- PRODUCT_QA   — a question about what RelayN does: features, channels, the shared
                 inbox, automation, broadcasting, catalogs, integrations, analytics,
                 onboarding, "does it support X". Set search_query to 2-4 keywords.
- PRICING      — anything about cost, plans, tiers, trials, discounts, "how much".
                 Set search_query to 2-4 keywords including the word "pricing".
- BOOK_DEMO    — the user wants a demo / to see it / to get set up / to be walked through it.
- CONTACT_SALES— the user wants to talk to sales / get a quote / discuss a rollout,
                 without specifically asking for a scheduled demo.
- HANDOFF      — the user asks for a human, a real person, an agent, or has a support
                 problem with an existing account.
- GENERAL_CHAT — greetings, thanks, small talk, or anything off-topic.

Rules:
- Judge on meaning, not wording. Users may write in any language or romanized script.
- A question is PRODUCT_QA or PRICING. Wanting to proceed is BOOK_DEMO or CONTACT_SALES.
- If unsure between a question and a greeting, prefer GENERAL_CHAT.
- search_query is null unless the intent is PRODUCT_QA or PRICING.
"""

# --------------------------------------------------------------------------
# GENERATOR PROMPT (voice + grounding rules). {{TOKEN}} slots, filled by render().
# --------------------------------------------------------------------------

GENERATOR_SYSTEM_PROMPT = """You are on the team at {{BUSINESS_NAME}}.

{{PERSONA}}

Tone: {{TONE}}

---------------------------------------------------------------
USER QUERY (answer THIS, directly)
---------------------------------------------------------------
{{USER_QUERY}}

---------------------------------------------------------------
CORE RULES
---------------------------------------------------------------
- Use ONLY the RETRIEVED CONTEXT below for facts about RelayN — features,
  channels, limits, pricing, plan names, numbers.
- NEVER invent a price, a plan, a limit, an integration, or a policy. If you
  did not retrieve it, you do not know it.
- If the RETRIEVED CONTEXT is "(nothing retrieved)" or does not cover the
  question: say plainly that you don't have that detail to hand, then point
  them to {{CONTACT_INFO}}. Do not guess.
- Every factual claim must trace to a line in RETRIEVED CONTEXT.

---------------------------------------------------------------
STYLE
---------------------------------------------------------------
- You are a real person on the team, not a bot. Never announce yourself as an AI.
- WhatsApp-style: 2-6 short lines. Greet only once per conversation.
- Reply in the user's own language / script.
- When the user shows buying interest (asks about fit, pricing, rollout, "can it
  do X for my team"), end by offering a demo — one natural sentence, e.g.
  "Want me to set up a quick demo?" Do not offer a demo on a pure greeting.
- End with a natural continuation, not a scripted sign-off.

---------------------------------------------------------------
RETRIEVED CONTEXT
---------------------------------------------------------------
{{TOOL_DATA}}
"""

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

_LEFTOVER_SLOT = re.compile(r"\{\{[A-Z_]+\}\}")


def render(template: str, **slots: str) -> str:
    """Fill {{TOKEN}} slots by literal replacement.

    str.format / ChatPromptTemplate are unsafe here: retrieved KB chunks and
    user queries routinely contain { and } and would raise KeyError. Slots the
    caller omits collapse to nothing rather than leaking a raw {{TOKEN}}.
    """
    out = template
    for key, value in slots.items():
        out = out.replace("{{" + key.upper() + "}}", (value or "").strip())
    return _LEFTOVER_SLOT.sub("", out)


def format_question(q: dict, n: int, total: int) -> str:
    """Render one wizard question, with an option list for button steps."""
    body = f"*{n}/{total}* {q['question']}"
    if q["type"] == "button":
        opts = "\n".join(f"{o['id']}) {o['title']}" for o in q["options"])
        body = f"{body}\n{opts}"
    return body
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add agent/state.py agent/prompts.py tests/test_prompts.py
git commit -m "feat: agent state and RelayN prompt/content module

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 4: Knowledge-base retrieval tool

**Files:**
- Create: `agent/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `clients.embeddings`, `clients.supabase_client`.
- Produces: `search_knowledge_base(organization_id: str, workflow_id: str, query: str, k: int = 5) -> list[str]` — returns chunk `content` strings; `[]` when the RPC returns no rows.

- [ ] **Step 1: Write the failing test — `tests/test_tools.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.tools'`

- [ ] **Step 3: Write `agent/tools.py`**

```python
# agent/tools.py
import logging

from clients import embeddings, supabase_client

logger = logging.getLogger("relayn_agents.tools")

MATCH_COUNT = 5


def search_knowledge_base(
    organization_id: str, workflow_id: str, query: str, k: int = MATCH_COUNT
) -> list[str]:
    """Vector search over ONE workflow's ingested knowledge base.

    Scoped on both keys, not just workflow_id. The org filter is redundant on a
    correct call — service.py has already checked the workflow belongs to the
    requesting org — and that is the point: it still holds if workflow_id is
    wrong. Both ids come from the verified request, never from model output, so
    the model's only influence on a search is the query string.
    """
    query_embedding = embeddings.embed_query(query)
    res = supabase_client.rpc(
        "match_workflow_kb_chunks",
        {
            "p_organization_id": organization_id,
            "p_workflow_id": workflow_id,
            "query_embedding": query_embedding,
            "match_count": k,
        },
    ).execute()
    return [row["content"] for row in (res.data or [])]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/tools.py tests/test_tools.py
git commit -m "feat: KB retrieval via match_workflow_kb_chunks RPC

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 5: Router node

**Files:**
- Create: `agent/router.py`
- Test: `tests/test_router_intents.py`

**Interfaces:**
- Consumes: `agent.state.AgentState`, `agent.prompts.ROUTER_SYSTEM_PROMPT`, `config.settings`.
- Produces:
  - `STOP_WORDS: set[str]`
  - `RouteDecision` — pydantic model `{intent: Literal["GENERAL_CHAT","PRODUCT_QA","PRICING","BOOK_DEMO","CONTACT_SALES","HANDOFF"], search_query: str | None}`
  - module-level `structured_router` (a `Runnable`) — tests monkeypatch this
  - `async router_node(state: AgentState) -> dict` returning some of `{intent, search_query, tool_data}`. `intent` is one of the six model values **plus** `STOP_DEMO` / `STOP_SALES` from the active-flow locks. Sets `tool_data="START_DEMO"` / `"START_SALES"` when a demo/sales flow should initialise.

- [ ] **Step 1: Write the failing test — `tests/test_router_intents.py`**

```python
import asyncio
import types

from langchain_core.messages import AIMessage, HumanMessage


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {}}
    base.update(extra)
    return base


def _run(state):
    import agent.router as router
    return asyncio.run(router.router_node(state))


def test_active_demo_lock_holds_plain_message_in_flow():
    assert _run(_state("acme corp", demo_step=1, demo_completed=False))["intent"] == "BOOK_DEMO"


def test_active_demo_lock_detects_stop_word():
    assert _run(_state("cancel", demo_step=2, demo_completed=False))["intent"] == "STOP_DEMO"


def test_completed_demo_does_not_lock():
    # demo_completed True -> falls through to fast paths / LLM, not BOOK_DEMO lock
    out = _run(_state("hello", demo_step=0, demo_completed=True))
    assert out["intent"] != "STOP_DEMO"


def test_active_sales_lock():
    assert _run(_state("we are 20 people", sales_step=1, sales_completed=False))["intent"] == "CONTACT_SALES"
    assert _run(_state("stop", sales_step=1, sales_completed=False))["intent"] == "STOP_SALES"


def test_keyword_fast_path_book_demo_sets_start_marker():
    out = _run(_state("book demo"))
    assert out["intent"] == "BOOK_DEMO"
    assert out["tool_data"] == "START_DEMO"


def test_keyword_fast_path_contact_sales_and_handoff():
    assert _run(_state("contact sales"))["tool_data"] == "START_SALES"
    assert _run(_state("talk to a human"))["intent"] == "HANDOFF"


def test_llm_fallback_used_for_freeform_message(monkeypatch):
    import agent.router as router

    async def fake_ainvoke(_payload):
        return router.RouteDecision(intent="PRICING", search_query="pricing plans")

    monkeypatch.setattr(router, "structured_router",
                        types.SimpleNamespace(ainvoke=fake_ainvoke))

    out = asyncio.run(router.router_node(_state("what do your plans cost?")))
    assert out["intent"] == "PRICING"
    assert out["search_query"] == "pricing plans"


def test_llm_fallback_book_demo_gets_start_marker(monkeypatch):
    import agent.router as router

    async def fake_ainvoke(_payload):
        return router.RouteDecision(intent="BOOK_DEMO", search_query=None)

    monkeypatch.setattr(router, "structured_router",
                        types.SimpleNamespace(ainvoke=fake_ainvoke))

    out = asyncio.run(router.router_node(_state("can you show me how it works live")))
    assert out["intent"] == "BOOK_DEMO"
    assert out["tool_data"] == "START_DEMO"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_router_intents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.router'`

- [ ] **Step 3: Write `agent/router.py`**

```python
# agent/router.py
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agent.prompts import ROUTER_SYSTEM_PROMPT
from agent.state import AgentState
from config import settings

STOP_WORDS = {"stop", "quit", "exit", "cancel", "back", "menu"}

_DEMO_TRIGGERS = {"book demo", "book a demo", "schedule demo", "schedule a demo",
                  "get a demo", "demo", "see a demo"}
_SALES_TRIGGERS = {"talk to sales", "contact sales", "speak to sales", "sales team"}
_HANDOFF_TRIGGERS = {"human", "agent", "real person", "talk to a human",
                     "talk to a person", "support", "help me"}


class RouteDecision(BaseModel):
    intent: Literal[
        "GENERAL_CHAT", "PRODUCT_QA", "PRICING",
        "BOOK_DEMO", "CONTACT_SALES", "HANDOFF",
    ]
    search_query: Optional[str] = Field(default=None)


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    openai_api_key=settings.OPENAI_API_KEY,
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

structured_router = _prompt | llm.with_structured_output(RouteDecision)


def _norm(text: str) -> str:
    return text.strip().lower()


def _has_stop_word(text: str) -> bool:
    return bool(STOP_WORDS.intersection(_norm(text).split()))


async def router_node(state: AgentState) -> dict:
    messages = state["messages"]
    raw = messages[-1].content
    msg = _norm(raw)

    demo_step = state.get("demo_step", 0)
    demo_completed = state.get("demo_completed", False)
    sales_step = state.get("sales_step", 0)
    sales_completed = state.get("sales_completed", False)

    # 1. ACTIVE DEMO LOCK
    if demo_step > 0 and not demo_completed:
        return {"intent": "STOP_DEMO" if _has_stop_word(raw) else "BOOK_DEMO"}

    # 2. ACTIVE SALES LOCK
    if sales_step > 0 and not sales_completed:
        return {"intent": "STOP_SALES" if _has_stop_word(raw) else "CONTACT_SALES"}

    # 3. KEYWORD FAST PATHS
    if msg in _DEMO_TRIGGERS:
        return {"intent": "BOOK_DEMO", "tool_data": "START_DEMO"}
    if msg in _SALES_TRIGGERS:
        return {"intent": "CONTACT_SALES", "tool_data": "START_SALES"}
    if msg in _HANDOFF_TRIGGERS:
        return {"intent": "HANDOFF"}

    # 4. LLM FALLBACK
    result = await structured_router.ainvoke(
        {"history": messages[-6:-1], "input": raw}
    )
    out: dict = {"intent": result.intent, "search_query": result.search_query, "tool_data": None}
    if result.intent == "BOOK_DEMO":
        out["tool_data"] = "START_DEMO"
    elif result.intent == "CONTACT_SALES":
        out["tool_data"] = "START_SALES"
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_router_intents.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/router.py tests/test_router_intents.py
git commit -m "feat: router node — locks, fast paths, LLM fallback

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 6: Demo sub-flow nodes

**Files:**
- Create: `agent/nodes.py` (with `demo_node`, `demo_end_node`, and the shared helper `_match_button`)
- Test: `tests/test_demo_flow.py`

**Interfaces:**
- Consumes: `agent.state.AgentState`, `agent.prompts.DEMO_QUESTIONS`, `agent.prompts.format_question`.
- Produces:
  - `_match_button(user_msg: str, options: list[dict]) -> dict | None` — matches on option `id` or case-insensitive `title`
  - `async demo_node(state: AgentState) -> dict`
  - `async demo_end_node(state: AgentState) -> dict`
  - Sentinel string `[DEMO_BOOKED]` appears in the final message on completion.

- [ ] **Step 1: Write the failing test — `tests/test_demo_flow.py`**

```python
import asyncio

from langchain_core.messages import HumanMessage


def _run(fn_name, state):
    import agent.nodes as nodes
    return asyncio.run(getattr(nodes, fn_name)(state))


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {"user_name": "Sam"}}
    base.update(extra)
    return base


def test_start_demo_emits_first_question():
    out = _run("demo_node", _state("book demo", tool_data="START_DEMO"))
    assert out["demo_step"] == 1
    assert out["demo_completed"] is False
    assert out["intent"] == "BOOK_DEMO"
    assert "*1/6*" in out["messages"][0].content


def test_open_answer_is_stored_and_flow_advances():
    out = _run("demo_node", _state("Acme Corp", demo_step=1, demo_data={}))
    assert out["demo_data"]["business_name"] == "Acme Corp"
    assert out["demo_step"] == 2
    assert "*2/6*" in out["messages"][0].content


def test_button_step_rejects_unknown_option_and_re_asks():
    out = _run("demo_node", _state("banana", demo_step=2, demo_data={"business_name": "Acme"}))
    assert out["demo_step"] == 2                            # not advanced
    assert out["demo_data"] == {"business_name": "Acme"}    # nothing stored, nothing lost
    text = out["messages"][0].content
    assert "Please tap one" in text
    assert "ch_all) All of them" in text                   # options shown again


def test_button_step_accepts_option_id_and_stores_title():
    out = _run("demo_node", _state("ch_all", demo_step=2, demo_data={"business_name": "Acme"}))
    assert out["demo_data"]["channels"] == "All of them"
    assert out["demo_step"] == 3


def test_final_answer_completes_and_emits_sentinel():
    data = {
        "business_name": "Acme", "channels": "WhatsApp", "monthly_volume": "500 to 2k",
        "contact_name": "Sam", "contact_info": "sam@acme.com",
    }
    out = _run("demo_node", _state("Tuesday afternoon", demo_step=6, demo_data=data))
    assert out["demo_completed"] is True
    assert out["demo_step"] == 0
    assert out["intent"] == "BOOK_DEMO"
    assert "[DEMO_BOOKED]" in out["messages"][0].content
    assert out["demo_data"]["preferred_time"] == "Tuesday afternoon"
    assert set(out["demo_data"]) == set(data) | {"preferred_time"}


def test_demo_end_resets_and_reports_count():
    out = _run("demo_end_node", _state("stop", demo_step=3, demo_data={"business_name": "Acme", "channels": "WhatsApp"}))
    assert out["demo_step"] == 0
    assert out["demo_data"] == {}
    assert out["demo_completed"] is False
    assert out["intent"] == "STOP_DEMO"
    assert "2" in out["messages"][0].content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_demo_flow.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.nodes'`

- [ ] **Step 3: Write `agent/nodes.py` (demo portion only)**

Tasks 7 and 8 append to this file and each adds only the imports it needs.

```python
# agent/nodes.py
from langchain_core.messages import SystemMessage

from agent.prompts import DEMO_QUESTIONS, format_question
from agent.state import AgentState


# ---------------------------------------------------------------------------
# SHARED
# ---------------------------------------------------------------------------

def _match_button(user_msg: str, options: list[dict]) -> dict | None:
    probe = user_msg.strip().lower()
    for o in options:
        if probe == o["id"].lower() or probe == o["title"].lower():
            return o
    return None


def _wizard_summary(data: dict) -> str:
    return "\n".join(f"• {k.replace('_', ' ').title()}: {v}" for k, v in data.items())


# ---------------------------------------------------------------------------
# DEMO FLOW
# ---------------------------------------------------------------------------

async def demo_node(state: AgentState) -> dict:
    step = state.get("demo_step", 0)
    data = dict(state.get("demo_data") or {})
    tool_data = state.get("tool_data")
    user_msg = state["messages"][-1].content.strip()
    total = len(DEMO_QUESTIONS)

    # 1. INIT
    if tool_data == "START_DEMO":
        return {
            "messages": [SystemMessage(content=format_question(DEMO_QUESTIONS[0], 1, total))],
            "demo_step": 1,
            "demo_data": {},
            "demo_completed": False,
            "intent": "BOOK_DEMO",
            "tool_data": None,
        }

    # 2. VALIDATE + STORE PREVIOUS ANSWER
    if step > 0:
        prev = DEMO_QUESTIONS[step - 1]
        if prev["type"] == "button":
            match = _match_button(user_msg, prev["options"])
            if match is None:
                nudge = "Please tap one of the options below:\n" + format_question(
                    prev, step, total
                )
                return {
                    "messages": [SystemMessage(content=nudge)],
                    "demo_step": step,
                    "demo_data": data,
                    "demo_completed": False,
                    "intent": "BOOK_DEMO",
                    "tool_data": None,
                }
            data[prev["key"]] = match["title"]
        else:
            data[prev["key"]] = user_msg

    # 3. COMPLETE
    if step >= total:
        msg = (
            f"Perfect — here's what I've got:\n\n{_wizard_summary(data)}\n\n"
            "Our team will reach out to lock in a time. Thanks! [DEMO_BOOKED]"
        )
        return {
            "messages": [SystemMessage(content=msg)],
            "demo_step": 0,
            "demo_data": data,
            "demo_completed": True,
            "intent": "BOOK_DEMO",
            "tool_data": None,
        }

    # 4. ASK NEXT
    q = DEMO_QUESTIONS[step]
    return {
        "messages": [SystemMessage(content=format_question(q, step + 1, total))],
        "demo_step": step + 1,
        "demo_data": data,
        "intent": "BOOK_DEMO",
        "tool_data": None,
    }


async def demo_end_node(state: AgentState) -> dict:
    answered = len(state.get("demo_data") or {})
    name = state.get("user_data", {}).get("user_name", "there")
    if answered == 0:
        msg = (
            f"No problem, {name} — I've cancelled the demo booking. "
            "What else can I help with?"
        )
    else:
        msg = (
            f"Got it, {name} — I've stopped the demo booking. You'd answered "
            f"{answered} question(s); nothing was saved. Just say 'book demo' "
            "whenever you want to start again."
        )
    return {
        "messages": [SystemMessage(content=msg)],
        "demo_step": 0,
        "demo_data": {},
        "demo_completed": False,
        "intent": "STOP_DEMO",
        "tool_data": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_demo_flow.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_demo_flow.py
git commit -m "feat: demo booking sub-flow (demo_node, demo_end_node)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 7: Sales sub-flow nodes

**Files:**
- Modify: `agent/nodes.py` (append `sales_node`, `sales_end_node`)
- Test: `tests/test_sales_flow.py`

**Interfaces:**
- Consumes: `agent.prompts.SALES_QUESTIONS`, `agent.prompts.format_question`, `_wizard_summary` (from Task 6).
- Produces:
  - `async sales_node(state: AgentState) -> dict`
  - `async sales_end_node(state: AgentState) -> dict`
  - Sentinel string `[SALES_REQUEST]` on completion.

- [ ] **Step 1: Write the failing test — `tests/test_sales_flow.py`**

```python
import asyncio

from langchain_core.messages import HumanMessage


def _run(fn_name, state):
    import agent.nodes as nodes
    return asyncio.run(getattr(nodes, fn_name)(state))


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)], "user_data": {"user_name": "Sam"}}
    base.update(extra)
    return base


def test_start_sales_emits_first_question():
    out = _run("sales_node", _state("contact sales", tool_data="START_SALES"))
    assert out["sales_step"] == 1
    assert out["sales_completed"] is False
    assert out["intent"] == "CONTACT_SALES"
    assert "*1/3*" in out["messages"][0].content


def test_sales_answers_accumulate():
    out = _run("sales_node", _state("rollout for support team", sales_step=1, sales_data={}))
    assert out["sales_data"]["need"] == "rollout for support team"
    assert out["sales_step"] == 2


def test_sales_completes_with_sentinel():
    data = {"need": "rollout", "company": "Acme, 40 people"}
    out = _run("sales_node", _state("sam@acme.com", sales_step=3, sales_data=data))
    assert out["sales_completed"] is True
    assert out["sales_step"] == 0
    assert out["intent"] == "CONTACT_SALES"
    assert "[SALES_REQUEST]" in out["messages"][0].content
    assert out["sales_data"]["contact_info"] == "sam@acme.com"


def test_sales_end_resets():
    out = _run("sales_end_node", _state("stop", sales_step=2, sales_data={"need": "x"}))
    assert out["sales_step"] == 0
    assert out["sales_data"] == {}
    assert out["sales_completed"] is False
    assert out["intent"] == "STOP_SALES"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sales_flow.py -v`
Expected: FAIL — `AttributeError: module 'agent.nodes' has no attribute 'sales_node'`

- [ ] **Step 3: Append to `agent/nodes.py`**

First extend the prompts import at the top of the file to add `SALES_QUESTIONS`:

```python
from agent.prompts import DEMO_QUESTIONS, SALES_QUESTIONS, format_question
```

Then append:

```python
# ---------------------------------------------------------------------------
# SALES FLOW
# ---------------------------------------------------------------------------

async def sales_node(state: AgentState) -> dict:
    step = state.get("sales_step", 0)
    data = dict(state.get("sales_data") or {})
    tool_data = state.get("tool_data")
    user_msg = state["messages"][-1].content.strip()
    total = len(SALES_QUESTIONS)

    # 1. INIT
    if tool_data == "START_SALES":
        return {
            "messages": [SystemMessage(content=format_question(SALES_QUESTIONS[0], 1, total))],
            "sales_step": 1,
            "sales_data": {},
            "sales_completed": False,
            "intent": "CONTACT_SALES",
            "tool_data": None,
        }

    # 2. STORE PREVIOUS ANSWER (all sales questions are open)
    if step > 0:
        data[SALES_QUESTIONS[step - 1]["key"]] = user_msg

    # 3. COMPLETE
    if step >= total:
        msg = (
            f"Thanks — I've passed this to our sales team:\n\n{_wizard_summary(data)}\n\n"
            "Someone will be in touch shortly. [SALES_REQUEST]"
        )
        return {
            "messages": [SystemMessage(content=msg)],
            "sales_step": 0,
            "sales_data": data,
            "sales_completed": True,
            "intent": "CONTACT_SALES",
            "tool_data": None,
        }

    # 4. ASK NEXT
    q = SALES_QUESTIONS[step]
    return {
        "messages": [SystemMessage(content=format_question(q, step + 1, total))],
        "sales_step": step + 1,
        "sales_data": data,
        "intent": "CONTACT_SALES",
        "tool_data": None,
    }


async def sales_end_node(state: AgentState) -> dict:
    answered = len(state.get("sales_data") or {})
    name = state.get("user_data", {}).get("user_name", "there")
    if answered == 0:
        msg = f"No problem, {name} — cancelled. What else can I help with?"
    else:
        msg = (
            f"Got it, {name} — I've stopped that. You'd answered {answered} "
            "question(s); nothing was saved. Say 'contact sales' to start again."
        )
    return {
        "messages": [SystemMessage(content=msg)],
        "sales_step": 0,
        "sales_data": {},
        "sales_completed": False,
        "intent": "STOP_SALES",
        "tool_data": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sales_flow.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_sales_flow.py
git commit -m "feat: contact-sales sub-flow (sales_node, sales_end_node)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 8: Executor, generator, and handoff nodes

**Files:**
- Modify: `agent/nodes.py` (append `executor_node`, `generator_node`, `handoff_node`; add module-level `gen_llm`)
- Test: `tests/test_nodes.py`, `tests/test_handoff.py`

**Interfaces:**
- Consumes: `agent.tools.search_knowledge_base`, `agent.prompts.render`, `agent.prompts.GENERATOR_SYSTEM_PROMPT`, `agent.prompts.RELAYN_BUSINESS`, `agent.prompts.RELAYN_CONTACT`, `agent.prompts.HANDOFF_MESSAGE`.
- Produces:
  - module-level `gen_llm` (a `ChatOpenAI`) — tests monkeypatch it
  - `async executor_node(state) -> dict` — sets `tool_data` (retrieved-context string or `None`) and `search_query` (the query actually used)
  - `async generator_node(state) -> dict` — returns `{"messages": [AIMessage(...)]}`
  - `async handoff_node(state) -> dict` — returns `{"messages": [SystemMessage(HANDOFF_MESSAGE)], "intent": "HANDOFF", "tool_data": None}`

- [ ] **Step 1: Write the failing tests**

`tests/test_nodes.py`:

```python
import asyncio
import types

from langchain_core.messages import AIMessage, HumanMessage


def _state(text, **extra):
    base = {"messages": [HumanMessage(content=text)],
            "user_data": {"user_name": "Sam", "org_id": "o1", "workflow_id": "w1"}}
    base.update(extra)
    return base


def test_executor_uses_router_query_when_present(monkeypatch):
    import agent.nodes as nodes

    seen = {}

    def fake_search(org, wf, query, k=5):
        seen["args"] = (org, wf, query)
        return ["RelayN has a shared inbox", "and automation"]

    monkeypatch.setattr(nodes, "search_knowledge_base", fake_search)

    out = asyncio.run(nodes.executor_node(_state("x", intent="PRODUCT_QA", search_query="shared inbox")))
    assert seen["args"] == ("o1", "w1", "shared inbox")
    assert out["tool_data"] == "- RelayN has a shared inbox\n\n- and automation"
    assert out["search_query"] == "shared inbox"


def test_executor_falls_back_to_intent_default_query(monkeypatch):
    import agent.nodes as nodes
    monkeypatch.setattr(nodes, "search_knowledge_base", lambda o, w, q, k=5: [])

    out = asyncio.run(nodes.executor_node(_state("x", intent="PRICING", search_query=None)))
    assert out["search_query"] == "RelayN pricing and plans"
    assert out["tool_data"] is None


def test_executor_prefixes_pricing_query(monkeypatch):
    import agent.nodes as nodes
    captured = {}
    monkeypatch.setattr(nodes, "search_knowledge_base",
                        lambda o, w, q, k=5: captured.setdefault("q", q) or [])

    asyncio.run(nodes.executor_node(_state("x", intent="PRICING", search_query="team plan")))
    assert captured["q"] == "pricing plans team plan"


def test_generator_builds_prompt_and_returns_ai_message(monkeypatch):
    import agent.nodes as nodes

    captured = {}

    class FakeLLM:
        async def ainvoke(self, msgs):
            captured["system"] = msgs[0].content
            return AIMessage(content="Sure — RelayN unifies your channels.")

    monkeypatch.setattr(nodes, "gen_llm", FakeLLM())

    out = asyncio.run(nodes.generator_node(_state("what is relayn?", tool_data="- RelayN unifies channels")))
    assert isinstance(out["messages"][0], AIMessage)
    assert "RelayN" in captured["system"]
    assert "what is relayn?" in captured["system"]
    assert "- RelayN unifies channels" in captured["system"]


def test_generator_handles_missing_tool_data(monkeypatch):
    import agent.nodes as nodes

    class FakeLLM:
        async def ainvoke(self, msgs):
            return AIMessage(content="ok")

    monkeypatch.setattr(nodes, "gen_llm", FakeLLM())
    out = asyncio.run(nodes.generator_node(_state("hi")))
    assert out["messages"][0].content == "ok"
```

`tests/test_handoff.py`:

```python
import asyncio

from langchain_core.messages import HumanMessage


def test_handoff_node_returns_canned_message_and_intent():
    import agent.nodes as nodes
    from agent.prompts import HANDOFF_MESSAGE

    out = asyncio.run(nodes.handoff_node(
        {"messages": [HumanMessage(content="I want a human")], "user_data": {}}
    ))
    assert out["messages"][0].content == HANDOFF_MESSAGE
    assert out["intent"] == "HANDOFF"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nodes.py tests/test_handoff.py -v`
Expected: FAIL — `AttributeError: module 'agent.nodes' has no attribute 'executor_node'`

- [ ] **Step 3: Update imports, then append the nodes**

Replace the import block at the top of `agent/nodes.py` with the full set the
finished module needs:

```python
# agent/nodes.py
import asyncio

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts import (
    DEMO_QUESTIONS,
    GENERATOR_SYSTEM_PROMPT,
    HANDOFF_MESSAGE,
    RELAYN_BUSINESS,
    RELAYN_CONTACT,
    SALES_QUESTIONS,
    format_question,
    render,
)
from agent.state import AgentState
from agent.tools import search_knowledge_base
from config import settings
```

Then append:

```python
# ---------------------------------------------------------------------------
# RAG PATH: EXECUTOR + GENERATOR
# ---------------------------------------------------------------------------

gen_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.7,
    openai_api_key=settings.OPENAI_API_KEY,
)

_DEFAULT_QUERY = {
    "PRICING": "RelayN pricing and plans",
    "PRODUCT_QA": "RelayN features and capabilities",
}


async def executor_node(state: AgentState) -> dict:
    intent = state.get("intent")
    query = state.get("search_query")
    if not query:
        query = _DEFAULT_QUERY.get(intent, "RelayN features and capabilities")
    elif intent == "PRICING":
        query = f"pricing plans {query}"

    ud = state.get("user_data", {})
    chunks = await asyncio.to_thread(
        search_knowledge_base, ud["org_id"], ud["workflow_id"], query
    )
    tool_data = "\n\n".join(f"- {c}" for c in chunks) if chunks else None
    return {"tool_data": tool_data, "search_query": query}


async def generator_node(state: AgentState) -> dict:
    latest_user_query = ""
    for m in reversed(state["messages"]):
        if getattr(m, "type", None) == "human":
            latest_user_query = m.content
            break

    system = render(
        GENERATOR_SYSTEM_PROMPT,
        business_name=RELAYN_BUSINESS["name"],
        persona=RELAYN_BUSINESS["persona"],
        tone=RELAYN_BUSINESS["tone"],
        contact_info=RELAYN_CONTACT,
        tool_data=state.get("tool_data") or "(nothing retrieved)",
        user_query=latest_user_query,
    )
    history = state["messages"][-20:]
    response = await gen_llm.ainvoke([SystemMessage(content=system), *history])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# HANDOFF
# ---------------------------------------------------------------------------

async def handoff_node(state: AgentState) -> dict:
    return {
        "messages": [SystemMessage(content=HANDOFF_MESSAGE)],
        "intent": "HANDOFF",
        "tool_data": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nodes.py tests/test_handoff.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add agent/nodes.py tests/test_nodes.py tests/test_handoff.py
git commit -m "feat: executor, generator, and handoff nodes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 9: Graph assembly

**Files:**
- Create: `agent/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: every node from Tasks 5–8, `agent.state.AgentState`.
- Produces:
  - `route_decision(state: AgentState) -> str` → one of `"demo"|"demo_end"|"sales"|"sales_end"|"handoff"|"executor"|"generator"`
  - `build_workflow() -> StateGraph` — an **uncompiled** `StateGraph` (caller compiles with a checkpointer)

- [ ] **Step 1: Write the failing test — `tests/test_graph.py`**

```python
import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


def test_route_decision_maps_every_intent():
    from agent.graph import route_decision

    cases = {
        "BOOK_DEMO": "demo", "STOP_DEMO": "demo_end",
        "CONTACT_SALES": "sales", "STOP_SALES": "sales_end",
        "HANDOFF": "handoff",
        "PRODUCT_QA": "executor", "PRICING": "executor",
        "GENERAL_CHAT": "generator", None: "generator",
    }
    for intent, node in cases.items():
        assert route_decision({"intent": intent}) == node


def test_demo_flow_persists_step_across_invocations():
    """Router locks the flow; checkpointer carries demo_step turn to turn.

    No LLM is mocked because no LLM is reached: the keyword fast path starts the
    flow, then the active-demo lock holds every following turn.
    """
    import agent.graph as graph_mod

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}

    s1 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="book demo")],
         "user_data": {"user_name": "Sam"}}, cfg))
    assert s1["demo_step"] == 1
    assert "*1/6*" in s1["messages"][-1].content

    s2 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="Acme Corp")]}, cfg))
    assert s2["demo_data"]["business_name"] == "Acme Corp"
    assert s2["demo_step"] == 2

    s3 = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="cancel")]}, cfg))
    assert s3["demo_step"] == 0
    assert s3["intent"] == "STOP_DEMO"


def test_handoff_path_runs(monkeypatch):
    import agent.graph as graph_mod
    from agent.prompts import HANDOFF_MESSAGE

    app = graph_mod.build_workflow().compile(checkpointer=MemorySaver())
    out = asyncio.run(app.ainvoke(
        {"messages": [HumanMessage(content="talk to a human")],
         "user_data": {"user_name": "Sam"}},
        {"configurable": {"thread_id": "t2"}}))
    assert out["messages"][-1].content == HANDOFF_MESSAGE
    assert out["intent"] == "HANDOFF"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent.graph'`

- [ ] **Step 3: Write `agent/graph.py`**

```python
# agent/graph.py
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    demo_end_node,
    demo_node,
    executor_node,
    generator_node,
    handoff_node,
    sales_end_node,
    sales_node,
)
from agent.router import router_node
from agent.state import AgentState

_INTENT_TO_NODE = {
    "BOOK_DEMO": "demo",
    "STOP_DEMO": "demo_end",
    "CONTACT_SALES": "sales",
    "STOP_SALES": "sales_end",
    "HANDOFF": "handoff",
    "PRODUCT_QA": "executor",
    "PRICING": "executor",
}


def route_decision(state: AgentState) -> str:
    return _INTENT_TO_NODE.get(state.get("intent"), "generator")


def build_workflow() -> StateGraph:
    """Uncompiled graph. main.py compiles it with the Redis checkpointer."""
    wf = StateGraph(AgentState)

    wf.add_node("router", router_node)
    wf.add_node("executor", executor_node)
    wf.add_node("generator", generator_node)
    wf.add_node("demo", demo_node)
    wf.add_node("demo_end", demo_end_node)
    wf.add_node("sales", sales_node)
    wf.add_node("sales_end", sales_end_node)
    wf.add_node("handoff", handoff_node)

    wf.add_edge(START, "router")
    wf.add_conditional_edges(
        "router",
        route_decision,
        {
            "demo": "demo",
            "demo_end": "demo_end",
            "sales": "sales",
            "sales_end": "sales_end",
            "handoff": "handoff",
            "executor": "executor",
            "generator": "generator",
        },
    )

    wf.add_edge("executor", "generator")
    wf.add_edge("generator", END)
    wf.add_edge("demo", END)
    wf.add_edge("demo_end", END)
    wf.add_edge("sales", END)
    wf.add_edge("sales_end", END)
    wf.add_edge("handoff", END)

    return wf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: PASS (all tests from Tasks 1–9)

- [ ] **Step 6: Commit**

```bash
git add agent/graph.py tests/test_graph.py
git commit -m "feat: assemble the LangGraph state machine

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 10: Redis checkpointer (ported)

**Files:**
- Create: `db/redis.py`
- Test: `tests/test_redis.py`

**Interfaces:**
- Consumes: `config.settings`.
- Produces:
  - `PruningAsyncRedisSaver` (subclass of `AsyncRedisSaver`) with per-thread size pruning, 30-day TTL, semaphore-guarded ops, and safe `aget` / `aget_tuple` that return `None` on a stale index instead of raising.
  - `init_semantic_cache() -> None`
  - module constants `MAX_MEMORY_PER_THREAD`, `KEEP_LATEST`, `INDEX_PREFIX`, `THREAD_TTL_SECONDS`.

- [ ] **Step 1: Write the failing test — `tests/test_redis.py`**

```python
def test_module_constants_present():
    from db.redis import (
        INDEX_PREFIX,
        KEEP_LATEST,
        MAX_MEMORY_PER_THREAD,
        PruningAsyncRedisSaver,
        THREAD_TTL_SECONDS,
        init_semantic_cache,
    )

    assert KEEP_LATEST >= 1
    assert MAX_MEMORY_PER_THREAD > 0
    assert THREAD_TTL_SECONDS == 60 * 60 * 24 * 30
    assert INDEX_PREFIX
    assert callable(init_semantic_cache)


def test_prune_keeps_at_least_keep_latest(monkeypatch):
    """The pruning loop must never drop below KEEP_LATEST entries even if the
    running total is still over budget."""
    import asyncio

    from db.redis import KEEP_LATEST, MAX_MEMORY_PER_THREAD, PruningAsyncRedisSaver

    entries = [f"checkpoint:t:{i}:{MAX_MEMORY_PER_THREAD}" for i in range(KEEP_LATEST + 3)]
    deleted = []

    class FakeRedis:
        async def lrange(self, *_a):
            return list(entries)

        async def delete(self, key):
            deleted.append(key)

        async def lpop(self, *_a):
            entries.pop(0)

    saver = PruningAsyncRedisSaver.__new__(PruningAsyncRedisSaver)
    saver._my_redis = FakeRedis()
    asyncio.run(saver._prune_checkpoints("t", "idx"))

    assert len(entries) >= KEEP_LATEST
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_redis.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'db.redis'`

- [ ] **Step 3: Write `db/redis.py`**

Copy `D:\Personal\Learnings\Internship\Angan - Copy\angan_services\db\redis.py`
**verbatim**, then change only the logger name:

```python
logger = logging.getLogger("relayn_agents")
```

The full file (for reference — it must match the source byte-for-byte apart from
that one line):

```python
# db/redis.py
import logging
import asyncio
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisCache
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain_core.runnables import RunnableConfig
from config import settings

logger = logging.getLogger("relayn_agents")

MAX_MEMORY_PER_THREAD = 5 * 1024 * 1024  # 5 MB
KEEP_LATEST = 5
INDEX_PREFIX = "checkpoint_index:"
THREAD_TTL_SECONDS = 60 * 60 * 24 * 30   # 30 days inactivity expiry
MAX_CONCURRENT_REDIS_OPS = 50

_redis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REDIS_OPS)


def init_semantic_cache():
    from redis import Redis
    print(f"🧠 Initializing Standard Redis Cache at {settings.REDIS_HOST}")
    sync_client = Redis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
    set_llm_cache(RedisCache(redis_=sync_client))


class PruningAsyncRedisSaver(AsyncRedisSaver):
    def __init__(self, redis_client, *args, **kwargs):
        super().__init__(redis_client=redis_client, *args, **kwargs)
        self._my_redis = redis_client

    async def aput(self, config: RunnableConfig, checkpoint: dict, metadata: dict, new_versions: dict) -> RunnableConfig:
        async with _redis_semaphore:
            updated_config = await super().aput(config, checkpoint, metadata, new_versions)

            thread_id = config["configurable"]["thread_id"]
            checkpoint_id = updated_config["configurable"]["checkpoint_id"]

            checkpoint_key = f"checkpoint:{thread_id}:{checkpoint_id}"
            index_key = f"{INDEX_PREFIX}{thread_id}"

            type_str, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
            size = len(serialized_checkpoint)

            await self._my_redis.rpush(index_key, f"{checkpoint_key}:{size}")

            await self._my_redis.expire(index_key, THREAD_TTL_SECONDS)
            await self._my_redis.expire(checkpoint_key, THREAD_TTL_SECONDS)

            await self._prune_checkpoints(thread_id, index_key)

            return updated_config

    async def aget(self, config: RunnableConfig, *args, **kwargs):
        async with _redis_semaphore:
            try:
                return await super().aget(config, *args, **kwargs)
            except Exception as e:
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                logger.warning(
                    f"⚠️ Checkpoint read failed for {thread_id} — "
                    f"likely TTL expiry mid-conversation. Starting fresh. Error: {e}"
                )
                return None

    async def aget_tuple(self, config: RunnableConfig):
        async with _redis_semaphore:
            try:
                return await super().aget_tuple(config)
            except Exception as e:
                thread_id = config.get("configurable", {}).get("thread_id", "unknown")
                logger.warning(
                    f"⚠️ Checkpoint tuple read failed for {thread_id} — "
                    f"starting fresh thread. Error: {e}"
                )
                return None

    async def _prune_checkpoints(self, thread_id: str, index_key: str):
        try:
            entries = await self._my_redis.lrange(index_key, 0, -1)
            if not entries:
                return

            entries = [e.decode("utf-8") if isinstance(e, bytes) else e for e in entries]
            total_size = sum(int(entry.rsplit(":", 1)[1]) for entry in entries)

            while total_size > MAX_MEMORY_PER_THREAD and len(entries) > KEEP_LATEST:
                oldest = entries.pop(0)
                old_key, old_size_str = oldest.rsplit(":", 1)

                await self._my_redis.delete(old_key)
                await self._my_redis.lpop(index_key)

                total_size -= int(old_size_str)
                logger.info(f"🧹 Pruned checkpoint for {thread_id} ({old_size_str} bytes)")

        except Exception as e:
            logger.error(f"⚠️ Checkpoint Pruning Error: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_redis.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add db/redis.py tests/test_redis.py
git commit -m "feat: port PruningAsyncRedisSaver checkpointer from angan_services

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 11: `service.py` — orchestration, guards, intent collapse

**Files:**
- Create: `service.py`
- Test: `tests/test_service_guard.py`, `tests/test_intent_collapse.py`

**Interfaces:**
- Consumes: `schemas.GenerateReplyRequest`, `schemas.GenerateReplyResponse`, `schemas.FlowCapture`, `clients.supabase_client`, `config.settings`.
- Produces:
  - `COLLAPSE: dict[str, str]` mapping every internal intent → `"RAG"|"ORDER"|"CHAT"`
  - `_fetch_workflow(workflow_id: str) -> dict | None`
  - `async generate_reply(payload: GenerateReplyRequest, agent_app) -> GenerateReplyResponse` — `agent_app` is a compiled LangGraph app exposing `await agent_app.ainvoke(state, config)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_intent_collapse.py`:

```python
def test_collapse_covers_every_internal_intent():
    from service import COLLAPSE

    expected = {
        "BOOK_DEMO": "ORDER", "CONTACT_SALES": "ORDER",
        "PRODUCT_QA": "RAG", "PRICING": "RAG",
        "GENERAL_CHAT": "CHAT", "HANDOFF": "CHAT",
        "STOP_DEMO": "CHAT", "STOP_SALES": "CHAT",
    }
    assert COLLAPSE == expected
    assert set(COLLAPSE.values()) == {"RAG", "ORDER", "CHAT"}
```

`tests/test_service_guard.py`:

```python
import asyncio
import types

from schemas import GenerateReplyRequest


class _SpyApp:
    def __init__(self, final_state):
        self.final_state = final_state
        self.calls = []

    async def ainvoke(self, state, config):
        self.calls.append((state, config))
        return self.final_state


def _payload(**over):
    base = dict(org_id="org-relayn-test", asset_id="a1", asset_type="whatsapp",
                workflow_id="wf-relayn-test", conversation_id="c1",
                message={"body": "hi"}, user_name="Sam")
    base.update(over)
    return GenerateReplyRequest(**base)


def _patch_workflow(monkeypatch, row):
    import service
    monkeypatch.setattr(service, "_fetch_workflow", lambda wid: row)


def test_wrong_org_id_returns_empty_and_skips_graph(monkeypatch):
    import service
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(org_id="someone-else"), app))
    assert out.reply is None and out.intent is None
    assert app.calls == []


def test_wrong_workflow_id_returns_empty_and_skips_graph(monkeypatch):
    import service
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(workflow_id="wf-other"), app))
    assert out.intent is None
    assert app.calls == []


def test_inactive_workflow_returns_empty(monkeypatch):
    import service
    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": False, "workflow_type": "ai_chatbot"})
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(), app))
    assert out.intent is None
    assert app.calls == []


def test_org_mismatch_on_row_returns_empty(monkeypatch):
    import service
    _patch_workflow(monkeypatch, {"organization_id": "different-org",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    app = _SpyApp({})
    out = asyncio.run(service.generate_reply(_payload(), app))
    assert out.intent is None
    assert app.calls == []


def test_happy_path_invokes_graph_with_thread_id_and_collapses_intent(monkeypatch):
    import service
    from langchain_core.messages import AIMessage

    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    final = {"messages": [AIMessage(content="RelayN unifies your channels.")],
             "intent": "PRODUCT_QA", "search_query": "channels"}
    app = _SpyApp(final)

    out = asyncio.run(service.generate_reply(_payload(), app))

    state, config = app.calls[0]
    assert config["configurable"]["thread_id"] == "wf-relayn-test:c1"
    assert state["user_data"]["org_id"] == "org-relayn-test"
    assert out.reply == "RelayN unifies your channels."
    assert out.intent == "RAG"
    assert out.topic == "channels"
    assert out.capture is None
    assert out.handoff_requested is False


def test_demo_completion_produces_capture_and_strips_sentinel(monkeypatch):
    import service
    from langchain_core.messages import SystemMessage

    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    final = {"messages": [SystemMessage(content="All set. Thanks! [DEMO_BOOKED]")],
             "intent": "BOOK_DEMO", "demo_data": {"business_name": "Acme"}}
    out = asyncio.run(service.generate_reply(_payload(), _SpyApp(final)))

    assert "[DEMO_BOOKED]" not in out.reply
    assert out.reply == "All set. Thanks!"
    assert out.intent == "ORDER"
    assert out.capture.type == "demo"
    assert out.capture.data == {"business_name": "Acme"}


def test_handoff_intent_sets_flag(monkeypatch):
    import service
    from langchain_core.messages import SystemMessage

    _patch_workflow(monkeypatch, {"organization_id": "org-relayn-test",
                                  "is_active": True, "workflow_type": "ai_chatbot"})
    final = {"messages": [SystemMessage(content="Bringing in a human.")], "intent": "HANDOFF"}
    out = asyncio.run(service.generate_reply(_payload(), _SpyApp(final)))
    assert out.handoff_requested is True
    assert out.intent == "CHAT"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_service_guard.py tests/test_intent_collapse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'service'`

- [ ] **Step 3: Write `service.py`**

```python
import logging

from langchain_core.messages import HumanMessage

from clients import supabase_client
from config import settings
from schemas import FlowCapture, GenerateReplyRequest, GenerateReplyResponse

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_service_guard.py tests/test_intent_collapse.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add service.py tests/test_service_guard.py tests/test_intent_collapse.py
git commit -m "feat: generate_reply orchestration, guards, intent collapse

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 12: `main.py` — FastAPI app and lifespan

**Files:**
- Create: `main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `service.generate_reply`, `schemas`, `agent.graph.build_workflow`, `db.redis.PruningAsyncRedisSaver`, `db.redis.init_semantic_cache`, `clients` (for a Redis client factory — see step 3).
- Produces: `app` (FastAPI), routes `GET /health`, `POST /generate-reply`. `app.state.agent_app` holds the compiled graph; `app.state.is_ready: bool`.

- [ ] **Step 1: Write the failing test — `tests/test_main.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'main'`

- [ ] **Step 3: Write `main.py`**

```python
import logging
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException

from agent.graph import build_workflow
from config import settings
from db.redis import PruningAsyncRedisSaver, init_semantic_cache
from schemas import GenerateReplyRequest, GenerateReplyResponse
from service import generate_reply

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("relayn_agents")


def _redis_client():
    return aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        health_check_interval=30,
        retry_on_timeout=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_client = _redis_client()
    try:
        init_semantic_cache()
        checkpointer = PruningAsyncRedisSaver(redis_client=redis_client)
        try:
            await checkpointer.setup()
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
        app.state.agent_app = build_workflow().compile(checkpointer=checkpointer)
        app.state.is_ready = True
        logger.info("✅ relayn_agents online")
        yield
    except Exception as e:
        logger.error(f"❌ Startup error: {e}", exc_info=True)
        app.state.is_ready = False
        yield
    finally:
        await redis_client.aclose()
        logger.info("🛑 relayn_agents shutdown")


app = FastAPI(title="relayn_agents", lifespan=lifespan)
app.state.agent_app = None
app.state.is_ready = False


@app.get("/health")
async def health():
    if not app.state.is_ready:
        raise HTTPException(status_code=503, detail="Initializing")
    return {"status": "ok"}


@app.post("/generate-reply", response_model=GenerateReplyResponse)
async def generate_reply_endpoint(payload: GenerateReplyRequest):
    if app.state.agent_app is None:
        raise HTTPException(status_code=503, detail="Graph initializing")
    return await generate_reply(payload, app.state.agent_app)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: PASS (everything)

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: FastAPI app, lifespan graph compile, /generate-reply

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 13: KB scraper script

**Files:**
- Create: `scripts/__init__.py`, `scripts/scrape_relayn_site.py`
- Test: `tests/test_scrape.py`

**Interfaces:**
- Consumes: `clients.supabase_client`, `config.settings`, `httpx`, `bs4.BeautifulSoup`.
- Produces:
  - `SEED_URLS: list[str]`
  - `resolve_urls(client: httpx.Client) -> list[str]` — sitemap first, `SEED_URLS` on any failure
  - `extract_text(html: str) -> str` — main content, nav/footer/script/style stripped, whitespace collapsed
  - `upsert_kb_row(url: str, text: str) -> None` — one `workflow_knowledge_base` row per URL for the pinned workflow
  - `trigger_ingest(client: httpx.Client) -> None` — `POST {RELAYN_SERVICES_URL}/ingest-knowledge-base {workflow_id}`
  - `main() -> None`

- [ ] **Step 1: Write the failing test — `tests/test_scrape.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scrape.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.scrape_relayn_site'`

- [ ] **Step 3: Create `scripts/__init__.py`**

Single newline.

- [ ] **Step 4: Write `scripts/scrape_relayn_site.py`**

```python
"""Offline: scrape relayn.com into the pinned workflow's knowledge base.

Run manually or on a cron. NOT part of the request path.

    python -m scripts.scrape_relayn_site

Steps: resolve URLs (sitemap, else SEED_URLS) -> fetch + extract main text ->
upsert one `workflow_knowledge_base` row per URL for RELAYN_WORKFLOW_ID ->
POST the shared relayn_services /ingest-knowledge-base to chunk + embed.

OI-2 / OI-3 in the spec: confirm `workflow_knowledge_base` has a `source_url`
column and that relayn.com serves /sitemap.xml before the first real run.
"""
import logging
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from clients import supabase_client
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("relayn_agents.scraper")

SITEMAP_URL = "https://relayn.com/sitemap.xml"

SEED_URLS = [
    "https://relayn.com/",
    "https://relayn.com/features",
    "https://relayn.com/pricing",
    "https://relayn.com/integrations",
    "https://relayn.com/faq",
    "https://relayn.com/about",
]

_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "form")


def resolve_urls(client: httpx.Client) -> list[str]:
    try:
        resp = client.get(SITEMAP_URL, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        locs = [
            el.text.strip()
            for el in root.iter()
            if el.tag.endswith("loc") and el.text and el.text.strip()
        ]
        if locs:
            return locs
    except Exception as e:  # network error, 404, malformed XML — all fall back
        logger.warning("sitemap unavailable (%s); using SEED_URLS", e)
    return SEED_URLS


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    text = root.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


def upsert_kb_row(url: str, text: str) -> None:
    table = supabase_client.table("workflow_knowledge_base")
    # Idempotent per source URL: drop this workflow's prior row for the URL,
    # then insert fresh. ingest re-chunks by knowledge_base_id, so replacing
    # the row is enough.
    table.delete().eq("workflow_id", settings.RELAYN_WORKFLOW_ID).eq("source_url", url).execute()
    table.insert(
        {
            "workflow_id": settings.RELAYN_WORKFLOW_ID,
            "organization_id": settings.RELAYN_ORG_ID,
            "source_type": "text",
            "source_url": url,
            "raw_text": text,
            "ingest_status": "pending",
        }
    ).execute()


def trigger_ingest(client: httpx.Client) -> None:
    resp = client.post(
        f"{settings.RELAYN_SERVICES_URL.rstrip('/')}/ingest-knowledge-base",
        json={"workflow_id": settings.RELAYN_WORKFLOW_ID},
        timeout=30,
    )
    resp.raise_for_status()


def main() -> None:
    with httpx.Client(follow_redirects=True, headers={"User-Agent": "relayn-agents-scraper"}) as client:
        urls = resolve_urls(client)
        logger.info("scraping %d urls", len(urls))
        for url in urls:
            try:
                resp = client.get(url, timeout=20)
                resp.raise_for_status()
                text = extract_text(resp.text)
                if len(text) < 80:
                    logger.warning("skipping %s — too little text (%d chars)", url, len(text))
                    continue
                upsert_kb_row(url, text)
                logger.info("stored %s (%d chars)", url, len(text))
            except Exception as e:
                logger.error("failed %s: %s", url, e)
        trigger_ingest(client)
        logger.info("ingest triggered for workflow %s", settings.RELAYN_WORKFLOW_ID)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_scrape.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add scripts/ tests/test_scrape.py
git commit -m "feat: relayn.com KB scraper + shared-ingest trigger

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Task 14: Deployment files and docs

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `docs/DB_MIGRATION.md`, `README.md`
- Test: none automated (documentation + container config)

**Interfaces:** none.

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8002
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8002", "--workers", "2", "--timeout", "120"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  relayn-agents:
    build: .
    ports:
      - "8002:8002"          # beside relayn_services (:8001) and a custom-tier (:8000)
    env_file: .env
    restart: unless-stopped
```

- [ ] **Step 3: Write `docs/DB_MIGRATION.md`**

```markdown
# DB migration — `chat_turns.flow_capture`

The **only** shared-schema change this service needs. Additive and nullable, so
`angan_services` and `relayn_services` are unaffected (both insert explicit
column lists; the lead engine reads only known keys off `select("*")`).

Run once against the shared Supabase project:

```sql
ALTER TABLE public.chat_turns
  ADD COLUMN IF NOT EXISTS flow_capture jsonb;

COMMENT ON COLUMN public.chat_turns.flow_capture IS
  'Set by the dashboard from relayn_agents GenerateReplyResponse.capture on a '
  'demo/sales completion turn: {"type":"demo"|"sales","data":{...}}. Null otherwise.';
```

Rollback (safe — no reader depends on it):

```sql
ALTER TABLE public.chat_turns DROP COLUMN IF EXISTS flow_capture;
```

## Dashboard changes required (out of this repo)

1. Call `POST {relayn_agents}/generate-reply` from the Meta webhook / `runWorkflow`
   path for RelayN's `ai_chatbot` workflow, with the body in `schemas.GenerateReplyRequest`.
2. Stamp `intent`, `topic`, and `flow_capture` (from `capture`) onto the
   `chat_turns` row it already writes.
3. On `handoff_requested: true`, pause automation for that conversation.
```

- [ ] **Step 4: Write `README.md`**

```markdown
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
```

- [ ] **Step 5: Run the full suite one last time**

Run: `pytest -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add Dockerfile docker-compose.yml docs/DB_MIGRATION.md README.md
git commit -m "chore: Dockerfile, compose, migration doc, README

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HUDYkj7kNSZG8J7rPCRcZv"
```

---

## Appendix: Spec coverage map

| Spec section | Task(s) |
| --- | --- |
| §3.1 integration boundary / request-response | 2, 11, 12 |
| §3.2 LangGraph + checkpointer rationale | 9, 10 |
| §3.3 component table | 1–13 |
| §3.4 schemas | 2 |
| §4.1 `AgentState` | 3 |
| §4.2 nodes + routing | 9 |
| §4.3 router (locks, fast paths, LLM fallback) | 5 |
| §4.4 `executor_node` | 8 |
| §4.5 `generator_node` | 8 |
| §4.6 `demo_node` | 6 |
| §4.7 `sales_node` | 7 |
| §4.8 `demo_end` / `sales_end` | 6, 7 |
| §4.9 `handoff_node` | 8 |
| §5.1 intent collapse | 11 |
| §5.2 no lead scoring | (enforced by omission; Global Constraints) |
| §6.1 KB ingestion script | 13 |
| §6.2 KB retrieval | 4 |
| §7.1 `chat_turns.flow_capture` | 14 (`docs/DB_MIGRATION.md`) |
| §8 configuration | 1 |
| §9 `service.py` control flow | 11 |
| §10 testing | every task (TDD) |
| §11 dashboard assumptions | 14 (`docs/DB_MIGRATION.md`) |
| §12 open items | 13 (script comments), 14 (README) |
