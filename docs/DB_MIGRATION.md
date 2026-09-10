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
