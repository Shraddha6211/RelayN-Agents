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

DEMO_EXTRACTION_SYSTEM_PROMPT = """You collect information for scheduling a RelayN product demo.

Extract every value present in the user's latest message for these fields:
- business_name: the business or company name
- channels: channels the business wants to manage; use the listed option title when clear
- monthly_volume: monthly customer-message volume; use the listed option title when clear
- contact_name: the person the demo invite should address
- contact_info: an email address or phone number
- preferred_time: preferred day or time for the demo

The latest message may answer several fields at once, regardless of the question previously
asked. Return null for fields not stated clearly in the latest message. Never infer or invent a
value from the current state. Existing values are supplied only as context so you can recognize
new information; the caller decides how to merge the result.

Available channels: WhatsApp, Instagram, Facebook, All of them.
Available monthly volumes: Under 500, 500 to 2k, 2k to 10k, 10k or more.
"""

SALES_EXTRACTION_SYSTEM_PROMPT = """You collect information for connecting a RelayN prospect with sales.

Extract every value present in the user's latest message for these fields:
- need: what the prospect wants help with or is looking to accomplish
- company: the company name and any team size or company detail they provide
- contact_info: an email address or phone number

The latest message may answer several fields at once, regardless of the question previously
asked. Return null for fields not stated clearly in the latest message. Never infer or invent a
value from the current state. Existing values are supplied only as context so the caller can
merge the result; keep the existing company field combined if the user provides a company name
and team size together.
"""

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
- GENERAL_CHAT — greetings, thanks, small talk, or anything outside RelayN.
                 This includes requests to teach or troubleshoot programming
                 languages, frameworks, homework, general coding, or unrelated
                 products and services.

Rules:
- Judge on meaning, not wording. Users may write in any language or romanized script.
- A question is PRODUCT_QA or PRICING. Wanting to proceed is BOOK_DEMO or CONTACT_SALES.
- Only classify a message as PRODUCT_QA or PRICING when it is specifically about RelayN.
- Do not treat a programming question, including Java, JavaScript, Python, or SQL,
  as a RelayN question merely because it mentions a technical term.
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
- SCOPE GATE: You only help with RelayN, including RelayN's product capabilities,
    supported channels, pricing, onboarding, demos, sales, and existing-account
    support. Do not teach, explain, debug, or generate code for Java, JavaScript,
    Python, SQL, or any other programming language or unrelated topic.
- If the user asks for anything outside RelayN, reply briefly: "I can help with
    RelayN's product, pricing, demos, or support. I can't help with that topic."
    Do not answer any part of the out-of-scope request, provide examples, or offer
    a tutorial. Keep this boundary response in the user's language when clear.
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
