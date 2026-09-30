def test_demo_turn_prompt_requires_all_slots_and_reply_intents():
    from agent.prompts import DEMO_TURN_SYSTEM_PROMPT

    for key in (
        "contact_name", "contact_info",
        "reply_intent", "side_question_query", "language",
    ):
        assert key in DEMO_TURN_SYSTEM_PROMPT
    for intent in ("ANSWER", "CONFIRM", "DENY", "QUESTION", "SKIP", "OTHER"):
        assert intent in DEMO_TURN_SYSTEM_PROMPT
    assert "Return null" in DEMO_TURN_SYSTEM_PROMPT


def test_demo_reply_writer_prompt_keeps_it_conversational():
    from agent.prompts import DEMO_COMPOSER_SYSTEM_PROMPT

    assert "RECENT CHAT" in DEMO_COMPOSER_SYSTEM_PROMPT
    assert "BRIEF" in DEMO_COMPOSER_SYSTEM_PROMPT
    assert "at most one thing" in DEMO_COMPOSER_SYSTEM_PROMPT
    assert "No lists" in DEMO_COMPOSER_SYSTEM_PROMPT


def test_every_reply_prompt_mirrors_the_users_script():
    from agent.prompts import DEMO_COMPOSER_SYSTEM_PROMPT, GENERATOR_SYSTEM_PROMPT, MIRROR_SCRIPT_RULE

    assert "Devanagari only if the customer writes in Devanagari" in MIRROR_SCRIPT_RULE
    assert "romanized Nepali" in MIRROR_SCRIPT_RULE
    assert "Never answer Hindi in Nepali" in MIRROR_SCRIPT_RULE
    assert MIRROR_SCRIPT_RULE in DEMO_COMPOSER_SYSTEM_PROMPT
    assert MIRROR_SCRIPT_RULE in GENERATOR_SYSTEM_PROMPT


def test_demo_turn_prompt_separates_script_and_allows_unknown():
    from agent.prompts import DEMO_TURN_SYSTEM_PROMPT

    assert "Nepali (romanized)" in DEMO_TURN_SYSTEM_PROMPT
    assert "Nepali (Devanagari)" in DEMO_TURN_SYSTEM_PROMPT
    assert '"unknown"' in DEMO_TURN_SYSTEM_PROMPT


def test_sales_extraction_prompt_requires_all_slots():
    from agent.prompts import SALES_EXTRACTION_SYSTEM_PROMPT

    for key in ("need", "company", "contact_info"):
        assert key in SALES_EXTRACTION_SYSTEM_PROMPT
    assert "several fields at once" in SALES_EXTRACTION_SYSTEM_PROMPT
    assert "Return null" in SALES_EXTRACTION_SYSTEM_PROMPT


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


def test_format_question_open():
    from agent.prompts import SALES_QUESTIONS, format_question

    assert format_question(SALES_QUESTIONS[0], 1, 3).startswith("*1/3*")


def test_generator_prompt_has_expected_slots():
    from agent.prompts import GENERATOR_SYSTEM_PROMPT

    for token in ("{{BUSINESS_NAME}}", "{{PERSONA}}", "{{TONE}}",
                  "{{CONTACT_INFO}}", "{{TOOL_DATA}}", "{{USER_QUERY}}"):
        assert token in GENERATOR_SYSTEM_PROMPT


def test_prompts_keep_unrelated_requests_out_of_scope():
    from agent.prompts import GENERATOR_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT

    assert "Do not teach, explain, debug, or generate code" in GENERATOR_SYSTEM_PROMPT
    assert all(language in GENERATOR_SYSTEM_PROMPT for language in ("Java", "JavaScript", "Python", "SQL"))
    assert "outside RelayN" in ROUTER_SYSTEM_PROMPT
    assert "programming question" in ROUTER_SYSTEM_PROMPT
