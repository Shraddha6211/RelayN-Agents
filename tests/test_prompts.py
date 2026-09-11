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


def test_demo_extraction_prompt_requires_all_slots():
    from agent.prompts import DEMO_EXTRACTION_SYSTEM_PROMPT

    for key in (
        "business_name", "channels", "monthly_volume",
        "contact_name", "contact_info", "preferred_time",
    ):
        assert key in DEMO_EXTRACTION_SYSTEM_PROMPT
    assert "several fields at once" in DEMO_EXTRACTION_SYSTEM_PROMPT
    assert "Return null" in DEMO_EXTRACTION_SYSTEM_PROMPT


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


def test_prompts_keep_unrelated_requests_out_of_scope():
    from agent.prompts import GENERATOR_SYSTEM_PROMPT, ROUTER_SYSTEM_PROMPT

    assert "Do not teach, explain, debug, or generate code" in GENERATOR_SYSTEM_PROMPT
    assert all(language in GENERATOR_SYSTEM_PROMPT for language in ("Java", "JavaScript", "Python", "SQL"))
    assert "outside RelayN" in ROUTER_SYSTEM_PROMPT
    assert "programming question" in ROUTER_SYSTEM_PROMPT
