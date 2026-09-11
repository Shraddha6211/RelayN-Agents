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
