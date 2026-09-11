import json

import pytest


def test_load_gold_set_reads_one_json_object_per_line(tmp_path):
    from scripts.evaluate_rag import load_gold_set

    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({"question": "q", "relevant_context_ids": ["chunk-id"]}) + "\n", encoding="utf-8")

    assert load_gold_set(path) == [{"question": "q", "relevant_context_ids": ["chunk-id"]}]


def test_score_retrieval_calculates_precision_and_recall_at_k():
    from scripts.evaluate_rag import score_retrieval

    result = score_retrieval(
        [{"id": "id-1"}, {"id": "id-miss"}, {"id": "id-2"}],
        ["id-1", "id-2"],
        k=3,
    )

    assert result["hits"] == 2
    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == 1.0


def test_evaluate_calls_retriever_and_keeps_chunks():
    from scripts.evaluate_rag import evaluate

    calls = []

    def retrieve(question, k):
        calls.append((question, k))
        return [{"id": "relevant", "content": "Relevant chunk"}, {"id": "other", "content": "Other chunk"}]

    results = evaluate(
        [{"question": "Where?", "ground_truth": "There.", "relevant_context_ids": ["relevant"]}],
        retrieve,
        k=2,
    )

    assert calls == [("Where?", 2)]
    assert results[0]["retrieved"][0]["id"] == "relevant"
    assert results[0]["precision_at_k"] == 0.5
    assert results[0]["recall_at_k"] == 1.0


def test_evaluate_can_retrieve_the_same_chunk_for_multiple_questions():
    from scripts.evaluate_rag import evaluate

    def retrieve(_question, _k):
        return [{"id": "shared-id", "content": "Shared chunk"}]

    records = [
        {"question": "q1", "relevant_context_ids": ["shared-id"]},
        {"question": "q2", "relevant_context_ids": ["shared-id"]},
    ]

    results = evaluate(records, retrieve, k=1)

    assert [result["hits"] for result in results] == [1, 1]
