"""Evaluate the existing RelayN retriever against a JSONL gold set.

Run from the RelayN-Agents directory after configuring .env:

    python -m scripts.evaluate_rag --org ORG_ID --workflow WORKFLOW_ID

The gold set must contain one JSON object per line with ``question`` and
``relevant_context_ids``: a list of chunk IDs from ``workflow_kb_chunks``.
"""
import argparse
import json
from pathlib import Path


def load_gold_set(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict) or "question" not in record:
                raise ValueError(f"Line {line_number} must contain a question object")
            context_ids = record.get("relevant_context_ids")
            if not isinstance(context_ids, list) or not all(isinstance(item, str) for item in context_ids):
                raise ValueError(f"Line {line_number} relevant_context_ids must be a list of strings")
            record["relevant_context_ids"] = context_ids
            records.append(record)
    if not records:
        raise ValueError(f"Gold set is empty: {path}")
    return records


def score_retrieval(retrieved: list[dict], relevant_context_ids: list[str], k: int) -> dict:
    top_k = retrieved[:k]
    relevant = set(relevant_context_ids)
    hits = sum(chunk["id"] in relevant for chunk in top_k)
    precision = hits / k if k else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    return {
        "hits": hits,
        "retrieved_count": len(top_k),
        "relevant_count": len(relevant),
        "precision_at_k": precision,
        "recall_at_k": recall,
    }


def evaluate(records: list[dict], retrieve, k: int) -> list[dict]:
    results = []
    for record in records:
        retrieved = retrieve(record["question"], k)
        results.append({
            "question": record["question"],
            "ground_truth": record.get("ground_truth"),
            "relevant_context_ids": record["relevant_context_ids"],
            "retrieved": retrieved,
            **score_retrieval(retrieved, record["relevant_context_ids"], k),
        })
    return results


def _print_results(results: list[dict], k: int) -> None:
    for index, result in enumerate(results, 1):
        print(f"\n[{index}] {result['question']}")
        print(
            f"P@{k}: {result['precision_at_k']:.3f}  "
            f"R@{k}: {result['recall_at_k']:.3f}  "
            f"hits: {result['hits']}/{result['relevant_count']}"
        )
        print("Retrieved chunks:")
        for chunk_number, chunk in enumerate(result["retrieved"], 1):
            print(f"  {chunk_number}. [{chunk['id']}] {chunk['content']}")

    average_precision = sum(item["precision_at_k"] for item in results) / len(results)
    average_recall = sum(item["recall_at_k"] for item in results) / len(results)
    print(f"\nOverall macro average: Precision@{k}={average_precision:.3f}, Recall@{k}={average_recall:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-set", default="tests/gold_set.jsonl")
    parser.add_argument("--org", help="organization id; defaults to RELAYN_ORG_ID")
    parser.add_argument("--workflow", help="workflow id; defaults to RELAYN_WORKFLOW_ID")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be at least 1")

    from clients import embeddings, supabase_client
    from config import settings

    org_id = args.org or settings.RELAYN_ORG_ID
    workflow_id = args.workflow or settings.RELAYN_WORKFLOW_ID
    records = load_gold_set(args.gold_set)
    rows = (supabase_client.table("workflow_kb_chunks")
        .select("id, content")
        .eq("organization_id", org_id)
        .eq("workflow_id", workflow_id)
        .execute().data or [])
    ids_by_content = {}
    for row in rows:
        ids_by_content.setdefault(row["content"], []).append(row["id"])

    def retrieve(question: str, k: int) -> list[dict]:
        query_embedding = embeddings.embed_query(question)
        response = supabase_client.rpc("match_workflow_kb_chunks", {
            "p_organization_id": org_id,
            "p_workflow_id": workflow_id,
            "query_embedding": query_embedding,
            "match_count": k,
        }).execute()
        retrieved = []
        for row in response.data or []:
            matching_ids = ids_by_content.get(row["content"], [])
            if not matching_ids:
                raise ValueError("Retrieved content could not be mapped to workflow_kb_chunks.id")
            retrieved.append({
                "id": matching_ids[0],
                "content": row["content"],
                "similarity": row.get("similarity"),
            })
        return retrieved

    results = evaluate(records, retrieve, args.k)
    _print_results(results, args.k)


if __name__ == "__main__":
    main()
