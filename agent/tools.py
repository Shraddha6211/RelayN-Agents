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
