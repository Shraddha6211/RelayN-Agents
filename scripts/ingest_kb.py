"""Self-contained knowledge-base ingest for the RelayN agent.

Chunks + embeds text and writes it straight into Supabase
(`workflow_knowledge_base` + `workflow_kb_chunks`) scoped to the pinned
`RELAYN_ORG_ID` / `RELAYN_WORKFLOW_ID`. No dependency on the shared
`relayn_services` deployment.

Typical use:

    # one-time: create the RelayN workflow row and print the .env lines
    python -m scripts.ingest_kb --setup --org <org-uuid>

    # ingest every kb/*.md and kb/*.txt (the default source)
    python -m scripts.ingest_kb

    # ingest specific files or URLs
    python -m scripts.ingest_kb --file kb/relayn.md --file kb/pricing.md
    python -m scripts.ingest_kb --url https://example.com/docs

Re-running replaces this workflow's previously ingested `text` rows and their
chunks (idempotent). PDF-sourced KB rows are left untouched.
"""
import argparse
import glob
import os
import sys
import uuid
import xml.etree.ElementTree as ET

# config.py requires these; resolve real values from flags / .env before import.
_ORG_PLACEHOLDER = "__unset__"
_WF_PLACEHOLDER = "__unset__"


def _bootstrap_env(org: str | None, workflow: str | None) -> None:
    if org:
        os.environ["RELAYN_ORG_ID"] = org
    if workflow:
        os.environ["RELAYN_WORKFLOW_ID"] = workflow
    try:
        from dotenv import dotenv_values

        file_vals = dotenv_values(".env")
    except Exception:
        file_vals = {}
    for key, placeholder in (
        ("RELAYN_ORG_ID", _ORG_PLACEHOLDER),
        ("RELAYN_WORKFLOW_ID", _WF_PLACEHOLDER),
        ("RELAYN_SERVICES_URL", "http://localhost:8001"),
    ):
        if not os.environ.get(key):
            os.environ[key] = file_vals.get(key) or placeholder


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
_STRIP_TAGS = ("script", "style", "nav", "footer", "header", "noscript", "form")


def chunk_text(text: str) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [c.strip() for c in splitter.split_text(text or "") if c.strip()]


def extract_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    lines = [ln.strip() for ln in root.get_text(separator="\n").splitlines() if ln.strip()]
    return "\n".join(lines)


def _sitemap_locs(client, sitemap_url: str) -> list[str]:
    resp = client.get(sitemap_url, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    return [el.text.strip() for el in root.iter() if el.tag.endswith("loc") and el.text and el.text.strip()]


def load_sources(files: list[str], urls: list[str], sitemap: str | None) -> list[tuple[str, str]]:
    """Return (label, text) pairs. Defaults to kb/*.md + kb/*.txt when nothing is passed."""
    out: list[tuple[str, str]] = []

    if not files and not urls and not sitemap:
        files = sorted(glob.glob("kb/*.md") + glob.glob("kb/*.txt"))
        if not files:
            sys.exit("No sources: pass --file / --url / --sitemap, or add kb/*.md files.")

    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            out.append((path, fh.read()))

    if urls or sitemap:
        import httpx

        with httpx.Client(follow_redirects=True, headers={"User-Agent": "relayn-agents-kb"}) as client:
            all_urls = list(urls)
            if sitemap:
                all_urls += _sitemap_locs(client, sitemap)
            for url in all_urls:
                resp = client.get(url, timeout=20)
                resp.raise_for_status()
                out.append((url, extract_text(resp.text)))

    return out


def verify_or_create_workflow(sb, workflow_id: str, org_id: str, *, create: bool, name: str) -> None:
    resp = (
        sb.table("workflows")
        .select("id, organization_id, workflow_type, is_active")
        .eq("id", workflow_id)
        .limit(1)
        .execute()
    )
    row = (resp.data or [None])[0] if resp is not None else None
    if row:
        if row["organization_id"] != org_id:
            sys.exit(f"workflow {workflow_id} belongs to org {row['organization_id']}, not {org_id}")
        if not row.get("is_active"):
            print(f"warning: workflow {workflow_id} is not active")
        return
    if not create:
        sys.exit(
            f"No workflows row for {workflow_id}.\n"
            f"Create it in the dashboard, or re-run with:  --setup --org {org_id}"
        )
    sb.table("workflows").insert(
        {
            "id": workflow_id,
            "organization_id": org_id,
            "name": name,
            "workflow_type": "ai_chatbot",
            "is_active": True,
        }
    ).execute()
    print(f"created workflows row {workflow_id} ('{name}') in org {org_id}")


def clear_existing_text_kb(sb, workflow_id: str) -> int:
    kb_rows = (
        sb.table("workflow_knowledge_base")
        .select("id")
        .eq("workflow_id", workflow_id)
        .eq("source_type", "text")
        .execute()
        .data
        or []
    )
    ids = [r["id"] for r in kb_rows]
    if ids:
        sb.table("workflow_kb_chunks").delete().in_("knowledge_base_id", ids).execute()
        sb.table("workflow_knowledge_base").delete().in_("id", ids).execute()
    return len(ids)


def ingest_source(sb, embeddings, workflow_id: str, org_id: str, label: str, text: str) -> int:
    chunks = chunk_text(text)
    if not chunks:
        print(f"  {label}: no text, skipped")
        return 0

    kb_id = (
        sb.table("workflow_knowledge_base")
        .insert(
            {
                "workflow_id": workflow_id,
                "organization_id": org_id,
                "source_type": "text",
                "raw_text": text,
                "ingest_status": "pending",
            }
        )
        .execute()
        .data[0]["id"]
    )

    vectors = embeddings.embed_documents(chunks)
    rows = [
        {
            "knowledge_base_id": kb_id,
            "workflow_id": workflow_id,
            "organization_id": org_id,
            "content": chunk,
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    for i in range(0, len(rows), 100):
        sb.table("workflow_kb_chunks").insert(rows[i : i + 100]).execute()

    sb.table("workflow_knowledge_base").update({"ingest_status": "indexed"}).eq("id", kb_id).execute()
    print(f"  {label}: {len(chunks)} chunks")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--org", help="RELAYN_ORG_ID (uuid)")
    parser.add_argument("--workflow", help="RELAYN_WORKFLOW_ID (uuid); generated if --setup and omitted")
    parser.add_argument("--setup", action="store_true", help="create the RelayN workflow row, then exit")
    parser.add_argument("--workflow-name", default="RelayN Assistant", help="name for a created workflow row")
    parser.add_argument("--file", action="append", default=[], help="local text/markdown file (repeatable)")
    parser.add_argument("--url", action="append", default=[], help="page URL to fetch + extract (repeatable)")
    parser.add_argument("--sitemap", help="sitemap.xml URL to crawl")
    args = parser.parse_args()

    _bootstrap_env(args.org, args.workflow)

    from clients import embeddings, supabase_client
    from config import settings

    org_id = settings.RELAYN_ORG_ID
    if org_id == _ORG_PLACEHOLDER:
        sys.exit("RELAYN_ORG_ID is not set. Pass --org <uuid> or add it to .env.")

    workflow_id = settings.RELAYN_WORKFLOW_ID
    if workflow_id == _WF_PLACEHOLDER:
        if not args.setup:
            sys.exit("RELAYN_WORKFLOW_ID is not set. Run with --setup --org <uuid> first, or pass --workflow <uuid>.")
        workflow_id = str(uuid.uuid4())

    verify_or_create_workflow(
        supabase_client, workflow_id, org_id, create=args.setup, name=args.workflow_name
    )

    if args.setup:
        print("\nAdd these to .env:\n")
        print(f"RELAYN_ORG_ID={org_id}")
        print(f"RELAYN_WORKFLOW_ID={workflow_id}")
        print("\nThen run:  python -m scripts.ingest_kb")
        return

    sources = load_sources(args.file, args.url, args.sitemap)
    removed = clear_existing_text_kb(supabase_client, workflow_id)
    if removed:
        print(f"replaced {removed} existing text KB row(s)")

    total = 0
    print(f"ingesting {len(sources)} source(s) into workflow {workflow_id}:")
    for label, text in sources:
        total += ingest_source(supabase_client, embeddings, workflow_id, org_id, label, text)
    print(f"done — {total} chunks written to workflow_kb_chunks")


if __name__ == "__main__":
    main()
