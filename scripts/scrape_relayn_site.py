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
