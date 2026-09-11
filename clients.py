from langchain_openai import OpenAIEmbeddings
from supabase import Client, create_client
from supabase.client import ClientOptions

from config import settings

# 1536 dims — must match vector(1536) on public.workflow_kb_chunks.
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=settings.OPENAI_API_KEY,
)

supabase_client: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SERVICE_KEY,
    options=ClientOptions(auto_refresh_token=False, persist_session=False),
)
