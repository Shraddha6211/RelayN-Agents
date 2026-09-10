"""Give config.py values to read before anything imports it.

clients.py builds Supabase and embeddings clients at import time; without
these the suite fails on a missing-env ValidationError before any test runs.
Neither client makes a network call while being constructed.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "test-service-key")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("REDIS_PASSWORD", "")
os.environ.setdefault("REDIS_DB", "0")
os.environ.setdefault("RELAYN_ORG_ID", "org-relayn-test")
os.environ.setdefault("RELAYN_WORKFLOW_ID", "wf-relayn-test")
os.environ.setdefault("RELAYN_SERVICES_URL", "http://localhost:8001")
