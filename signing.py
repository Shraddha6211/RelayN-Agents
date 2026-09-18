"""Verify requests signed by relayn_gateway.

Scheme (identical in relayn_gateway/src/signing.ts): HMAC-SHA256 over
"<timestamp>.<raw body>", sent as x-relayn-signature: sha256=<hex> with
x-relayn-timestamp in unix seconds, rejected beyond 300 seconds of skew.
"""
import hashlib
import hmac
import time

from fastapi import HTTPException, Request

from config import settings

MAX_SKEW_SECONDS = 300


def expected_signature(secret: str, timestamp: str, raw_body: bytes) -> str:
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + raw_body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def is_valid_signature(
    secret: str,
    timestamp: str | None,
    signature: str | None,
    raw_body: bytes,
    now: int | None = None,
) -> bool:
    # An unset secret must never verify: the signature of "" is computable by anyone.
    if not secret or not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except ValueError:
        return False
    current = int(time.time()) if now is None else now
    if abs(current - ts) > MAX_SKEW_SECONDS:
        return False
    return hmac.compare_digest(expected_signature(secret, timestamp, raw_body), signature)


async def require_gateway_signature(request: Request) -> None:
    """FastAPI dependency. Starlette caches the body, so the route still parses it."""
    raw = await request.body()
    if not is_valid_signature(
        settings.GATEWAY_HMAC_SECRET,
        request.headers.get("x-relayn-timestamp"),
        request.headers.get("x-relayn-signature"),
        raw,
    ):
        raise HTTPException(status_code=401, detail="invalid signature")
