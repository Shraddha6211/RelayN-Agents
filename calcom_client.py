"""Small Cal.com v2 client used by the demo-booking flow."""
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from config import settings

CAL_API_BASE = "https://api.cal.com/v2"
CAL_SLOTS_API_VERSION = "2024-09-04"
CAL_BOOKINGS_API_VERSION = "2024-08-13"


class CalComError(Exception):
    """Raised when Cal.com cannot fulfill a request."""


def _headers(api_version: str) -> dict[str, str]:
    if not settings.CAL_API_KEY or not settings.CAL_EVENT_TYPE_ID:
        raise CalComError("Cal.com is not configured")
    return {
        "Authorization": f"Bearer {settings.CAL_API_KEY}",
        "cal-api-version": api_version,
    }


async def get_available_slots(days_ahead: int = 7) -> dict[str, list[str]]:
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=days_ahead)
    params = {
        "eventTypeId": settings.CAL_EVENT_TYPE_ID,
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "timeZone": settings.CAL_TIMEZONE,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{CAL_API_BASE}/slots", headers=_headers(CAL_SLOTS_API_VERSION), params=params
            )
    except httpx.HTTPError as exc:
        raise CalComError(f"slots fetch failed: {exc}") from exc
    if response.status_code != 200:
        raise CalComError(f"slots fetch failed: {response.status_code} {response.text}")
    data = response.json().get("data", {})
    return {date: [slot["start"] for slot in slots] for date, slots in data.items()}


async def create_booking(*, name: str, email: str, start_iso: str, notes: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "start": start_iso,
        "eventTypeId": settings.CAL_EVENT_TYPE_ID,
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": settings.CAL_TIMEZONE,
        },
    }
    if notes:
        body["metadata"] = {"notes": notes}
    headers = {**_headers(CAL_BOOKINGS_API_VERSION), "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{CAL_API_BASE}/bookings", headers=headers, json=body)
    except httpx.HTTPError as exc:
        raise CalComError(f"booking failed: {exc}") from exc
    if response.status_code not in (200, 201):
        raise CalComError(f"booking failed: {response.status_code} {response.text}")
    return response.json().get("data", {})
