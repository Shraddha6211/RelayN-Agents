# agent/scheduling.py
"""Demo-slot rules, independent of Cal.com's own availability settings.

Cal.com's event may open earlier than we want to offer, so the bot only offers
what it promises: Monday to Friday, 15-minute slots starting 10:30 am to 4:45 pm
(the last one ends at 5 pm), in CAL_TIMEZONE.
"""
from datetime import date, datetime, time, timedelta
from typing import Iterable
from zoneinfo import ZoneInfo

from config import settings

FIRST_START = time(10, 30)
LAST_START = time(16, 45)


def tz() -> ZoneInfo:
    return ZoneInfo(settings.CAL_TIMEZONE)


def now() -> datetime:
    return datetime.now(tz())


def within_hours(t: time) -> bool:
    return FIRST_START <= t <= LAST_START


def next_weekday(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def plan_day(requested: date | None, at: time | None, current: datetime) -> tuple[date, bool]:
    """The day to look for slots on, and whether it moved off a weekend.

    No date given: today, or the next weekday if the requested time has already
    passed today. Dates in the past become today.
    """
    today = current.date()
    day = requested or today
    if requested is None and at is not None and datetime.combine(today, at, current.tzinfo) <= current:
        day = today + timedelta(days=1)
    day = max(day, today)
    moved = requested is not None and requested.weekday() >= 5
    return next_weekday(day), moved


def open_slots(starts: Iterable[str]) -> list[datetime]:
    """Cal.com slot starts (ISO strings) cut down to the hours we offer, in order."""
    local = (datetime.fromisoformat(s).astimezone(tz()) for s in starts)
    return sorted(s for s in local if s.weekday() < 5 and within_hours(s.time()))


def closest(slots: list[datetime], target: time, exclude: Iterable[datetime], count: int) -> list[datetime]:
    """The `count` slots nearest to `target` that are not in `exclude`, in time order."""
    skip = set(exclude)
    target_min = target.hour * 60 + target.minute
    nearest = sorted(
        (s for s in slots if s not in skip),
        key=lambda s: (abs(s.hour * 60 + s.minute - target_min), s),
    )
    return sorted(nearest[:count])


def fmt_day(day: date) -> str:
    return f"{day:%A}, {day.day} {day:%B}"


def fmt_time(slot: datetime) -> str:
    return f"{slot.hour % 12 or 12}:{slot:%M} {'am' if slot.hour < 12 else 'pm'}"
