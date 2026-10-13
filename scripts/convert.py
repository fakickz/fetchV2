#!/usr/bin/env python3
"""convert: OUTPUT/detail.json -> Website-Importformat (OUTPUT/events-final.json).

Rein deterministisch, kein LLM, kein Netzwerk. Events, die der Import nicht
tragen kann (z.B. ohne Startzeit oder volle Anschrift), landen mit Grund in
OUTPUT/events-excluded.json.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import eventlib as ev

IN_DETAIL = ev.OUT_DIR / "detail.json"
OUT_FINAL = ev.OUT_DIR / "events-final.json"
OUT_EXCLUDED = ev.OUT_DIR / "events-excluded.json"

VIENNA = ZoneInfo("Europe/Vienna")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
TYPES = {"workshop", "conference"}


def slugify(value: str) -> str:
    value = value.strip().lower()
    for old, new in {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}.items():
        value = value.replace(old, new)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "event"


def parse_time(value: Any) -> tuple[int, int] | None:
    match = TIME_RE.match(str(value or "").strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    return (hour, minute) if hour < 24 and minute < 60 else None


def is_full_address(value: str) -> bool:
    return bool(re.search(r"\b\d{4}\b", value) and re.search(r"[^\W\d_][\w.\-]*\s+\d+", value))


def strip_city(value: str) -> str:
    return re.sub(r"\s{2,}", " ", re.sub(r"(1\d{3})\s+(Wien|Vienna)\b", r"\1", value)).strip()


def convert(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: set[str] = set()
    slugs: dict[str, int] = {}
    today = datetime.now(VIENNA).date().isoformat()

    for row in rows:
        title = str(row.get("title") or "").strip()
        address = strip_city(str(row.get("venue_address") or "").strip())
        start_time = parse_time(row.get("start_time"))
        event_url = str(row.get("event_url") or "").strip()
        org_id = str(row.get("orga_id") or "").strip()
        try:
            start_date = date.fromisoformat(str(row.get("start_date") or ""))
        except ValueError:
            start_date = None

        reason = ""
        if not title:
            reason = "missing_title"
        elif start_date is None:
            reason = "missing_start_date"
        elif start_time is None:
            reason = "missing_start_time"
        elif not is_full_address(address):
            reason = "missing_full_address"
        elif not UUID_RE.match(org_id):
            reason = "missing_org_id"
        key = event_url.lower().rstrip("/") or f"{title.lower()}|{row.get('start_date')}|{row.get('start_time')}"
        if not reason and key in seen:
            reason = "duplicate_event"
        if reason:
            excluded.append({**row, "exclusion_reason": reason, "excluded_at_stage": "convert"})
            continue
        seen.add(key)

        base = slugify(title)
        slugs[base] = slugs.get(base, 0) + 1
        slug = base if slugs[base] == 1 else f"{base}-{slugs[base]}"

        starts_at = datetime(start_date.year, start_date.month, start_date.day, *start_time, tzinfo=VIENNA)
        ends_at = None
        end_time = parse_time(row.get("end_time"))
        if end_time:
            try:
                end_date = date.fromisoformat(str(row.get("end_date") or "")) if row.get("end_date") else start_date
                ends_at = datetime(end_date.year, end_date.month, end_date.day, *end_time, tzinfo=VIENNA).isoformat()
            except ValueError:
                pass
        language = str(row.get("language") or "").strip()

        events.append({
            "title": title,
            "slug": slug,
            "description": str(row.get("description") or "").strip() or None,
            "type": row.get("category") if row.get("category") in TYPES else "other",
            "status": "published",
            "language_codes": [language] if language else [],
            "starts_at": starts_at.isoformat(),
            "ends_at": ends_at,
            "is_recurring": False,
            "recurrence_rule": None,
            "is_online": False,
            "online_url": event_url or None,
            "location_override": address,
            "cost_type": None,
            "price_min": None,
            "price_max": None,
            "currency": None,
            "registration_url": None,
            "registration_required": row.get("registration_required") or "unbekannt",
            "org_id": org_id,
            "venue_id": None,
            "created_by": None,
            "cover_image_id": None,
            "created_at": today,
        })
    return events, excluded


def main() -> None:
    data = json.loads(IN_DETAIL.read_text(encoding="utf-8"))
    events, excluded = convert(data["events"])
    OUT_FINAL.write_text(json.dumps({"events": events}, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_EXCLUDED.write_text(json.dumps({"events": excluded}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Convert fertig: {len(events)} übernommen, {len(excluded)} ausgeschlossen")
    for item in excluded:
        print(f"  {item['exclusion_reason']}: {item.get('orga')} | {item.get('title')}")
    print(f"-> {OUT_FINAL}\n-> {OUT_EXCLUDED}")


if __name__ == "__main__":
    main()
