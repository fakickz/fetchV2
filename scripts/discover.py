#!/usr/bin/env python3
"""discover: inventarisieren, welche Events eine Übersichtsseite listet.

Bewusst schmal. Dieses Skript entscheidet nur, WELCHE Events es gibt und wo sie
stehen. Es extrahiert keine Facts: die Rückgabe sind Identifikatoren.

Der Grund für die Schmalspur: alles, was hier als Fakt durchginge, würde von
basis/detail blind übernommen und könnte später nicht mehr von einer echten
Extraktion unterschieden werden. Ein falsch zugeordneter Titel auf dieser Stufe
vergiftet alle nachfolgenden Stufen still.

URLs werden deterministisch absolutisiert (urljoin gegen die Quellseite), nicht
vom Modell gebaut.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import eventlib as ev
from costs import UsageTracker, record_usage
from llm import ClientRouter, require_api_key
from progress import Progress, load_or_start_run, record_step

OUT_DISCOVERY = ev.OUT_DIR / "discovery.json"
API_SOURCES = ev.OUT_DIR / "api_sources.json"
DISCOVER_TEXT_CAP = 24000
_ARCHIVE_TITLE_RE = re.compile(
    r"^monatsprogramm(?:\s+\d{1,2}(?:/\d{2,4})?|"
    r"\s+[a-zäöü]+(?:\s+\d{4})?)?$",
    re.IGNORECASE,
)


def load_api_sources(path: Path | None = None) -> dict[str, Any]:
    """Gate-Ergebnis: {source_url: tag}. Fehlt die Datei -> {} (alles HTML)."""
    data = ev.read_json(path or API_SOURCES, {}) or {}
    sources = data.get("sources", {}) if isinstance(data, dict) else {}
    return sources if isinstance(sources, dict) else {}


def normalize_event(
    source_url: str,
    raw: dict[str, Any],
    page_links: list[dict[str, Any]] | None = None,
) -> dict[str, str] | None:
    title = _flat(str(raw.get("title") or ""))
    if not title:
        return None
    if _ARCHIVE_TITLE_RE.fullmatch(title):
        return None
    href = _flat(str(raw.get("url") or ""))
    event_url = ""
    if href and not href.startswith("#"):
        candidate = ev.absolute_url(source_url, href)
        if ev.is_http_url(candidate) and not ev.same_page_link(candidate, source_url):
            candidate = ev.norm_url(candidate)
            if page_links is None:
                event_url = candidate
            else:
                known_urls = {
                    ev.norm_url(str(link.get("href") or "")).lower()
                    for link in page_links if isinstance(link, dict)
                }
                if candidate.lower() in known_urls:
                    event_url = candidate
    return {
        "title": title[:300],
        "event_url": event_url,
        "listed_date_text": _flat(str(raw.get("listed_date_text") or ""))[:120],
    }


def _flat(value: str) -> str:
    return " ".join(value.split()).strip()


def dedupe(events: list[dict[str, str]]) -> list[dict[str, str]]:
    """Dasselbe Event mehrfach auf der Seite (Liste + Teaser) -> einmal."""
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for event in events:
        key = (event["title"].lower(), event["event_url"].lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(event)
    return out


def link_index(page: dict[str, Any], source_url: str, limit: int = 200) -> str:
    """Kompakter Linkindex: Ankertext -> URL. Der bereinigte Text enthält keine
    Ziele mehr, also bekommt das Modell sie hier deterministisch dazu."""
    rows: list[str] = []
    links = page.get("links") if isinstance(page, dict) else []
    for item in links or []:
        if not isinstance(item, dict) or len(rows) >= limit:
            break
        href = ev.norm_url(str(item.get("href") or ""))
        if not href or ev.same_page_link(href, source_url):
            continue
        label = " ".join(str(item.get("text") or "").split())[:80]
        rows.append(f"{label} -> {href}" if label else href)
    return "\n".join(rows)


async def discover_source(
    client: ClientRouter,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    source: dict[str, str],
    semaphore: asyncio.Semaphore,
    tracker: UsageTracker,
    max_tokens: int,
    api_tags: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    url = ev.norm_url(source["url"])
    page = ev.load_page(url)
    api_tags = api_tags or {}
    status = "ok"
    error = ""
    events: list[dict[str, str]] = []
    truncated = False

    if not page or page.get("status") != "ok":
        status = "missing_page" if not page else str(page.get("status") or "error")
        error = "" if page else "Seite nicht im Corpus"
    else:
        text = ev.page_text(page)
        if not text:
            status = "empty"
            error = "Corpus-Seite ohne Text"
        else:
            payload = {
                "source_url": url,
                "orga": source.get("orga", ""),
                "source_text": text[:DISCOVER_TEXT_CAP],
                "page_links": link_index(page, url),
            }
            data, errors = await ev.llm_json(
                client, model, prompt, payload, semaphore,
                schema=schema, max_tokens=max_tokens, tracker=tracker,
            )
            if data is None:
                status = "error"
                error = "; ".join(errors)[:400]
            else:
                raw_events = data.get("events")
                if isinstance(raw_events, list):
                    for item in raw_events:
                        if not isinstance(item, dict):
                            continue
                        links = page.get("links")
                        normalized = normalize_event(
                            url, item, links if isinstance(links, list) else [])
                        if normalized:
                            events.append({**normalized, "orga_id": source.get("orga_id", ""),
                                            "orga": source.get("orga", ""), "source_url": url})
                events = dedupe(events)
                truncated = bool(data.get("possibly_truncated"))
                if not events:
                    status = "empty"
    row = {
        "orga_id": source.get("orga_id", ""),
        "orga": source.get("orga", ""),
        "source_url": url,
        "status": status,
        "event_count": len(events),
        "possibly_truncated": truncated,
        "error": error,
        # Gate-Tag, kein LLM-Urteil. Solange das API-Leseskript fehlt, läuft
        # das HTML-Listing als Fallback weiter — das Tag ist nur Routing-Info.
        "api_source": api_tags.get(url),
    }
    return row, events


async def run(args: argparse.Namespace) -> int:
    sources = ev.read_sources(Path(args.input))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        print(f"Keine Quellen in {args.input}", flush=True)
        return 1
    model = ev.model_for("DISCOVER", args.model)
    prompt = ev.load_prompt("discover")
    schema = ev.load_schema("discover")
    require_api_key(model)
    await ev.probe_llm(model)

    state = load_or_start_run()
    started = time.monotonic()
    semaphore = asyncio.Semaphore(max(1, args.jobs))
    client = ClientRouter()
    tracker = UsageTracker()
    print(f"Discover: {len(sources)} Übersichten, Modell {model}", flush=True)

    progress_done = 0
    rows: list[dict[str, Any]] = []
    all_events: list[dict[str, str]] = []
    api_tags = load_api_sources()
    progress = Progress("Discover", len(sources), state["started_at"], parallelism=args.jobs)

    async def worker(source: dict[str, str]) -> None:
        nonlocal progress_done
        unit_started = time.monotonic()
        row, events = await discover_source(
            client, model, prompt, schema, source, semaphore, tracker, args.max_tokens,
            api_tags,
        )
        rows.append(row)
        all_events.extend(events)
        progress_done += 1
        progress.update(progress_done, time.monotonic() - unit_started)

    await asyncio.gather(*(worker(source) for source in sources))
    progress.finish()
    await client.close()

    rows.sort(key=lambda row: (row["orga"], row["source_url"]))
    payload = {
        "generated_at": ev.now_iso(),
        "model": model,
        "sources": rows,
        "events": all_events,
    }
    ev.write_json(Path(args.output), payload)
    record_usage(state, tracker)
    record_step(state, "discover", time.monotonic() - started)

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    with_url = sum(1 for event in all_events if event["event_url"])
    print(f"Discover fertig: {len(all_events)} Events aus {len(rows)} Seiten "
          f"({with_url} mit eigener Seite, {len(all_events) - with_url} inline)", flush=True)
    print("Status: " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())), flush=True)
    flagged = [row for row in rows if row["possibly_truncated"]]
    if flagged:
        print(f"Trunkierungsverdacht bei {len(flagged)} Seiten: "
              + ", ".join(row["orga"] for row in flagged[:8]), flush=True)
    api_rows = [row for row in rows
                if isinstance(row.get("api_source"), dict)
                and row["api_source"].get("reachable")]
    if api_rows:
        print("API-Quellen: " + ", ".join(
            f"{row['orga']} ({row['api_source']['component']})"
            for row in api_rows), flush=True)
    for row in rows:
        if row["status"] not in {"ok", "empty"}:
            print(f"  {row['orga']}: {row['status']} {row['error'][:160]}", flush=True)
    print(tracker.render("Discover LLM-Nutzung"), flush=True)
    print(f"-> {args.output}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Events auf Übersichtsseiten inventarisieren.")
    parser.add_argument("--input", default=str(ev.DEFAULT_SOURCES))
    parser.add_argument("--output", default=str(OUT_DISCOVERY))
    parser.add_argument("--model", default=None, help="überschreibt MODEL_DISCOVER")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=6000)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
