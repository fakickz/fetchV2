#!/usr/bin/env python3
"""api_gate: API-Quellen erkennen, prüfen, taggen. Deterministisch, kein LLM.

Läuft zwischen fetch --mode overviews und discover. Liest NUR den Corpus
(kein Crawl) und schreibt OUTPUT/api_sources.json: pro Quelle entweder ein
nutzbares api_source-Objekt oder nichts.

Zwei Wege zur Config:
  Tribe Events  automatisch: Marker im Overview-HTML -> öffentliche REST-API,
                kein Key nötig. Probe per Live-Request.
  Google Cal.   manuell: INPUT/api_sources.csv (Kalender-ID + Key stehen nur
                im JS der Seite, nicht im HTML — nicht auto-detektierbar).
                Gate prüft nur Reachability.

Nur reachable Quellen werden getaggt (DerFunke-Lehre: Marker ohne Endpoint
ist kein Tag). Discover hängt das Tag ans Source-Row; das HTML-Listing läuft
als Fallback weiter, bis das parallele API-Leseskript existiert.
"""
from __future__ import annotations

import argparse
import json
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import eventlib as ev

OUT_API_SOURCES = ev.OUT_DIR / "api_sources.json"
DEFAULT_CONFIG = ev.INPUT_DIR / "api_sources.csv"

TRIBE_PATH = "/wp-json/tribe/events/v1/events"
GCAL_BASE = "https://content.googleapis.com/calendar/v3/calendars"
TIMEOUT = 25
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/18.5 Safari/605.1.15")


def detect_tribe(html: str) -> bool:
    """Tribe-Events-Marker im HTML. Beide Bedingungen: Plugin-Spuren."""
    low = (html or "").lower()
    return ("the-events-calendar" in low
            and ("tribe" in low or "wp-json" in low))


def tribe_endpoint(page_url: str) -> str:
    parts = urllib.parse.urlsplit(ev.norm_url(page_url))
    origin = f"{parts.scheme}://{parts.netloc}"
    return origin + TRIBE_PATH


def gcal_url(calendar_id: str, key: str,
             now: datetime | None = None) -> str:
    """Probe-URL: 1 Event im Fenster gestern -> +90 Tage. Reicht für 200."""
    moment = now or datetime.now(timezone.utc)
    params = urllib.parse.urlencode({
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": "1",
        "timeMin": (moment - timedelta(days=1)).isoformat(),
        "timeMax": (moment + timedelta(days=90)).isoformat(),
        "key": key.strip(),
    })
    return f"{GCAL_BASE}/{urllib.parse.quote(calendar_id.strip(), safe='')}/events?{params}"


def _ssl_context() -> ssl.SSLContext:
    """System-CA-Bündel (certifi), damit HTTPS-Proben nicht an der
    Zertifikatsprüfung scheitern."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover
        return ssl.create_default_context()


def probe_json(url: str, timeout: int = TIMEOUT) -> tuple[Any | None, str]:
    """GET -> (JSON, Fehlertext). Fehlertext leer bei Erfolg."""
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=timeout,
                                     context=_ssl_context()) as response:
            if response.status != 200:
                return None, f"HTTP {response.status}"
            try:
                return json.loads(response.read().decode("utf-8", "replace")), ""
            except (json.JSONDecodeError, ValueError):
                return None, "kein JSON"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:160]


def probe_tribe(endpoint: str) -> tuple[bool, str]:
    data, error = probe_json(endpoint + "?per_page=1")
    if error:
        return False, error
    if isinstance(data, dict) and isinstance(data.get("events"), list):
        return True, ""
    return False, "unerwartete Antwortform"


def probe_gcal(url: str) -> tuple[bool, str]:
    data, error = probe_json(url)
    if error:
        return False, error
    if isinstance(data, dict) and data.get("kind") == "calendar#events":
        return True, ""
    if isinstance(data, dict) and "error" in data:
        return False, f"API-Fehler: {data['error']}"
    return False, "unerwartete Antwortform"


def read_manual_config(path: Path) -> dict[str, dict[str, str]]:
    """INPUT/api_sources.csv -> {norm_url: {component, calendar_id, key}}.

    Nutzt bewusst read_csv_rows statt read_sources: read_sources projiziert
    auf orga/orga_id/url und würde die Config-Spalten verwerfen.
    """
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    for row in ev.read_csv_rows(path):
        url = ev.norm_url(ev.pick(row, "url"))
        if not url:
            continue
        out[url] = {
            "component": ev.pick(row, "component").lower(),
            "calendar_id": ev.pick(row, "calendar_id", "calendarid"),
            "key": ev.pick(row, "key"),
        }
    return out


def check_source(source: dict[str, str],
                 manual: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Eine Quelle: detektieren, Config bauen, Reachability prüfen."""
    url = ev.norm_url(source["url"])
    tag: dict[str, Any] = {
        "orga": source.get("orga", ""),
        "source_url": url,
        "component": "",
        "endpoint": "",
        "reachable": False,
        "checked_at": ev.now_iso(),
        "error": "",
    }
    page = ev.load_page(url)
    if not page or page.get("status") != "ok":
        tag["error"] = "keine Overview-Seite im Corpus"
        return tag
    html = page.get("html") or ""

    manual_cfg = manual.get(url)
    if manual_cfg and manual_cfg["component"] == "gcal":
        if not manual_cfg["calendar_id"] or not manual_cfg["key"]:
            tag.update(component="gcal",
                       error="api_sources.csv unvollständig (calendar_id/key)")
            return tag
        endpoint = gcal_url(manual_cfg["calendar_id"], manual_cfg["key"])
        ok, error = probe_gcal(endpoint)
        tag.update(component="gcal", endpoint=endpoint,
                   reachable=ok, error=error)
        if ok:
            tag["calendar_id"] = manual_cfg["calendar_id"]
            tag["key"] = manual_cfg["key"]
        return tag

    if detect_tribe(html):
        endpoint = tribe_endpoint(url)
        ok, error = probe_tribe(endpoint)
        tag.update(component="tribe", endpoint=endpoint,
                   reachable=ok, error=error)
        return tag

    tag["error"] = "keine API-Signatur im HTML"
    return tag


def run(args: argparse.Namespace) -> int:
    sources = ev.read_sources(Path(args.input))
    if args.limit:
        sources = sources[:args.limit]
    if not sources:
        print(f"Keine Quellen in {args.input}", flush=True)
        return 1
    manual = read_manual_config(Path(args.config))

    results: dict[str, dict[str, Any]] = {}
    for source in sources:
        results[ev.norm_url(source["url"])] = check_source(source, manual)

    ev.write_json(Path(args.output), {
        "generated_at": ev.now_iso(),
        "sources": results,
    })
    tagged = sum(1 for tag in results.values() if tag["reachable"])
    print(f"API-Gate: {tagged}/{len(results)} Quellen mit nutzbarer API",
          flush=True)
    for tag in results.values():
        state = "API:" + tag["component"] if tag["reachable"] else (
            f"kein Tag ({tag['error'][:80]})")
        print(f"  {tag['orga'][:45]:45} {state}", flush=True)
    print(f"-> {args.output}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="API-Quellen erkennen und Reachability prüfen.")
    parser.add_argument("--input", default=str(ev.DEFAULT_SOURCES))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(OUT_API_SOURCES))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
