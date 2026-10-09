#!/usr/bin/env python3
"""basis: Basisinformationen eines Events bestimmen (oberste Priorität).

Kette pro Event:

    identity  (Titel, Startdatum, Startzeit, Event_URL)
        |
    GATE 1    past      deterministisch
        |
    venue     (Anschrift, Venue-Name)
        |
    GATE 2    vienna    deterministisch

Zwei Entscheidungen absichtlich ausserhalb dieses Skripts und ausserhalb jedes
LLMs:

1. Die Gates sind determinisch. Ein LLM bringt hier nur Varianz, keine Info.
2. Past schliesst nur bei gesichertem Datum aus (unklar = durchlassen).
   Wien lässt nur eine gesicherte Wiener Adresse durch: Nicht-Wien geht nach
   basis-excluded.csv, alles Unklare (z. B. keine Adresse) nach
   basis-review.csv zur manuellen Prüfung, nicht nach basis.json.

Keine Zitatpflicht: das Modell entscheidet, Werte werden nur auf Format
geprüft. Erschlossene Adressen sind mit venue_address_source = "inferred"
markiert.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import time
from pathlib import Path
from typing import Any

import eventlib as ev
from costs import UsageTracker, record_usage
from llm import ClientRouter, require_api_key
from progress import Progress, load_or_start_run, record_step

OUT_DISCOVERY = ev.OUT_DIR / "discovery.json"
OUT_BASIS = ev.OUT_DIR / "basis.json"
OUT_EXCLUDED = ev.OUT_DIR / "basis-excluded.csv"
OUT_REVIEW = ev.OUT_DIR / "basis-review.csv"

EXCLUDED_FIELDS = [
    "orga_id", "orga", "source_url", "event_url", "listed_title", "title",
    "start_date", "venue_address", "venue_address_source", "venue_confidence", "exclusion_reason",
    "excluded_at_stage", "listed_date_text", "note",
]


def attach(target: dict[str, Any], data: dict[str, Any] | None, errors: list[str],
           fields: tuple[str, ...], prefix: str) -> list[str]:
    """Werte des Modells übernehmen, nur Format prüfen. Gibt Fehler zurück.

    Keine Zitatpflicht: das Modell entscheidet. Datum wird auf ISO, Zeit auf
    HH:MM gebracht; Unparsebares wird geleert. Fehlt das Jahr, gilt das
    aktuelle Jahr.
    """
    if data is None:
        for field in fields:
            target[field] = ""
            target[f"confidence_{field}"] = "none"
        return [f"{prefix}: {'; '.join(errors)[:300]}"] if errors else [f"{prefix}: keine Antwort"]
    problems: list[str] = []
    year = str(ev.current_date().year)
    for field in fields:
        value = str(data.get(field) or "").strip()
        if field == "start_date" and value:
            value = value if ev.iso_date(value) else ev.parse_de_date(value, year)
        elif field == "start_time" and value:
            value = value if re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value) else ev.parse_de_time(value)
        if not value:
            if str(data.get(field) or "").strip():
                problems.append(f"{prefix}.{field}: Format nicht lesbar")
            target[field] = ""
            target[f"confidence_{field}"] = "none"
            continue
        target[field] = value
        target[f"confidence_{field}"] = "llm"
    return problems


def run_vienna_gate(record: dict[str, Any]) -> tuple[bool, str]:
    """Nur gesichert Wien kommt durch. Nicht-Wien wird ausgeschlossen, Unklares
    (z. B. keine Adresse) geht zur manuellen Prüfung in eine eigene Datei."""
    verdict = ev.is_vienna_address(record.get("venue_address", ""))
    if verdict is False:
        return False, "outside_vienna"
    if verdict is None:
        return False, "needs_review"
    return True, ""


def reconcile_event_url(record: dict[str, Any], supplied: str, resolved: str) -> list[str]:
    """identity darf eine Entdeckungs-URL bestätigen oder verwerfen, nie ersetzen.

    event_url ist eine Identifikation aus der Entdeckungsstufe. identity kann
    keine Seite abrufen und weiss nicht, was hinter einer URL liegt. Ersetzen
    hat hier die Ausstellung tatsächlich auf die Seite ihrer Eröffnung gebogen:
    beide Records lasen dieselbe Seite, teilten sich die Beschreibung und
    bekamen einander fremde Enddaten. Abweichungen bleiben deshalb Vermerk,
    der Datensatz behält die entdeckte URL.
    """
    supplied = ev.norm_url(supplied or "")
    resolved = ev.norm_url(resolved or "")
    if not supplied:
        return []
    if not resolved:
        record["event_url_dropped"] = True
        return ["identity: Detail-URL vom Modell verworfen"]
    if resolved == supplied:
        return []
    return [f"identity: Modell nannte abweichende URL {resolved[:120]}"]


def apply_venue(record: dict[str, Any], data: dict[str, Any] | None,
                errors: list[str], source_text: str) -> list[str]:
    """Venue-Antwort prüfen und übernehmen. Rein, damit testbar.

    Das Schema heisst die Felder address/venue_name, der Datensatz führt sie als
    venue_address/venue_name. Diese Übersetzung passiert ausschliesslich hier.

    Keine Zitatpflicht: das Modell entscheidet. Eine unvollständige Anschrift
    bleibt erhalten und wird nur vermerkt.

    Ein erklärtes "online" wird nur geglaubt, wenn sonst nichts gefunden wurde:
    ein Modell, das online behauptet und gleichzeitig eine Strasse zitiert, hat
    die Strasse gewonnen.
    """
    problems: list[str] = []
    if data is None:
        record["venue_address"] = ""
        record["venue_name"] = ""
        record["venue_address_source"] = ""
        record["venue_confidence"] = "none"
        record["confidence_venue_address"] = "none"
        record["confidence_venue_name"] = "none"
        return [f"venue: {'; '.join(errors)[:300]}" if errors else "venue: keine Antwort"]

    address = " ".join(str(data.get("address") or "").split()).strip()
    venue_name = " ".join(str(data.get("venue_name") or "").split()).strip()
    declared = str(data.get("confidence") or "").strip().lower()

    address_ok = bool(address)
    name_ok = bool(venue_name)
    if address_ok and not ev.is_full_address(address):
        problems.append("venue.address: keine volle Anschrift (Straße+Nr+PLZ+Ort)")

    source = str(data.get("address_source") or "").strip().lower()
    record["venue_address"] = address
    record["venue_address_source"] = source if address and source in {"page", "inferred"} else ""
    record["venue_name"] = venue_name
    record["confidence_venue_address"] = "llm" if address_ok else "none"
    record["confidence_venue_name"] = "llm" if name_ok else "none"
    if address_ok:
        record["venue_confidence"] = "high"
    elif name_ok:
        record["venue_confidence"] = "venue_only"
    elif declared == "online":
        record["venue_confidence"] = "online"
    else:
        record["venue_confidence"] = "none"
    return problems


async def extract_one(
    client: ClientRouter,
    semaphore: asyncio.Semaphore,
    event: dict[str, Any],
    cfg: dict[str, Any],
    tracker: UsageTracker | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "orga_id": event.get("orga_id", ""),
        "orga": event.get("orga", ""),
        "source_url": event.get("source_url", ""),
        "event_url": event.get("event_url", ""),
        "listed_date_text": event.get("listed_date_text", ""),
        # Locator aus dem Listing, kein Faktum. Bewusst getrennt von "title",
        # damit ein unverifizierter Titel nie als Extraktion durchgeht.
        "listed_title": event.get("title", ""),
        "title": "",
        "start_date": "",
        "start_time": "",
        "venue_address": "",
        "venue_name": "",
        "venue_address_source": "",
        "venue_confidence": "none",
        "field_errors": [],
    }
    llm_kwargs = {"tracker": tracker} if tracker is not None else {}

    text, source = ev.event_text(event, cfg["text_source"])
    record["text_source"] = source
    if not text:
        record["field_errors"].append("text: kein Quelltext im Corpus")
        return record

    identity_fields = ("title", "start_date", "start_time")
    identity_payload = {
        "event_title_hint": event.get("title", ""),
        "listed_date_text": event.get("listed_date_text", ""),
        "event_url": event.get("event_url", ""),
        "orga": event.get("orga", ""),
        "source_url": event.get("source_url", ""),
        "jsonld_hint": ev.jsonld_hint_for(event) if cfg["jsonld"] else "",
        "default_year": str(ev.current_date().year),
        "source_text": text,
    }
    data, errors = await ev.llm_json(
        client, cfg["identity"], cfg["prompts"]["identity"], identity_payload, semaphore,
        schema=cfg["schemas"]["identity"], max_tokens=cfg["max_tokens"], **llm_kwargs,
    )
    record["field_errors"].extend(attach(record, data, errors, identity_fields, "identity"))
    if data is not None:
        record["field_errors"].extend(reconcile_event_url(
            record, str(event.get("event_url") or ""), str(data.get("event_url") or "")))

    # War die Detailseite leer für dieses Event, ist sie vermutlich die falsche
    # Seite (Discovery-Irrtum). Einmal mit Übersichtstext neu versuchen, statt
    # das Event aufgrund einer fremden Seite zu verlieren.
    if source == "detail" and not record.get("title") and not record.get("start_date"):
        fallback, fallback_source = ev.event_text({**event, "event_url": ""}, cfg["text_source"])
        if fallback and fallback != text:
            fdata, ferr = await ev.llm_json(
                client, cfg["identity"], cfg["prompts"]["identity"],
                {**identity_payload, "source_text": fallback}, semaphore,
                schema=cfg["schemas"]["identity"], max_tokens=cfg["max_tokens"], **llm_kwargs,
            )
            trial: dict[str, Any] = {}
            trial_errors = attach(trial, fdata, ferr, identity_fields, "identity")
            if trial.get("title") or trial.get("start_date"):
                for field in identity_fields:
                    record[field] = trial[field]
                    record[f"confidence_{field}"] = trial[f"confidence_{field}"]
                record["field_errors"] = [
                    e for e in record["field_errors"] if not e.startswith("identity")
                ] + trial_errors
                text, source = fallback, fallback_source
                record["text_source"] = fallback_source
            else:
                record["field_errors"].append("identity: Fallback auf Übersicht ohne Ertrag")

    if ev.is_past(record.get("start_date", "")) is True:
        record["exclusion_reason"] = "past_event"
        record["excluded_at_stage"] = "past"
        return record

    hint = ev.jsonld_hint_for(event) if cfg["jsonld"] else ""
    venue_payload = {
        "title": record.get("title", ""),
        "start_date": record.get("start_date", ""),
        "orga": record.get("orga", ""),
        "event_url": record.get("event_url", ""),
        "source_url": record.get("source_url", ""),
        "jsonld_hint": hint,
        "source_text": text,
    }
    vdata, verr = await ev.llm_json(
        client, cfg["venue"], cfg["prompts"]["venue"], venue_payload, semaphore,
        schema=cfg["schemas"]["venue"], max_tokens=cfg["max_tokens"], **llm_kwargs,
    )
    record["field_errors"].extend(apply_venue(record, vdata, verr, text))

    ok, reason = run_vienna_gate(record)
    if not ok:
        record["exclusion_reason"] = reason
        record["excluded_at_stage"] = "location"
        return record
    return record


async def run(args: argparse.Namespace) -> int:
    discovery = ev.read_json(Path(args.discovery), {}) or {}
    events = discovery.get("events", []) if isinstance(discovery, dict) else []
    if args.limit:
        events = events[:args.limit]
    if not events:
        print(f"Keine Events in {args.discovery}", flush=True)
        return 1

    text_source = args.text_source or str(
        ev.load_input_config().get("basis_text_source") or "raw")
    if text_source not in ev.TEXT_SOURCES:
        print(f"Unbekannte Textquelle: {text_source} (raw|content)", flush=True)
        return 1

    cfg = {
        "identity": ev.model_for("IDENTITY", args.model_identity),
        "venue": ev.model_for("VENUE", args.model_venue),
        "prompts": {name: ev.load_prompt(name) for name in ("identity", "venue")},
        "schemas": {name: ev.load_schema(name) for name in ("identity", "venue")},
        "jsonld": not args.no_jsonld,
        "text_source": text_source,
        "max_tokens": args.max_tokens,
    }
    for role in ("identity", "venue"):
        require_api_key(cfg[role])
    await ev.probe_llm(cfg["venue"])
    if cfg["identity"] != cfg["venue"]:
        await ev.probe_llm(cfg["identity"])

    state = load_or_start_run()
    started = time.monotonic()
    semaphore = asyncio.Semaphore(max(1, args.jobs))
    client = ClientRouter()
    tracker = UsageTracker()
    print(f"Basis: {len(events)} Events | text={text_source} identity={cfg['identity']} venue={cfg['venue']}", flush=True)

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    done = 0
    progress = Progress("Basis", len(events), state["started_at"], parallelism=args.jobs)

    async def worker(event: dict[str, Any]) -> None:
        nonlocal done
        unit_started = time.monotonic()
        try:
            record = await extract_one(client, semaphore, event, cfg, tracker)
        except Exception as exc:
            record = {**event, "field_errors": [f"{type(exc).__name__}: {exc}"],
                      "text_source": "error"}
        reason = record.get("exclusion_reason")
        (review if reason == "needs_review" else excluded if reason else kept).append(record)
        done += 1
        progress.update(done, time.monotonic() - unit_started)

    await asyncio.gather(*(worker(event) for event in events))
    progress.finish()
    await client.close()

    payload = {
        "generated_at": ev.now_iso(),
        "models": {"identity": cfg["identity"], "venue": cfg["venue"]},
        "jsonld_hint": cfg["jsonld"],
        "text_source_mode": cfg["text_source"],
        "events": kept,
    }
    ev.write_json(Path(args.output), payload)
    ev.write_csv(Path(args.excluded), EXCLUDED_FIELDS, excluded)
    ev.write_csv(Path(args.review), EXCLUDED_FIELDS, review)
    record_usage(state, tracker)
    record_step(state, "basis", time.monotonic() - started)

    reasons: dict[str, int] = {}
    for record in excluded:
        key = f"{record.get('exclusion_reason', '?')}"
        reasons[key] = reasons.get(key, 0) + 1
    venues: dict[str, int] = {}
    for record in kept:
        key = str(record.get("venue_confidence") or "?")
        venues[key] = venues.get(key, 0) + 1
    errored = sum(1 for record in kept if record.get("field_errors"))

    print(f"Basis fertig: {len(kept)} behalten, {len(excluded)} ausgeschlossen, {len(review)} zur Prüfung", flush=True)
    if reasons:
        print("Ausschluss: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())), flush=True)
    print("Venue: " + ", ".join(f"{k} {v}" for k, v in sorted(venues.items())), flush=True)
    print(f"Mit Feldfehlern: {errored}", flush=True)
    print(tracker.render("Basis LLM-Nutzung"), flush=True)
    print(f"-> {args.output}\n-> {args.excluded}\n-> {args.review}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Basisinformationen extrahieren.")
    parser.add_argument("--discovery", default=str(OUT_DISCOVERY))
    parser.add_argument("--output", default=str(OUT_BASIS))
    parser.add_argument("--excluded", default=str(OUT_EXCLUDED))
    parser.add_argument("--review", default=str(OUT_REVIEW))
    parser.add_argument("--model-identity", default=None)
    parser.add_argument("--model-venue", default=None)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--text-source", choices=sorted(ev.TEXT_SOURCES), default=None,
                        help="Eingabetext: raw (markdown) oder content (content_markdown); "
                             "Default aus config/input.json")
    parser.add_argument("--no-jsonld", action="store_true",
                        help="JSON-LD-Hinweis nicht in den Venue-Prompt geben")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
