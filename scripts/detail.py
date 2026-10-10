#!/usr/bin/env python3
"""detail: Feininformationen für die von basis behaltenen Events.

Liest ausschliesslich OUTPUT/basis.json, ruft nie Webseiten auf und liest denselben
Corpus wie basis. Die Prompts laufen parallel pro Event; jedes Feld hat ein
eigenes Prompt, ein eigenes Schema und die gleiche Beweispflicht.

Das basis-JSON wird jedem Prompt als Kontext mitgegeben. Ein Prompt, der weiss,
dass es um "Konzert im Depot am 23.08." geht, liefert eine präzisere
Beschreibung und Kategorie als derselbe Prompt ohne diesen Rahmen.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Any

import eventlib as ev
from costs import UsageTracker, record_usage
from llm import ClientRouter, require_api_key
from progress import Progress, load_or_start_run, record_step

OUT_BASIS = ev.OUT_DIR / "basis.json"
OUT_DETAIL = ev.OUT_DIR / "detail.json"

DESCRIPTION_CAP = 2000

# Rolle -> (Felder, Prüfart, Modell-Umweltvariable)
#   quote    : Evidenzfeld enthält die wörtliche Fundstelle; die wird geprüft
#   verbatim : das Feld selbst muss im Quelltext stehen
#   derived  : normalisierter Wert; Beleg ist die Quellform, rückwärts parsiert
FIELDS: dict[str, dict[str, Any]] = {
    "language": {
        "fields": ("language", "language_name"),
        "check": {"language": ("quote", "evidence"), "language_name": ("verbatim", None)},
        "model_role": "LANGUAGE",
    },
    "registration": {
        "fields": ("registration_required",),
        "check": {"registration_required": ("quote", "evidence")},
        "model_role": "REGISTRATION",
    },
    "category": {
        "fields": ("category",),
        "check": {"category": ("quote", "evidence")},
        "model_role": "CATEGORY",
    },
    "timing": {
        "fields": ("end_date", "end_time"),
        "check": {
            "end_date": ("derived", "evidence_end_date", ev.parse_de_dates),
            "end_time": ("derived", "evidence_end_time", ev.parse_de_times),
        },
        "model_role": "TIMING",
    },
    "description": {
        "fields": ("description",),
        "check": {"description": ("prose", None)},
        "model_role": "DESCRIPTION",
    },
}

# Legitime Antworten ohne Belegmöglichkeit: Sie sagen aus, dass der Text nichts
# hergibt. Ein Zitat kann es dafür per Definition nicht geben; verlangt man
# eines, wird die richtige Antwort als Fehler verworfen.
NO_EVIDENCE_VALUES = {
    "language": {"unknown"},
    "registration_required": {"unbekannt"},
    "category": {"sonstiges"},
}

ALLOWED = {
    "language": {"de", "en", "other", "unknown"},
    "registration_required": {"ja", "nein", "unbekannt"},
    "category": {"workshop", "demo", "talk", "meetup", "kultur", "sonstiges"},
}


def attach(target: dict[str, Any], data: dict[str, Any] | None, errors: list[str],
           spec: dict[str, Any], name: str, source_text: str) -> list[str]:
    problems: list[str] = []
    if data is None:
        for field in spec["fields"]:
            target[field] = ""
            target[f"confidence_{field}"] = "none"
        return [f"{name}: {'; '.join(errors)[:300]}"] if errors else [f"{name}: keine Antwort"]
    for field in spec["fields"]:
        value = str(data.get(field) or "").strip()
        if not value:
            target[field] = ""
            target[f"confidence_{field}"] = "none"
            continue
        rule = spec["check"][field]
        if field == "description":
            value = " ".join(value.split())[:DESCRIPTION_CAP].strip()
        if value.lower() in NO_EVIDENCE_VALUES.get(field, ()):
            target[field] = value.lower()
            target[f"confidence_{field}"] = "default"
            continue
        verdict = _verdict(value, rule, data, source_text)
        if verdict[1] == "none":
            target[field] = ""
            target[f"confidence_{field}"] = "none"
            problems.append(f"{name}.{field}: Beleg fehlt, nicht im Quelltext oder werterwidrig")
            continue
        final = verdict[0]
        allowed = ALLOWED.get(field)
        if allowed and final.lower() not in allowed:
            target[field] = ""
            target[f"confidence_{field}"] = "none"
            problems.append(f"{name}.{field}: Wert {final!r} ausserhalb der erlaubten Menge")
            continue
        target[field] = final.lower() if allowed else final
        target[f"confidence_{field}"] = verdict[1]
    return problems


def _verdict(value: str, rule: tuple, data: dict[str, Any], source_text: str) -> tuple[str, str]:
    """Gibt (wert, konfidenz) zurück. Wert kann korrigiert sein."""
    kind = rule[0]
    if kind == "verbatim":
        return (value, "high") if ev.verify_evidence(value, source_text) else ("", "none")
    if kind == "prose":
        return (value, "high") if ev.verify_prose(value, source_text) else ("", "none")
    if kind == "derived":
        quote = str(data.get(rule[1]) or "").strip()
        return ev.resolve_derived(value, quote, source_text, rule[2])
    quote = str(data.get(rule[1]) or "").strip()
    return (value, "high") if ev.verify_evidence(quote, source_text) else ("", "none")


async def extract_one(
    client: ClientRouter,
    semaphore: asyncio.Semaphore,
    event: dict[str, Any],
    cfg: dict[str, Any],
    tracker: UsageTracker | None,
) -> dict[str, Any]:
    record = dict(event)
    record["field_errors"] = list(event.get("field_errors") or [])
    llm_kwargs = {"tracker": tracker} if tracker is not None else {}

    text, source = ev.event_text(event)
    record["detail_text_source"] = source
    if not text:
        record["field_errors"].append("text: kein Quelltext im Corpus")
        for spec in FIELDS.values():
            for field in spec["fields"]:
                record.setdefault(field, "")
                record[f"confidence_{field}"] = "none"
        return record

    base_context = {
        "title": event.get("title", ""),
        "start_date": event.get("start_date", ""),
        "start_time": event.get("start_time", ""),
        "venue_address": event.get("venue_address", ""),
        "venue_name": event.get("venue_name", ""),
        "orga": event.get("orga", ""),
        "event_url": event.get("event_url", ""),
        "source_url": event.get("source_url", ""),
    }

    async def one(name: str) -> tuple[str, dict[str, Any] | None, list[str]]:
        spec = FIELDS[name]
        payload = {**base_context, "source_text": text}
        data, errors = await ev.llm_json(
            client, cfg["models"][name], cfg["prompts"][name], payload, semaphore,
            schema=cfg["schemas"][name], max_tokens=cfg["max_tokens"], **llm_kwargs,
        )
        return name, data, errors

    results = await asyncio.gather(
        *(one(name) for name in FIELDS), return_exceptions=True
    )
    for result in results:
        if isinstance(result, BaseException):
            record["field_errors"].append(f"detail: {type(result).__name__}: {result}")
            continue
        name, data, errors = result
        record["field_errors"].extend(
            attach(record, data, errors, FIELDS[name], name, text)
        )
    for spec in FIELDS.values():
        for field in spec["fields"]:
            record.setdefault(field, "")
            record.setdefault(f"confidence_{field}", "none")
    return record


async def run(args: argparse.Namespace) -> int:
    basis = ev.read_json(Path(args.basis), {}) or {}
    events = basis.get("events", []) if isinstance(basis, dict) else []
    if args.limit:
        events = events[:args.limit]
    if not events:
        print(f"Keine Events in {args.basis}", flush=True)
        return 1

    names = list(FIELDS)
    cfg = {
        "models": {name: ev.model_for(FIELDS[name]["model_role"], getattr(args, f"model_{name}"))
                   for name in names},
        "prompts": {name: ev.load_prompt(name) for name in names},
        "schemas": {name: ev.load_schema(name) for name in names},
        "max_tokens": args.max_tokens,
    }
    distinct = sorted(set(cfg["models"].values()))
    for model in distinct:
        require_api_key(model)
    for model in distinct:
        await ev.probe_llm(model)

    state = load_or_start_run()
    started = time.monotonic()
    semaphore = asyncio.Semaphore(max(1, args.jobs))
    client = ClientRouter()
    tracker = UsageTracker()
    print(f"Detail: {len(events)} Events x {len(names)} Felder", flush=True)
    for name in names:
        print(f"  {name}: {cfg['models'][name]}", flush=True)

    out: list[dict[str, Any]] = []
    done = 0
    progress = Progress("Detail", len(events), state["started_at"], parallelism=args.jobs)

    async def worker(event: dict[str, Any]) -> None:
        nonlocal done
        unit_started = time.monotonic()
        try:
            record = await extract_one(client, semaphore, event, cfg, tracker)
        except Exception as exc:
            record = {**event, "field_errors": [f"{type(exc).__name__}: {exc}"]}
        out.append(record)
        done += 1
        progress.update(done, time.monotonic() - unit_started)

    await asyncio.gather(*(worker(event) for event in events))
    progress.finish()
    await client.close()

    payload = {
        "generated_at": ev.now_iso(),
        "models": cfg["models"],
        "events": out,
    }
    ev.write_json(Path(args.output), payload)
    record_usage(state, tracker)
    record_step(state, "detail", time.monotonic() - started)

    print(f"Detail fertig: {len(out)} Events", flush=True)
    for name in names:
        for field in FIELDS[name]["fields"]:
            filled = sum(1 for record in out if str(record.get(field) or "").strip())
            print(f"  {field}: {filled}/{len(out)} belegt", flush=True)
    errored = sum(1 for record in out if record.get("field_errors"))
    print(f"Mit Feldfehlern: {errored}", flush=True)
    print(tracker.render("Detail LLM-Nutzung"), flush=True)
    print(f"-> {args.output}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Feininformationen extrahieren.")
    parser.add_argument("--basis", default=str(OUT_BASIS))
    parser.add_argument("--output", default=str(OUT_DETAIL))
    parser.add_argument("--jobs", type=int, default=6,
                        help="parallele Events (jedes Event feuert selbst 5 Prompts)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2500)
    for name in FIELDS:
        parser.add_argument(f"--model-{name}", default=None, dest=f"model_{name}",
                            help=f"überschreibt MODEL_{FIELDS[name]['model_role']}")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
