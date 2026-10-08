#!/usr/bin/env python3
"""fetch: Seiten in den Corpus laden. Kein LLM, kein Urteil, keine Extraktion.

Der Corpus ist der Cache der Pipeline. Er wird einmal gefüllt und danach von
discover/basis/detail gelesen, sodass Prompts und Modelle beliebig oft neu
ausprobiert werden können, ohne je wieder ins Netz zu müssen.

Modi:
  --mode overviews   INPUT/sources.csv  -> Übersichtsseiten, kind=overview
  --mode urls        out/discovery.json -> Event-Detailseiten, kind=detail

Speicherform: eine Datei pro Seite unter out/pages/<hash>.json plus index.json.
Einzelne Dateien statt eines Blobs erlauben selektiertes Nachladen, sicheres
paralleles Schreiben und --retry-failed ohne Restschaden.

Zusätzlich schreibt jeder Lauf ein lesbare Spiegel unter out/pages_readable/,
gruppiert nach Organisation und mit URL-Segmente statt Hash als Dateinamen.
Das Spiegel ist abgeleitet und nie autoritativ: gesucht, gelesen und referenziert
wird ausschliesslich der Hash-Corpus.

  --readable-only    kein Netz, kein Crawl: Spiegel aus dem vorhandenen Corpus
                     neu aufbauen (für Seiten, die vor Einführung des Spiegels
                     geladen wurden, oder nach einer Namensänderung)
  --no-readable      kein Spiegel schreiben
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import time
from pathlib import Path
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

import eventlib as ev
from progress import Progress, load_or_start_run, record_step, start_run

OUT_DISCOVERY = ev.OUT_DIR / "discovery.json"


def build_run_config(scroll_steps: int, scroll_delay: float, timeout: int) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        word_count_threshold=1,
        page_timeout=timeout,
        scan_full_page=True,
        scroll_delay=scroll_delay,
        max_scroll_steps=scroll_steps,
        wait_until="networkidle",
        delay_before_return_html=1.5,
    )


def targets_from_sources(path: Path, limit: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in ev.read_sources(path):
        url = ev.norm_url(row["url"])
        if not url or not ev.is_http_url(url) or url.lower() in seen:
            continue
        seen.add(url.lower())
        out.append({"url": url, "kind": "overview",
                    "orga": row["orga"], "orga_id": row["orga_id"]})
        if limit and len(out) >= limit:
            break
    return out


def targets_from_discovery(path: Path, limit: int) -> list[dict[str, str]]:
    data = ev.read_json(path, {}) or {}
    events = data.get("events", []) if isinstance(data, dict) else []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        url = ev.norm_url(str(event.get("event_url") or ""))
        source = ev.norm_url(str(event.get("source_url") or ""))
        if not url or not ev.is_http_url(url):
            continue
        if ev.same_page_link(url, source) or url.lower() in seen:
            continue
        seen.add(url.lower())
        out.append({"url": url, "kind": "detail",
                    "orga": str(event.get("orga") or ""),
                    "orga_id": str(event.get("orga_id") or "")})
        if limit and len(out) >= limit:
            break
    return out


def orga_lookup(sources_path: Path, discovery_path: Path) -> dict[str, dict[str, str]]:
    """URL -> Orga, aus beiden Quellen zusammengetragen.

    Für den Backfill: der Corpus selbst trägt bei älteren Aufzeichnungen keine
    Orga-Spalte, aber beide Zuordnungsquellen liegen noch auf der Platte.
    """
    mapping: dict[str, dict[str, str]] = {}
    for target in targets_from_sources(sources_path, 0):
        mapping[target["url"]] = {k: target[k] for k in ("orga", "orga_id")}
    for target in targets_from_discovery(discovery_path, 0):
        mapping[target["url"]] = {k: target[k] for k in ("orga", "orga_id")}
    return mapping


def already_fetched(url: str, refresh: bool, retry_failed: bool) -> bool:
    if refresh:
        return False
    record = ev.load_index().get(ev.norm_url(url))
    if not record:
        return False
    status = str(record.get("status") or "")
    if retry_failed:
        return status == "ok"
    return True


async def fetch_one(
    crawler: AsyncWebCrawler,
    target: dict[str, str],
    config: CrawlerRunConfig,
    semaphore: asyncio.Semaphore,
    readable: bool,
) -> dict[str, Any]:
    url = target["url"]
    record = ev.new_page_record(url, target["kind"], target["orga"], target["orga_id"])
    async with semaphore:
        try:
            result = await crawler.arun(url=url, config=config)
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:400])
            ev.save_page(record, readable=readable)
            return record
    html = str(getattr(result, "html", "") or "")
    if not getattr(result, "success", False) or not html.strip():
        message = getattr(result, "error_message", None) or "kein HTML"
        record.update(status="error", error=str(message)[:400])
        ev.save_page(record, readable=readable)
        return record
    metadata = getattr(result, "metadata", None)
    page_title = str(metadata.get("title") or "")[:300] if isinstance(metadata, dict) else ""
    record.update(
        status="ok",
        fetched_at=ev.now_iso(),
        markdown=ev.result_markdown(result),
        content_markdown=ev.content_markdown(html),
        text=ev.clean_text(html),
        html=html,
        links=ev.extract_links(html, url),
        images=ev.extract_images(html, url),
        jsonld=ev.extract_jsonld(html),
        title=page_title,
    )
    if not record["markdown"] and not record["text"]:
        record.update(status="empty", error="Seite ohne lesbaren Text")
    ev.save_page(record, readable=readable)
    return record


def rebuild_readable(sources_path: Path, discovery_path: Path) -> tuple[int, int]:
    """Spiegel aus dem vorhandenen Corpus neu aufbauen. Kein Netz, kein LLM."""
    index = ev.load_index()
    lookup = orga_lookup(sources_path, discovery_path)
    root = ev.readable_dir()
    # Ganzes Baum verwerfen, nicht nur die Dateien: ein Umbenennen der Orga-
    # Slugs liesse sonst leere Alt-Ordner zurück, und die sind nicht mehr
    # von echten Inhalten zu unterscheiden. Das Spiegel ist ein Derivat,
    # es hat keinen eigenen Zustand.
    stale = sum(1 for path in root.rglob("*.json")) if root.exists() else 0
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    written = 0
    taken: dict[str, str] = {}
    for url in sorted(index):
        record = ev.load_page(url)
        if not record:
            continue
        known = lookup.get(ev.norm_url(url)) or {}
        record["orga"] = str(record.get("orga") or known.get("orga") or "")
        record["orga_id"] = str(record.get("orga_id") or known.get("orga_id") or "")
        rel = ev.resolve_readable(ev.readable_rel_path(record), url, taken)
        taken[rel] = ev.norm_url(url)
        ev.write_readable(record, rel)
        entry = index[url] if isinstance(index[url], dict) else {}
        entry = dict(entry)
        entry["readable"] = rel
        entry["kind"] = record.get("kind", entry.get("kind", ""))
        entry["orga"] = record["orga"]
        index[url] = entry
        written += 1
    ev.save_index(index)
    return written, stale


def backfill_content(sources_path: Path | None = None,
                     discovery_path: Path | None = None) -> tuple[int, int]:
    """content_markdown aus gespeichertem html nachrechnen. Kein Netz, kein LLM."""
    index = ev.load_index()
    lookup = orga_lookup(sources_path or ev.DEFAULT_SOURCES,
                         discovery_path or OUT_DISCOVERY)
    done = 0
    empty = 0
    for url in sorted(index):
        record = ev.load_page(url)
        if not record or record.get("status") != "ok":
            continue
        if not record.get("orga"):
            known = lookup.get(ev.norm_url(url)) or {}
            record["orga"] = str(known.get("orga") or "")
            record["orga_id"] = str(known.get("orga_id") or "")
        text = ev.content_markdown(record.get("html") or "")
        if not text:
            empty += 1
        record["content_markdown"] = text
        ev.save_page(record)
        done += 1
    return done, empty


def run_summary(targets: list[dict[str, str]], index: dict[str, Any]) -> None:
    """Orga-Weise Rückmeldung, was das Spiegel hergibt."""
    per_orga: dict[str, dict[str, int]] = {}
    for target in targets:
        folder = f"{target['orga_id'][:8]}_{ev.slugify(target['orga'])}" if (
            target["orga_id"] or target["orga"]) else "_unassigned"
        counts = per_orga.setdefault(folder, {"overview": 0, "detail": 0})
        counts[target["kind"]] = counts.get(target["kind"], 0) + 1
    for folder in sorted(per_orga):
        counts = per_orga[folder]
        print(f"  {folder:<48} overview {counts.get('overview', 0)} | "
              f"detail {counts.get('detail', 0)}", flush=True)


async def run(args: argparse.Namespace) -> int:
    if args.backfill_content:
        done, empty = backfill_content()
        print(f"content_markdown nachgerechnet: {done} Seiten, "
              f"{empty} ohne verwertbaren Inhalt -> {ev.PAGES_DIR}", flush=True)
        return 0
    if args.readable_only:
        written, stale = rebuild_readable(Path(args.input), Path(args.urls_from))
        print(f"Spiegel neu aufgebaut: {written} Seiten geschrieben, "
              f"{stale} alte Spiegel-Dateien ersetzt -> {ev.readable_dir()}", flush=True)
        return 0

    if args.mode == "overviews":
        targets = targets_from_sources(Path(args.input), args.limit)
    else:
        targets = targets_from_discovery(Path(args.urls_from), args.limit)
    if not targets:
        print("Keine Ziele gefunden.", flush=True)
        return 1

    pending = [t for t in targets if not already_fetched(t["url"], args.refresh, args.retry_failed)]
    skipped = len(targets) - len(pending)
    # Der Übersichts-Durchlauf ist der Kopf der Pipeline und eröffnet die
    # Gesamtuhr; die Folgestufen (und --mode urls) hängen sich daran. Ohne das
    # würde ein alter Rest-State von gestern die Gesamtzeit unbrauchbar machen.
    state = start_run() if args.mode == "overviews" else load_or_start_run()
    started = time.monotonic()
    print(f"Fetch [{args.mode}]: {len(targets)} Ziele, {skipped} bereits im Corpus, "
          f"{len(pending)} zu laden ({args.jobs} parallel)", flush=True)
    if not pending:
        run_summary(targets, ev.load_index())
        return 0

    browser_config = BrowserConfig(headless=True, verbose=False)
    config = build_run_config(args.max_scroll_steps, args.scroll_delay, args.timeout)
    tracker = Progress("Fetch", len(pending), state["started_at"], parallelism=args.jobs)
    semaphore = asyncio.Semaphore(max(1, args.jobs))
    counts = {"ok": 0, "error": 0, "empty": 0}

    async with AsyncWebCrawler(config=browser_config) as crawler:
        async def worker(target: dict[str, str]) -> None:
            unit_started = time.monotonic()
            try:
                record = await fetch_one(crawler, target, config, semaphore, not args.no_readable)
            except Exception as exc:  # ein Ausfall darf den Batch nicht stoppen
                record = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            status = str(record.get("status", "error"))
            counts[status] = counts.get(status, 0) + 1
            tracker.update(sum(counts.values()), time.monotonic() - unit_started)

        await asyncio.gather(*(worker(target) for target in pending))

    tracker.finish()
    record_step(state, f"fetch_{args.mode}", time.monotonic() - started)
    print(f"Fetch fertig: ok {counts.get('ok', 0)} / leer {counts.get('empty', 0)} "
          f"/ Fehler {counts.get('error', 0)}", flush=True)
    if counts.get("error"):
        print("Fehlerhafte Seiten bleiben im Corpus (status=error) und können mit "
              "--retry-failed erneut geholt werden.", flush=True)
    if not args.no_readable:
        run_summary(targets, ev.load_index())
        print(f"Spiegel: {ev.readable_dir()}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Seiten in den Corpus laden (kein LLM).")
    parser.add_argument("--mode", choices=["overviews", "urls"], default="overviews")
    parser.add_argument("--input", default=str(ev.DEFAULT_SOURCES), help="Quellen-CSV")
    parser.add_argument("--urls-from", default=str(OUT_DISCOVERY), help="discovery.json für --mode urls")
    parser.add_argument("--jobs", type=int, default=4, help="parallele Seitenabrufe")
    parser.add_argument("--limit", type=int, default=0, help="max. Ziele (0 = alle)")
    parser.add_argument("--refresh", action="store_true", help="Corpus ignorieren und neu laden")
    parser.add_argument("--retry-failed", action="store_true", help="nur nicht-ok Seiten neu laden")
    parser.add_argument("--no-readable", action="store_true", help="kein lesbares Spiegel schreiben")
    parser.add_argument("--readable-only", action="store_true",
                        help="nur Spiegel aus dem vorhandenen Corpus aufbauen, kein Netz")
    parser.add_argument("--backfill-content", action="store_true",
                        help="nur content_markdown aus gespeichertem html nachrechnen, kein Netz")
    parser.add_argument("--max-scroll-steps", type=int, default=12)
    parser.add_argument("--scroll-delay", type=float, default=0.8)
    parser.add_argument("--timeout", type=int, default=120000, help="Seiten-Timeout in ms")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
