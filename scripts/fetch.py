#!/usr/bin/env python3
"""fetch: Seiten in den Corpus laden. Kein LLM, kein Urteil, keine Extraktion.

Jeder reguläre Lauf crawlt seine ausgewählten Ziele frisch und aktualisiert
den Corpus. Nachfolgende Stufen lesen diesen Snapshot, sodass Prompts und
Modelle beliebig oft neu ausprobiert werden können, ohne erneut zu crawlen.

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
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

import eventlib as ev
from progress import Progress, load_or_start_run, record_step, start_run

OUT_DISCOVERY = ev.OUT_DIR / "discovery.json"

# Consent UI observed in the stored corpus. Keep selectors provider-specific so
# ordinary event content mentioning cookies is not removed.
MARKDOWN_EXCLUDED_SELECTOR = (
    ".jelka-consent, #cmplz-cookiebanner-container, #cmplz-manage-consent, "
    ".cky-consent-container, .cky-preference-center, .cky-btn-revisit-wrapper"
)


class CorpusStage:
    """Isolierter Corpus-Schreibstand mit Commit- oder Rollback-Verhalten."""

    def __init__(self) -> None:
        self.pages_dir = ev.PAGES_DIR
        self.pages_index = ev.PAGES_INDEX
        self.readable_dir = ev.readable_dir()
        self.root: Path | None = None
        self.stage_pages: Path | None = None
        self.stage_readable: Path | None = None
        self._preserve_root = False

    def __enter__(self) -> "CorpusStage":
        self.root = Path(tempfile.mkdtemp(
            prefix=".fetch-corpus-", dir=self.pages_dir.parent))
        self.stage_pages = self.root / "pages"
        self.stage_readable = self.root / "pages_readable"
        try:
            if self.pages_dir.exists():
                shutil.copytree(self.pages_dir, self.stage_pages)
            else:
                self.stage_pages.mkdir(parents=True)
            if self.readable_dir.exists():
                shutil.copytree(self.readable_dir, self.stage_readable)
            else:
                self.stage_readable.mkdir(parents=True)
        except Exception:
            shutil.rmtree(self.root)
            self.root = None
            raise
        ev.PAGES_DIR = self.stage_pages
        ev.PAGES_INDEX = self.stage_pages / "index.json"
        return self

    def prune_kind(self, kind: str, keep_urls: set[str]) -> int:
        index = ev.load_index()
        removed = 0
        for url, entry in list(index.items()):
            if not isinstance(entry, dict) or entry.get("kind") != kind:
                continue
            if ev.norm_url(url) in keep_urls:
                continue
            ev.page_path(url).unlink(missing_ok=True)
            del index[url]
            removed += 1
        ev.save_index(index)
        return removed

    def publish(
        self,
        update_readable: bool,
        sources_path: Path,
        discovery_path: Path,
    ) -> None:
        if self.root is None or self.stage_pages is None or self.stage_readable is None:
            raise RuntimeError("CorpusStage wurde nicht gestartet")
        if update_readable:
            rebuild_readable(sources_path, discovery_path)

        backup_pages = self.root / "pages.previous"
        backup_readable = self.root / "pages_readable.previous"
        moved_pages = moved_readable = installed_pages = installed_readable = False
        try:
            if self.pages_dir.exists():
                os.replace(self.pages_dir, backup_pages)
                moved_pages = True
            if update_readable and self.readable_dir.exists():
                os.replace(self.readable_dir, backup_readable)
                moved_readable = True
            os.replace(self.stage_pages, self.pages_dir)
            installed_pages = True
            if update_readable:
                os.replace(self.stage_readable, self.readable_dir)
                installed_readable = True
        except Exception:
            try:
                if installed_readable and self.readable_dir.exists():
                    os.replace(self.readable_dir, self.stage_readable)
                if installed_pages and self.pages_dir.exists():
                    os.replace(self.pages_dir, self.stage_pages)
                if moved_readable and backup_readable.exists():
                    os.replace(backup_readable, self.readable_dir)
                if moved_pages and backup_pages.exists():
                    os.replace(backup_pages, self.pages_dir)
            except Exception as rollback_error:
                self._preserve_root = True
                raise RuntimeError(
                    f"Corpus-Rollback fehlgeschlagen; Sicherung bleibt unter {self.root}"
                ) from rollback_error
            raise

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        ev.PAGES_DIR = self.pages_dir
        ev.PAGES_INDEX = self.pages_index
        if self.root is not None and not self._preserve_root:
            shutil.rmtree(self.root)


def build_run_config(scroll_steps: int, scroll_delay: float, timeout: int) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        verbose=False,
        word_count_threshold=1,
        excluded_selector=MARKDOWN_EXCLUDED_SELECTOR,
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


def targets_to_fetch(targets: list[dict[str, str]], retry_failed: bool) -> list[dict[str, str]]:
    """Normale Läufe crawlen alle Ziele; retry-failed ist ein expliziter Teillauf."""
    if not retry_failed:
        return targets
    index = ev.load_index()
    pending: list[dict[str, str]] = []
    for target in targets:
        entry = index.get(ev.norm_url(target["url"]))
        if (not isinstance(entry, dict) or entry.get("status") != "ok"
                or entry.get("last_fetch_error")):
            pending.append(target)
    return pending


def save_fetch_failure(
    record: dict[str, Any],
    message: str,
    readable: bool,
    status: str = "error",
) -> dict[str, Any]:
    """Einen Crawlfehler protokollieren, ohne einen guten Corpusstand zu verlieren."""
    previous = ev.load_page(record["url"])
    if previous and previous.get("status") == "ok":
        previous["last_fetch_attempt_at"] = ev.now_iso()
        previous["last_fetch_error"] = message
        ev.save_page(previous, readable=readable)
        return {**record, "status": status, "error": message}
    record.update(status=status, error=message, fetched_at=ev.now_iso())
    ev.save_page(record, readable=readable)
    return record


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
            return save_fetch_failure(
                record, f"{type(exc).__name__}: {exc}"[:400], readable)
    html = str(getattr(result, "html", "") or "")
    if not getattr(result, "success", False) or not html.strip():
        message = getattr(result, "error_message", None) or "kein HTML"
        return save_fetch_failure(record, str(message)[:400], readable)
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
        return save_fetch_failure(record, "Seite ohne lesbaren Text",
                                  readable, status="empty")
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

    pending = targets_to_fetch(targets, args.retry_failed)
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

    with CorpusStage() as staged:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            async def worker(target: dict[str, str]) -> None:
                unit_started = time.monotonic()
                try:
                    record = await fetch_one(
                        crawler, target, config, semaphore, not args.no_readable)
                except Exception as exc:  # ein Ausfall darf den Batch nicht stoppen
                    record = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                status = str(record.get("status", "error"))
                counts[status] = counts.get(status, 0) + 1
                tracker.update(sum(counts.values()), time.monotonic() - unit_started)

            await asyncio.gather(*(worker(target) for target in pending))

        complete = counts.get("error", 0) == 0 and counts.get("empty", 0) == 0
        if complete:
            full_run = args.limit == 0 and not args.retry_failed
            if full_run:
                keep_urls = {ev.norm_url(target["url"]) for target in targets}
                kind = "overview" if args.mode == "overviews" else "detail"
                removed = staged.prune_kind(kind, keep_urls)
                if removed:
                    print(f"Veraltete {kind}-Seiten entfernt: {removed}", flush=True)
            staged.publish(
                update_readable=not args.no_readable,
                sources_path=Path(args.input),
                discovery_path=Path(args.urls_from),
            )

    tracker.finish()
    record_step(state, f"fetch_{args.mode}", time.monotonic() - started)
    print(f"Fetch fertig: ok {counts.get('ok', 0)} / leer {counts.get('empty', 0)} "
          f"/ Fehler {counts.get('error', 0)}", flush=True)
    if not complete:
        print("Fehlgeschlagene Abrufe können mit --retry-failed erneut geholt werden; "
              "der bisherige Corpus bleibt unverändert.", flush=True)
    if not args.no_readable:
        run_summary(targets, ev.load_index())
        print(f"Spiegel: {ev.readable_dir()}", flush=True)
    return 0 if complete else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Seiten in den Corpus laden (kein LLM).")
    parser.add_argument("--mode", choices=["overviews", "urls"], default="overviews")
    parser.add_argument("--input", default=str(ev.DEFAULT_SOURCES), help="Quellen-CSV")
    parser.add_argument("--urls-from", default=str(OUT_DISCOVERY), help="discovery.json für --mode urls")
    parser.add_argument("--jobs", type=int, default=4, help="parallele Seitenabrufe")
    parser.add_argument("--limit", type=int, default=0, help="max. Ziele (0 = alle)")
    parser.add_argument("--refresh", action="store_true",
                        help="kompatibler Alt-Parameter; normale Läufe crawlen immer frisch")
    parser.add_argument("--retry-failed", action="store_true",
                        help="nur Ziele ohne erfolgreichen letzten Abruf neu laden")
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
