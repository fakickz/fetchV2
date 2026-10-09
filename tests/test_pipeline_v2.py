"""Tests für Pipeline v2: nur reine Funktionen, kein Netzwerk, keine LLMs."""
import asyncio
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name: str, relative_path: str):
    """Skript unter seinem echten Modulnamen laden.

    Die Eintragung in sys.modules VOR dem Ausführen ist zwingend: fetch.py und
    Co. importieren eventlib normal per Namen. Ohne den Eintrag bekäme jedes
    Skript eine zweite eventlib-Instanz mit unpatched Pfadkonstanten, und ein
    Test, der PAGES_DIR ins Temp-Verzeichnis biegt, schreibt still ins echte
    out/pages.
    """
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = load_script("eventlib", "scripts/eventlib.py")
discover = load_script("discover", "scripts/discover.py")
basis = load_script("basis", "scripts/basis.py")
fetch = load_script("fetch", "scripts/fetch.py")
detail = load_script("detail", "scripts/detail.py")
api_gate = load_script("api_gate", "scripts/api_gate.py")


class UrlTests(unittest.TestCase):
    def test_norm_url_strips_trailing_slash(self):
        self.assertEqual(ev.norm_url("  https://a.org/x/ "), "https://a.org/x")

    def test_absolute_url_resolves_relative(self):
        self.assertEqual(
            ev.absolute_url("https://a.org/termine/", "../event/x"),
            "https://a.org/event/x",
        )

    def test_same_page_link_ignores_fragment_and_slash(self):
        self.assertTrue(ev.same_page_link("https://a.org/t#e1", "https://a.org/t/"))
        self.assertFalse(ev.same_page_link("https://a.org/t/e1", "https://a.org/t"))

    def test_page_hash_stable_and_distinct(self):
        self.assertEqual(ev.page_hash("https://a.org/x"), ev.page_hash("https://a.org/x/"))
        self.assertNotEqual(ev.page_hash("https://a.org/x"), ev.page_hash("https://a.org/y"))

    def test_is_http_url_rejects_junk(self):
        for bad in ("", "#frag", "mailto:a@b.c", "javascript:void(0)", "/pfad"):
            self.assertFalse(ev.is_http_url(bad), bad)
        self.assertTrue(ev.is_http_url("https://a.org"))


class DateTests(unittest.TestCase):
    def test_parse_de_date_variants(self):
        self.assertEqual(ev.parse_de_date("So 23.08.2026 Sacco"), "2026-08-23")
        self.assertEqual(ev.parse_de_date("23.8.26"), "2026-08-23")
        self.assertEqual(ev.parse_de_date("05.11.", default_year="2026"), "2026-11-05")
        self.assertEqual(ev.parse_de_date("kein datum"), "")

    def test_parse_de_time_rejects_impossible(self):
        self.assertEqual(ev.parse_de_time("Beginn: 20:00"), "20:00")
        self.assertEqual(ev.parse_de_time("19.30 Uhr"), "19:30")
        self.assertEqual(ev.parse_de_time("25:00"), "")
        self.assertEqual(ev.parse_de_time("12:61"), "")

    def test_parse_de_time_tolerates_poster_typography(self):
        """Plakate setzen die Stunde gern als "20°°" oder "20••" statt "20:00"."""
        self.assertEqual(ev.parse_de_time("Lesung 20°°"), "20:00")
        self.assertEqual(ev.parse_de_time("Beginn 20••"), "20:00")
        self.assertEqual(ev.parse_de_time("20 .. 00"), "20:00")


class MarkupTests(unittest.TestCase):
    def test_strip_markup_removes_bold_and_link_targets(self):
        self.assertEqual(ev.strip_markup("**Zeit |**_Roman_"), "Zeit |Roman")
        self.assertEqual(ev.strip_markup("[Depot](https://a.org/d)"), "Depot")
        self.assertEqual(ev.strip_markup("![](bild.jpg)Text"), " Text")

    def test_normalize_quote_ignores_markup(self):
        self.assertEqual(ev.normalize_quote("**A**"), ev.normalize_quote("A"))
        self.assertEqual(ev.normalize_quote("[Text](http://x.y)"), ev.normalize_quote("Text"))

    def test_quote_inside_markdown_page_still_verifies(self):
        src = "Wir sind in der **Sanettystraße 1**, 1080 Wien"
        self.assertTrue(ev.verify_evidence("Sanettystraße 1", src))

    def test_is_past_none_when_no_date(self):
        today = date(2026, 10, 8)
        self.assertIsNone(ev.is_past("", today))
        self.assertIsNone(ev.is_past("unsicher", today))

    def test_is_past_decides_only_on_valid_date(self):
        today = date(2026, 10, 8)
        self.assertTrue(ev.is_past("2026-08-23", today))
        self.assertFalse(ev.is_past("2026-11-05", today))
        self.assertFalse(ev.is_past("2026-10-08", today))


class AddressTests(unittest.TestCase):
    def test_is_full_address_needs_street_number_postcode(self):
        self.assertTrue(ev.is_full_address("Sanettystraße 1, 1080 Wien"))
        self.assertTrue(ev.is_full_address("Breite Gasse 3, 1070 Wien"))
        self.assertFalse(ev.is_full_address("Depot"))
        self.assertFalse(ev.is_full_address("1070 Wien"))
        self.assertFalse(ev.is_full_address(""))

    def test_vienna_gate_verdicts(self):
        self.assertTrue(ev.is_vienna_address("Sanettystraße 1, 1080 Wien"))
        self.assertFalse(ev.is_vienna_address("Hauptplatz 3, 3500 Krems"))
        self.assertFalse(ev.is_vienna_address("Wiener Neustadt 1, 2700 Wr. Neustadt"))
        self.assertIsNone(ev.is_vienna_address("Depot"))
        self.assertIsNone(ev.is_vienna_address(""))

    def test_year_like_numbers_are_not_postcodes(self):
        "1984 als Zahl darf nicht als Fremd-PLZ durchgehen."
        self.assertIsNone(ev.is_vienna_address("Roman 1984, Verlag"))

    def test_wien_without_postcode_is_wien(self):
        self.assertTrue(ev.is_vienna_address("Irgendwo in Wien"))


class EvidenceTests(unittest.TestCase):
    def test_verify_evidence_normalises_whitespace_and_case(self):
        src = "Der   Text sagt: Breite\tGasse 3, 1070 Wien."
        self.assertTrue(ev.verify_evidence("breite gasse 3", src))
        self.assertTrue(ev.verify_evidence("Breite Gasse 3, 1070 Wien", src))

    def test_verify_evidence_rejects_hallucination(self):
        self.assertFalse(ev.verify_evidence("Falkengasse 9", "Breite Gasse 3"))
        self.assertFalse(ev.verify_evidence("", "irgendwas"))

    def test_dashes_and_quotes_are_equivalent(self):
        src = "Er sagte \u201eJa\u201c \u2013 und ging."
        self.assertTrue(ev.verify_evidence('er sagte "ja" - und ging', src))

    def test_verify_prose_allows_dropped_junk(self):
        src = "Erster Satz hier. SHARE BUTTON. Zweiter Satz auch."
        self.assertTrue(ev.verify_prose("Erster Satz hier. Zweiter Satz auch.", src))

    def test_verify_prose_rejects_invented_sentences(self):
        src = "Erster Satz hier. Zweiter Satz auch."
        self.assertFalse(ev.verify_prose("Erster Satz hier. Erfundener Zusatz.", src))


class SchemaTests(unittest.TestCase):
    def test_validate_schema_reports_missing_and_typed(self):
        schema = ev.load_schema("venue")
        errors = ev.validate_schema({"address": 1}, schema)
        joined = " ".join(errors)
        self.assertIn("venue_name", joined)
        self.assertIn("erwartet string", joined)

    def test_validate_schema_accepts_full_object(self):
        schema = ev.load_schema("venue")
        ok = {"address": "", "address_source": "", "venue_name": "", "confidence": "none"}
        self.assertEqual(ev.validate_schema(ok, schema), [])

    def test_validate_schema_array_items(self):
        schema = ev.load_schema("discover")
        errors = ev.validate_schema({"events": [{"title": "x"}], "possibly_truncated": False}, schema)
        self.assertTrue(any("url" in e for e in errors))

    def test_enum_violation(self):
        schema = ev.load_schema("category")
        errors = ev.validate_schema({"category": "konzert", "evidence": ""}, schema)
        self.assertTrue(errors)

    def test_all_shipped_schemas_are_valid_objects(self):
        for name in ("discover", "identity", "venue", "timing", "language",
                     "registration", "category", "description"):
            schema = ev.load_schema(name)
            self.assertEqual(schema["type"], "object", name)
            self.assertTrue(schema.get("required"), name)


class JsonParseTests(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(ev.parse_json_object('{"a": 1}'), {"a": 1})

    def test_markdown_fenced(self):
        self.assertEqual(ev.parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_trailing_prose_tolerated(self):
        self.assertEqual(ev.parse_json_object('{"a": 1}\nHoffentlich hilft das!'), {"a": 1})

    def test_list_wrapped_as_events(self):
        self.assertEqual(ev.parse_json_object('[{"a": 1}]'), {"events": [{"a": 1}]})

    def test_garbage_returns_none(self):
        self.assertIsNone(ev.parse_json_object("kein json"))
        self.assertIsNone(ev.parse_json_object(""))


class HtmlTests(unittest.TestCase):
    HTML = """
    <html><body>
      <h1>Titel</h1>
      <p>Am 23.08.2026 im <a href="/event/x">Depot</a>.</p>
      <a href="mailto:a@b.c">Mail</a>
      <a href="#top">Sprung</a>
      <a href="https://other.org/a">Extern</a>
      <a href="/event/x">Duplikat</a>
      <img src="/img/p.jpg" alt="Sanettystra&szlig;e 1, 1080 Wien" title="Plakat">
      <script type="application/ld+json">
        {"@context":"https://schema.org","@type":"Event","name":"Konzert",
         "startDate":"2026-08-23T20:00","location":{"@type":"Place","name":"Depot"}}
      </script>
      <style>p{color:red}</style>
    </body></html>
    """

    def test_strip_html_removes_script_and_style(self):
        text = ev.strip_html(self.HTML)
        self.assertNotIn("color:red", text)
        self.assertIn("Titel", text)

    def test_clean_text_unescapes_entities(self):
        self.assertEqual(ev.clean_text("<p>Sanettystra&szlig;e 1</p>"), "Sanettystraße 1")

    def test_extract_links_absolute_dedup_and_filter(self):
        links = ev.extract_links(self.HTML, "https://a.org/termine/")
        hrefs = [link["href"] for link in links]
        self.assertIn("https://a.org/event/x", hrefs)
        self.assertIn("https://other.org/a", hrefs)
        self.assertNotIn("mailto:a@b.c", hrefs)
        self.assertEqual(len(hrefs), len(set(h.lower() for h in hrefs)))
        self.assertNotIn("https://a.org/termine/#top", hrefs)

    def test_extract_link_keeps_anchor_text(self):
        links = {link["href"]: link for link in ev.extract_links(self.HTML, "https://a.org/")}
        self.assertEqual(links["https://a.org/event/x"]["text"], "Depot")

    def test_extract_images_carries_alt(self):
        images = ev.extract_images(self.HTML, "https://a.org/")
        self.assertEqual(images[0]["src"], "https://a.org/img/p.jpg")
        self.assertIn("1080 Wien", images[0]["alt"])
        self.assertEqual(images[0]["title"], "Plakat")

    def test_extract_jsonld_parses_block(self):
        data = ev.extract_jsonld(self.HTML)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["@type"], "Event")

    def test_jsonld_hints_compact(self):
        hint = ev.jsonld_hints(ev.extract_jsonld(self.HTML))
        self.assertIn("Event", hint)
        self.assertIn("Depot", hint)

    def test_jsonld_broken_block_skipped_not_fatal(self):
        self.assertEqual(ev.extract_jsonld('<script type="application/ld+json">{unsinn</script>'), [])

    def test_result_markdown_handles_object_and_string(self):
        class Inner:
            raw_markdown = "a\n\n\n\nb"
            fit_markdown = ""
        class Obj:
            markdown = Inner()
        class Plain:
            markdown = "c\n\n\n\nd"
        self.assertEqual(ev.result_markdown(Obj()), "a\n\nb")
        self.assertEqual(ev.result_markdown(Plain()), "c\n\nd")
        self.assertEqual(ev.result_markdown(object()), "")


class ContentMarkdownTests(unittest.TestCase):
    HTML = """
    <html><body>
      <a href="#primary">Skip to content</a>
      <nav><ul><li>Kalender</li><li>Spenden</li></ul></nav>
      <header><div class="site-menu">Menüpunkt</div></header>
      <main>
        <h2>Linkswende kennenlernen &amp; aktiv werden</h2>
        <h2></h2>
        <p>Heute, 18:45 in der Breite Gasse 3, 1070 Wien.</p>
        <p>Anmeldung unter <a href="https://a.org/x">Depot-Seite</a>.</p>
        <img src="/p.jpg" alt="Plakatmotiv">
        <div class="pagination">Ältere Termine laden</div>
      </main>
      <aside class="sidebar">Werbung</aside>
      <footer>Impressum</footer>
    </body></html>
    """

    def test_nav_and_frame_removed_content_kept_in_order(self):
        out = ev.content_markdown(self.HTML)
        for junk in ("Kalender", "Spenden", "Menüpunkt", "Werbung",
                      "Impressum", "Skip to content"):
            self.assertNotIn(junk, out)
        self.assertLess(out.index("Linkswende kennenlernen"),
                        out.index("Breite Gasse 3"))

    def test_headings_and_facts_verbatim(self):
        out = ev.content_markdown(self.HTML)
        self.assertIn("## Linkswende kennenlernen & aktiv werden", out)
        self.assertIn("Heute, 18:45", out)
        self.assertIn("Breite Gasse 3, 1070 Wien", out)

    def test_empty_headings_removed(self):
        out = ev.content_markdown(self.HTML)
        for line in out.split("\n"):
            self.assertFalse(line.strip().strip("#").strip() == "",
                             f"leere Überschrift: {line!r}")

    def test_link_targets_and_image_syntax_dropped(self):
        out = ev.content_markdown(self.HTML)
        self.assertIn("Depot-Seite", out)
        self.assertNotIn("https://a.org/x", out)
        self.assertNotIn("![", out)
        self.assertIn("Plakatmotiv", out)

    def test_pagination_hint_survives_as_truncation_signal(self):
        self.assertIn("Ältere Termine laden",
                      ev.content_markdown(self.HTML))

    def test_empty_or_broken_html_yields_empty(self):
        self.assertEqual(ev.content_markdown(""), "")
        self.assertEqual(ev.content_markdown("<nav>nur</nav>"), "")

    def test_consecutive_duplicate_lines_collapsed(self):
        out = ev.content_markdown(
            "<main><p>Zeile</p><p>Zeile</p><p>Text</p></main>")
        self.assertEqual(out, "Zeile\nText")


class CutSectionTests(unittest.TestCase):
    def test_locator_in_heading_keeps_own_heading(self):
        text = "# A\ntext a\n## 23.08.2026 Konzert\nBeschreibung X\n## Nächstes\nandere"
        self.assertEqual(ev.cut_section(text, "23.08.2026"),
                         "## 23.08.2026 Konzert\nBeschreibung X")

    def test_inline_locator_cuts_at_next_heading(self):
        text = "Vorspann\nAm 23.08.2026 Konzert.\n## Nächstes\nanderer text"
        self.assertEqual(ev.cut_section(text, "23.08.2026"), "Am 23.08.2026 Konzert.")

    def test_date_before_heading_keeps_event_title_and_description(self):
        text = (
            "Heute, 18:45\n"
            "## Linkswende kennenlernen & aktiv werden\n"
            "Im Oktober stehen jede Menge Proteste an.\n"
            "Morgen, 12:00\n"
            "## Schulstreik gegen das Kopftuchverbot\n"
            "Der nächste Eventtext."
        )
        section = ev.cut_section(
            text, "Heute, 18:45", title="Linkswende kennenlernen & aktiv werden")
        self.assertIn("Heute, 18:45", section)
        self.assertIn("## Linkswende kennenlernen & aktiv werden", section)
        self.assertIn("Im Oktober stehen jede Menge Proteste an.", section)
        self.assertNotIn("Morgen, 12:00", section)
        self.assertNotIn("Schulstreik", section)

    def test_missing_locator_returns_head_of_text(self):
        self.assertEqual(ev.cut_section("ganz was anderes", "23.08.2026"), "ganz was anderes")

    def test_empty_locator_returns_head(self):
        self.assertEqual(ev.cut_section("abc def", ""), "abc def")

    def test_cap_respected(self):
        self.assertLessEqual(len(ev.cut_section("wort " * 5000, "wort", cap=200)), 200)


class SlugTests(unittest.TestCase):
    def test_slugify_umlauts_and_hooks(self):
        self.assertEqual(ev.slugify("Mala Z\u00ear\u00een"), "mala-zerin")
        self.assertEqual(ev.slugify("Anarchistische Bibliothek"), "anarchistische-bibliothek")
        self.assertEqual(ev.slugify("Pal\u00e4stina Solidarit\u00e4t \u00d6sterreich"),
                         "palaestina-solidaritaet-oesterreich")
        self.assertEqual(ev.slugify("DerFunke / RKP"), "derfunke-rkp")
        self.assertEqual(ev.slugify(""), "")

    def test_url_slug_skips_generic_segments(self):
        self.assertEqual(ev.url_slug("https://a.org/2026-09-28/buchvorstellung-x/"),
                         "buchvorstellung-x")
        self.assertEqual(ev.url_slug("https://mala-zerin.at/cinema-politica"),
                         "cinema-politica")
        self.assertEqual(ev.url_slug("https://derfunke.at/termin/klassiker-des-marxismus"),
                         "klassiker-des-marxismus")

    def test_url_slug_falls_back_to_generic_last_segment(self):
        self.assertEqual(ev.url_slug("https://linkswende.org/kalender/"), "kalender")
        self.assertEqual(ev.url_slug("https://a.org/"), "")

    def test_url_slug_ignores_query_and_fragment(self):
        self.assertEqual(ev.url_slug("https://a.org/event/x?troop=1#top"), "x")

    def test_url_slug_strips_file_extension(self):
        self.assertEqual(ev.url_slug("https://a.org/bericht.html"), "bericht")


class ReadablePathTests(unittest.TestCase):
    def test_overview_named_by_orga_folder(self):
        record = {"url": "https://a.org/t", "kind": "overview",
                  "orga": "Mala Z\u00ear\u00een", "orga_id": "64667744-e190-46b8"}
        self.assertEqual(ev.readable_rel_path(record),
                         "64667744_mala-zerin/overview.json")

    def test_detail_uses_url_segment(self):
        record = {"url": "https://mala-zerin.at/cinema-politica", "kind": "detail",
                  "orga": "Jelka", "orga_id": "3c8373bd-636d"}
        self.assertEqual(ev.readable_rel_path(record),
                         "3c8373bd_jelka/detail_cinema-politica.json")

    def test_missing_orga_lands_in_unassigned(self):
        record = {"url": "https://a.org/x", "kind": "detail", "orga": "", "orga_id": ""}
        self.assertTrue(ev.readable_rel_path(record).startswith("_unassigned/"))

    def test_collision_gets_hash_suffix_only_for_other_url(self):
        rel = "3c8373bd_jelka/detail_x.json"
        self.assertEqual(ev.resolve_readable(rel, "https://a.org/x", {}), rel)
        self.assertEqual(
            ev.resolve_readable(rel, "https://a.org/x", {rel: "https://a.org/x"}), rel)
        other = ev.resolve_readable(rel, "https://b.org/x", {rel: "https://a.org/x"})
        self.assertNotEqual(other, rel)
        self.assertIn(ev.page_hash("https://b.org/x")[:8], other)
        self.assertTrue(other.startswith("3c8373bd_jelka/detail_x-"))


class ReadableMirrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._pages, self._index = ev.PAGES_DIR, ev.PAGES_INDEX
        ev.PAGES_DIR, ev.PAGES_INDEX = self.dir / "pages", self.dir / "pages" / "index.json"

    def tearDown(self):
        ev.PAGES_DIR, ev.PAGES_INDEX = self._pages, self._index
        self.tmp.cleanup()

    def _record(self, url="https://a.org/e1", kind="detail", orga="Jelka", oid="3c8373bd-1"):
        record = ev.new_page_record(url, kind, orga, oid)
        record.update(status="ok", markdown="# Hi", text="Hi", html="<h1>Hi</h1>" * 40)
        return record

    def test_mirror_is_written_and_drops_html(self):
        ev.save_page(self._record())
        mirror = ev.readable_dir() / "3c8373bd_jelka" / "detail_e1.json"
        self.assertTrue(mirror.exists())
        data = json.loads(mirror.read_text(encoding="utf-8"))
        self.assertNotIn("html", data)
        self.assertEqual(data["markdown"], "# Hi")
        self.assertEqual(data["orga"], "Jelka")

    def test_index_records_readable_path(self):
        ev.save_page(self._record())
        entry = ev.load_index()["https://a.org/e1"]
        self.assertEqual(entry["readable"], "3c8373bd_jelka/detail_e1.json")

    def test_readable_false_writes_no_mirror(self):
        ev.save_page(self._record(), readable=False)
        self.assertFalse(any((ev.readable_dir()).rglob("*.json")))
        self.assertNotIn("readable", ev.load_index()["https://a.org/e1"])

    def test_readable_false_does_not_keep_stale_index_reference(self):
        ev.save_page(self._record())
        updated = self._record()
        updated["markdown"] = "# Updated"
        ev.save_page(updated, readable=False)
        self.assertNotIn("readable", ev.load_index()["https://a.org/e1"])

    def test_failed_refresh_keeps_last_good_corpus(self):
        previous = self._record()
        ev.save_page(previous)
        attempted = ev.new_page_record(previous["url"], "detail", "Jelka", "3c8373bd-1")
        result = fetch.save_fetch_failure(attempted, "Timeout", readable=False)
        saved = ev.load_page(previous["url"])
        entry = ev.load_index()[previous["url"]]
        self.assertEqual(result["status"], "error")
        self.assertEqual(saved["status"], "ok")
        self.assertEqual(saved["html"], previous["html"])
        self.assertEqual(saved["last_fetch_error"], "Timeout")
        self.assertEqual(entry["last_fetch_error"], "Timeout")
        self.assertIn("readable", entry)
        empty_result = fetch.save_fetch_failure(
            attempted, "Seite ohne lesbaren Text", readable=False, status="empty")
        self.assertEqual(empty_result["status"], "empty")
        self.assertEqual(ev.load_page(previous["url"])["html"], previous["html"])

    def test_original_keeps_html(self):
        ev.save_page(self._record())
        self.assertIn("html", ev.load_page("https://a.org/e1"))

    def test_second_distinct_url_does_not_overwrite_first(self):
        ev.save_page(self._record("https://a.org/x"))
        ev.save_page(self._record("https://b.org/x"))
        folder = ev.readable_dir() / "3c8373bd_jelka"
        urls = {json.loads(p.read_text(encoding="utf-8"))["url"] for p in folder.glob("*.json")}
        self.assertEqual(len(urls), 2)
        self.assertEqual(urls, {"https://a.org/x", "https://b.org/x"})
        for entry in ev.load_index().values():
            self.assertTrue((ev.readable_dir() / entry["readable"]).exists())

    def test_resave_same_url_reuses_same_mirror_name(self):
        ev.save_page(self._record("https://a.org/x"))
        ev.save_page(self._record("https://a.org/x"))
        files = list((ev.readable_dir() / "3c8373bd_jelka").glob("*.json"))
        self.assertEqual(len(files), 1)


class ReadableBackfillTests(unittest.TestCase):
    """--readable-only: Spiegel aus bestehendem Corpus, ohne Netz."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._pages, self._index = ev.PAGES_DIR, ev.PAGES_INDEX
        ev.PAGES_DIR, ev.PAGES_INDEX = self.dir / "pages", self.dir / "pages" / "index.json"
        ev.PAGES_DIR.mkdir(parents=True)
        self.sources = self.dir / "sources.csv"
        self.sources.write_text(
            "orga,orga_id,url\n"
            "Jelka,3c8373bd-636d-4f19,https://jelka.at/veranstaltungen\n", encoding="utf-8")
        self.discovery = self.dir / "discovery.json"

    def tearDown(self):
        ev.PAGES_DIR, ev.PAGES_INDEX = self._pages, self._index
        self.tmp.cleanup()

    def test_backfill_recovers_orga_from_sources(self):
        """Alter Corpus-Eintrag ohne orga-Spalte muss trotzdem benannt werden."""
        record = ev.new_page_record("https://jelka.at/veranstaltungen", "overview")
        record.update(status="ok", markdown="x", html="<p>x</p>")
        ev.save_page(record, readable=False)
        self.assertNotIn("orga", ev.load_index()["https://jelka.at/veranstaltungen"])
        written, stale = fetch.rebuild_readable(self.sources, self.discovery)
        self.assertEqual(written, 1)
        entry = ev.load_index()["https://jelka.at/veranstaltungen"]
        self.assertEqual(entry["readable"], "3c8373bd_jelka/overview.json")
        mirror = ev.readable_dir() / "3c8373bd_jelka" / "overview.json"
        self.assertEqual(json.loads(mirror.read_text(encoding="utf-8"))["orga"], "Jelka")

    def test_backfill_replaces_stale_mirror_files(self):
        record = ev.new_page_record("https://jelka.at/veranstaltungen", "overview", "Jelka",
                                    "3c8373bd-636d-4f19")
        record.update(status="ok", markdown="x", html="<p>x</p>")
        ev.save_page(record)
        stray = ev.readable_dir() / "_unassigned" / "ghost.json"
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_text("{}", encoding="utf-8")
        written, stale = fetch.rebuild_readable(self.sources, self.discovery)
        self.assertEqual(stale, 2)          # echtes Mirror + Geist werden beide ersetzt
        self.assertFalse(stray.exists())
        self.assertEqual(written, 1)
        self.assertTrue((ev.readable_dir() / "3c8373bd_jelka" / "overview.json").exists())

    def test_backfill_removes_orphan_directories(self):
        """Ein Umbenennen der Orga-Slugs darf keine leeren Alt-Ordner hinterlassen."""
        record = ev.new_page_record("https://jelka.at/x", "overview", "Jelka", "3c8373bd-636d-4f19")
        record.update(status="ok", markdown="x", html="<p>x</p>")
        ev.save_page(record)
        orphan = ev.readable_dir() / "3c8373bd_ganzer-alter-vereinsname-mit-x-belang"
        orphan.mkdir(parents=True, exist_ok=True)
        (orphan / "detail_alt.json").write_text("{}", encoding="utf-8")
        fetch.rebuild_readable(self.sources, self.discovery)
        self.assertFalse(orphan.exists())
        self.assertEqual(sorted(d.name for d in ev.readable_dir().iterdir() if d.is_dir()),
                         ["3c8373bd_jelka"])

    def test_backfill_is_idempotent(self):
        record = ev.new_page_record("https://jelka.at/x", "detail", "Jelka", "3c8373bd-636d-4f19")
        record.update(status="ok", markdown="x", html="<p>x</p>")
        ev.save_page(record)
        first, _ = fetch.rebuild_readable(self.sources, self.discovery)
        second, _ = fetch.rebuild_readable(self.sources, self.discovery)
        self.assertEqual(first, second)
        files = list(ev.readable_dir().rglob("*.json"))
        self.assertEqual(len(files), 1)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._pages = ev.PAGES_DIR
        self._index = ev.PAGES_INDEX
        ev.PAGES_DIR = self.dir
        ev.PAGES_INDEX = self.dir / "index.json"

    def tearDown(self):
        ev.PAGES_DIR = self._pages
        ev.PAGES_INDEX = self._index
        self.tmp.cleanup()

    def test_save_load_roundtrip_updates_index(self):
        record = ev.new_page_record("https://a.org/x/", "overview")
        record.update(status="ok", markdown="# Hi")
        ev.save_page(record)
        loaded = ev.load_page("https://a.org/x")
        self.assertEqual(loaded["markdown"], "# Hi")
        index = ev.load_index()
        self.assertIn("https://a.org/x", index)
        self.assertEqual(index["https://a.org/x"]["kind"], "overview")

    def test_load_missing_page_returns_none(self):
        self.assertIsNone(ev.load_page("https://a.org/niemals"))

    def test_new_record_shape(self):
        record = ev.new_page_record("https://a.org", "detail")
        for key in ("url", "kind", "status", "markdown", "text", "html",
                    "links", "images", "jsonld"):
            self.assertIn(key, record)
        self.assertEqual(record["status"], "pending")

    def test_corrupt_index_treated_as_empty(self):
        ev.PAGES_INDEX.write_text("{kaputt", encoding="utf-8")
        self.assertEqual(ev.load_index(), {})


class CorpusStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._pages, self._index = ev.PAGES_DIR, ev.PAGES_INDEX
        ev.PAGES_DIR, ev.PAGES_INDEX = self.dir / "pages", self.dir / "pages" / "index.json"
        self.sources = self.dir / "sources.csv"
        self.sources.write_text("orga,orga_id,url\n", encoding="utf-8")
        self.discovery = self.dir / "discovery.json"
        self.discovery.write_text('{"events":[]}', encoding="utf-8")

    def tearDown(self):
        ev.PAGES_DIR, ev.PAGES_INDEX = self._pages, self._index
        self.tmp.cleanup()

    def _save(self, url, kind, markdown):
        record = ev.new_page_record(url, kind, "Orga", "orga-id")
        record.update(status="ok", markdown=markdown, text=markdown, html="<p>Text</p>")
        ev.save_page(record)

    def test_success_prunes_stale_same_kind_and_preserves_other_kind(self):
        overview_url = "https://a.org/overview"
        stale_overview = "https://a.org/old-overview"
        detail_url = "https://a.org/event"
        self._save(overview_url, "overview", "Old overview")
        self._save(stale_overview, "overview", "Stale overview")
        self._save(detail_url, "detail", "Keep detail")

        with fetch.CorpusStage() as staged:
            updated = ev.load_page(overview_url)
            updated["markdown"] = "Fresh overview"
            ev.save_page(updated)
            removed = staged.prune_kind("overview", {overview_url})
            self.assertEqual(removed, 1)
            staged.publish(True, self.sources, self.discovery)

        self.assertEqual(ev.load_page(overview_url)["markdown"], "Fresh overview")
        self.assertIsNone(ev.load_page(stale_overview))
        self.assertEqual(ev.load_page(detail_url)["markdown"], "Keep detail")
        index = ev.load_index()
        self.assertEqual(set(index), {overview_url, detail_url})
        mirror_files = list(ev.readable_dir().rglob("*.json"))
        self.assertEqual(len(mirror_files), 2)

    def test_unpublished_stage_rolls_back_all_writes(self):
        url = "https://a.org/overview"
        self._save(url, "overview", "Original")

        with fetch.CorpusStage():
            changed = ev.load_page(url)
            changed["markdown"] = "Uncommitted"
            ev.save_page(changed)
            self._save("https://a.org/new", "overview", "Uncommitted page")

        self.assertEqual(ev.load_page(url)["markdown"], "Original")
        self.assertIsNone(ev.load_page("https://a.org/new"))
        self.assertEqual(set(ev.load_index()), {url})

    def test_success_without_pruning_keeps_non_target_pages(self):
        old_url = "https://a.org/old-event"
        self._save(old_url, "detail", "Keep during partial run")

        with fetch.CorpusStage() as staged:
            self._save("https://a.org/new-event", "detail", "New event")
            staged.publish(True, self.sources, self.discovery)

        self.assertIsNotNone(ev.load_page(old_url))
        self.assertIsNotNone(ev.load_page("https://a.org/new-event"))


class EventTextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self._pages, self._index = ev.PAGES_DIR, ev.PAGES_INDEX
        ev.PAGES_DIR, ev.PAGES_INDEX = self.dir, self.dir / "index.json"

    def tearDown(self):
        ev.PAGES_DIR, ev.PAGES_INDEX = self._pages, self._index
        self.tmp.cleanup()

    def _seed(self, url, markdown, status="ok"):
        record = ev.new_page_record(url, "detail")
        record.update(status=status, markdown=markdown)
        ev.save_page(record)

    def test_markdown_preferred_over_content_markdown(self):
        record = ev.new_page_record("https://a.org/e1", "detail")
        record.update(status="ok", markdown="NAV VOLLTEXT",
                      content_markdown="BEREINIGT")
        ev.save_page(record)
        self.assertEqual(ev.page_text(ev.load_page("https://a.org/e1")),
                         "NAV VOLLTEXT")

    def test_content_markdown_not_used_when_markdown_is_empty(self):
        record = ev.new_page_record("https://a.org/e2", "detail")
        record.update(status="ok", content_markdown="ALT CLEANED TEXT",
                      text="ALT PLAINTEXT")
        ev.save_page(record)
        self.assertEqual(ev.page_text(ev.load_page("https://a.org/e2")),
                         "")

    def test_content_source_selects_content_markdown(self):
        record = ev.new_page_record("https://a.org/e3", "detail")
        record.update(status="ok", markdown="NAV VOLLTEXT",
                      content_markdown="BEREINIGT")
        ev.save_page(record)
        page = ev.load_page("https://a.org/e3")
        self.assertEqual(ev.page_text(page, "content"), "BEREINIGT")
        text, source = ev.event_text(
            {"event_url": "https://a.org/e3", "source_url": ""}, "content")
        self.assertEqual((text, source), ("BEREINIGT", "detail"))

    def test_content_source_has_no_silent_fallback_to_raw(self):
        self._seed("https://a.org/e4", "NUR ROH")
        text, source = ev.event_text(
            {"event_url": "https://a.org/e4", "source_url": ""}, "content")
        self.assertEqual((text, source), ("", "none"))

    def test_detail_preferred_over_overview(self):
        self._seed("https://a.org/e1", "DETAILTEXT")
        self._seed("https://a.org/t", "UEBERSICHT")
        text, source = ev.event_text({
            "event_url": "https://a.org/e1", "source_url": "https://a.org/t",
            "listed_date_text": "23.08",
        })
        self.assertEqual(source, "detail")
        self.assertEqual(text, "DETAILTEXT")

    def test_inline_event_uses_cut_overview(self):
        self._seed("https://a.org/t", "Intro\n## 23.08 Konzert\nNur hier.\n## 24.08 Anderes\nweg")
        text, source = ev.event_text({
            "event_url": "", "source_url": "https://a.org/t", "listed_date_text": "23.08",
        })
        self.assertEqual(source, "overview_cut")
        self.assertIn("Nur hier.", text)
        self.assertNotIn("weg", text)

    def test_inline_event_date_before_title_keeps_its_own_section(self):
        self._seed(
            "https://a.org/t",
            "Heute, 18:45\n## Linkswende kennenlernen\nBeschreibung eins.\n"
            "Morgen, 12:00\n## Schulstreik\nBeschreibung zwei.",
        )
        text, source = ev.event_text({
            "event_url": "",
            "source_url": "https://a.org/t",
            "listed_date_text": "Heute, 18:45",
            "title": "Linkswende kennenlernen",
        })
        self.assertEqual(source, "overview_cut")
        self.assertIn("Linkswende kennenlernen", text)
        self.assertIn("Beschreibung eins.", text)
        self.assertNotIn("Morgen, 12:00", text)
        self.assertNotIn("Schulstreik", text)

    def test_missing_detail_page_falls_back_to_overview(self):
        self._seed("https://a.org/t", "UEBERSICHTSTEXT")
        text, source = ev.event_text({
            "event_url": "https://a.org/nichtda", "source_url": "https://a.org/t",
            "listed_date_text": "23.08",
        })
        self.assertEqual(source, "overview")
        self.assertEqual(text, "UEBERSICHTSTEXT")

    def test_error_page_not_used(self):
        self._seed("https://a.org/e1", "DETAIL", status="error")
        self._seed("https://a.org/t", "UEBERSICHT")
        _, source = ev.event_text({
            "event_url": "https://a.org/e1", "source_url": "https://a.org/t",
            "listed_date_text": "",
        })
        self.assertEqual(source, "overview")

    def test_nothing_in_corpus(self):
        text, source = ev.event_text({"event_url": "", "source_url": "https://a.org/x"})
        self.assertEqual(text, "")
        self.assertEqual(source, "none")

    def test_jsonld_hint_read_from_detail_first(self):
        self._seed("https://a.org/e1", '<p>x</p>')
        ev.save_page({**ev.new_page_record("https://a.org/e1", "detail"),
                      "status": "ok",
                      "jsonld": [{"@type": "Event", "name": "Konzert",
                                  "location": {"name": "Depot"}}]})
        hint = ev.jsonld_hint_for({"event_url": "https://a.org/e1", "source_url": ""})
        self.assertIn("Depot", hint)


class DiscoverTests(unittest.TestCase):
    def test_normalize_event_absolutizes_relative_url(self):
        out = discover.normalize_event("https://a.org/termine/",
                                       {"title": " X ", "url": "/event/1", "listed_date_text": "23.08"})
        self.assertEqual(out["event_url"], "https://a.org/event/1")
        self.assertEqual(out["title"], "X")

    def test_normalize_event_inline_stays_empty(self):
        for raw in ({"title": "T", "url": "", "listed_date_text": ""},
                    {"title": "T", "url": "#anchor", "listed_date_text": ""},
                    {"title": "T", "url": "https://a.org/termine", "listed_date_text": ""}):
            self.assertEqual(discover.normalize_event("https://a.org/termine/", raw)["event_url"], "", raw)

    def test_normalize_event_drops_titleless(self):
        self.assertIsNone(discover.normalize_event("https://a.org/", {"title": "  ", "url": "", "listed_date_text": ""}))

    def test_normalize_event_rejects_non_http(self):
        out = discover.normalize_event("https://a.org/", {"title": "T", "url": "javascript:x", "listed_date_text": ""})
        self.assertEqual(out["event_url"], "")

    def test_normalize_event_requires_url_in_source_links(self):
        links = [{"href": "https://a.org/events/known", "text": "T"}]
        accepted = discover.normalize_event(
            "https://a.org/", {"title": "T", "url": "/events/known"}, links)
        rejected = discover.normalize_event(
            "https://a.org/", {"title": "T", "url": "/events/invented"}, links)
        self.assertEqual(accepted["event_url"], "https://a.org/events/known")
        self.assertEqual(rejected["event_url"], "")

    def test_normalize_event_drops_monthly_archive_pages(self):
        for title in ("Monatsprogramm 09/2026", "Monatsprogramm 08/26",
                      "Monatsprogramm 07/2026", "Monatsprogramm März 2026"):
            with self.subTest(title=title):
                self.assertIsNone(discover.normalize_event(
                    "https://a.org/", {"title": title, "url": "/archive"}))
        self.assertIsNotNone(discover.normalize_event(
            "https://a.org/", {"title": "Herbstfest 2026", "url": "/herbstfest"}))

    def test_dedupe_collapses_title_url_pairs(self):
        events = [{"title": "A", "event_url": "u", "listed_date_text": ""},
                  {"title": "a", "event_url": "u", "listed_date_text": ""},
                  {"title": "A", "event_url": "v", "listed_date_text": ""}]
        self.assertEqual(len(discover.dedupe(events)), 2)


class FetchTargetTests(unittest.TestCase):
    TARGETS = [
        {"url": "https://a.org/overview", "kind": "overview"},
        {"url": "https://a.org/event", "kind": "detail"},
    ]

    def test_run_config_excludes_observed_consent_ui(self):
        config = fetch.build_run_config(0, 0, 1000)
        self.assertIn(".jelka-consent", config.excluded_selector)
        self.assertIn("#cmplz-cookiebanner-container", config.excluded_selector)
        self.assertIn(".cky-consent-container", config.excluded_selector)

    def test_consent_ui_removed_before_markdown_without_losing_page_content(self):
        from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        html = """
        <html><body>
          <section class="jelka-consent">Cookie consent notice</section>
          <main>
            <h1>Event title</h1>
            <p>The event discusses cookie laws.</p>
          </main>
        </body></html>
        """
        config = fetch.build_run_config(0, 0, 1000)
        scraped = LXMLWebScrapingStrategy().scrap(
            "https://example.org/events",
            html,
            excluded_selector=config.excluded_selector,
            word_count_threshold=config.word_count_threshold,
        )
        markdown = DefaultMarkdownGenerator().generate_markdown(
            scraped.cleaned_html
        ).raw_markdown

        self.assertNotIn("Cookie consent notice", markdown)
        self.assertIn("Event title", markdown)
        self.assertIn("The event discusses cookie laws.", markdown)

    def test_regular_run_reloads_every_target(self):
        original = ev.load_index
        ev.load_index = lambda: {
            "https://a.org/overview": {"status": "ok"},
            "https://a.org/event": {"status": "ok"},
        }
        try:
            self.assertEqual(fetch.targets_to_fetch(self.TARGETS, retry_failed=False),
                             self.TARGETS)
        finally:
            ev.load_index = original

    def test_retry_failed_only_selects_missing_or_failed_targets(self):
        original = ev.load_index
        ev.load_index = lambda: {
            "https://a.org/overview": {"status": "ok"},
            "https://a.org/event": {"status": "ok", "last_fetch_error": "Timeout"},
        }
        try:
            self.assertEqual(fetch.targets_to_fetch(self.TARGETS, retry_failed=True),
                             [self.TARGETS[1]])
        finally:
            ev.load_index = original


class ApiGateTests(unittest.TestCase):
    FLAT = "  a \n b \t c "

    def test_flat_collapses_whitespace(self):
        self.assertEqual(discover._flat(self.FLAT), "a b c")

    TRIBE_HTML = ('<html><head>'
                  '<link rel="stylesheet" href="/wp-content/plugins/the-events-calendar/src/resources/css/x">'
                  '<script>var tribe = {};</script>'
                  '<link rel="https://api.w.org/" href="https://a.org/wp-json/">'
                  '</head><body>Events</body></html>')

    def test_detect_tribe_needs_plugin_marker(self):
        self.assertTrue(api_gate.detect_tribe(self.TRIBE_HTML))
        self.assertFalse(api_gate.detect_tribe("<html><body>Kalender</body></html>"))
        self.assertFalse(api_gate.detect_tribe(""))

    def test_tribe_endpoint_uses_origin(self):
        self.assertEqual(api_gate.tribe_endpoint("https://jelka.at/veranstaltungen/"),
                         "https://jelka.at/wp-json/tribe/events/v1/events")

    def test_gcal_url_carries_calendar_and_key(self):
        url = api_gate.gcal_url("linkswendejetzt@gmail.com", "KEY123")
        self.assertIn("linkswendejetzt%40gmail.com", url)
        self.assertIn("key=KEY123", url)
        self.assertIn("maxResults=1", url)

    def test_probe_tribe_accepts_event_list(self):
        orig = api_gate.probe_json
        api_gate.probe_json = lambda url, timeout=25: ({"events": []}, "")
        try:
            self.assertEqual(api_gate.probe_tribe("https://a.org/x"), (True, ""))
        finally:
            api_gate.probe_json = orig

    def test_probe_tribe_rejects_wrong_shape(self):
        orig = api_gate.probe_json
        api_gate.probe_json = lambda url, timeout=25: ({"code": "rest_not_logged_in"}, "")
        try:
            ok, error = api_gate.probe_tribe("https://a.org/x")
            self.assertFalse(ok)
            self.assertTrue(error)
        finally:
            api_gate.probe_json = orig

    def test_probe_gcal_accepts_calendar_events(self):
        orig = api_gate.probe_json
        api_gate.probe_json = lambda url, timeout=25: ({"kind": "calendar#events"}, "")
        try:
            self.assertEqual(api_gate.probe_gcal("https://a.org/x"), (True, ""))
        finally:
            api_gate.probe_json = orig

    def test_probe_gcal_reports_api_error(self):
        orig = api_gate.probe_json
        api_gate.probe_json = lambda url, timeout=25: ({"error": {"code": 403}}, "")
        try:
            ok, error = api_gate.probe_gcal("https://a.org/x")
            self.assertFalse(ok)
            self.assertIn("API-Fehler", error)
        finally:
            api_gate.probe_json = orig

    def test_probe_json_reports_http_error(self):
        import urllib.request as ureq

        class BadResponse:
            status = 401

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        real = ureq.urlopen
        ureq.urlopen = lambda req, timeout=None, context=None: BadResponse()
        try:
            data, error = api_gate.probe_json("https://a.org/x")
            self.assertIsNone(data)
            self.assertIn("401", error)
        finally:
            ureq.urlopen = real

    def test_manual_config_missing_file_is_empty(self):
        self.assertEqual(api_gate.read_manual_config(Path("/nope/none.csv")), {})

    def test_check_source_without_corpus_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_pages, old_index = ev.PAGES_DIR, ev.PAGES_INDEX
            ev.PAGES_DIR = Path(tmp)
            ev.PAGES_INDEX = Path(tmp) / "index.json"
            try:
                tag = api_gate.check_source(
                    {"orga": "X", "url": "https://a.org/nie-da"}, {})
                self.assertFalse(tag["reachable"])
                self.assertEqual(tag["component"], "")
            finally:
                ev.PAGES_DIR, ev.PAGES_INDEX = old_pages, old_index

    def test_check_source_tags_reachable_tribe(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_pages, old_index = ev.PAGES_DIR, ev.PAGES_INDEX
            ev.PAGES_DIR = Path(tmp)
            ev.PAGES_INDEX = Path(tmp) / "index.json"
            try:
                record = ev.new_page_record("https://a.org/t", "overview")
                record.update(status="ok", html=self.TRIBE_HTML)
                ev.save_page(record, readable=False)
                orig = api_gate.probe_tribe
                api_gate.probe_tribe = lambda endpoint: (True, "")
                try:
                    tag = api_gate.check_source(
                        {"orga": "A", "url": "https://a.org/t"}, {})
                finally:
                    api_gate.probe_tribe = orig
                self.assertTrue(tag["reachable"])
                self.assertEqual(tag["component"], "tribe")
                self.assertIn("/wp-json/tribe/events/v1/events", tag["endpoint"])
            finally:
                ev.PAGES_DIR, ev.PAGES_INDEX = old_pages, old_index

    def test_check_source_skips_unreachable_tribe(self):
        """DerFunke-Fall: Marker da, Endpoint 401 -> kein Tag."""
        with tempfile.TemporaryDirectory() as tmp:
            old_pages, old_index = ev.PAGES_DIR, ev.PAGES_INDEX
            ev.PAGES_DIR = Path(tmp)
            ev.PAGES_INDEX = Path(tmp) / "index.json"
            try:
                record = ev.new_page_record("https://a.org/t", "overview")
                record.update(status="ok", html=self.TRIBE_HTML)
                ev.save_page(record, readable=False)
                orig = api_gate.probe_tribe
                api_gate.probe_tribe = lambda endpoint: (False, "HTTP 401")
                try:
                    tag = api_gate.check_source(
                        {"orga": "A", "url": "https://a.org/t"}, {})
                finally:
                    api_gate.probe_tribe = orig
                self.assertFalse(tag["reachable"])
                self.assertEqual(tag["component"], "tribe")
            finally:
                ev.PAGES_DIR, ev.PAGES_INDEX = old_pages, old_index

    def test_discover_load_api_sources_missing_is_empty(self):
        self.assertEqual(discover.load_api_sources(Path("/nope/none.json")), {})

    def test_link_index_skips_same_page_and_formats(self):
        page = {"links": [
            {"href": "https://a.org/termine#e1", "text": "Sprung"},
            {"href": "/event/1", "text": "Konzert "},
            {"href": "https://other.org/x", "text": ""},
        ]}
        out = discover.link_index(page, "https://a.org/termine/")
        self.assertNotIn("Sprung", out)
        self.assertIn("Konzert -> /event/1", out)
        self.assertIn("https://other.org/x", out)


    """Normalisierte Felder: Beleg ist die Quellform, nicht der ISO-Wert."""

    SRC = "Fr 25.09.2026 19:30 VoKü | 20:00 Film"

    def test_iso_from_german_date_is_high(self):
        self.assertEqual(
            ev.resolve_derived("2026-09-25", "Fr 25.09.2026", self.SRC,
                               ev.parse_de_dates), ("2026-09-25", "high"))

    def test_time_from_text_is_high(self):
        self.assertEqual(
            ev.resolve_derived("20:00", "20:00 Film", self.SRC,
                               ev.parse_de_times), ("20:00", "high"))

    def test_quoted_date_disagreeing_with_value_is_none(self):
        "Text A zitieren und Wert B behaupten muss durchfallen."
        self.assertEqual(
            ev.resolve_derived("2026-08-23", "Fr 25.09.2026", self.SRC,
                               ev.parse_de_dates), ("", "none"))

    def test_invented_quote_is_none(self):
        self.assertEqual(
            ev.resolve_derived("2026-09-25", "25.09.2099", self.SRC,
                               ev.parse_de_dates), ("", "none"))

    def test_relative_wording_without_number_is_derived(self):
        self.assertEqual(
            ev.resolve_derived("2026-09-25", "nächsten Freitag",
                               "nächsten Freitag im Depot",
                               ev.parse_de_dates), ("2026-09-25", "derived"))

    def test_yearless_date_backfilled_from_value(self):
        self.assertEqual(
            ev.resolve_derived("2026-09-25", "25.09.", "Fr 25.09. Film",
                               ev.parse_de_dates), ("2026-09-25", "high"))

    def test_empty_value_is_none(self):
        self.assertEqual(
            ev.resolve_derived("", "x", self.SRC, ev.parse_de_dates), ("", "none"))

    def test_second_of_two_times_is_accepted(self):
        """Der reale Fall: Küche und Lesung im selben Absatz."""
        quote = "Beginn: Vo-Kü 19:30 Buchvorstellung 20:00"
        self.assertEqual(
            ev.resolve_derived("20:00", quote, quote, ev.parse_de_times), ("20:00", "high"))
        self.assertEqual(
            ev.resolve_derived("19:30", quote, quote, ev.parse_de_times), ("19:30", "high"))

    def test_value_absent_from_multi_candidate_quote_is_none(self):
        quote = "Vo-Kü 19:30 Buchvorstellung 20:00"
        self.assertEqual(
            ev.resolve_derived("21:00", quote, quote, ev.parse_de_times), ("", "none"))

    def test_empty_value_with_two_candidates_is_none(self):
        """Modell zitiert den Stretch, legt sich aber nicht fest: keine Wahl."""
        quote = "Vo-Kü 19:30 Buchvorstellung 20:00"
        self.assertEqual(
            ev.resolve_derived("", quote, quote, ev.parse_de_times), ("", "none"))

    def test_second_of_two_dates_is_accepted(self):
        quote = "Ausstellung: 06.11. – 27.11.2026"
        self.assertEqual(
            ev.resolve_derived("2026-11-27", quote, quote, ev.parse_de_dates),
            ("2026-11-27", "high"))
        self.assertEqual(
            ev.resolve_derived("2026-11-06", quote, quote, ev.parse_de_dates),
            ("2026-11-06", "high"))

    def test_poster_typography_in_value_is_canonised(self):
        """Der reale Dinev-Fall: Modell schreibt "20°°" in das Wertfeld."""
        quote = "Beginn | Vo-Kü 19:30 | Lesung 20°°"
        self.assertEqual(
            ev.resolve_derived("20°°", quote, quote, ev.parse_de_times), ("20:00", "high"))

    def test_german_date_in_value_is_canonised(self):
        self.assertEqual(
            ev.resolve_derived("25.09.2026", "Fr 25.09.2026", self.SRC,
                               ev.parse_de_dates), ("2026-09-25", "high"))

    def test_empty_quote_is_none(self):
        self.assertEqual(
            ev.resolve_derived("20:00", "", self.SRC, ev.parse_de_times), ("", "none"))


class PluralParserTests(unittest.TestCase):
    def test_parse_de_times_returns_all_in_order(self):
        self.assertEqual(ev.parse_de_times("19:30 und 20:00 und 19:30"),
                         ["19:30", "20:00"])

    def test_parse_de_times_mixed_typography(self):
        self.assertEqual(ev.parse_de_times("Vo-Kü 19:30 | Lesung 20°°"),
                         ["19:30", "20:00"])

    def test_parse_de_times_skips_impossible(self):
        self.assertEqual(ev.parse_de_times("25:00 und 20:00"), ["20:00"])

    def test_parse_de_dates_returns_all_in_order(self):
        self.assertEqual(ev.parse_de_dates("06.11. – 27.11.2026"),
                         ["2026-11-06", "2026-11-27"])

    def test_parse_de_dates_rejects_impossible_day(self):
        self.assertEqual(ev.parse_de_dates("31.02.2026"), [])

    def test_singular_wrappers_return_first(self):
        self.assertEqual(ev.parse_de_time("19:30 und 20:00"), "19:30")
        self.assertEqual(ev.parse_de_date("06.11. – 27.11.2026"), "2026-11-06")


class BasisAttachTests(unittest.TestCase):
    FIELDS = ("title", "start_date", "start_time")

    def _attach(self, data):
        target = {}
        errors = basis.attach(target, data, [], self.FIELDS, "identity")
        return target, errors

    def test_values_taken_without_quote(self):
        target, errors = self._attach(
            {"title": "Konzert", "start_date": "2026-08-23", "start_time": "20:00"})
        self.assertEqual(errors, [])
        self.assertEqual(target["title"], "Konzert")
        self.assertEqual(target["start_date"], "2026-08-23")
        self.assertEqual(target["start_time"], "20:00")

    def test_german_date_is_normalised(self):
        target, _ = self._attach({"title": "K", "start_date": "Do 08.10.", "start_time": "20.00"})
        self.assertEqual(target["start_date"], f"{ev.current_date().year}-10-08")
        self.assertEqual(target["start_time"], "20:00")

    def test_unreadable_date_cleared_with_error(self):
        target, errors = self._attach({"title": "K", "start_date": "bald", "start_time": ""})
        self.assertEqual(target["start_date"], "")
        self.assertTrue(any("start_date" in e for e in errors))

    def test_empty_value_is_none_not_error(self):
        target, errors = self._attach({"title": "", "start_date": "", "start_time": ""})
        self.assertEqual(errors, [])
        self.assertEqual(target["confidence_title"], "none")

    def test_no_answer_marks_all_none(self):
        target = {}
        errors = basis.attach(target, None, ["Timeout"], self.FIELDS, "identity")
        self.assertEqual(errors, ["identity: Timeout"])
        self.assertEqual(target["confidence_start_time"], "none")
        self.assertEqual(target["title"], "")


class VenueApplyTests(unittest.TestCase):
    SOURCE = "Im Depot, Breite Gasse 3, 1070 Wien. Saal: Alois Wagner-Saal."

    def _venue(self, data, errors=None, source=None):
        record = {}
        problems = basis.apply_venue(record, data, errors or [], source or self.SOURCE)
        return record, problems

    def test_full_address_accepted(self):
        record, problems = self._venue({
            "address": "Breite Gasse 3, 1070 Wien", "venue_name": "Depot",
            "confidence": "high",
        })
        self.assertEqual(problems, [])
        self.assertEqual(record["venue_address"], "Breite Gasse 3, 1070 Wien")
        self.assertEqual(record["venue_confidence"], "high")
        self.assertEqual(record["confidence_venue_address"], "llm")

    def test_address_source_kept_only_with_address(self):
        record, _ = self._venue({
            "address": "Breite Gasse 3, 1070 Wien", "address_source": "inferred",
            "venue_name": "Depot", "confidence": "high",
        })
        self.assertEqual(record["venue_address_source"], "inferred")
        record, _ = self._venue({
            "address": "", "address_source": "inferred",
            "venue_name": "Depot", "confidence": "venue_only",
        }, source="Im Depot")
        self.assertEqual(record["venue_address_source"], "")

    def test_partial_address_kept_and_noted(self):
        record, problems = self._venue({
            "address": "Breite Gasse", "venue_name": "",
            "confidence": "high",
        })
        self.assertEqual(record["venue_address"], "Breite Gasse")
        self.assertTrue(any("volle Anschrift" in p for p in problems))

    def test_address_not_in_source_is_kept(self):
        record, problems = self._venue({
            "address": "Falkengasse 9, 1020 Wien", "venue_name": "",
            "confidence": "high",
        })
        self.assertEqual(record["venue_address"], "Falkengasse 9, 1020 Wien")
        self.assertEqual(problems, [])

    def test_venue_name_only(self):
        record, _ = self._venue({
            "address": "", "venue_name": "Depot",
            "address": "", "venue_name": "Depot", "confidence": "venue_only",
        }, source="Findet im Depot statt.")
        self.assertEqual(record["venue_address"], "")
        self.assertEqual(record["venue_name"], "Depot")
        self.assertEqual(record["venue_confidence"], "venue_only")

    def test_online_claim_with_address_yields_address(self):
        record, _ = self._venue({
            "address": "Breite Gasse 3, 1070 Wien", "venue_name": "",
            "confidence": "online",
        })
        self.assertEqual(record["venue_confidence"], "high")

    def test_online_without_anything(self):
        record, _ = self._venue({
            "address": "", "venue_name": "", "confidence": "online",
        })
        self.assertEqual(record["venue_confidence"], "online")

    def test_no_answer_resets_everything(self):
        record, problems = self._venue(None, ["Timeout"])
        self.assertEqual(record["venue_confidence"], "none")
        self.assertEqual(record["confidence_venue_name"], "none")
        self.assertEqual(problems, ["venue: Timeout"])


class EventUrlReconcileTests(unittest.TestCase):
    """identity darf eine entdeckte URL bestätigen oder verwerfen, nie ersetzen."""

    SUPPLIED = "https://a.org/2026/09/28/ausstellung-06-11-27-11-2026"
    OTHER = "https://a.org/2026/09/28/fr-06-11-1800-ausstellungseroeffnung"

    def test_identical_url_is_accepted_silently(self):
        record = {"event_url": self.SUPPLIED}
        self.assertEqual(basis.reconcile_event_url(record, self.SUPPLIED, self.SUPPLIED), [])
        self.assertEqual(record["event_url"], self.SUPPLIED)

    def test_different_url_is_noted_but_never_written(self):
        """Der Korruptionsfall: Ersetzt hätte die Ausstellung auf die
        Eröffnungsseite gebogen."""
        record = {"event_url": self.SUPPLIED}
        problems = basis.reconcile_event_url(record, self.SUPPLIED, self.OTHER)
        self.assertEqual(len(problems), 1)
        self.assertIn("abweichende URL", problems[0])
        self.assertEqual(record["event_url"], self.SUPPLIED)
        self.assertNotIn("event_url_dropped", record)

    def test_empty_resolution_marks_dropped(self):
        record = {"event_url": self.SUPPLIED}
        problems = basis.reconcile_event_url(record, self.SUPPLIED, "")
        self.assertTrue(problems)
        self.assertTrue(record["event_url_dropped"])
        self.assertEqual(record["event_url"], self.SUPPLIED)

    def test_inline_event_has_nothing_to_reconcile(self):
        record = {"event_url": ""}
        self.assertEqual(basis.reconcile_event_url(record, "", self.OTHER), [])
        self.assertEqual(record["event_url"], "")

    def test_trailing_slash_difference_is_not_a_conflict(self):
        record = {"event_url": self.SUPPLIED}
        self.assertEqual(
            basis.reconcile_event_url(record, self.SUPPLIED, self.SUPPLIED + "/"), [])


class VenueGateTests(unittest.TestCase):
    def test_gate_wien_ausschluss_review(self):
        for address, expected in (
            ("Breite Gasse 3, 1070 Wien", (True, "")),
            ("Hauptplatz 3, 3500 Krems", (False, "outside_vienna")),
            ("", (False, "needs_review")),
            ("Parkplatz Bahnhof", (False, "needs_review")),
        ):
            self.assertEqual(basis.run_vienna_gate({"venue_address": address}), expected)


class DetailAttachTests(unittest.TestCase):
    SOURCE = "Workshop auf Farsi. Eintritt frei. Anmeldung erbeten bis Friday."

    def _attach(self, name, data, source=None):
        record = {}
        errors = detail.attach(record, data, [], detail.FIELDS[name], name,
                               source or self.SOURCE)
        return record, errors

    def test_language_enum_accepted(self):
        record, errors = self._attach("language", {
            "language": "OTHER", "language_name": "Farsi", "evidence": "auf Farsi",
        })
        self.assertEqual(record["language"], "other")
        self.assertEqual(errors, [])

    def test_language_out_of_range_cleared(self):
        record, errors = self._attach("language", {
            "language": "farsi", "language_name": "", "evidence": "auf Farsi",
        })
        self.assertEqual(record["language"], "")
        self.assertTrue(any("ausserhalb" in e for e in errors))

    def test_registration_requires_evidence_for_ja(self):
        record, errors = self._attach("registration", {
            "registration_required": "ja", "evidence": "Anmeldung erbeten",
        })
        self.assertEqual(record["registration_required"], "ja")
        record2, errors2 = self._attach("registration", {
            "registration_required": "ja", "evidence": "nichts im text",
        })
        self.assertEqual(record2["registration_required"], "")
        self.assertTrue(errors2)

    def test_category_enum(self):
        record, _ = self._attach("category", {"category": "workshop", "evidence": "Workshop"})
        self.assertEqual(record["category"], "workshop")

    def test_registration_unbekannt_is_legitimate_without_evidence(self):
        """"unbekannt" ist die korrekte Antwort auf Seiten ohne Anmeldehinweis."""
        record, errors = self._attach("registration", {
            "registration_required": "unbekannt", "evidence": "",
        })
        self.assertEqual(record["registration_required"], "unbekannt")
        self.assertEqual(record["confidence_registration_required"], "default")
        self.assertEqual(errors, [])

    def test_language_unknown_is_legitimate_without_evidence(self):
        record, errors = self._attach("language", {
            "language": "unknown", "language_name": "", "evidence": "",
        })
        self.assertEqual(record["language"], "unknown")
        self.assertEqual(record["confidence_language"], "default")
        self.assertEqual(errors, [])

    def test_category_sonstiges_is_legitimate_without_evidence(self):
        record, errors = self._attach("category", {"category": "sonstiges", "evidence": ""})
        self.assertEqual(record["category"], "sonstiges")
        self.assertEqual(record["confidence_category"], "default")
        self.assertEqual(errors, [])

    def test_description_needs_no_separate_quote(self):
        src = "Erster Satz hier. TEASER. Zweiter Satz auch."
        record, errors = self._attach("description", {
            "description": "Erster Satz hier. Zweiter Satz auch.",
        }, source=src)
        self.assertEqual(record["description"], "Erster Satz hier. Zweiter Satz auch.")
        self.assertEqual(errors, [])

    def test_description_markdown_bold_in_source_does_not_break_match(self):
        """Der reale Fall: Quelltext hat **fett**, Prompt räumt es weg."""
        src = "**Zeit der Mutigen |**_Roman_\nAls das Dienstmädchen Eva verliert ihre Unschuld."
        record, errors = self._attach("description", {
            "description": "Zeit der Mutigen | Roman\nAls das Dienstmädchen Eva verliert ihre Unschuld.",
        }, source=src)
        self.assertNotEqual(record["description"], "")
        self.assertEqual(errors, [])

    def test_description_invented_content_cleared(self):
        record, errors = self._attach("description", {
            "description": "Erster Satz hier. Und frei erfundene Zusatzinfo.",
        }, source="Erster Satz hier.")
        self.assertEqual(record["description"], "")
        self.assertTrue(errors)

    def test_empty_description_is_legitimate(self):
        record, errors = self._attach("description", {"description": ""})
        self.assertEqual(record["description"], "")
        self.assertEqual(errors, [])
        self.assertEqual(record["confidence_description"], "none")

    def test_timing_endpoints(self):
        source = "Vom 23.08.2026 bis 25.08.2026, jeweils Beginn 20:00."
        record, errors = self._attach("timing", {
            "end_date": "2026-08-25", "end_time": "",
            "evidence_end_date": "25.08.2026", "evidence_end_time": "",
        }, source=source)
        self.assertEqual(record["end_date"], "2026-08-25")
        self.assertEqual(errors, [])
        self.assertEqual(record["confidence_end_time"], "none")

    def test_timing_invented_end_cleared(self):
        record, errors = self._attach("timing", {
            "end_date": "2026-09-30", "end_time": "",
            "evidence_end_date": "30.09.2026", "evidence_end_time": "",
        }, source="Am 23.08.2026.")
        self.assertEqual(record["end_date"], "")
        self.assertTrue(errors)


class ModelConfigTests(unittest.TestCase):
    def test_bare_model_strips_providers(self):
        self.assertEqual(ev.bare_model("openrouter/qwen/qwen3-32b"), "qwen/qwen3-32b")
        self.assertEqual(ev.bare_model("openai/gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(ev.bare_model("gpt-4o-mini"), "gpt-4o-mini")

    def test_model_for_prefers_override_then_role_then_default(self):
        import os
        old = {k: os.environ.get(k) for k in ("MODEL_VENUE", "MODEL_DEFAULT")}
        try:
            os.environ["MODEL_VENUE"] = "qwen/qwen3.8-27b"
            os.environ["MODEL_DEFAULT"] = "gpt-4o-mini"
            self.assertEqual(ev.model_for("VENUE", "cli/x"), "cli/x")
            self.assertEqual(ev.model_for("VENUE"), "qwen/qwen3.8-27b")
            self.assertEqual(ev.model_for("LANGUAGE"), "gpt-4o-mini")
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class SourcesCsvTests(unittest.TestCase):
    def test_real_sources_file_reads_clean(self):
        rows = ev.read_sources(ev.INPUT_DIR / "sources.csv")
        self.assertTrue(rows)
        self.assertTrue(all(ev.is_http_url(row["url"]) for row in rows))
        joined = " ".join(row["orga"] for row in rows)
        self.assertNotIn("Ã¶", joined)
        self.assertIn("Österreich", joined)

    def test_missing_column_tolerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.csv"
            path.write_text("url\nhttps://a.org/x\n\nhttps://a.org/y\n", encoding="utf-8")
            rows = ev.read_sources(path)
            self.assertEqual([r["url"] for r in rows], ["https://a.org/x", "https://a.org/y"])


class IoTests(unittest.TestCase):
    def test_write_json_atomic_leaves_no_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.json"
            ev.write_json(path, {"x": 1})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"x": 1})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_write_csv_flattens_nested(self):
        import csv as csvmod
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.csv"
            ev.write_csv(path, ["a", "b"], [{"a": {"k": 1}, "b": None}])
            with path.open(newline="", encoding="utf-8") as handle:
                row = next(csvmod.DictReader(handle))
            self.assertEqual(json.loads(row["a"]), {"k": 1})
            self.assertEqual(row["b"], "")

    def test_read_json_missing_returns_default(self):
        self.assertEqual(ev.read_json(Path("/nope/none.json"), {"d": 1}), {"d": 1})


class LlmJsonTests(unittest.TestCase):
    """llm_json gegen einen Fake-Client: Retry-Pfad und Ehrlichkeit bei Müll."""

    class Resp:
        def __init__(self, content):
            self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
            self.usage = None

    class BadClient:
        def __init__(self, replies):
            self.replies = list(replies)
            self.calls = 0
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            self.calls += 1
            reply = self.replies[min(self.calls - 1, len(self.replies) - 1)]
            if isinstance(reply, Exception):
                raise reply
            return reply

    def _run(self, client, schema=None):
        sem = asyncio.Semaphore(1)
        return asyncio.run(ev.llm_json(
            client, "m", "prompt", {"source_text": "x"}, sem,
            schema=schema, retries=1,
        ))

    def test_valid_first_try(self):
        client = self.BadClient([self.Resp('{"a": "hi"}')])
        data, errors = self._run(client, {"type": "object", "properties": {"a": {"type": "string"}},
                                          "required": ["a"]})
        self.assertEqual(data, {"a": "hi"})
        self.assertEqual(errors, [])
        self.assertEqual(client.calls, 1)

    def test_schema_violent_retry_then_success(self):
        client = self.BadClient([self.Resp('{"a": 1}'), self.Resp('{"a": "ok"}')])
        data, errors = self._run(client, {"type": "object", "properties": {"a": {"type": "string"}},
                                          "required": ["a"]})
        self.assertEqual(data, {"a": "ok"})
        self.assertEqual(errors, [])
        self.assertEqual(client.calls, 2)

    def test_persistent_garbage_returns_none_not_guess(self):
        client = self.BadClient([self.Resp("kein json"), self.Resp("immer noch nicht")])
        data, errors = self._run(client)
        self.assertIsNone(data)
        self.assertTrue(errors)

    def test_non_retryable_error_stops_after_one_call(self):
        client = self.BadClient([RuntimeError("401 unauthorized")])
        data, errors = self._run(client)
        self.assertIsNone(data)
        self.assertEqual(client.calls, 1)
        self.assertTrue(any("401" in e for e in errors))

    def test_retryable_error_is_retried(self):
        client = self.BadClient([RuntimeError("429 rate limited"), self.Resp('{"a": 1}')])
        data, _ = self._run(client)
        self.assertEqual(client.calls, 2)
        self.assertEqual(data, {"a": 1})

    def test_korrektur_hinweis_wird_mitgegeben(self):
        seen = []
        client = self.BadClient([self.Resp('{"a": 1}'), self.Resp('{"a": "ok"}')])
        original = client.create

        async def spy(**kwargs):
            seen.append(kwargs["messages"][1]["content"])
            return await original(**kwargs)

        client.create = spy
        self._run(client, {"type": "object", "properties": {"a": {"type": "string"}},
                           "required": ["a"]})
        self.assertNotIn("_korrektur", seen[0])
        self.assertIn("_korrektur", seen[1])


if __name__ == "__main__":
    unittest.main()
