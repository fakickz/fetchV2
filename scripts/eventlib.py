#!/usr/bin/env python3
"""Gemeinsame Bausteine der Pipeline v2: Corpus, Schema-Validierung, Beweispflicht, LLM.

Keine Extraktionslogik hier drin — nur Werkzeuge, die fetch/discover/basis/detail
gemeinsam nutzen, damit nichts doppelt gepflegt wird.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import date, datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urldefrag, urlparse
from zoneinfo import ZoneInfo

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from costs import UsageTracker
from llm import ClientRouter, is_openrouter_model, reasoning_kwargs, require_api_key

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"
CONFIG_DIR = ROOT / "config"
SCHEMAS_DIR = ROOT / "schemas"
INPUT_DIR = ROOT / "INPUT"
OUT_DIR = ROOT / "out"
PAGES_DIR = OUT_DIR / "pages"
PAGES_INDEX = PAGES_DIR / "index.json"
DEFAULT_SOURCES = INPUT_DIR / "sources.csv"
VIENNA = ZoneInfo("Europe/Vienna")

load_dotenv(ROOT / ".env")

SECTION_CAP = 6000
MAX_LINKS = 600
RETRYABLE = ("429", "ratelimit", "rate_limit", "500", "502", "503", "504",
             "overloaded", "timeout", "timed out", "connection")


# --------------------------------------------------------------------------
# Modell-Auflösung
# --------------------------------------------------------------------------

def bare_model(model: str) -> str:
    """'openrouter/qwen/qwen3-32b' -> 'qwen/qwen3-32b'; 'openai/gpt-4o-mini' -> 'gpt-4o-mini'."""
    parts = (model or "").strip().split("/")
    if len(parts) > 2 and parts[0].lower() in {"openrouter", "openai"}:
        return "/".join(parts[1:])
    if len(parts) == 2 and parts[0].lower() == "openai":
        return parts[1]
    return model


def model_for(role: str, override: str | None = None) -> str:
    """Modell für eine Prompt-Rolle: CLI-Override > MODEL_<ROLE> > MODEL_DEFAULT."""
    if override:
        return override.strip()
    value = (os.getenv(f"MODEL_{role}") or "").strip()
    if value:
        return value
    return (os.getenv("MODEL_DEFAULT") or "gpt-4o-mini").strip()


# --------------------------------------------------------------------------
# URLs und Pfade
# --------------------------------------------------------------------------

def norm_url(value: str) -> str:
    return (value or "").strip().rstrip("/")


def absolute_url(base: str, href: str) -> str:
    return urljoin(base, (href or "").strip()) if href else ""


def is_http_url(value: str) -> bool:
    try:
        return urlparse(value).scheme.lower() in {"http", "https"} and bool(urlparse(value).netloc)
    except ValueError:
        return False


def same_page_link(candidate: str, source: str) -> bool:
    """True wenn candidate dieselbe Seite wie source adressiert (auch mit #fragment)."""
    if not candidate or not source:
        return False
    cand_base, frag = urldefrag(norm_url(candidate))
    src_base = norm_url(source)
    return cand_base.lower() == src_base.lower()


def readable_dir() -> Path:
    """Lesbares Spiegelverzeichnis, abgeleitet aus PAGES_DIR.

    Ableitung statt Modulkonstante: Tests patchen PAGES_DIR und isolieren damit
    automatisch auch das Spiegel. Eine separate Konstante hätte still das echte
    out/pages_readable verschmutzt.
    """
    return PAGES_DIR.parent / "pages_readable"


def page_hash(url: str) -> str:
    return hashlib.sha1(norm_url(url).encode("utf-8")).hexdigest()[:16]


def page_path(url: str) -> Path:
    return PAGES_DIR / f"{page_hash(url)}.json"


def slugify(value: str) -> str:
    """'Mala Zêrîn' -> 'mala-zerin'. Deterministisch, ASCII, keine Leerzeichen."""
    text = (value or "").strip().lower()
    for old, new in {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "ê": "e", "î": "i"}.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


# Segmentnamen, die als Dateiname nichts aussagen (Listen- und Navigationspfade).
GENERIC_SLUGS = {"veranstaltungen", "events", "termine", "kalender", "category",
                 "tag", "thema", "page", "aktuelles", "liste", "monat", "heute",
                                 "index", "home", "start"}


def url_slug(url: str) -> str:
    """Letztes sinnvolles Pfadsegment einer URL als Lesename.

    Bewusst aus der URL und nicht aus einem LLM-Feld: die URL steht beim Fetch
    fest, ein Discovery-Titel ist unverifiziert und kann sich von Lauf zu Lauf
    ändern.
    """
    path = urlparse(norm_url(url)).path
    parts = [re.sub(r"\.[a-zA-Z0-9]{1,5}$", "", part)
           for part in path.split("/") if part]
    for part in reversed(parts):
        if part.lower() not in GENERIC_SLUGS:
            return slugify(part)
    return slugify(parts[-1]) if parts else ""


# --------------------------------------------------------------------------
# HTML -> Text / Strukturen
# --------------------------------------------------------------------------

def strip_html(html: str) -> str:
    text = re.sub(r"<(script|style|noscript|template)\b[^>]*>.*?</\1>", " ", html or "",
                  flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|h[1-6]|tr|section|article)>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(html: str) -> str:
    return re.sub(r"\s+", " ", strip_html(html or "")).strip()


def result_markdown(result: Any) -> str:
    markdown = getattr(result, "markdown", "") or ""
    if not isinstance(markdown, str):
        markdown = (getattr(markdown, "fit_markdown", "")
                    or getattr(markdown, "raw_markdown", "") or "")
    text = re.sub(r"[ \t\r\f\v]+", " ", markdown or "").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


# Tags ohne Event-Nutzwert: Navigation, Rahmen, Formulare, Einbettungen.
_JUNK_TAGS = {"nav", "header", "footer", "aside", "form", "iframe", "select",
              "button", "dialog"}
# Klassen/IDs/Rollen ohne Event-Nutzwert. "pagination" steht bewusst NICHT
# drin: "mehr laden / ältere Termine"-Hinweise braucht discover als
# Trunkierungssignal. "banner" fehlt bewusst: Event-Banner wären Kollateral.
# Token-Match statt Substring: "mobmenu-push-wrap" ist Struktur und kein Menü.
_JUNK_TOKENS = {"nav", "menu", "sidebar", "breadcrumb", "social", "share",
                "follow", "cookie", "cookies", "consent", "zustimmung",
                "einwilligung", "search", "newsletter", "login",
                "skip-link", "complementary", "cmplz", "klaro", "borlabs",
                "usercentrics", "consentmanager", "cookiebot", "onetrust"}
_JUNK_ROLE_RE = re.compile(r"navigation|search|contentinfo|complementary", re.I)
_SKIP_LINK_RE = re.compile(r"skip to content|zum inhalt springen", re.I)


def _has_junk_token(value: str) -> bool:
    return any(token in _JUNK_TOKENS
               for token in re.split(r"[^a-z0-9]+", (value or "").lower()))


def content_markdown(html: str) -> str:
    """DOM-bereinigtes Markdown aus vorhandenem html. Kein Netz, kein LLM.

    Entfernt Navigationsrahmen (Tags + Menu-/Sidebar-Klassen), Linkziele und
    Bildsyntax, behält aber Reihenfolge, Überschriftenstruktur und Wortlaut
    (Termine, Adressen, Zeiten). Leere Überschriften (nacktes "##") fallen weg:
    sie sind Navigationsrest und würden später Abschnitte falsch schneiden.
    Fällt auf "" zurück, wenn nichts übrig ist — dann gilt das alte markdown.
    """
    if not html or not html.strip():
        return ""
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return ""
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag is None or not getattr(tag, "attrs", None) and tag.name is None:
            continue
        attrs = tag.attrs if isinstance(getattr(tag, "attrs", None), dict) else {}
        name = (tag.name or "").lower()
        if name in _JUNK_TAGS:
            tag.decompose()
            continue
        # Strukturcontainer nie per Klasse/Rolle entfernen: Layout-Klassen wie
        # "right-sidebar" am body würden sonst die ganze Seite töten.
        if name in {"body", "html", "main", "article"}:
            continue
        if name == "a":
            anchor_text = tag.get_text(strip=True) if hasattr(tag, "get_text") else ""
            if not anchor_text or _SKIP_LINK_RE.search(
                    anchor_text + str(attrs.get("href") or "")):
                tag.decompose()
                continue
        classes = " ".join(attrs.get("class") or "")
        ident = str(attrs.get("id") or "")
        role = str(attrs.get("role") or "")
        if _has_junk_token(classes + " " + ident) \
                or _JUNK_ROLE_RE.search(role):
            tag.decompose()
            continue
        if "hidden" in attrs or attrs.get("aria-hidden") == "true":
            tag.decompose()
            continue
        style = str(attrs.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()
            continue
    for img in soup.find_all("img"):
        alt = re.sub(r"\s+", " ", img.get("alt") or "").strip()
        img.replace_with(alt)
    for tag in soup.find_all("a"):
        tag.replace_with(tag.get_text())
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip()
        if not text or text == "#":
            heading.decompose()
            continue
        level = int(heading.name[1])
        heading.replace_with(f"\n\n{'#' * level} {text}\n\n")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    root = soup.find("body") or soup
    text = root.get_text("\n")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line and line.strip("#").strip()]
    deduped = [line for i, line in enumerate(lines)
               if i == 0 or line != lines[i - 1]]
    return "\n".join(deduped).strip()


def extract_links(html: str, base_url: str) -> list[dict[str, str]]:
    """Alle http(s)-Links, absolutiert, dedupliziert, mit ankertext."""
    from bs4 import BeautifulSoup

    if not html:
        return []
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    for tag in soup.find_all("a", href=True):
        raw = (tag.get("href") or "").strip()
        if not raw or raw.lower().startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        url = absolute_url(base_url, raw)
        if not is_http_url(url):
            continue
        key = norm_url(url).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "href": url,
            "text": re.sub(r"\s+", " ", tag.get_text(" ", strip=True))[:300],
            "rel": " ".join(tag.get("rel") or []),
        })
        if len(out) >= MAX_LINKS:
            break
    return out


def extract_images(html: str, base_url: str) -> list[dict[str, str]]:
    """Bild-URLs samt alt/titel — reine Referenz, v1 wertet sie nicht aus."""
    from bs4 import BeautifulSoup

    if not html:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return []
    out: list[dict[str, str]] = []
    for tag in soup.find_all("img"):
        raw = (tag.get("src") or tag.get("data-src") or "").strip()
        if not raw or raw.startswith("data:"):
            continue
        out.append({
            "src": absolute_url(base_url, raw),
            "alt": re.sub(r"\s+", " ", tag.get("alt") or "").strip()[:400],
            "title": re.sub(r"\s+", " ", tag.get("title") or "").strip()[:400],
        })
    return out


def extract_jsonld(html: str) -> list[Any]:
    """Rohe JSON-LD-Objekte; Hinweis für die Prompts, nie Autorität."""
    if not html:
        return []
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html, flags=re.S | re.I,
    )
    out: list[Any] = []
    for raw in blocks:
        try:
            out.append(json.loads(raw.strip()))
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def jsonld_hints(data: Any, limit: int = 1200) -> str:
    """Kompakter Hinweis-Text aus JSON-LD (Typ + Name + Ort + Datum)."""
    if not data:
        return ""
    interesting: list[str] = []

    def walk(node: Any) -> None:
        if len(interesting) >= 8:
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        types = node.get("@type", "")
        if isinstance(types, str):
            types = [types]
        label = str(node.get("name") or node.get("headline") or "")[:120]
        if any(str(t).lower() in {"event", "posteropeningevent", "interactioncounter"}
               for t in types) or label:
            loc = node.get("location")
            loc_name = ""
            if isinstance(loc, dict):
                loc_name = str(loc.get("name") or "")[:120]
            elif isinstance(loc, str):
                loc_name = loc[:120]
            start = str(node.get("startDate") or "")[:40]
            bits = [f"type={','.join(str(t) for t in types)}"]
            if label:
                bits.append(f"name={label}")
            if loc_name:
                bits.append(f"location={loc_name}")
            if start:
                bits.append(f"startDate={start}")
            interesting.append(" ".join(bits))
        for key in ("@graph", "itemListElement", "mainEntity", "hasPart", "event"):
            if key in node:
                walk(node[key])

    walk(data)
    text = "; ".join(interesting)
    return text[:limit]


# --------------------------------------------------------------------------
# Corpus (Cache: eine Datei pro Seite)
# --------------------------------------------------------------------------

def readable_rel_path(record: dict[str, Any]) -> str:
    """Zielenamen für das lesbare Spiegelverzeichnis.

    Ordner nach Orga (UUID-Präfix bleibt vornedran, damit die Zuordnung zum
 Zielsystem auch ohne Orga-Nachschlagen eindeutig ist), Datei nach Art und
    URL-Segment. Der Hash taucht nur noch bei echter Kollision auf.
    """
    orga_id = str(record.get("orga_id") or "").strip()
    orga = str(record.get("orga") or "").strip()
    # Voller Vereinsname als Ordner ist unlesbar ("...-verein-zur-foerderung-von-
    # demokratie-und-interkultureller-teilhabe"). Das UUID-Präfix trägt die
    # Eindeutigkeit bereits, der Slug ist nur Lesehilfe.
    folder = f"{orga_id[:8]}_{slugify(orga)[:36].strip('-')}" if (orga_id or orga) else "_unassigned"
    if str(record.get("kind") or "") == "overview":
        name = "overview"
    else:
        slug = url_slug(str(record.get("url") or ""))
        name = f"detail_{slug}" if slug else "detail"
    return f"{folder}/{name}.json"


def resolve_readable(rel: str, url: str, taken: dict[str, str]) -> str:
    """Kollision auflösen: gleicher Pfad, andere URL -> Hash anhängen."""
    owner = taken.get(rel)
    if owner is None or owner == norm_url(url):
        return rel
    stem, slash, tail = rel.rpartition("/")
    return f"{stem}{slash}{Path(tail).stem}-{page_hash(url)[:8]}.json"


def write_readable(record: dict[str, Any], rel: str) -> None:
    """Kopie ohne html-Feld. Das html ist 87% des Corpus und liest keine Folgestufe."""
    target = readable_dir() / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    slim = {key: value for key, value in record.items() if key != "html"}
    write_json(target, slim)


def load_index() -> dict[str, Any]:
    try:
        data = json.loads(PAGES_INDEX.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_index(index: dict[str, Any]) -> None:
    write_json(PAGES_INDEX, index)


def load_page(url: str) -> dict[str, Any] | None:
    path = page_path(url)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def save_page(record: dict[str, Any], readable: bool = True) -> None:
    url = norm_url(str(record.get("url") or ""))
    if not url:
        raise ValueError("Corpus-Eintrag ohne url")
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    write_json(page_path(url), record)
    index = load_index()
    entry = {
        "file": page_path(url).name,
        "kind": record.get("kind", ""),
        "status": record.get("status", ""),
        "fetched_at": record.get("fetched_at", ""),
    }
    for key in ("last_fetch_attempt_at", "last_fetch_error"):
        if record.get(key):
            entry[key] = record[key]
    if readable:
        taken = {str(item.get("readable")): owner
                 for owner, item in index.items()
                 if isinstance(item, dict) and item.get("readable")}
        rel = resolve_readable(readable_rel_path(record), url, taken)
        write_readable(record, rel)
        entry["readable"] = rel
    elif record.get("last_fetch_error"):
        previous = index.get(url)
        if isinstance(previous, dict) and previous.get("readable"):
            entry["readable"] = previous["readable"]
    index[url] = entry
    save_index(index)


def new_page_record(url: str, kind: str, orga: str = "", orga_id: str = "") -> dict[str, Any]:
    return {
        "url": norm_url(url),
        "kind": kind,
        "orga": orga,
        "orga_id": orga_id,
        "status": "pending",
        "error": "",
        "fetched_at": now_iso(),
        "markdown": "",
        "content_markdown": "",
        "text": "",
        "html": "",
        "links": [],
        "images": [],
        "jsonld": [],
    }


# --------------------------------------------------------------------------
# Datum / Zeit
# --------------------------------------------------------------------------

_DE_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b")
_EXPLICIT_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# Trennzeichen zwischen Stunde und Minute: Doppelpunkt, Punkt und die
# Typografie-Varianten, die auf Plakaten vorkommen ("20°°", "20••", "20..").
_DE_TIME_RE = re.compile(
    r"\b(\d{1,2})\s*[:.\uFF0E\u2022\u2219\u00B7\u2025]{1,3}\s*(\d{2})\s*(?:uhr)?\b",
    re.IGNORECASE,
)
# "20°°" / "20••": zwei gleiche Marken nach der Stunde meinen volle Stunden.
# Kein \b am Ende — ° ist kein Wortzeichen, die Grenze würde nie greifen.
_DE_TIME_HOUR_RE = re.compile(r"\b(\d{1,2})\s*([\u00B0\u2022])\s*\2(?![\u00B0\u2022])")


def _norm_year(year: str | None) -> str:
    if not year:
        return ""
    year = year.strip()
    return f"20{year}" if len(year) == 2 else year


def parse_de_dates(text: str, default_year: str = "") -> list[str]:
    """Alle deutschen Datumsformen in der Reihenfolge ihres Auftretens.

    Jahresbereiche werden aufgelöst: "06.11. – 27.11.2026" nennt das Jahr nur
    einmal, am Ende. Fehlt einem Datum das Jahr, wird deshalb das letzte im
    Text genannte Jahr ergänzt, nicht verworfen.
    """
    body = text or ""
    if not default_year:
        years = _EXPLICIT_YEAR_RE.findall(body)
        default_year = years[-1] if years else ""
    out: list[str] = []
    for m in _DE_DATE_RE.finditer(body):
        day, month, year = m.group(1), m.group(2), _norm_year(m.group(3))
        year = year or default_year
        if not year:
            continue
        try:
            value = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
            date.fromisoformat(value)   # 31.02. ist kein Datum
        except ValueError:
            continue
        if value not in out:
            out.append(value)
    return out


def parse_de_date(text: str, default_year: str = "") -> str:
    found = parse_de_dates(text, default_year)
    return found[0] if found else ""


def parse_de_times(text: str, _default_year: str = "") -> list[str]:
    """Alle Uhrzeitkandidaten in der Reihenfolge ihres Auftretens.

    Mehrere Kandidaten sind der Normalfall: "Beginn: Vo-Kü 19:30
    Buchvorstellung 20:00" nennt Küche und Lesung im selben Absatz. Wer nur
    den ersten Treffer nimmt, verwirft die richtige Antwort.
    """
    out: list[str] = []
    for m in _DE_TIME_RE.finditer(text or ""):
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            continue
        value = f"{hour:02d}:{minute:02d}"
        if value not in out:
            out.append(value)
    for m in _DE_TIME_HOUR_RE.finditer(text or ""):
        hour = int(m.group(1))
        if hour > 23:
            continue
        value = f"{hour:02d}:00"
        if value not in out:
            out.append(value)
    return out


def parse_de_time(text: str) -> str:
    found = parse_de_times(text)
    return found[0] if found else ""


def iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip())
    except ValueError:
        return None


def current_date() -> date:
    return datetime.now(VIENNA).date()


def is_past(start_date: str, today: date | None = None) -> bool | None:
    """True/False bei gesichertem Datum, None wenn nicht beurteilbar.

    None bedeutet: durchlassen. Ein Event wird nur bei gesicherter
    Vergangenheit ausgeschlossen, niemals bei Ungewissheit.
    """
    parsed = iso_date(start_date)
    if parsed is None:
        return None
    return parsed < (today if today is not None else current_date())


def now_iso() -> str:
    return datetime.now(VIENNA).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Adresse / Wien-Gate
# --------------------------------------------------------------------------

_NON_VIENNA_CITIES = (
    "wiener neustadt", "st. pölten", "sankt pölten", "eisenstadt", "bregenz",
    "linz", "wels", "steyr", "salzburg", "innsbruck", "klagenfurt", "villach",
    "graz", "klosterneuburg", "mödling", "baden", "schwechat", "stockerau",
    "korneuburg", "tulln", "krems", "vösendorf", "traiskirchen", "hollabrunn",
    "katsdorf", "sittendorf", "hainburg", "bruck an der leitha", "deutsch-wagram",
)


def is_full_address(value: str) -> bool:
    """Straße + Hausnummer + 4-stellige PLZ."""
    text = (value or "").strip()
    if not text or not re.search(r"\b\d{4}\b", text):
        return False
    return bool(re.search(r"[A-Za-zÄÖÜäöüß][\wÄÖÜäöüß.\-]*\s+\d+[a-zA-Z]?", text))


VIENNA_PLZ = range(1010, 1240)
_YEAR_LOOKALIKE = range(1900, 2200)


def is_vienna_address(value: str) -> bool | None:
    """Deterministischer Wien-Check: True/False, None bei unklar.

    Vierstelllige Zahlen sind nicht automatisch PLZ: Jahreszahlen (1900-2199,
    z.B. "Roman 1984" oder "1989") fallen ins gleiche Muster wie eine Wiener
    PLZ und werden deshalb verworfen. Bleibt eine echte PLZ übrig, entscheidet
    der Bereich 1010-1239. Gemischte Angaben lassen wir als Wien gelten, denn
    das Gate schliesst nur bei gesichertem Nicht-Wien aus.
    """
    text = (value or "").strip().lower()
    if not text:
        return None
    codes = [int(m.group(1)) for m in re.finditer(r"\b(\d{4})\b", text)]
    plz = [code for code in codes if code not in _YEAR_LOOKALIKE]
    if any(code in VIENNA_PLZ for code in plz):
        return True
    if plz:
        return False
    if "wiener neustadt" in text:
        return False
    if re.search(r"\bwien\b|\bvienna\b|\b1\.[1-9] bezirk\b", text):
        return True
    for city in _NON_VIENNA_CITIES:
        if city in text:
            return False
    return None


# --------------------------------------------------------------------------
# Beweispflicht
# --------------------------------------------------------------------------

_QUOTES = {
    "\u201a": "'", "\u2018": "'", "\u2019": "'", "\u00ab": '"', "\u00bb": '"',
    "\u201e": '"', "\u201c": '"', "\u201d": '"', "\u2032": "'", "\u2033": '"',
}
_DASHES = {
    "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u00ad": "",
}
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def strip_markup(text: str) -> str:
    """Markdown-Deko entfernen, die ein Prompt laut Auftrag säubern soll.

    Ohne diesen Schritt wäre jede korrekt bereinigte Beschreibung ein
    Verifikationsfehler: der Quelltext enthält **fett** und Bild-Links, das
    Modell darf und soll sie entfernen. Verglichen wird deshalb beide Seiten
    ohne Deko, aber weiterhin Zeichen für Zeichen.
    """
    t = _MD_IMAGE.sub(" ", text or "")
    t = _MD_LINK.sub(r"\1", t)
    for token in ("**", "__", "`", "*", "_"):
        t = t.replace(token, "")
    return t


def normalize_quote(text: str) -> str:
    """Vergleichsform: NFKC, kleine Buchstaben, Einheitsanführungszeichen, Leerzeichen."""
    value = strip_markup(text or "")
    value = unicodedata.normalize("NFKC", value)
    for src, dst in {**_QUOTES, **_DASHES}.items():
        value = value.replace(src, dst)
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    # Leerzeichen um Interpunktion fallen lassen: entfernt nur Typografie,
    # nicht Inhalt. Nötig, weil Markdown-Restgrössen ("|**_Roman_") nach dem
    # Markup-Strip ein anderes Leerzeichenmuster lassen als der bereinigte Text.
    value = re.sub(r"\s*([|/.,;:!?(){}[\]\-–\"'])\s*", r"\1", value)
    return value


def verify_evidence(quote: str, source_text: str) -> bool:
    """Ist das Zitat wirklich im Quelltext? False bei leerem Zitat."""
    needle = normalize_quote(quote)
    if not needle:
        return False
    return needle in normalize_quote(source_text)


def _prose_chunks(text: str, min_len: int = 12) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+|\n+", text or "")
    return [p for p in (chunk.strip() for chunk in parts) if len(p) >= min_len]


def verify_prose(value: str, source_text: str, threshold: float = 0.8) -> bool:
    """Langtext-Prüfung: muss ein Exzerpt sein, darf aber Junk-Sätze auslassen.

    Einzelne Sätze werden gegen den Quelltext geprüft; genug müssen vorkommen.
    """
    chunks = _prose_chunks(value)
    if not chunks:
        return verify_evidence(value, source_text)
    hay = normalize_quote(source_text)
    hits = sum(1 for chunk in chunks if normalize_quote(chunk) in hay)
    return hits / len(chunks) >= threshold


def _candidates(parser: Any, text: str, year_hint: str) -> list[str]:
    try:
        found = parser(text, year_hint)
    except TypeError:
        found = parser(text)
    if isinstance(found, str):
        found = [found] if found else []
    return list(found or [])


def resolve_derived(value: str, quote: str, source_text: str, parser: Any) -> tuple[str, str]:
    """Normalisiertes Feld auflösen. Gibt (wert, konfidenz) zurück.

    Der Beleg ist die QUELLFORM, nicht der normalisierte Wert: "2026-09-25"
    steht auf deutschen Seiten als "Fr 25.09.2026", und eine Plakatstunde steht
    als "20°°" da. Zwei Konsequenzen:

    * Gibt das Modell die Rohform im WERTfeld an, wird sie kanonisiert statt
      verworfen — die Information ist korrekt, nur das Format weicht ab.
    * Weicht der Wert nachweislich vom Zitat ab, wird abgelehnt. Das Zitat ist
      Boden, der Modellwert nur eine Behauptung darüber.

    Liefert ("", "none") bei Ablehnung, (wert, "derived") wenn das Zitat kein
    Zahlformat enthält (relative Angabe, Wert bleibt wie geliefert).
    """
    value = (value or "").strip()
    if not quote or not verify_evidence(quote, source_text):
        return "", "none"
    year_hint = value[:4] if len(value) >= 4 and value[:4].isdigit() else ""
    candidates = _candidates(parser, quote, year_hint)
    own = _candidates(parser, value, year_hint)
    if own:
        if not candidates or own[0] in candidates:
            return own[0], "high" if candidates else "derived"
        return "", "none"
    if value and value in candidates:
        return value, "high"
    if not candidates:
        return value, "derived"
    return "", "none"


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt fehlt: {path}") from exc
    if not prompt:
        raise ValueError(f"Prompt ist leer: {path}")
    return prompt


def load_input_config() -> dict[str, Any]:
    """config/input.json: Eingabequelle je Schritt. Fehlende Datei = Defaults."""
    path = CONFIG_DIR / "input.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Config ist kein gültiges JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config ist kein Objekt: {path}")
    return data


def load_schema(name: str) -> dict[str, Any]:
    path = SCHEMAS_DIR / f"{name}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Schema fehlt: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema ist kein gültiges JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Schema ist kein Objekt: {path}")
    return data


_TYPE_MAP: dict[str, Any] = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "number": (int, float), "integer": int, "null": type(None),
}


def validate_schema(data: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Minimale JSON-Schema-Prüfung: type, required, properties, items, enum."""
    errors: list[str] = []
    expected = schema.get("type")
    if expected:
        py = _TYPE_MAP.get(str(expected))
        if py is not None:
            if expected == "integer":
                ok = isinstance(data, int) and not isinstance(data, bool)
            elif expected == "number":
                ok = isinstance(data, (int, float)) and not isinstance(data, bool)
            else:
                ok = isinstance(data, py)
            if not ok:
                return [f"{path}: erwartet {expected}, gefunden {type(data).__name__}"]

    enum = schema.get("enum")
    if enum is not None and data not in enum:
        errors.append(f"{path}: Wert {data!r} nicht in {enum}")

    if isinstance(data, dict):
        for key in schema.get("required") or []:
            if key not in data:
                errors.append(f"{path}.{key}: Pflichtfeld fehlt")
        props = schema.get("properties") or {}
        for key, value in data.items():
            if key in props:
                errors.extend(validate_schema(value, props[key], f"{path}.{key}"))
    if isinstance(data, list) and "items" in schema:
        for index, item in enumerate(data):
            errors.extend(validate_schema(item, schema["items"], f"{path}[{index}]"))
    return errors


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """LLM-Antwort zu Objekt machen; toleriert Markdown-Fences und Beitsatz."""
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"events": parsed} if parsed and isinstance(parsed[0], dict) else None
    return None


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

def build_system_prompt(prompt: str, schema: dict[str, Any] | None = None) -> str:
    if not schema:
        return prompt
    return f"{prompt}\n\nAntwortschema (Pflichtfelder, exakte Keys):\n{json.dumps(schema, ensure_ascii=False)}"


async def llm_json(
    client: ClientRouter,
    model: str,
    prompt: str,
    payload: dict[str, Any],
    semaphore: asyncio.Semaphore,
    schema: dict[str, Any] | None = None,
    max_tokens: int = 1500,
    retries: int = 1,
    tracker: UsageTracker | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Ein Feld-Call: JSON-Objekt zurückgeben, das dem Schema genügt.

    Liefert (daten, fehler). fehler leer => verwendbar. Bei Schema-Verstoß
    wird ein Retry mit angehängtem Validierungsfehler versucht, danach ehrlich
    (None, fehler) statt geratener Daten.
    """
    system = build_system_prompt(prompt, schema)
    errors: list[str] = []
    async with semaphore:
        for attempt in range(retries + 1):
            user_payload = dict(payload)
            if errors:
                user_payload["_korrektur"] = (
                    "Deine vorige Antwort war unbrauchbar: "
                    + "; ".join(errors[:6])
                    + ". Antworte erneut, nur mit den geforderten Feldern."
                )
            try:
                response = await _create(client, model, system, user_payload, max_tokens)
            except Exception as exc:
                message = str(exc)
                errors = [f"{type(exc).__name__}: {message[:200]}"]
                if any(marker in message.lower() for marker in RETRYABLE) and attempt < retries:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                return None, errors
            if tracker is not None:
                tracker.add_response(model, response)
            raw = ""
            try:
                raw = response.choices[0].message.content or ""
            except (AttributeError, IndexError):
                raw = ""
            data = parse_json_object(raw)
            if data is None:
                errors = ["Antwort ist kein JSON-Objekt"]
                if attempt < retries:
                    continue
                return None, errors
            if schema is not None:
                found = validate_schema(data, schema)
                if found:
                    errors = found
                    if attempt < retries:
                        continue
                    return None, found
            return data, []
    return None, errors or ["LLM-Aufruf ohne Ergebnis"]


async def _create(client: ClientRouter, model: str, system: str,
                  user_payload: dict[str, Any], max_tokens: int) -> Any:
    kwargs: dict[str, Any] = {
        "model": bare_model(model),
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    }
    kwargs.update(reasoning_kwargs(model))
    try:
        return await client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Nicht jede Provider-Strecke kann json_object; ohne Response-Format
        # erneut versuchen, parse_json_object toleriert Fences trotzdem.
        if not any(marker in str(exc).lower()
                   for marker in ("response_format", "json_object", "json mode", "unsupported")):
            raise
        kwargs.pop("response_format", None)
        return await client.chat.completions.create(**kwargs)


async def probe_llm(model: str) -> None:
    client = ClientRouter()
    try:
        await client.chat.completions.create(
            model=bare_model(model),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        hint = ""
        if "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
            hint = " (API-Key prüfen: rotiert oder falscher Provider)"
        elif "402" in lowered or "insufficient" in lowered:
            hint = " (Guthaben prüfen)"
        elif "404" in lowered:
            hint = f" (Modell-Slug prüfen: {model})"
        raise SystemExit(f"LLM-Probe fehlgeschlagen [{model}]: {message[:200]}{hint}")
    finally:
        await client.close()


# --------------------------------------------------------------------------
# CSV / JSON IO
# --------------------------------------------------------------------------

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except (UnicodeDecodeError, UnicodeError) as exc:
            last_error = exc
            continue
    raise last_error or ValueError(f"Kann {path} nicht lesen")


def pick(row: dict[str, str], *names: str) -> str:
    lowered = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
    for name in names:
        if name in lowered and lowered[name].strip():
            return lowered[name].strip()
    return ""


def read_sources(path: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in read_csv_rows(path):
        url = pick(row, "url", "target_url", "targeturl")
        if not url:
            continue
        out.append({
            "orga": pick(row, "orga", "name", "slug"),
            "orga_id": pick(row, "orga_id", "orgid", "id"),
            "url": url,
        })
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fields})


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# --------------------------------------------------------------------------
# Textauswahl pro Event
# --------------------------------------------------------------------------

TEXT_SOURCES = {"raw": "markdown", "content": "content_markdown"}


def page_text(page: dict[str, Any] | None, source: str = "raw") -> str:
    """Lesetext einer Corpus-Seite. raw = Crawl-Markdown, content = DOM-bereinigt.

    Kein stilles Fallback zwischen den Fassungen: leer bleibt leer.
    """
    if not isinstance(page, dict):
        return ""
    return str(page.get(TEXT_SOURCES[source]) or "").strip()


_HEADING_RE = re.compile(r"^#{1,4}\s", re.MULTILINE)
_LISTING_DATE_LINE_RE = re.compile(
    r"^(?:(?:heute|morgen|übermorgen|diese woche|nächste woche)\b|"
    r"(?:mo|di|mi|do|fr|sa|so)\.?\s+\d{1,2}\b|"
    r"\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b|"
    r"(?:jan(?:uar)?|feb(?:ruar)?|märz|mar|apr(?:il)?|mai|jun(?:i)?|"
    r"jul(?:i)?|aug(?:ust)?|sep(?:t(?:ember)?)?|okt(?:ober)?|nov(?:ember)?|"
    r"dez(?:ember)?|jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}\b)",
    re.IGNORECASE,
)


def cut_section(text: str, locator: str, cap: int = SECTION_CAP,
                title: str = "") -> str:
    """Abschnitt um einen Locator herum, für Events ohne eigene Detailseite.

    Der Locator stammt aus einem unverifizierten Listing-Pass; er dient nur
    dazu, die Stelle im Übersichtstext zu finden. Der optionale Titel ist
    ebenfalls nur ein Locator und hilft, wenn das Datum vor der Event-Überschrift
    steht. Kein Treffer => ganzer Text.
    """
    if not text:
        return ""
    lines = text.splitlines(keepends=True)

    def matching_lines(value: str) -> list[int]:
        needle = normalize_quote(value)
        if not needle:
            return []
        return [i for i, line in enumerate(lines) if needle in normalize_quote(line)]

    locator_lines = matching_lines(locator)
    title_lines = matching_lines(title)
    locator_line = locator_lines[0] if locator_lines else None
    title_line = None
    if title_lines:
        if locator_line is None:
            title_line = title_lines[0]
        else:
            after_locator = [line for line in title_lines if line >= locator_line]
            title_line = after_locator[0] if after_locator else title_lines[-1]
    if locator_line is None and title_line is None:
        return text[:cap].strip()

    start_line = min(line for line in (locator_line, title_line) if line is not None)
    headings = [i for i, line in enumerate(lines) if _HEADING_RE.match(line)]
    if title_line is not None:
        event_heading = title_line
    else:
        later_headings = [line for line in headings if line > start_line]
        event_heading = None
        locator_is_standalone = (
            locator_line is not None
            and normalize_quote(lines[locator_line]) == normalize_quote(locator)
        )
        if later_headings and locator_is_standalone:
            candidate = later_headings[0]
            between = lines[start_line + 1:candidate]
            if all(not line.strip() for line in between):
                event_heading = candidate

    end_line = len(lines)
    if event_heading is not None:
        later_headings = [line for line in headings if line > event_heading]
        if later_headings:
            end_line = later_headings[0]
        for i in range(event_heading + 1, end_line):
            if _LISTING_DATE_LINE_RE.match(lines[i].strip()):
                end_line = i
                break
    else:
        later_headings = [line for line in headings if line > start_line]
        if later_headings:
            end_line = later_headings[0]

    section = "".join(lines[start_line:end_line]).strip()
    return (section or "".join(lines[start_line:])).strip()[:cap].strip()


def event_text(event: dict[str, Any], source: str = "raw") -> tuple[str, str]:
    """Lesematerial + Herkunft. Detailseite > Übersicht (auf Abschnitt gekürzt).

    Discovery-Ergebnisse werden nie als Fakt behandelt: title/listed_date_text
    aus dem Listing sind nur Locator.
    """
    detail_url = str(event.get("event_url") or "").strip()
    if detail_url:
        page = load_page(detail_url)
        if page and page.get("status") == "ok":
            text = page_text(page, source)
            if text:
                return text, "detail"
    overview_url = str(event.get("source_url") or "").strip()
    if overview_url:
        page = load_page(overview_url)
        if page and page.get("status") == "ok":
            full = page_text(page, source)
            if full:
                if detail_url:
                    return full, "overview"
                locator = str(event.get("listed_date_text") or "")
                title = str(event.get("title") or "")
                if locator or title:
                    return cut_section(full, locator, title=title), "overview_cut"
                return full, "overview"
    return "", "none"


def jsonld_hint_for(event: dict[str, Any]) -> str:
    for key in ("event_url", "source_url"):
        url = str(event.get(key) or "").strip()
        if not url:
            continue
        page = load_page(url)
        if page and page.get("status") == "ok":
            hint = jsonld_hints(page.get("jsonld"))
            if hint:
                return hint
    return ""
