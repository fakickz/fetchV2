# fetchV2 — Event-Pipeline: erst perfekte Basisinfos, dann Details

```
INPUT/sources.csv
→ fetch --mode overviews   -> out/pages/ + out/pages_readable/
→ api-gate                 -> out/api_sources.json
→ discover                 -> out/discovery.json
→ fetch --mode urls        -> out/pages/ (Detailseiten)
→ basis                    -> out/basis.json + out/basis-excluded.csv + out/basis-review.csv
→ detail                   -> out/detail.json
```

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/crawl4ai-setup   # Browser für den Crawler
cp .env.example .env       # Keys eintragen
```

## Lauf

```sh
npm run v2     # alles
npm run crawl  # overviews -> api-gate -> discovery -> detailseiten
npm run fetch:overview
npm run fetch:details
npm test       # reine Funktionstests, kein Netz, kein LLM
```

## Prinzipien

- Fetch crawlt nur (kein LLM, kein Urteil). Bei jedem normalen Lauf werden
  alle ausgewählten Overview- und Detailziele frisch gecrawlt; der aktuelle
  Corpus liegt unter `out/pages/`. Bereits vorhandene gute Corpusdaten bleiben
  bei einem fehlgeschlagenen erneuten Abruf erhalten.
- Ein erfolgreicher vollständiger Fetch-Lauf ersetzt nur die Seiten derselben
  Art: alte Overviews werden nach einem Overview-Lauf entfernt; alte
  Detailseiten nach einem Detail-Lauf. Der jeweils andere Seitentyp bleibt
  erhalten. Bei Abruffehlern bleibt der bisherige Corpus des Laufs unverändert.
- `--limit` und `--retry-failed` sind Teilläufe und entfernen keine nicht
  angefragten Seiten. Fehler in einem vollständigen Lauf liefern einen
  Fehlerstatus, statt einen unvollständigen Corpus zu veröffentlichen.
- Die Crawl4AI-Markdown-Ausgabe schließt bekannte Cookie-/Consent-Banner
  anhand quellenspezifischer HTML-Selektoren aus. Das gespeicherte HTML bleibt
  unverändert; allgemeiner Seiteninhalt wird nicht nach Cookie-Schlüsselwörtern
  gefiltert.
- Discovery verwendet ausschließlich `markdown` aus dem Fetch-Corpus; basis
  standardmäßig auch (siehe `config/input.json`). `text` dient nicht als
  stille Fallback-Quelle.
- `--limit` begrenzt einen Lauf bewusst. `--retry-failed` ist ein expliziter
  Teillauf und lädt nur Ziele ohne erfolgreichen letzten Abruf neu.
- Gates (past, Wien) sind deterministisch. Past und Nicht-Wien gehen nach
  `basis-excluded.csv`; Events ohne gesicherte Wiener Adresse nach
  `basis-review.csv` (`needs_review`) und nicht in `basis.json`.
- Eingabetext von basis: `config/input.json` (`basis_text_source`: `raw` =
  `markdown`, `content` = DOM-bereinigtes `content_markdown`), überschreibbar
  mit `--text-source`. `npm run basis:raw` und `npm run basis:content` schreiben
  getrennte Ausgaben (`out/basis-raw*`, `out/basis-content*`) für A/B-Vergleiche.
  Kein stilles Fallback zwischen den Fassungen. discover bleibt auf `markdown`.
- Identity (Titel, Datum, Startzeit) hat vorerst keine Zitatpflicht und keine
  Textlängenbegrenzung. Fehlendes Jahr gilt als aktuelles Jahr. JSON-LD geht
  als Hinweis in den Prompt.
- `api_gate` erkennt API-Quellen (Tribe/gcal) und prüft ihre Erreichbarkeit.
  Die Event-Discovery ignoriert API-Daten vorerst und arbeitet mit den
  gecrawlten HTML-Seiten.
