# fetchV2 — Event-Pipeline: erst perfekte Basisinfos, dann Details

```
INPUT/sources.csv
→ fetch --mode overviews   -> OUTPUT/corpus/ + OUTPUT/corpus_readable/
→ api-gate                 -> OUTPUT/api_sources.json
→ discover                 -> OUTPUT/discovery.json
→ fetch --mode urls        -> OUTPUT/corpus/ (Detailseiten)
→ basis                    -> OUTPUT/basis.json + OUTPUT/basis-excluded.csv + OUTPUT/basis-review.csv
→ detail                   -> OUTPUT/detail.json
```

`npm run crawl` = Phase 1 (fetch:overview, api-gate, discover, fetch:details),
`npm run enrich` = Phase 2 (`basis` und danach `detail`; `npm run details` ist
ein Alias für `detail`), `npm run v2` = beides.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/crawl4ai-setup   # Browser für den Crawler
cp .env.example .env       # Keys eintragen
cp INPUT/sources.example.csv INPUT/sources.csv
```

## Lauf

```sh
npm run v2     # alles
npm run crawl  # overviews -> api-gate -> discovery -> detailseiten
npm run fetch:overview
npm run fetch:details
npm run enrich
npm run convert
npm test       # reine Funktionstests, kein Netz, kein LLM
```

## Prinzipien

- Fetch crawlt nur (kein LLM, kein Urteil). Bei jedem normalen Lauf werden
  alle ausgewählten Overview- und Detailziele frisch gecrawlt; der aktuelle
  Corpus liegt unter `OUTPUT/corpus/`. Bereits vorhandene gute Corpusdaten bleiben
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
  getrennte Ausgaben (`OUTPUT/basis-raw*`, `OUTPUT/basis-content*`) für A/B-Vergleiche.
  Kein stilles Fallback zwischen den Fassungen. discover bleibt auf `markdown`.
- Identity (Titel, Datum, Startzeit) hat vorerst keine Zitatpflicht und keine
  Textlängenbegrenzung. Fehlendes Jahr gilt als aktuelles Jahr. JSON-LD geht
  als Hinweis in den Prompt.
- `api_gate` erkennt API-Quellen (Tribe/gcal) und prüft ihre Erreichbarkeit.
  Die Event-Discovery ignoriert API-Daten vorerst und arbeitet mit den
  gecrawlten HTML-Seiten.
- `INPUT/sources.csv` und `offene-todos` sind lokale Arbeitsdateien (nicht im
  Repo). Nutze `INPUT/sources.example.csv` als Startpunkt für deine Quellen.
