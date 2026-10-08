# fetchV2 — Event-Pipeline: erst perfekte Basisinfos, dann Details

```
INPUT/sources.csv
→ fetch --mode overviews   -> out/pages/ + out/pages_readable/
→ api-gate                 -> out/api_sources.json
→ discover                 -> out/discovery.json
→ fetch --mode urls        -> out/pages/ (Detailseiten)
→ basis                    -> out/basis.json + out/basis-excluded.csv
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
npm test       # reine Funktionstests, kein Netz, kein LLM
```

## Prinzipien

- Fetch crawlt nur (kein LLM, kein Urteil). Corpus unter `out/pages/`.
- Gates (past, Wien) sind deterministisch; unklar = durchlassen.
- Jedes extrahierte Feld braucht ein wörtliches Zitat aus dem Quelltext.
- `api_gate` erkennt API-Quellen (Tribe/gcal) und taggt sie für das
  parallele API-Leseskript; HTML bleibt Fallback.
