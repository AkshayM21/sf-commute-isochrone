# SF Commute Isochrone

**Live at [sfcommutemap.com](https://sfcommutemap.com).** Free, no signup: enter a workplace
address and the map colors every SF neighborhood by door-to-door commute time. **Hover any
square to see — and watch drawn on the map — the actual route you'd take** (walk legs, transit
lines with labels, transfers, total time); **click to pin a card and compare alternative routes**,
one at a time. Hosted on Oracle Always Free ($0/mo) behind Cloudflare. Self-host instructions in
`deploy/`.

Find **where in San Francisco you can live and still reach a workplace you
choose within N minutes** by **walking + Muni + BART + Caltrain**, arriving on a
weekday morning — rendered as an interactive map.

For a 200m grid of origin points across SF, the tool routes door-to-door
(walk → transit incl. transfers → walk) to your chosen destination using real GTFS
schedules and the OpenStreetMap walking network. The live application uses its own
JVM-free reverse range-RAPTOR transit router plus a hill-aware walking graph. Results
are aggregated to the 117 SF "Find Neighborhoods" and ranked by travel time.

> **Note (2026-05-25): the live server now defaults to a JVM-free stack** — a self-built reverse
> RAPTOR transit router + a hill-aware (Tobler) walk router — so **Java/R5 is no longer required to
> run it** (~245 MB RSS, no JVM). The map answers **depart-after** ("leave during a ~8am window —
> how long to work?"), colored by door-to-door commute DURATION (so "arrive ~9am" is implicit). It
> has a scheduled/best-case toggle, a "bad-day" service-delay chip, and a slow/med/fast walk-speed
> control. The scheduled depart-after read is timed to the first vehicle, so controllable starting
> wait is not counted. **Walking faster can never increase the commute** (true-zero monotonicity,
> guaranteed).
> R5 is kept only for offline oracle/validation and as a fallback. See **RAPTOR.md** for the engine
> + the one-time bakes (`fetch_dem.sh` → `build_walk_graph.py` → `bake_walk_access.py`); the
> prereqs/quick-start below still describe the original R5 build and are being updated.

Two numbers per location:
- **scheduled** — timed to the first vehicle you choose to board; starting wait is not counted
- **bad-day** — the p90 service-delay tail for that committed plan

## Prerequisites

- **macOS or Linux**
- **Java 21** (R5 is a JVM routing engine) — e.g. `brew install openjdk@21`
- **uv** (Python env + package manager) — `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`
- **osmium-tool** (clips the OSM extract to SF) — `brew install osmium-tool`
- A **free 511.org API token** for the current Muni + Caltrain feeds — get one
  at <https://511.org/open-data/token>

`setup.sh` checks each of these and prints install hints if anything is missing.

## Quick start

```bash
git clone <this-repo> sf-commute-isochrone
cd sf-commute-isochrone

# 1. Configure API access (this file is gitignored)
cp .env.example .env
#    then edit .env and set:
#      API511_TOKEN=...         (your free 511.org token), and
#      GEOAPIFY_KEY=...         (free key for address search + autocomplete;
#                                omit to fall back to keyless Photon)
#    The live page does not inject a default workplace; visitors enter their own.

# 2. Install the environment and download all input data
bash scripts/setup.sh
```

`setup.sh` is idempotent: it creates the Python venv only if missing and
downloads each dataset only if it isn't already present, so it's safe to re-run.

## Usage

All commands use the venv created by `setup.sh` at
`.venv`.

### Live interactive server (recommended)

Boots once (warm R5 network + grid), then recomputes on demand as you change the
destination in the browser:

```bash
.venv/bin/python scripts/server.py
# then open http://127.0.0.1:8000
```

Tunable via environment variables:

| Var          | Default | Meaning                                          |
|--------------|---------|--------------------------------------------------|
| `GRID_M`     | `200`   | grid resolution in meters (smaller = finer/slower) |
| `WINDOW_MIN` | `30`    | departure-time window in minutes                 |
| `PORT`       | `8000`  | server port                                      |

```bash
GRID_M=200 PORT=8080 .venv/bin/python scripts/server.py
```

### Static explorers and rankings

Generate the standalone HTML map, ranked CSV, and per-cell travel-time grids in
`out/`:

```bash
PY=.venv/bin/python

# full network (Muni + BART + Caltrain)
$PY scripts/isochrone.py --tag full

# Muni only (for the "BART off" comparison)
$PY scripts/isochrone.py --gtfs muni_current.zip --tag munionly

# combine both grids into one interactive page (out/commute_explorer.html)
$PY scripts/make_interactive.py
```

`isochrone.py` writes, per `--tag`:
- `out/isochrone_map_<tag>.html` — interactive banded map
- `out/neighborhoods_ranked_<tag>.csv` — ranked table
- `out/grid_traveltimes_<tag>.gpkg` — per-cell travel times

Optional extras:

```bash
$PY scripts/itineraries.py    # example door-to-door itineraries
$PY scripts/route_map.py      # which transit routes dominate each area
```

#### `isochrone.py` CLI flags

| Flag      | Description                                                        |
|-----------|-------------------------------------------------------------------|
| `--tag`   | suffix for output filenames (e.g. `full`, `munionly`)             |
| `--gtfs`  | extra GTFS zip(s) added to the default `muni_current.zip` + `bart_gtfs.zip` + `caltrain.zip` feeds |
| `--limit` | randomly subsample N grid origins for a fast validation run        |

## How the destination works

The destination is configured once, in a gitignored `.env` (copy from
`.env.example`). `scripts/destination.py` resolves it in this order:

1. `DEST_LAT` + `DEST_LON` (explicit coordinates; `DEST_LABEL` optional)
2. `DEFAULT_ADDRESS` — geocoded once via OpenStreetMap Nominatim and cached in
   `.dest_cache.json` so later runs don't re-geocode
3. fallback: the geographic center of San Francisco

Change the destination by editing `.env`; delete `.dest_cache.json` if you want
to force a fresh geocode.

## Data sources & credits

All inputs are downloaded by `setup.sh` into `data/` (gitignored):

- **Muni GTFS** — [511 SF Bay Open Data](https://511.org/open-data) (current
  feed, operator `SF`, via `scripts/fetch_511.sh` → `data/muni_current.zip`)
- **Caltrain GTFS** — [511 SF Bay Open Data](https://511.org/open-data) (current
  feed, operator `CT`, via `scripts/fetch_511.sh` → `data/caltrain.zip`)
- **BART GTFS** — [BART developer feed](https://www.bart.gov/dev/schedules/google_transit.zip) (→ `data/bart_gtfs.zip`)
- **OpenStreetMap** — [Geofabrik](https://download.geofabrik.de/) NorCal
  extract, clipped to SF with `osmium` (© OpenStreetMap contributors)
- **Neighborhoods** — [DataSF](https://data.sfgov.org/) "SF Find
  Neighborhoods" (`gfpk-269f`)

## Methodology

The live route metric is a full scheduled door-to-door trip: physical walk to the
boarding stop → transit (including transfers) → physical walk to the destination.
It is anchored to the first scheduled vehicle, so controllable time spent arriving
early for that first boarding is reported separately as schedule allowance rather
than mislabeled as walking. A service-delay simulation supplies the bad-day impact.

Walking uses fixed Slow (3.4 km/h), Medium (4.2 km/h), and Fast/Brisk (5.2 km/h)
presets. Access, transfer, egress, and pure-walk legs all use the selected pace. See
[`WALK_SPEED_CALIBRATION_2026-08-09.md`](WALK_SPEED_CALIBRATION_2026-08-09.md) for
the bounded San Francisco calibration corpus and [`RAPTOR.md`](RAPTOR.md) for the
routing model. Conveyal R5 remains an offline validation oracle and fallback, not
the normal map or itinerary path.

## Notes

`out/`, `REPORT.md`, `.env`, and `.dest_cache.json` are gitignored: they are
personal/address-specific and stay on your machine. The Python venv lives in
`.venv` (gitignored) at the repo root.
