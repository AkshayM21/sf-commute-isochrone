# SF Commute Isochrone

**Find where in San Francisco you can live and still reach a workplace within a
chosen commute time.** The interactive map estimates weekday-morning,
door-to-door trips by walking, Muni, BART, and Caltrain. Hover or tap a grid cell
to inspect its route, compare practical alternatives, and open step-by-step
directions.

**Live at [sfcommutemap.com](https://sfcommutemap.com).** It is free and does not
require an account.

The served application uses a JVM-free reverse range-RAPTOR transit engine and a
hill-aware OpenStreetMap walking graph. It groups routes structurally by boarding
corridor and destination approach, surfaces trade-offs such as walking and
transfers, and provides scheduled, best-case, and bad-day views.

## What the estimate means

Every colored 200 m grid cell represents a physical door-to-door trip:

```text
walk to a boarding stop → scheduled transit, including transfers → walk to work
```

- **Scheduled** is timed to the first selected vehicle; optional early-arrival
  allowance before boarding is shown separately rather than counted as walking.
- **Best-case** represents a well-timed trip within the model window.
- **Bad-day** adds the model's service-delay tail for the selected plan; it is an
  estimate, not a prediction or live alert.
- Walking presets are Slow (3.4 km/h), Medium (4.2 km/h), and Fast/Brisk (5.2
  km/h). They apply consistently to access, transfer, egress, and walking-only
  legs.

The map uses a scheduled-data snapshot, not real-time vehicle positions,
disruptions, accessibility status, fares, or guaranteed arrival times. The service
date is selected from the GTFS feeds available when the bakes are created. Refresh
the feeds and rebuild the bakes when schedules change before relying on a local
deployment for a new service period.

Historical local benchmarks and R5-comparison results are documented in
[RAPTOR.md](RAPTOR.md). They describe a particular data snapshot and machine; they
are not performance or accuracy guarantees.

## Requirements

The supported development/runtime target is **Python 3.12** on macOS or Linux.

For the standard JVM-free application and its data bootstrap:

- [uv](https://docs.astral.sh/uv/) to create the virtual environment and install
  Python packages
- `curl` and `unzip` for input downloads
- [`osmium-tool`](https://osmcode.org/osmium-tool/) to clip the OpenStreetMap
  extract to San Francisco
- a free [511.org Open Data token](https://511.org/open-data/token) for the
  current Muni and Caltrain feeds

**Java is not required to run the served app after its bakes exist.** A JDK 21 and
the optional R5 dependencies are required to reproduce the one-time raw-data
access-table seed, run offline R5 validation, or use the legacy static isochrone
exporter. That distinction is intentional: the normal map, route inspection, and
API path do not import R5 or start a JVM.

## Quick start: reproduce a local JVM-free server

The repository deliberately does not commit transit feeds, OpenStreetMap data, or
derived bakes. A clean clone therefore needs the raw inputs and one local bake
cycle before it can serve the fully JVM-free path.

```bash
git clone --branch main --single-branch https://github.com/AkshayM21/sf-commute-isochrone sf-commute-isochrone
cd sf-commute-isochrone

# Configure data access. This file is ignored by Git.
cp .env.example .env
# Edit .env and set API511_TOKEN=... . GEOAPIFY_KEY is optional for search.

# Create Python 3.12 environment, install requirements.txt, and fetch raw inputs.
bash scripts/setup.sh
```

### One-time R5 seed, then JVM-free bakes

The current raw-data seed starts with R5 because the access table's canonical cell
set is bridged from its offline walk matrix. This is a **bootstrap/validation
operation**, not a request-time dependency. Install JDK 21, then run:

```bash
PY=.venv/bin/python

# One-time optional R5 toolchain and access seed.
uv pip install --python "$PY" -r requirements-r5.txt
ONLY_ACCESS=1 R5_MAX_MEMORY=4G "$PY" scripts/raptor_oracle.py

# Build the hill-aware, JVM-free walking artifacts.
bash scripts/fetch_dem.sh
"$PY" scripts/build_walk_graph.py
"$PY" scripts/bake_walk_access.py
```

`raptor_oracle.py` uses a JVM only for this seed. Once `walk_graph.npz` and an
`access_walk_*.npz` bake match the downloaded feeds, start the normal app without
R5 or Java:

```bash
.venv/bin/python scripts/server.py
# Open http://127.0.0.1:8000
```

The first JVM-free boot also writes `data/server_static.json`, a lightweight static
bundle keyed to the GTFS inputs. It is regenerated automatically after relevant
feed changes. If the raw data are refreshed, repeat the R5 seed and both walking
bakes above so their fingerprints and service date stay aligned.

### Dependency sets

| File | Purpose |
| --- | --- |
| [requirements.txt](requirements.txt) | Default JVM-free runtime and raw-data/build tools |
| [requirements-dev.txt](requirements-dev.txt) | Runtime stack plus unit and browser-test tooling |
| [requirements-r5.txt](requirements-r5.txt) | Optional offline R5 seed and validation tooling; requires JDK 21 |

All dependency ranges are intentionally bounded to compatible major/minor releases
rather than a machine-specific lockfile. Use the provided requirements files for a
repeatable supported installation; generate a lock in your own deployment
environment if you need artifact-level pinning.

## Usage

### Interactive server

```bash
PORT=8080 .venv/bin/python scripts/server.py
```

Useful runtime configuration:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `PORT` | `8000` | Local listening port |
| `GRID_M` | `200` | Grid resolution in metres; smaller is finer and more expensive |
| `WINDOW_MIN` | `30` | Departure-time window in minutes |
| `RAPTOR_SEMANTIC` | `departafter` | Served scheduled routing semantic; `arriveby` is legacy/opt-in |
| `RAPTOR_WALK_RELUCTANCE` | `1.15` | Tie-break preference for less walking among near-equal trips |

The browser keeps a workplace locally for convenience. Do not commit `.env`,
`.dest_cache.json`, downloaded data, generated output, or browser captures; the
repository's `.gitignore` excludes them by default.

### Optional R5 analyses and static exporters

These legacy offline scripts use R5. Install JDK 21 and the optional dependency
set before running them:

```bash
PY=.venv/bin/python
uv pip install --python "$PY" -r requirements-r5.txt

"$PY" scripts/isochrone.py --tag full
"$PY" scripts/isochrone.py --gtfs muni_current.zip --tag munionly
"$PY" scripts/make_interactive.py
```

Outputs are written to the ignored `out/` directory. `scripts/itineraries.py` and
`scripts/route_map.py` use the same optional R5 toolchain.

## Data sources and refreshes

`scripts/setup.sh` fetches external, untracked inputs into `data/`:

- Muni and Caltrain GTFS from [511 SF Bay Open Data](https://511.org/open-data)
  (requires `API511_TOKEN`)
- BART GTFS from the [BART developer feed](https://www.bart.gov/dev/schedules/google_transit.zip)
- OpenStreetMap's Northern California extract from
  [Geofabrik](https://download.geofabrik.de/), clipped locally to San Francisco
- San Francisco neighborhood geometry from
  [DataSF](https://data.sfgov.org/) (including the Find Neighborhoods source)
- a San Francisco elevation raster from USGS 3DEP for the hill-aware walking bake

The downloads and bakes are cached by feed fingerprints, but they are snapshots.
They can change, disappear, or acquire updated terms at their sources. Review the
applicable provider terms, refresh deliberately, and rerun the seed/bake workflow
when operating your own instance.

## Tests

Install contributor dependencies and the browser once:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m playwright install chromium
```

Run the repository checks from the root:

```bash
# Python suite; excludes the browser tests that need a separately running server.
.venv/bin/python -m pytest tests/ --ignore=tests/e2e -q

# Frontend logic suite (Node.js 20+).
node --test tests/test_viz.mjs

# Browser suite: start the server in another terminal first.
.venv/bin/python scripts/server.py
tests/e2e/run.sh
```

See [tests/e2e/README.md](tests/e2e/README.md) for targeted and opt-in route-family
stress tests. The test suite uses neutral public locations; avoid adding a real
home or workplace to source, fixtures, screenshots, or benchmark artifacts.

## Project documents

- [License (MIT)](LICENSE)
- [Contributing guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Privacy policy](PRIVACY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Routing model and historical validation](RAPTOR.md)
- [Walking-speed calibration](WALK_SPEED_CALIBRATION_2026-08-09.md)

## Credits

Built with public transit schedules and map data from 511, BART, OpenStreetMap
contributors, DataSF, Geofabrik, and USGS. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license and attribution
details.
