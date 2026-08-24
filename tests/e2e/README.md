# End-to-end browser tests — SF Commute Explorer

Durable Playwright (Python) browser tests for the Leaflet web app, covering **desktop**
(1280x800) and **mobile** (iPhone 390x844, touch). They drive the **already-running**
Graph-native Flask + RAPTOR + walk-graph server; they do not boot their own application process.

## Prerequisites (one-time)

The harness deps are installed into the repo-local `.venv` with **uv**:

```bash
uv pip install -p .venv/bin/python playwright pytest-playwright
.venv/bin/python -m playwright install chromium
```

The featured server must be running at `http://127.0.0.1:8000`:

```bash
.venv/bin/python scripts/server.py
```

## Run the suite

From anywhere in the repo:

```bash
tests/e2e/run.sh                     # full suite (desktop + mobile), headless chromium
tests/e2e/run.sh test_desktop.py     # desktop specs only
tests/e2e/run.sh test_mobile.py      # mobile specs only
tests/e2e/run.sh test_route_families.py  # bounded route-family hotspot + card audit
tests/e2e/run.sh -k autocomplete     # filter by test name
HEADED=1 tests/e2e/run.sh            # watch in a real browser window
E2E_BASE_URL=http://host:port tests/e2e/run.sh   # different server

# Broader deterministic SF destination × walk-speed sampler (opt-in, still rate-bounded):
ROUTE_FAMILY_HOTSPOT_SCAN=1 tests/e2e/run.sh test_route_families.py
```

`run.sh` checks the server is reachable, then runs `pytest` from this directory (so
`pytest.ini` + `conftest.py` apply). Equivalent manual invocation:

```bash
cd tests/e2e && ../../.venv/bin/python -m pytest
```

## What's covered

- `test_desktop.py` — specs 1-10: load/no-errors, autocomplete (+ dedup flag), set/exact
  map, no-refine surface, hover breakdown, color-by-line, sliders, export (CSV + clipboard),
  permalink round-trip, how-it-works modal.
- `test_mobile.py` — specs 11-14: bottom-sheet layout + toggle, address on mobile,
  tap-to-breakdown (the key touch interaction), controls reachable + legend clear.
- `test_route_families.py` — a bounded compute → variance → pinned-itinerary sampler plus a
  real-canvas replay of a saved public hotspot at desktop, tight-desktop, and mobile widths. It
  checks the recommendation-first inspector, durable per-choice identity, authoritative family /
  branch / service metadata, API→DOM labels and times, route focus, clipping, horizontal overflow,
  card overlap, and scroll reachability.
- `route_family_hotspots.py` — fixed public SF coordinates, deterministic seeded candidate
  selection/ranking, false-advertising invariants, DOM measurements, and JSON artifact helpers.
- `conftest.py` — shared fixtures/helpers. Notably `find_colored_cell_hover` /
  `find_colored_cell_tap` locate a colored Leaflet **canvas** cell by driving real
  mouse/touch events (there is no per-cell DOM). Map-computed is detected via the
  non-empty user-visible `#dest` label + the neighborhood list, not internal globals. RAPTOR
  intentionally omits the legacy internal `fast ~Nms` timing string.

Screenshots of key desktop + mobile states are written to `tests/e2e/screens/`.
The hotspot sampler writes `tests/e2e/screens/route_family_hotspots.json` (also ignored by git).

## Route-family hotspot controls

The normal committed suite scans one destination at one speed and replays one saved hotspot. The
broader scan is opt-in so routine E2E remains practical and does not cross the server's live rate
limits. Available controls:

```bash
ROUTE_FAMILY_HOTSPOT_SCAN=1          # enable broad scan
ROUTE_FAMILY_HOTSPOT_SEED=20260712  # deterministic tie/sample seed
ROUTE_FAMILY_HOTSPOT_DESTS=6        # seeded subset of fixed public catalog (max 8)
ROUTE_FAMILY_HOTSPOT_SPEEDS=slow,med,fast
ROUTE_FAMILY_HOTSPOT_PER_CONFIG=5   # clamped to 5
ROUTE_FAMILY_HOTSPOT_ARTIFACT=/tmp/route-family-hotspots.json
```

The full 8-destination catalog × 3 speeds × 5 origins would exceed the `/variance` 20/minute
limit. The broad test therefore selects at most 6 destinations with all three speeds by default;
the test hard-fails if the selected configuration exceeds 18 variance calls or 90 pinned itinerary
calls. A practical full scan is therefore:

```bash
ROUTE_FAMILY_HOTSPOT_SCAN=1 ROUTE_FAMILY_HOTSPOT_DESTS=6 \
  tests/e2e/run.sh test_route_families.py -k broad
```

## Expected skips and failures

There are no intentional expected failures. Under the graph-native engine, `/compute` is already
exact and the obsolete Refine control is absent. Autocomplete deduplication
and primary-line attribution are active passing regressions; do not restore the old expected-fail
documentation or weaken those assertions.

Privacy: tests only use neutral public addresses ("ferry building" / "1 Market St") and
clear localStorage + location.hash before each test.
