# End-to-end browser tests — SF Commute Explorer

Durable Playwright (Python) browser tests for the Leaflet web app, covering **desktop**
(1280x800) and **mobile** (iPhone 390x844, touch). They drive the **already-running**
featured Flask+R5 server — they do NOT boot their own R5.

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
tests/e2e/run.sh -k autocomplete     # filter by test name
HEADED=1 tests/e2e/run.sh            # watch in a real browser window
E2E_BASE_URL=http://host:port tests/e2e/run.sh   # different server
```

`run.sh` checks the server is reachable, then runs `pytest` from this directory (so
`pytest.ini` + `conftest.py` apply). Equivalent manual invocation:

```bash
cd tests/e2e && ../../.venv/bin/python -m pytest
```

## What's covered

- `test_desktop.py` — specs 1-10: load/no-errors, autocomplete (+ dedup flag), set/fast
  map, opt-in refine, hover breakdown, color-by-line, sliders, export (CSV + clipboard),
  permalink round-trip, how-it-works modal.
- `test_mobile.py` — specs 11-14: bottom-sheet layout + toggle, address on mobile,
  tap-to-breakdown (the key touch interaction), controls reachable + legend clear.
- `conftest.py` — shared fixtures/helpers. Notably `find_colored_cell_hover` /
  `find_colored_cell_tap` locate a colored Leaflet **canvas** cell by driving real
  mouse/touch events (there is no per-cell DOM). Map-computed is detected via the
  user-visible `#dest` "fast ~Nms" text + the neighborhood list, not internal globals.

Screenshots of key desktop + mobile states are written to `tests/e2e/screens/`.

## Known failures = real bugs (NOT harness flakes)

Two tests are written to the CORRECT expectation and **fail on purpose** to surface
genuine product bugs (do not "fix" them by relaxing the assertion):

1. `test_02b_autocomplete_dedup_FLAG` — `/autocomplete` returns exact-duplicate rows
   (same label + lat/lon) and the frontend renders them verbatim, so the dropdown shows
   duplicate suggestions. **Fix:** dedup results by label (and/or lat,lon) before render.
2. `test_06_color_by_line` — toggling "Color by → Primary line" flips the legend title to
   "Primary transit line per area" but `/attribution` returns `{}` (0 cells) for every
   destination tried (verified directly via curl: empty after a 36-169s build), so **no
   cells ever recolor**. The color-by-line feature is effectively broken server-side.

Privacy: tests only use neutral public addresses ("ferry building" / "1 Market St") and
clear localStorage + location.hash before each test.
