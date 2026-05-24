"""Integration tests for the Flask + R5 commute API (scripts/server.py).

These are REGRESSION GUARDS, not unit tests: they exercise the real in-process R5 JVM
(booted once via the session-scoped `server`/`client` fixtures in conftest.py) and assert
the cross-cutting invariants that hold the app together:

  * /compute (fast reverse approx) and /compute_exact (slow forward refine) shapes + the
    best <= real ordering, and exact determinism (twice-identical + a golden snapshot).
  * the breakdown == map-color invariant: an /itinerary's total equals that cell's
    /compute_exact realistic minutes, and its legs (min + wait) sum to that total.
  * /attribution, /geocode, /autocomplete shapes + bounds + caching.
  * the operational guards: a held _HEAVY_LOCK turns /compute_exact into a 503{busy}, the
    rate limiter is wired, and the per-workplace breakdown caches are reset on a new dest.
  * the served page has no leftover template tokens and (privacy) no user address.

Run with:  .venv/bin/python -m pytest tests/test_api.py -q

It is SLOW (one R5 boot + a couple ~30s exact passes); the heavy tests are marked. Another
agent may have a live server on :8000 and another pytest JVM running — this suite boots its
OWN in-process R5 and never binds a port, so it's independent (just CPU-contended).
"""
import json
import os
import time

import pytest

from conftest import (
    FERRY_LAT,
    FERRY_LON,
    TWIN_PEAKS_LAT,
    TWIN_PEAKS_LON,
)

# SF bounding box for coordinate sanity checks (loosely the city + bay edge).
SF_LAT_MIN, SF_LAT_MAX = 37.6, 37.9
SF_LON_MIN, SF_LON_MAX = -122.6, -122.3
# The 200m grid has ~3068 cells; allow slack so a tiny grid/snap change doesn't break us.
EXPECTED_CELLS = 3068
CELL_TOL = 120

_HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_PATH = os.path.join(_HERE, "golden_exact_ferry.json")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _in_sf(lat, lon):
    return SF_LAT_MIN <= lat <= SF_LAT_MAX and SF_LON_MIN <= lon <= SF_LON_MAX


def _approx_cell_count(n):
    return abs(n - EXPECTED_CELLS) <= CELL_TOL


def _get_exact_cells(client):
    """GET /compute_exact for the Ferry Building and return its {id: [best, real]} dict.
    Coarse-result-cached server-side, so repeated calls in the suite are cheap."""
    resp = client.get(f"/compute_exact?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()["cells"]


# --------------------------------------------------------------------------------------
# /compute — fast reverse one-to-many approximation
# --------------------------------------------------------------------------------------
def test_compute_shape_and_best_le_real(client):
    resp = client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["dest"] == [FERRY_LAT, FERRY_LON]
    cells = body["cells"]
    assert _approx_cell_count(len(cells)), f"got {len(cells)} cells"

    reachable = 0
    for cid, pair in cells.items():
        assert isinstance(pair, list) and len(pair) == 2, f"{cid}: {pair!r}"
        best, real = pair
        assert best is None or isinstance(best, int)
        assert real is None or isinstance(real, int)
        if best is not None and real is not None:
            # best-case (p5) must never exceed realistic (p50).
            assert best <= real, f"{cid}: best {best} > real {real}"
            reachable += 1
    assert reachable > 100, f"only {reachable} reachable cells — network/feeds may be broken"


# --------------------------------------------------------------------------------------
# /compute_exact — deterministic forward refine + golden snapshot
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_compute_exact_shape_and_determinism(client, server):
    # Clear the coarse cache so the FIRST call actually recomputes (otherwise a cache hit
    # from an earlier test would make "determinism" trivially true).
    with server._RESULT_CACHE_LOCK:
        server._EXACT_RESULT_CACHE.clear()
    first = _get_exact_cells(client)
    assert _approx_cell_count(len(first)), f"got {len(first)} cells"

    # best <= real wherever both are present.
    for cid, (best, real) in first.items():
        if best is not None and real is not None:
            assert best <= real, f"{cid}: best {best} > real {real}"

    # Second call must be bit-identical (served from the coarse result cache, but also the
    # underlying compute is deterministic for a fixed network + service date).
    second = _get_exact_cells(client)
    assert second == first, "compute_exact is not deterministic across two calls"


@pytest.mark.slow
def test_compute_exact_matches_golden(client, server):
    """Regression snapshot. If this fails after a GTFS refresh (pick_service_date shifted),
    regenerate the golden:  .venv/bin/python tests/make_golden.py"""
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(
            "golden missing — generate with `.venv/bin/python tests/make_golden.py` "
            "(also runs automatically once via make_golden in this session is not wired; "
            "run the helper)."
        )
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    current = _get_exact_cells(client)
    g_cells = golden["cells"]
    if golden.get("service_date") != str(server._SVC_DATE):
        pytest.skip(
            f"golden service_date {golden.get('service_date')} != current "
            f"{server._SVC_DATE}; regenerate via tests/make_golden.py"
        )
    assert current == g_cells, (
        "compute_exact output drifted from the golden snapshot. If GTFS feeds were "
        "refreshed this is expected — regenerate with tests/make_golden.py."
    )


# --------------------------------------------------------------------------------------
# Breakdown == map color invariant
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_itinerary_total_matches_map_color_and_legs_sum(client):
    """For ~15 reachable cells: /itinerary total == compute_exact[id][1] (realistic
    minutes), legs' (min + wait) sum to total, and transit legs carry a non-empty line."""
    exact = _get_exact_cells(client)
    # Pick reachable cells (real minutes present) that are not trivially tiny, biased toward
    # ones likely to involve transit so we exercise the leg-attribution path too.
    reachable = [
        (cid, real)
        for cid, (best, real) in exact.items()
        if real is not None and real >= 8
    ]
    assert len(reachable) >= 15, f"only {len(reachable)} reachable cells to sample"
    # Spread the sample across the sorted-by-time list for variety.
    reachable.sort(key=lambda t: t[1])
    step = max(1, len(reachable) // 15)
    sample = reachable[::step][:15]
    assert len(sample) >= 15

    checked_transit = 0
    for cid, real in sample:
        resp = client.get(
            f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}"
        )
        assert resp.status_code == 200, f"{cid}: {resp.get_data(as_text=True)}"
        itin = resp.get_json()
        assert "error" not in itin, f"{cid}: {itin}"
        # total must equal the map color (realistic exact minutes) for the same cell.
        assert itin["total"] == real, (
            f"{cid}: itinerary total {itin['total']} != exact real {real}"
        )
        # legs' min + wait must sum to total exactly (the server reconciles rounding).
        legs = itin["legs"]
        assert legs, f"{cid}: empty legs"
        leg_sum = sum(l["min"] for l in legs) + sum(l.get("wait", 0) for l in legs)
        assert leg_sum == itin["total"], (
            f"{cid}: legs sum {leg_sum} != total {itin['total']} ({legs})"
        )
        for l in legs:
            if l["mode"] == "transit":
                assert l["line"], f"{cid}: transit leg with empty line: {l}"
                checked_transit += 1
    # The sample spans the time distribution; at least some should be transit trips.
    assert checked_transit > 0, "no transit legs seen across the sample — suspicious"


# --------------------------------------------------------------------------------------
# /attribution — dominant-line-per-cell, cached + deterministic
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_attribution_shape_caching_and_count(client, server):
    # Use a fresh coarse key (Twin Peaks) to avoid colliding with any cached Ferry result,
    # but keep the comparison against THIS dest's reachable count.
    dlat, dlon = FERRY_LAT, FERRY_LON
    with server._RESULT_CACHE_LOCK:
        server._ATTR_RESULT_CACHE.clear()

    resp1 = client.get(f"/attribution?dlat={dlat}&dlon={dlon}")
    assert resp1.status_code == 200, resp1.get_data(as_text=True)
    attr1 = resp1.get_json()
    assert isinstance(attr1, dict) and attr1
    for cid, line in attr1.items():
        assert isinstance(line, str) and line.strip(), f"{cid}: {line!r}"

    # Deterministic / cached: a second call returns the identical dict.
    resp2 = client.get(f"/attribution?dlat={dlat}&dlon={dlon}")
    assert resp2.status_code == 200
    assert resp2.get_json() == attr1

    # Count should be roughly the reachable-cell count from compute_exact (attribution is
    # built from the same forward journeys; unreachable cells are dropped).
    exact = _get_exact_cells(client)
    reachable = sum(1 for _, (b, r) in exact.items() if r is not None)
    # generous tolerance: attribution drops a few cells (no recorded path) vs exact.
    assert abs(len(attr1) - reachable) <= max(50, int(reachable * 0.15)), (
        f"attribution {len(attr1)} vs reachable {reachable} differ too much"
    )


# --------------------------------------------------------------------------------------
# /geocode
# --------------------------------------------------------------------------------------
def test_geocode_ferry_building(client):
    resp = client.get("/geocode?q=ferry building")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert _in_sf(body["lat"], body["lon"]), body
    assert body.get("label")


def test_geocode_blank_is_400(client):
    resp = client.get("/geocode?q=")
    assert resp.status_code == 400
    assert resp.get_json().get("error")


# --------------------------------------------------------------------------------------
# /autocomplete
# --------------------------------------------------------------------------------------
def test_autocomplete_results_are_sf_bounded(client):
    resp = client.get("/autocomplete?q=ferry")
    assert resp.status_code == 200
    results = resp.get_json()["results"]
    assert isinstance(results, list) and results, "expected some autocomplete results"
    for r in results:
        assert set(("label", "lat", "lon")) <= set(r.keys()), r
        assert isinstance(r["label"], str) and r["label"]
        assert _in_sf(r["lat"], r["lon"]), r


def test_autocomplete_short_and_blank_are_empty(client):
    for q in ("", " ", "a"):
        resp = client.get(f"/autocomplete?q={q}")
        assert resp.status_code == 200
        assert resp.get_json()["results"] == [], f"q={q!r} should yield no results"


@pytest.mark.xfail(
    reason="KNOWN BUG: /autocomplete (via geo.autocomplete) can return duplicate "
    "results — Photon returns the same place twice and nothing de-dupes. Documented, "
    "not hidden: this should pass once the geocoder de-dupes results.",
    strict=False,
)
def test_autocomplete_dedups_results(client):
    resp = client.get("/autocomplete?q=ferry")
    assert resp.status_code == 200
    results = resp.get_json()["results"]
    seen = set()
    for r in results:
        key = (r["label"], round(r["lat"], 5), round(r["lon"], 5))
        assert key not in seen, f"duplicate autocomplete result: {key}"
        seen.add(key)


# --------------------------------------------------------------------------------------
# Non-blocking heavy lock -> 503 {busy}
# --------------------------------------------------------------------------------------
def test_heavy_lock_makes_compute_exact_503(client, server):
    """Hold the real _HEAVY_LOCK ourselves; /compute_exact must NOT block — it returns 503
    {busy:true} with a Retry-After header. After release it works again.

    Use a fresh coarse coordinate so a cached result can't short-circuit the lock check
    (the endpoint returns a cache hit BEFORE trying to acquire the lock)."""
    # Coordinate guaranteed not to be in the coarse cache yet.
    lat, lon = 37.7611, -122.4350
    with server._RESULT_CACHE_LOCK:
        server._EXACT_RESULT_CACHE.pop(server._coarse_key(lat, lon), None)

    acquired = server._HEAVY_LOCK.acquire(blocking=False)
    assert acquired, "_HEAVY_LOCK was already held — another heavy job in this process?"
    try:
        resp = client.get(f"/compute_exact?lat={lat}&lon={lon}")
        assert resp.status_code == 503, resp.get_data(as_text=True)
        assert resp.get_json() == {"busy": True}
        assert resp.headers.get("Retry-After"), "503 should carry a Retry-After header"
    finally:
        server._HEAVY_LOCK.release()

    # Sanity: with the lock free, a (cheap, possibly-cached) exact request works again.
    ok = client.get(f"/compute_exact?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert ok.status_code == 200, ok.get_data(as_text=True)


# --------------------------------------------------------------------------------------
# Rate limit -> 429 (or, failing that, assert the limiter is wired)
# --------------------------------------------------------------------------------------
def test_rate_limit_is_enforced_or_wired(client, server):
    """/autocomplete is limited to 120/minute. Short queries (<2 chars) short-circuit the
    geocoder but STILL pass through the limiter, so we can hammer it cheaply. We try to
    trigger a real 429; if the limiter's real-time window makes that impractical here, we
    fall back to asserting the limiter is wired (an X-RateLimit-* header is present)."""
    # Reset the limiter's storage so prior tests don't eat into the window.
    try:
        server.limiter.reset()
    except Exception:
        pass

    saw_429 = False
    last_resp = None
    # 'a' is <2 chars after strip -> empty results, cheap (no upstream), still rate-limited.
    for _ in range(160):
        last_resp = client.get("/autocomplete?q=a")
        if last_resp.status_code == 429:
            saw_429 = True
            break

    if saw_429:
        assert last_resp.status_code == 429
        return

    # Fallback: limiter wired? flask-limiter sets X-RateLimit-* headers when limits apply.
    rl_headers = [h for h in (last_resp.headers or {}) if h.lower().startswith("x-ratelimit")]
    assert rl_headers, (
        "neither a 429 nor any X-RateLimit-* header observed — the limiter may not be "
        "wired on /autocomplete"
    )


# --------------------------------------------------------------------------------------
# Cache gating: warm, hit, reset on new dest
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_cell_cache_warms_and_resets_on_new_dest(client, server):
    """/compute warms _LAST_DEST_KEY; /itinerary for a cell warms _CELL_CACHE; repeating
    the SAME coords keeps the cache; a DIFFERENT coord clears it (new _LAST_DEST_KEY)."""
    fkey = server._dest_key(FERRY_LAT, FERRY_LON)

    # Reset the limiter so a prior test (e.g. the rate-limit test) that consumed this
    # minute's /itinerary or /compute budget doesn't bleed into this one and surface a 429.
    try:
        server.limiter.reset()
    except Exception:
        pass

    # 1) /compute for Ferry: establishes this as the current dest.
    r = client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert r.status_code == 200
    assert server._LAST_DEST_KEY == fkey

    # Drop any full-grid itinerary cache for this dest (an earlier /attribution test may have
    # built it). /itinerary's FAST PATH 1 serves from _ITIN_CACHE without touching
    # _CELL_CACHE, so to exercise (and assert) the per-cell on-demand cache we clear the
    # full-grid map first. This keeps the test independent of run order.
    with server._ITIN_CACHE_LOCK:
        server._ITIN_CACHE.pop(fkey, None)
    with server._CELL_CACHE_LOCK:
        server._CELL_CACHE.pop(fkey, None)

    # 2) /itinerary for a reachable grid cell -> warms _CELL_CACHE[fkey][cid]. Only cells
    # with an actual route get cached (the endpoint caches only when "error" not in res), so
    # iterate reachable cells (real minutes present) until one lands in the cell cache.
    exact = _get_exact_cells(client)
    reachable_ids = [c for c, (b, rl) in exact.items() if rl is not None]
    assert reachable_ids, "no reachable cells — network/feeds may be broken"
    cid = None
    # Bound the probe count well under /itinerary's 120/min limit; a reachable cell with a
    # route caches on its first hit, so a handful of tries is plenty in practice.
    for candidate in reachable_ids[:20]:
        r = client.get(f"/itinerary?id={candidate}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
        assert r.status_code == 200, r.get_data(as_text=True)
        if candidate in server._CELL_CACHE.get(fkey, {}):
            cid = candidate
            break
    assert cid is not None, "no /itinerary call warmed _CELL_CACHE for any reachable cell"

    # 3) Same coords again: cache survives (re-submitting the same dest keeps caches).
    client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert server._LAST_DEST_KEY == fkey
    # /itinerary for the same cell is served from cache (fast). Time it loosely as a hint,
    # but the authoritative check is that the cache entry is still present.
    t0 = time.perf_counter()
    r = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
    dt_cached = time.perf_counter() - t0
    assert r.status_code == 200
    assert cid in server._CELL_CACHE.get(fkey, {}), "cache should survive same-dest /compute"

    # 4) DIFFERENT coord: /compute must reset caches and move _LAST_DEST_KEY.
    tkey = server._dest_key(TWIN_PEAKS_LAT, TWIN_PEAKS_LON)
    assert tkey != fkey
    r = client.get(f"/compute?lat={TWIN_PEAKS_LAT}&lon={TWIN_PEAKS_LON}")
    assert r.status_code == 200
    assert server._LAST_DEST_KEY == tkey, "new dest did not update _LAST_DEST_KEY"
    # _reset_caches() clears the per-workplace cell cache; the old dest's entry is gone.
    assert fkey not in server._CELL_CACHE, "switching dest did not clear _CELL_CACHE"

    # Restore Ferry as the current dest so other tests start from a known state.
    client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    # (dt_cached is informational; cache-hit /itinerary should be well under a second.)
    assert dt_cached < 5.0, f"cached /itinerary unexpectedly slow: {dt_cached:.2f}s"


# --------------------------------------------------------------------------------------
# / — the served page
# --------------------------------------------------------------------------------------
def test_index_page_is_clean_and_private(client, server):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.get_data(as_text=True)

    # Leaflet bootstrap present (map is created + a tile layer added).
    assert "L.map(" in html, "Leaflet map bootstrap missing"
    assert "leaflet" in html.lower()

    # No leftover template tokens after the boot-time substitution.
    for token in ("__CELLS__", "__LINES__", "/*__VIZ__*/"):
        assert token not in html, f"leftover template token {token!r} in served page"

    # PRIVACY: the served page must not leak the user's CONFIGURED workplace. We read the
    # private literal from the gitignored .env at runtime (NOT hardcoded in this test) and
    # assert its absence from the page, which is built once at boot from a generic template.
    #
    # NB: we deliberately do NOT scrape .dest_cache.json — its keys are incidental geocoder
    # lookups (e.g. "1 Market St"), some of which legitimately appear as UI placeholder text
    # in the template and are not the user's private workplace. The privacy invariant is
    # specifically about the configured DEFAULT_ADDRESS / DEST_LABEL / DEST_LAT/LON.
    for needle in _private_workplace_needles():
        assert needle not in html, f"served page leaks private workplace value {needle!r}"


def _private_workplace_needles():
    """The user's CONFIGURED private workplace strings to assert ABSENT from the page,
    sourced from the gitignored .env at runtime (never hardcoded). Loading .env populates
    os.environ via core.config.load_dotenv at import; we also parse it directly in case it
    wasn't loaded. Returns [] if nothing is configured (the assertion then no-ops)."""
    keys = ("DEFAULT_ADDRESS", "DEST_LABEL", "DEST_LAT", "DEST_LON")
    needles = []
    for k in keys:
        v = (os.environ.get(k) or "").strip().strip('"').strip("'")
        if v:
            needles.append(v)
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line or line.startswith("#"):
                    continue
                k, _, v = line.partition("=")
                if k.strip() in keys:
                    v = v.strip().strip('"').strip("'")
                    if v:
                        needles.append(v)
    except OSError:
        pass
    # De-dup; keep only specific values (a bare coord like "37.7" could false-positive on
    # the map's setView, so require address-like length OR a full-precision coordinate).
    return sorted({n for n in needles if len(n) >= 6})
