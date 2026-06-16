"""Integration tests for the Flask commute API (scripts/server.py).

These are REGRESSION GUARDS, not unit tests: they exercise the real in-process engine —
the JVM-FREE RAPTOR stack by default, pinned by conftest.py's setdefaults to the
production config (USE_RAPTOR=1, USE_WALK_GRAPH=1, RAPTOR_SEMANTIC=arriveby, RAPTOR_MC=1)
— booted once via the session-scoped `server`/`client` fixtures, and assert the
cross-cutting invariants that hold the app together:

  * /compute and /compute_exact shapes + the best <= real ordering, and exact determinism
    (twice-identical + a golden snapshot stamped with the engine identity).
  * the breakdown == map-color invariant: an /itinerary's total equals that cell's
    /compute_exact realistic minutes, and its legs (min + wait) sum to that total.
  * /variance (service-noise MC overlay): realistic >= perfect floor, frag/stuck bounds,
    alt-lines exclude the dominant line, caching + the ?speed= walk-scalar knob.
  * /attribution, /geocode, /autocomplete shapes + bounds + caching.
  * the operational guards: a held _HEAVY_LOCK turns /compute_exact into a 503{busy}, the
    rate limiter is wired, and the per-workplace breakdown caches are reset on a new dest.
  * the served page has no leftover template tokens and (privacy) no user address.

Run with:  .venv/bin/python -m pytest tests/test_api.py -q

Under the default RAPTOR boot the whole suite is fast (~1s boot, ~ms grid computes); the
`slow` marker survives for the heaviest full-grid passes, which take ~30s+ only on the
legacy USE_RAPTOR=0 R5 path. Never run beside a server that is still BOOTING (concurrent
numba JIT corrupts the shared .nbc cache — see CLAUDE.md); a long-running server on :8000
is fine (we never bind a port).
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
    """Regression snapshot. If this fails after a GTFS refresh (pick_service_date shifted)
    or an engine-default change, regenerate:  .venv/bin/python tests/make_golden.py"""
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(
            "golden missing — generate with `.venv/bin/python tests/make_golden.py`"
        )
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    if golden.get("service_date") != str(server._SVC_DATE):
        pytest.skip(
            f"golden service_date {golden.get('service_date')} != current "
            f"{server._SVC_DATE}; regenerate via tests/make_golden.py"
        )
    # Engine-identity guard (mirrors the service_date one): the snapshot is only comparable
    # when the booted engine matches the one it was baked under — e.g. USE_RAPTOR=0 (legacy
    # R5) legitimately differs from the RAPTOR golden by MAE ~0.75, and a GTFS-fingerprint
    # shift means a repull happened even if pick_service_date landed on the same date.
    from make_golden import engine_identity
    booted = engine_identity(server)
    if golden.get("engine") != booted:
        pytest.skip(
            f"golden engine {golden.get('engine')} != booted {booted}; "
            "regenerate via tests/make_golden.py (or unset the engine env overrides)"
        )
    current = _get_exact_cells(client)
    g_cells = golden["cells"]
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
# /variance — service-noise Monte-Carlo overlay (realistic + fragility + alt-lines)
# --------------------------------------------------------------------------------------
def _skip_unless_mc(server):
    if not (server.USE_RAPTOR and server.RAPTOR_SEMANTIC == "arriveby" and server.RAPTOR_MC):
        pytest.skip("/variance is only served under the RAPTOR arrive-by MC boot")


@pytest.mark.slow
def test_variance_realistic_floor_bounds_alt_and_cache(client, server):
    """The MC overlay's served contract: every realistic >= the perfect map's best minutes
    (the asserted perfect <= committed invariant, floored server-side), frag >= 0,
    0 <= stuck <= 1, alt-lines are capped at 4 and never include the cell's own dominant
    line, and a second identical GET returns the identical payload (LRU cache + the
    deterministic per-workplace sha256 seed)."""
    _skip_unless_mc(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    # Perfect map first (exactly what the frontend paints before layering /variance).
    r = client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert r.status_code == 200
    perfect = r.get_json()["cells"]
    # Dominant line per cell, to cross-check the alt-lines exclusion.
    ra = client.get(f"/attribution?dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert ra.status_code == 200
    dom = ra.get_json()

    resp = client.get(f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body["dest"] == [FERRY_LAT, FERRY_LON]
    realistic, variance = body["realistic"], body["variance"]
    assert realistic, "MC overlay returned no realistic cells"
    assert set(realistic) == set(variance), "realistic/variance cell sets diverge"

    seen_alt = frag_pos = 0
    for cid, rmin in realistic.items():
        pair = perfect.get(cid)
        assert pair is not None and pair[0] is not None, (
            f"{cid}: realistic cell missing from the perfect map (reachability must follow it)"
        )
        assert isinstance(rmin, int)
        assert rmin >= pair[0], f"{cid}: realistic {rmin} < perfect {pair[0]} (floor lost)"
        v = variance[cid]
        assert v["frag"] >= 0, f"{cid}: negative fragility {v['frag']}"
        frag_pos += v["frag"] > 0
        assert v["std"] >= 0
        assert 0.0 <= v["stuck"] <= 1.0, f"{cid}: stuck {v['stuck']} out of [0,1]"
        alt = v.get("alt")
        if alt:
            assert len(alt) <= 4, f"{cid}: alt has {len(alt)} entries (> 4 cap)"
            assert dom.get(cid) not in alt, (
                f"{cid}: alt {alt} includes the cell's own dominant line {dom.get(cid)!r}"
            )
            assert all(isinstance(n, int) and n >= 1 for n in alt.values()), alt
            seen_alt += 1
    assert seen_alt > 0, "no cell carried alt-lines — the traced-perturbation path is dead"
    assert frag_pos > 0, "MC produced zero fragility everywhere — delays are not applied"

    # Cache + determinism: bit-identical realistic/variance on a repeat GET (ms may differ).
    resp2 = client.get(f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert resp2.status_code == 200
    body2 = resp2.get_json()
    assert body2["realistic"] == realistic
    assert body2["variance"] == variance


@pytest.mark.slow
def test_variance_speed_slow_shifts_payload(client, server):
    """?speed=slow is a separate cache key with a real effect: walking is scaled slower, so
    reachability can only SHRINK (slow-reachable cells are a subset of medium-reachable) and
    the average realistic commute over common cells goes UP (walk legs dominate the shift;
    per-cell monotonicity is not asserted because the MC seed is keyed by speed)."""
    _skip_unless_mc(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    med = client.get(f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert med.status_code == 200, med.get_data(as_text=True)
    slow = client.get(f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}&speed=slow")
    assert slow.status_code == 200, slow.get_data(as_text=True)
    m_real = med.get_json()["realistic"]
    s_real = slow.get_json()["realistic"]
    assert s_real, "slow-speed MC overlay returned no cells"
    assert s_real != m_real, "?speed=slow returned the medium-speed payload (cache key collision)"
    # Slower walking is monotonic on the deterministic perfect map -> no NEW reachable cells.
    assert set(s_real) <= set(m_real), (
        f"{len(set(s_real) - set(m_real))} cells became reachable only at slow walk speed"
    )
    common = set(s_real) & set(m_real)
    assert len(common) > 100, f"only {len(common)} common cells between speeds"
    mean_m = sum(m_real[c] for c in common) / len(common)
    mean_s = sum(s_real[c] for c in common) / len(common)
    assert mean_s > mean_m, (
        f"slow-walk mean realistic {mean_s:.1f} not above medium {mean_m:.1f}"
    )


# --------------------------------------------------------------------------------------
# /itinerary geom — the drawn hover route (RAPTOR arrive-by only)
# --------------------------------------------------------------------------------------
def _haversine_m(lat1, lon1, lat2, lon2):
    import math
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _skip_unless_geom(server):
    if not (server.USE_RAPTOR and server.RAPTOR_SEMANTIC == "arriveby"):
        pytest.skip("route geometry is served on the RAPTOR arrive-by path only")


def _geom_sample_cells(server, dlat, dlon, n_generic=4):
    """Pick test cells from the traced tree itself: >=1 multi-transfer (2+ significant
    rides), >=1 walk-only, plus a spread of generic reachable cells."""
    entry = server._raptor_tree(dlat, dlon)
    tree = entry["tree"]
    multi = walk_only = None
    generic = []
    for cid, (best, real) in entry["cells"].items():
        if real is None:
            continue
        ci = server._RAPTOR.cell_index[cid]
        tr = tree._trace(ci)
        if tr is None:
            continue
        legs_raw, _lh = tr
        rides = [l for l in legs_raw if l[0] == "ride" and (l[3] - l[2]) >= 120]
        if walk_only is None and legs_raw[0][0] == "walk":
            walk_only = cid
        if multi is None and len(rides) >= 2:
            multi = cid
        if len(generic) < n_generic and real >= 8:
            generic.append(cid)
    assert multi is not None, "no multi-transfer cell found in the traced tree"
    assert walk_only is not None, "no walk-only cell found in the traced tree"
    cells = [multi, walk_only] + [c for c in generic if c not in (multi, walk_only)]
    return entry, cells


def test_itinerary_geom_matches_breakdown_and_endpoints(client, server):
    """The drawn route's contract: geom legs mirror the breakdown legs 1:1 (mode + name +
    minutes, same order), the path starts at the hovered cell and ends at the workplace
    (within ~250m), every ride leg's first/last points are the board/alight stop coords of
    the SAME traced journey, and the minutes are consistent with the cell's map value."""
    _skip_unless_geom(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    r = client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    assert r.status_code == 200
    map_cells = r.get_json()["cells"]
    entry, sample = _geom_sample_cells(server, FERRY_LAT, FERRY_LON)
    tree = entry["tree"]
    data = server._RAPTOR.data

    saw_transit = saw_walk_only = 0
    for cid in sample:
        resp = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
        assert resp.status_code == 200, resp.get_data(as_text=True)
        body = resp.get_json()
        assert "error" not in body, f"{cid}: {body}"
        assert "geom" in body, f"{cid}: /itinerary has no geom field"
        geom, legs = body["geom"], body["legs"]

        # 1:1 with the displayed breakdown: same count, mode, line name, minutes, order.
        assert len(geom) == len(legs), f"{cid}: geom {len(geom)} legs vs breakdown {len(legs)}"
        for g, l in zip(geom, legs):
            assert g["mode"] == l["mode"], f"{cid}: {g['mode']} != {l['mode']}"
            assert (g.get("name") or None) == (l.get("line") or None), f"{cid}: {g} vs {l}"
            assert g["min"] == l["min"], f"{cid}: geom min {g['min']} != leg min {l['min']}"
            if g["mode"] == "transit":
                assert g.get("tmode") in ("bart", "metro", "bus", "cable"), g
                assert g.get("feed"), g
            assert isinstance(g["pts"], list)
            for p in g["pts"]:
                assert SF_LAT_MIN - 0.2 <= p[0] <= SF_LAT_MAX + 0.2, f"{cid}: bad lat {p}"
                assert SF_LON_MIN - 0.2 <= p[1] <= SF_LON_MAX + 0.2, f"{cid}: bad lon {p}"

        # Endpoints: starts at the hovered cell's center, ends at the workplace (~250m).
        olat, olon = server.ORIGIN_LL[cid]
        first_pts, last_pts = geom[0]["pts"], geom[-1]["pts"]
        assert first_pts and last_pts, f"{cid}: empty endpoint geometry"
        assert _haversine_m(first_pts[0][0], first_pts[0][1], olat, olon) <= 250, (
            f"{cid}: route starts {first_pts[0]} far from cell center ({olat},{olon})"
        )
        assert _haversine_m(last_pts[-1][0], last_pts[-1][1], FERRY_LAT, FERRY_LON) <= 250, (
            f"{cid}: route ends {last_pts[-1]} far from the workplace"
        )

        # Ride legs' endpoints == the traced journey's board/alight stop coords (the
        # hover==map invariant extended to the drawn geometry: same trace, same stops).
        ci = server._RAPTOR.cell_index[cid]
        legs_raw, _lh = tree._trace(ci)
        rides = [l for l in legs_raw
                 if l[0] == "ride" and (l[3] - l[2]) >= 120]   # significant (un-folded) rides
        transit_geoms = [g for g in geom if g["mode"] == "transit"]
        assert len(transit_geoms) == len(rides), (
            f"{cid}: {len(transit_geoms)} transit geom legs vs {len(rides)} traced rides"
        )
        for g, ride in zip(transit_geoms, rides):
            pi, bpos, apos = ride[1], ride[4], ride[5]
            sbase = int(data["pat_stop_off"][pi])
            bstop = int(data["pat_stops"][sbase + bpos])
            astop = int(data["pat_stops"][sbase + apos])
            for stop, pt in ((bstop, g["pts"][0]), (astop, g["pts"][-1])):
                sla, slo = float(data["stop_lat"][stop]), float(data["stop_lon"][stop])
                if sla != sla or slo != slo:                   # NaN coords: skipped in pts
                    continue
                assert abs(pt[0] - sla) < 1e-4 and abs(pt[1] - slo) < 1e-4, (
                    f"{cid}: ride endpoint {pt} != stop {stop} ({sla},{slo})"
                )
            saw_transit += 1
        if not rides and len(legs) == 1 and legs[0]["mode"] == "walk":
            saw_walk_only += 1

        # Minutes consistent with the map: total == the cell's served map value, and the
        # geom legs' minutes (+waits) sum to it like the breakdown's do.
        assert body["total"] == map_cells[cid][1], (
            f"{cid}: itinerary total {body['total']} != map value {map_cells[cid][1]}"
        )
        gsum = sum(g["min"] for g in geom) + sum(g.get("wait", 0) for g in geom)
        assert gsum == body["total"], f"{cid}: geom minutes {gsum} != total {body['total']}"

    assert saw_transit > 0, "sample never exercised a transit ride's geometry"
    assert saw_walk_only > 0, "sample never exercised a walk-only journey's geometry"


def test_itinerary_geom_walk_paths_are_real_and_cached(client, server):
    """Walk legs are real street paths when the walk graph is loaded (many nodes, not a
    straight 2-point approx), and a repeat GET serves the identical cached payload."""
    _skip_unless_geom(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    entry, sample = _geom_sample_cells(server, FERRY_LAT, FERRY_LON)
    cid = sample[0]                                   # the multi-transfer cell
    r1 = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert r1.status_code == 200
    b1 = r1.get_json()
    walk_geoms = [g for g in b1["geom"] if g["mode"] == "walk" and g["min"] >= 2]
    if server.USE_WALK_GRAPH:
        assert walk_geoms, "multi-transfer journey has no walk legs >= 2 min"
        for g in walk_geoms:
            assert not g.get("approx"), f"walk leg marked approx with the graph loaded: {g}"
            assert len(g["pts"]) >= 4, (
                f"walk leg of {g['min']} min has only {len(g['pts'])} pts — not a street path"
            )
    else:
        for g in walk_geoms:
            assert g.get("approx") is True, "graphless walk leg must be marked approx"
    # cached: the per-cell geometry is assembled once per workplace tree (bit-identical).
    r2 = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert r2.get_json() == b1


# --------------------------------------------------------------------------------------
# /itinerary alts — the drawn ALTERNATIVE routes (under-delays re-routes; RAPTOR MC only)
# --------------------------------------------------------------------------------------
def _cell_with_alt_chips(server, dlat, dlon):
    """A reachable cell id whose MC overlay surfaced alt chips (so /itinerary can draw alts) +
    its served alt-line set, peeking the built MC entry directly. None if no cell qualifies."""
    mc = server._mc_peek(dlat, dlon, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED)
    if not mc:
        return None, None
    for ci, lines in mc.get("alt_chips", {}).items():
        if lines:
            return server._RAPTOR.cell_ids[ci], list(lines)
    return None, None


@pytest.mark.slow
def test_itinerary_alts_lines_subset_chips_and_geometry_contract(client, server):
    """After /variance has built the MC, /itinerary for a cell with alt chips returns ``alts``
    whose lines are a SUBSET of that cell's chip lines, and each alt's legs satisfy the SAME
    geometry contract as the primary route: the first walk leg starts at the cell center and the
    last ends at the workplace (within ~250m), and every ride leg's first/last points are the
    board/alight stop coords. Determinism: a second identical request returns identical alts."""
    _skip_unless_mc(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    # Paint the perfect map, then build the MC (the only path that captures the alt bundle).
    client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    rv = client.get(f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert rv.status_code == 200, rv.get_data(as_text=True)
    cid, chips = _cell_with_alt_chips(server, FERRY_LAT, FERRY_LON)
    assert cid is not None, "MC surfaced no alt chips on any cell — alt-route path can't be tested"

    resp = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert "alts" in body, "/itinerary missing the alts field"
    alts = body["alts"]
    assert isinstance(alts, list) and alts, f"{cid}: expected drawn alts, got {alts!r}"
    assert len(alts) <= 4, f"{cid}: {len(alts)} alts (> 4 cap)"

    data = server._RAPTOR.data
    olat, olon = server.ORIGIN_LL[cid]
    chip_set = set(chips)
    seen_ride = 0
    for alt in alts:
        # line ⊆ the cell's chip lines (the same exclusion + cap /variance applied).
        assert alt["line"] in chip_set, f"{cid}: alt line {alt['line']!r} not in chips {chips}"
        assert isinstance(alt["min"], int) and alt["min"] > 0, alt
        legs = alt["legs"]
        assert legs, f"{cid}: alt {alt['line']} has no legs"
        # same geometry contract as the primary route ----------------------------------------
        first_pts, last_pts = legs[0]["pts"], legs[-1]["pts"]
        assert first_pts and last_pts, f"{cid}: empty endpoint geometry on alt {alt['line']}"
        assert _haversine_m(first_pts[0][0], first_pts[0][1], olat, olon) <= 250, (
            f"{cid}: alt {alt['line']} starts {first_pts[0]} far from cell ({olat},{olon})")
        assert _haversine_m(last_pts[-1][0], last_pts[-1][1], FERRY_LAT, FERRY_LON) <= 250, (
            f"{cid}: alt {alt['line']} ends {last_pts[-1]} far from the workplace")
        # the alt's dominant transit line IS its label (longest ride carries the line name).
        transit = [l for l in legs if l["mode"] == "transit"]
        assert transit, f"{cid}: alt {alt['line']} has no transit leg"
        assert alt["line"] in {l["name"] for l in transit}, (
            f"{cid}: alt label {alt['line']} not among its transit legs {[l['name'] for l in transit]}")
        for l in transit:
            assert l.get("tmode") in ("bart", "metro", "bus", "cable"), l
            assert l.get("feed"), l
            # ride pts must start/end at REAL stop coords (within ~120m; snapped polyline ends).
            for end in (l["pts"][0], l["pts"][-1]):
                assert SF_LAT_MIN - 0.2 <= end[0] <= SF_LAT_MAX + 0.2, l
                assert SF_LON_MIN - 0.2 <= end[1] <= SF_LON_MAX + 0.2, l
            seen_ride += 1
        # geom minutes (+waits) sum to the alt total, like the primary route.
        gsum = sum(l["min"] for l in legs) + sum(l.get("wait", 0) for l in legs)
        assert gsum == alt["min"], f"{cid}: alt {alt['line']} legs {gsum} != total {alt['min']}"
    assert seen_ride > 0, "no alt exercised a transit ride's geometry"

    # Cross-check ride endpoints against the access stop the dominance window picked for each alt
    # (the alt geometry is traced from the SAME unperturbed cached primary tree via the bundle's
    # per-cell alt_stop map — JourneyTree.itinerary_via_stop — not a recompute or a perturbed draw).
    mc = server._mc_peek(FERRY_LAT, FERRY_LON, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED)
    bundle = mc["alt_bundle"]
    assert bundle["draws"] == [], "alt_bundle should carry no perturbed draws (tree-window source)"
    ci = server._RAPTOR.cell_index[cid]
    cell_alt_stop = bundle["alt_stop"][ci]
    walk_scalar = server.config.WALK_KMH / server.WALK_SPEEDS.get(server.DEFAULT_SPEED, server.config.WALK_KMH)
    tree = server._raptor_tree(FERRY_LAT, FERRY_LON, server.DEFAULT_MAX_RIDES,
                               server.DEFAULT_SPEED, walk_scalar)["tree"]
    for alt in alts:
        s_star = cell_alt_stop[alt["line"]]              # the window's access stop for this line
        legs_raw, _lh = tree._trace_from(int(s_star), 0, 0)   # aw/home irrelevant for stop coords
        rides = [l for l in legs_raw if l[0] == "ride" and (l[3] - l[2]) >= 120]
        transit_geoms = [l for l in alt["legs"] if l["mode"] == "transit"]
        assert len(transit_geoms) == len(rides), (
            f"{cid}/{alt['line']}: {len(transit_geoms)} geom rides vs {len(rides)} traced rides")
        for g, ride in zip(transit_geoms, rides):
            pi, bpos, apos = ride[1], ride[4], ride[5]
            sbase = int(data["pat_stop_off"][pi])
            for pos, pt in ((bpos, g["pts"][0]), (apos, g["pts"][-1])):
                stop = int(data["pat_stops"][sbase + pos])
                sla, slo = float(data["stop_lat"][stop]), float(data["stop_lon"][stop])
                if sla != sla or slo != slo:                  # NaN coords skipped in pts
                    continue
                assert abs(pt[0] - sla) < 1e-4 and abs(pt[1] - slo) < 1e-4, (
                    f"{cid}/{alt['line']}: ride endpoint {pt} != stop {stop} ({sla},{slo})")

    # Determinism: a second identical request reproduces the alts byte-for-byte.
    resp2 = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert resp2.status_code == 200
    assert resp2.get_json()["alts"] == alts, "alts differ across two identical requests"


def test_itinerary_alts_empty_before_variance_built(client, server):
    """A fresh workplace whose MC has NOT been requested yet returns ``alts: []`` — /itinerary
    must NEVER trigger the ~1s MC build on the hover path (the frontend re-hovers after
    /variance lands). We evict any stale MC entry for this coord first to guarantee the miss."""
    _skip_unless_mc(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    lat, lon = TWIN_PEAKS_LAT, TWIN_PEAKS_LON
    key = server._coarse_key(lat, lon, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED)
    # Guarantee a clean MC state for this workplace (other tests / ordering may have built it).
    server._RAPTOR_MC_CACHE.pop(key)
    assert server._mc_peek(lat, lon, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED) is None
    # Paint the map (builds the traced tree, NOT the MC) and hover a reachable cell.
    rc = client.get(f"/compute?lat={lat}&lon={lon}")
    assert rc.status_code == 200
    cells = rc.get_json()["cells"]
    cid = next(c for c, pair in cells.items() if pair[1] is not None)
    resp = client.get(f"/itinerary?id={cid}&dlat={lat}&dlon={lon}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body.get("alts") == [], f"alts must be [] before /variance, got {body.get('alts')!r}"
    # The hover must NOT have built the MC (no cost paid on the hover path).
    assert server._mc_peek(lat, lon, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED) is None, (
        "/itinerary triggered the MC build on the hover path (must stay lazy)")


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


def test_autocomplete_dedups_results(client):
    """Guards the server-side _dedup wiring in geo.autocomplete: whatever the upstream
    returns, no two results may share the same (label, 5-decimal lat/lon) key."""
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
def test_heavy_lock_compute_exact_contract(client, server):
    """The _HEAVY_LOCK contract on /compute_exact, which INVERTS with the engine:

    * RAPTOR (the pinned default): /compute_exact is the instant engine result (map ==
      refine, ~ms) and bypasses _HEAVY_LOCK BY DESIGN — so it must stay responsive (200,
      non-empty cells) even while a heavy job holds the lock.
    * Legacy R5 (USE_RAPTOR=0): a held lock must turn it into a non-blocking 503
      {busy:true} with a Retry-After header; after release it works again."""
    # Coordinate guaranteed not to be in the coarse cache yet (the legacy endpoint returns
    # a cache hit BEFORE trying to acquire the lock).
    lat, lon = 37.7611, -122.4350

    acquired = server._HEAVY_LOCK.acquire(blocking=False)
    assert acquired, "_HEAVY_LOCK was already held — another heavy job in this process?"
    try:
        if server.USE_RAPTOR:
            resp = client.get(f"/compute_exact?lat={lat}&lon={lon}")
            assert resp.status_code == 200, resp.get_data(as_text=True)
            body = resp.get_json()
            assert body["dest"] == [lat, lon]
            assert body["cells"], "RAPTOR /compute_exact returned no cells under a held lock"
        else:
            with server._RESULT_CACHE_LOCK:
                server._EXACT_RESULT_CACHE.pop(server._coarse_key(lat, lon), None)
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
    """/compute warms _LAST_DEST_KEY; /itinerary for a cell warms the per-workplace
    breakdown cache; repeating the SAME coords keeps it.

    The cache LAYER differs by engine: under RAPTOR arrive-by (the pinned default)
    breakdowns come from _RAPTOR_TREE_CACHE (a bounded LRU keyed by ~110m bucket + rides
    + speed; new-dest hygiene is LRU eviction, not _reset_caches) and the legacy
    _CELL_CACHE must stay EMPTY on that path. Under legacy R5 (USE_RAPTOR=0), /itinerary
    warms _CELL_CACHE and a DIFFERENT coord clears it (new _LAST_DEST_KEY)."""
    fkey = server._dest_key(FERRY_LAT, FERRY_LON)

    # Reset the limiter so a prior test (e.g. the rate-limit test) that consumed this
    # minute's /itinerary or /compute budget doesn't bleed into this one and surface a 429.
    try:
        server.limiter.reset()
    except Exception:
        pass

    if server.USE_RAPTOR and server.RAPTOR_SEMANTIC == "arriveby":
        # 1) /compute sets the current dest AND builds the traced tree -> tree cache warm.
        r = client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
        assert r.status_code == 200
        assert server._LAST_DEST_KEY == fkey
        tkey = server._coarse_key(FERRY_LAT, FERRY_LON,
                                  server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED)
        entry = server._RAPTOR_TREE_CACHE.get(tkey)
        assert entry is not None, "/compute did not warm _RAPTOR_TREE_CACHE (arrive-by)"
        # 2) /itinerary serves from that tree; its total must equal the cell's map value
        #    (hover == map) and the legacy R5 _CELL_CACHE must remain untouched.
        cells = entry["cells"]
        cid = next(c for c, (b, rl) in cells.items() if rl is not None)
        r = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert "error" not in body, body
        assert body["total"] == cells[cid][1], (
            f"breakdown total {body['total']} != map value {cells[cid][1]} for cell {cid}"
        )
        assert fkey not in server._CELL_CACHE, (
            "_CELL_CACHE was populated on the RAPTOR arrive-by path — legacy cache leak"
        )
        # 3) Same coords again: the tree entry survives (re-submit keeps caches).
        client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
        assert server._LAST_DEST_KEY == fkey
        assert server._RAPTOR_TREE_CACHE.get(tkey) is not None, (
            "tree cache should survive a same-dest /compute"
        )
        # 4) A DIFFERENT coord moves _LAST_DEST_KEY (tree entries age out via LRU, not reset).
        tpkey = server._dest_key(TWIN_PEAKS_LAT, TWIN_PEAKS_LON)
        assert tpkey != fkey
        r = client.get(f"/compute?lat={TWIN_PEAKS_LAT}&lon={TWIN_PEAKS_LON}")
        assert r.status_code == 200
        assert server._LAST_DEST_KEY == tpkey, "new dest did not update _LAST_DEST_KEY"
        # Restore Ferry as the current dest so other tests start from a known state.
        client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
        return

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
    # The page NEVER injects a default workplace (cfg.default_wp is always null) — the old
    # db727c0 .env DEFAULT_ADDRESS injection was removed so a fresh visitor sees the
    # onboarding prompt and types their own address instead. So there is NO whitelist
    # exception: the private workplace strings must be absent from the ENTIRE page.
    assert '"default_wp": null' in html or '"default_wp":null' in html, (
        "cfg.default_wp must be null in the served page (no .env workplace injection)"
    )
    #
    # NB: we deliberately do NOT scrape .dest_cache.json — its keys are incidental geocoder
    # lookups (e.g. "1 Market St"), some of which legitimately appear as UI placeholder text
    # in the template and are not the user's private workplace. The privacy invariant is
    # specifically about the configured DEFAULT_ADDRESS / DEST_LABEL / DEST_LAT/LON.
    for needle in _private_workplace_needles():
        assert needle not in html, (
            f"served page leaks private workplace value {needle!r}"
        )


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
