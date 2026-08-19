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
    """Regression snapshot for the ARRIVE-BY engine (the OPT-IN path since the 2026-06-17
    default flip). The in-process `server` fixture is pinned to arrive-by by conftest.py
    (hard pin: os.environ["RAPTOR_SEMANTIC"]="arriveby" — exported env cannot override),
    so this compares against the arrive-by golden
    (tests/golden_exact_ferry.json). The SERVED DEFAULT (depart-after) has its own,
    non-skipping golden coverage in test_compute_exact_matches_golden_departafter below.

    If this fails after a GTFS refresh (pick_service_date shifted) or an engine change,
    regenerate:  .venv/bin/python tests/make_golden.py arriveby"""
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip(
            "arrive-by golden missing — generate with "
            "`.venv/bin/python tests/make_golden.py arriveby`"
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


# Depart-after golden — the SERVED DEFAULT path (since the 2026-06-17 flip). The in-process
# `server` fixture is arrive-by-pinned, so the depart-after served map is snapshotted in a
# CHILD process booted with RAPTOR_SEMANTIC=departafter (mirrors the _DEPARTAFTER_DRIVER
# pattern). This gives the default path REAL golden coverage that does not depend on, and does
# not skip under, the in-process arrive-by pin.
GOLDEN_PATH_DEPARTAFTER = os.path.join(_HERE, "golden_exact_ferry_departafter.json")
_GOLDEN_DA_DRIVER = r'''
import json, sys, os
sys.path.insert(0, %(scripts)r)
import server
assert server.RAPTOR_SEMANTIC == "departafter", server.RAPTOR_SEMANTIC
assert server._NEED_R5 is False, "depart-after golden boot flagged _NEED_R5 (would load the JVM)"
jvm = sorted(m for m in sys.modules
             if m == "r5py" or m.startswith("r5py.") or m.startswith("com.conveyal")
             or m == "jpype")
assert not jvm, "depart-after golden boot is NOT JVM-free: %%s" %% jvm
c = server.app.test_client()
with server._RESULT_CACHE_LOCK:
    server._EXACT_RESULT_CACHE.clear()
r = c.get("/compute_exact?lat=%(flat)s&lon=%(flon)s")
assert r.status_code == 200, r.get_data(as_text=True)
print(json.dumps({
    "service_date": str(server._SVC_DATE),
    "engine": {"use_raptor": bool(server.USE_RAPTOR),
               "raptor_semantic": str(server.RAPTOR_SEMANTIC),
               "use_walk_graph": bool(server.USE_WALK_GRAPH),
               "gtfs_fp": server._gtfs_fp(),
               "walk_reference_kmh": float(server.config.WALK_KMH),
               "walk_speeds": {key: float(value)
                               for key, value in sorted(server.WALK_SPEEDS.items())},
               "default_walk_speed": str(server.DEFAULT_SPEED)},
    "cells": r.get_json()["cells"],
}))
'''


def test_compute_exact_matches_golden_departafter():
    """Regression snapshot for the DEPART-AFTER engine — the SERVED DEFAULT since 2026-06-17.
    Boots a child server pinned to RAPTOR_SEMANTIC=departafter and asserts /compute_exact ==
    the depart-after golden (tests/golden_exact_ferry_departafter.json). This is the default
    path's golden guard: it does NOT skip just because the in-process suite pins arrive-by
    (the served default and the in-process fixture are deliberately different engines).
    It also re-asserts the depart-after boot is JVM-free.

    If this fails after a GTFS refresh or an engine change, regenerate:
        .venv/bin/python tests/make_golden.py departafter"""
    import subprocess
    import sys as _sys
    if not os.path.exists(GOLDEN_PATH_DEPARTAFTER):
        pytest.skip(
            "depart-after golden missing — generate with "
            "`.venv/bin/python tests/make_golden.py departafter`"
        )
    with open(GOLDEN_PATH_DEPARTAFTER) as f:
        golden = json.load(f)

    scripts = os.path.join(_HERE, "..", "scripts")
    driver = _GOLDEN_DA_DRIVER % {"scripts": scripts, "flat": FERRY_LAT, "flon": FERRY_LON}
    env = dict(os.environ)
    env.update(USE_RAPTOR="1", USE_WALK_GRAPH="1", RAPTOR_SEMANTIC="departafter", RAPTOR_MC="1")
    # Keep the child off the repo-default numba cache (the suite's .nbc gotcha) — share the
    # depart-after child cache used by test_itinerary_equals_map_departafter when present.
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(_HERE, ".nbcache_departafter"))
    proc = subprocess.run([_sys.executable, "-c", driver], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"depart-after golden boot/driver failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    res = json.loads(proc.stdout.strip().splitlines()[-1])

    # Skip-on-mismatch (mirrors the arrive-by golden) so a GTFS repull or engine change is a
    # regenerate signal, not a red test — but the engine HERE is the served default, so on the
    # default config this runs (does not skip).
    if golden.get("service_date") != res["service_date"]:
        pytest.skip(
            f"depart-after golden service_date {golden.get('service_date')} != current "
            f"{res['service_date']}; regenerate via tests/make_golden.py departafter")
    if golden.get("engine") != res["engine"]:
        pytest.skip(
            f"depart-after golden engine {golden.get('engine')} != booted {res['engine']}; "
            "regenerate via tests/make_golden.py departafter")
    assert res["cells"] == golden["cells"], (
        "depart-after compute_exact output drifted from the golden snapshot. If GTFS feeds "
        "were refreshed this is expected — regenerate with `tests/make_golden.py departafter`."
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
# BUG 3 regression: "uncolored" cells must still be inspectable / give clear feedback.
# --------------------------------------------------------------------------------------
# A user reported that hovering a cell that isn't lit up (outside the Max-commute time band, or
# genuinely unreachable) shows nothing. The Max-commute slider (`thr`, default 40) only DIMS the
# heatmap — it must NOT make a reachable cell un-inspectable — and a genuinely unreachable cell must
# return a clean "no route" error the frontend can render as "No transit route within ~75 min."
# (not a silent failure). This test pins the SERVER contract those two frontend behaviors rely on:
#   (a) a reachable cell with a HIGH commute (above any reasonable thr) STILL returns a full
#       /itinerary breakdown — proof the cell is inspectable even when dimmed; and
#   (b) an unreachable cell ([None,None] in /compute, no journey within the ~75 min cap) returns
#       {"error": ...} (no total/legs) — proof the no-route message path has real data behind it.
def test_itinerary_works_for_uncolored_cells(client):
    cells = _get_exact_cells(client)
    # (a) Find a REACHABLE cell whose realistic minutes are well above the default thr (40), i.e. a
    # cell that the heatmap leaves dimmed but that has a genuine commute. Pick the SLOWEST such cell
    # so it's unambiguously above any reasonable slider value.
    above_thr = sorted(
        ((cid, real) for cid, (best, real) in cells.items()
         if real is not None and real > 50),
        key=lambda t: t[1],
    )
    assert above_thr, "no reachable cell above 50 min — can't exercise the dimmed-but-reachable case"
    cid_hi, real_hi = above_thr[-1]                       # the slowest reachable cell
    resp = client.get(f"/itinerary?id={cid_hi}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    itin = resp.get_json()
    assert "error" not in itin, (
        f"dimmed-but-reachable cell {cid_hi} ({real_hi} min) must be inspectable, got {itin}"
    )
    assert itin.get("total") == real_hi, f"{cid_hi}: itinerary total {itin.get('total')} != {real_hi}"
    assert itin.get("legs"), f"{cid_hi}: reachable cell with empty legs: {itin}"
    assert itin.get("geom"), f"{cid_hi}: reachable cell with no geom to draw: {itin}"

    # (b) Find a genuinely UNREACHABLE cell ([None, None] in /compute) and assert /itinerary returns
    # a clean error (no total/legs), so the frontend can show the "no route" message rather than fail
    # silently. Some workplaces reach every cell; skip cleanly if none are unreachable here.
    unreachable = [cid for cid, (best, real) in cells.items() if best is None and real is None]
    if unreachable:
        cid_un = unreachable[0]
        resp = client.get(f"/itinerary?id={cid_un}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
        assert resp.status_code == 200, resp.get_data(as_text=True)
        itin = resp.get_json()
        assert "error" in itin, f"unreachable cell {cid_un} must return an error, got {itin}"
        assert itin.get("total") is None, f"unreachable cell {cid_un} carried a total: {itin}"
        assert not itin.get("legs"), f"unreachable cell {cid_un} carried legs: {itin}"


# --------------------------------------------------------------------------------------
# /itinerary == map under the DEPART-AFTER semantic (Stage 2 of the depart-after migration)
# --------------------------------------------------------------------------------------
# The session `server`/`client` fixtures boot ONE engine (arrive-by, the production default), so
# the depart-after path is exercised in a child process that imports server.py with
# RAPTOR_SEMANTIC=departafter and runs the same hover==map + geom assertions the arrive-by test
# above does. A subprocess (not a second in-process boot) keeps the two semantics from sharing a
# numba JIT / module-global state in the test process, and proves the depart-after boot is JVM-free.
_DEPARTAFTER_DRIVER = r'''
import json, sys, os
sys.path.insert(0, %(scripts)r)
import server
ALT_CAP = getattr(server.sr, "RAPTOR_ALT_CHIP_CAP", 6)

# JVM-free assertion: nothing from r5py / the Conveyal JVM may have been imported.
jvm = sorted(m for m in sys.modules
             if m == "r5py" or m.startswith("r5py.") or m.startswith("com.conveyal")
             or m == "jpype")
assert server.RAPTOR_SEMANTIC == "departafter", server.RAPTOR_SEMANTIC
assert server._NEED_R5 is False, "depart-after boot still flagged _NEED_R5 (would load the JVM)"
assert not jvm, ("depart-after boot is NOT JVM-free: %%s" %% jvm)

c = server.app.test_client()
FLAT, FLON = %(flat)r, %(flon)r

health = c.get("/healthz").get_json()

def route_sig(legs):
    return tuple((g.get("mode"), g.get("name") or g.get("line") or "",
                  int(g.get("min") or 0), int(g.get("wait") or 0))
                 for g in (legs or []))

def alt_total(a):
    for jk in ("typical", "best"):
        v = (a.get(jk) or {}).get("total")
        if isinstance(v, int):
            return v
    v = a.get("total")
    return v if isinstance(v, int) else None

def has_route_structure(route):
    family = route.get("family") or {}
    branch = route.get("branch") or {}
    return bool(family.get("key") and branch.get("key")
                and branch.get("kind") in ("walk", "transit"))

def family_branch_kinds(routes):
    out = {}
    for route in routes:
        family = route.get("family") or {}
        branch = route.get("branch") or {}
        family_key, kind = family.get("key"), branch.get("kind")
        if family_key and kind in ("walk", "transit"):
            out.setdefault(family_key, set()).add(kind)
    return {key: sorted(kinds) for key, kinds in out.items()}

r = c.get("/compute?lat=%%s&lon=%%s" %% (FLAT, FLON))
assert r.status_code == 200, r.get_data(as_text=True)
cells = r.get_json()["cells"]

# /compute returns [scheduled, scheduled] per reachable cell under planned depart-after.
shape_bad = [(cid, v) for cid, v in cells.items()
             if v[1] is not None and not (isinstance(v[0], int) and isinstance(v[1], int)
                                          and v[0] <= v[1])]
same_bad = [(cid, v) for cid, v in cells.items()
            if v[1] is not None and v[0] != v[1]]

# Sample reachable cells across the time distribution. /itinerary still returns both `.best` and
# `.typical` for compatibility, but planned depart-after mirrors the same scheduled journey into
# both slots. Each journey's legs reconcile to its own total and its geom mirrors its legs 1:1.
reach = [(cid, v[1]) for cid, v in cells.items() if v[1] is not None and v[1] >= 8]
reach.sort(key=lambda t: t[1])
step = max(1, len(reach) // 30)
sample = reach[::step][:30]

p50_match = p50_total = p5_match = p5_total = best_present = 0
legs50_bad = legs5_bad = first_wait_bad = transit = geom_checked = geom_bad = 0
mismatches = []
for cid, p50 in sample:
    it = c.get("/itinerary?id=%%s&dlat=%%s&dlon=%%s" %% (cid, FLAT, FLON)).get_json()
    if "error" in it:
        mismatches.append((cid, "error")); continue
    p5_exp, p50_exp = cells[cid][0], cells[cid][1]
    # Scheduled root/.typical total == cells[c][1].
    p50_total += 1
    if it["total"] == p50_exp and it.get("typical", {}).get("total") == p50_exp:
        p50_match += 1
    else:
        mismatches.append((cid, "p50", it["total"], it.get("typical", {}).get("total"), p50_exp))
    # Compatibility .best total == cells[c][0] (same scheduled value in planned mode).
    if "best" in it:
        best_present += 1; p5_total += 1
        if it["best"]["total"] == p5_exp:
            p5_match += 1
        else:
            mismatches.append((cid, "p5", it["best"]["total"], p5_exp))
    # legs (min + wait) reconcile to EACH journey's total; geom mirrors legs 1:1.
    for jk, exp_total, badctr in (("typical", p50_exp, "t"), ("best", p5_exp, "b")):
        j = it.get(jk)
        if not j:
            continue
        legs = j["legs"]
        ls = sum(l["min"] for l in legs) + sum(l.get("wait", 0) for l in legs)
        if ls != j["total"]:
            if jk == "typical": legs50_bad += 1
            else: legs5_bad += 1
        for l in legs:
            if l["mode"] == "transit" and l["line"]:
                transit += 1
                if l.get("wait", 0) != 0:
                    first_wait_bad += 1
                break
        geom = j.get("geom")
        if geom is not None:
            geom_checked += 1
            if len(geom) != len(legs):
                geom_bad += 1
            else:
                for g, l in zip(geom, legs):
                    if g["mode"] != l["mode"] or g["min"] != l["min"] \
                       or (g.get("name") or None) != (l.get("line") or None):
                        geom_bad += 1; break

attr = c.get("/attribution?dlat=%%s&dlon=%%s" %% (FLAT, FLON)).get_json()
attr_bad = sum(1 for v in attr.values() if not (isinstance(v, str) and v.strip()))

# Stage 3 target model: /variance under depart-after carries the service-noise overlay
# {frag, stuck, alt} ONLY — NO "realistic" headline (the typical headline is the bare p50 the map
# already paints, cells[c][1]). frag >= 0 per reachable cell; alts respect the configured cap
# and never include the cell's own line.
var = c.get("/variance?dlat=%%s&dlon=%%s" %% (FLAT, FLON))
vb = var.get_json()
has_realistic = "realistic" in vb
variance = vb.get("variance", {})
v_frag = v_altcap = v_altdom = alt_cells = frag_pos = 0
for cid, vv in variance.items():
    if vv.get("frag", 0) < 0: v_frag += 1
    frag_pos += vv.get("frag", 0) > 0
    a = vv.get("alt")
    if a:
        alt_cells += 1
        if len(a) > ALT_CAP: v_altcap += 1
        if attr.get(cid) in a: v_altdom += 1
# determinism: a second identical GET is byte-identical (LRU cache + per-workplace seed)
vb2 = c.get("/variance?dlat=%%s&dlon=%%s" %% (FLAT, FLON)).get_json()
var_cache_ok = (vb2.get("variance") == variance and ("realistic" in vb2) == has_realistic)

# /itinerary?pin=1 per-route p5/p50/frag: pick a cell with alt chips. Each route (primary + alts)
# carries its OWN best-case (p5) + typical (p50) journey + (on pin) its OWN frag (p90-p50). The
# primary's p50 == the cell's served p50 (cells[c][1]); every route p5 <= p50; frag >= 0.
pin = {}
cell_alt = next((cid for cid, vv in variance.items() if vv.get("alt")), None)
if cell_alt is not None:
    hov = c.get("/itinerary?id=%%s&dlat=%%s&dlon=%%s" %% (cell_alt, FLAT, FLON)).get_json()
    plain_no_frag = ("frag" not in hov) and all("frag" not in a for a in hov.get("alts", []))
    body = c.get("/itinerary?id=%%s&dlat=%%s&dlon=%%s&pin=1" %% (cell_alt, FLAT, FLON)).get_json()
    palts = body.get("alts", [])
    prim_p50_ok = (body.get("total") == cells[cell_alt][1]
                   and body.get("typical", {}).get("total") == cells[cell_alt][1])
    prim_p5_le_p50 = ("best" in body and body["best"]["total"] <= body["typical"]["total"])
    prim_frag = body.get("frag")
    # RECONCILIATION (the metric-contract fix): the PRIMARY pinned strip's frag must equal the cell's
    # served /variance frag (both = committed_p90 - the bare served p50, same seed) -> the primary
    # strip's "bad day" reconciles with the headline chip. And served_p50 + frag is the displayed
    # bad-day clock (>= served_p50; the per-route fragility is measured off the SAME served p50).
    served_frag = variance.get(cell_alt, {}).get("frag")
    prim_frag_matches_variance = (prim_frag is not None and prim_frag == served_frag)
    # each alt: alt's served p50 (its typical.total) + its frag must be a non-decreasing bad-day.
    alt_reconcile_bad = sum(1 for a in palts
                            if "frag" in a and "typical" in a and a["frag"] < 0)
    alt_both = sum(1 for a in palts if "best" in a and "typical" in a)
    alt_p5_gt_p50 = sum(1 for a in palts if "best" in a and "typical" in a
                        and a["best"]["total"] > a["typical"]["total"])
    alt_with_frag = sum(1 for a in palts if "frag" in a)
    alt_frag_bad = sum(1 for a in palts if "frag" in a and a["frag"] < 0)
    pin = dict(cell=cell_alt, n_alts=len(palts), plain_no_frag=plain_no_frag,
               prim_p50_ok=prim_p50_ok, prim_p5_le_p50=prim_p5_le_p50,
               prim_frag=prim_frag, prim_frag_ok=(prim_frag is not None and prim_frag >= 0),
               served_frag=served_frag,
               prim_frag_matches_variance=prim_frag_matches_variance,
               alt_reconcile_bad=alt_reconcile_bad,
               alt_both=alt_both, alt_p5_gt_p50=alt_p5_gt_p50,
               alt_with_frag=alt_with_frag, alt_frag_bad=alt_frag_bad)

# BUG 2 (depart-after): walk-only "alternatives" are pointless. Sweep the WALKABLE alt-cells (the
# closest by best-case minutes, where a line's hop is most likely a sub-2-min ride folded into walk)
# and assert NO served alt is transit-less. depart-after alts nest legs under best/typical.
def _alt_has_transit(a):
    for jk in ("best", "typical"):
        for l in (a.get(jk) or {}).get("legs", []):
            if (l or {}).get("mode") == "transit":
                return True
    return False
alt_cids = [cid for cid, vv in variance.items() if vv.get("alt")]
alt_cids.sort(key=lambda cid: (cells[cid][0] if cells.get(cid) and cells[cid][0] is not None else 10**6))
wo_scan_cells = 0; wo_alts = 0; wo_examples = []; transit_alts = 0
for cid in alt_cids[:90]:                                # walkable-first, bounded for the rate limit
    it = c.get("/itinerary?id=%%s&dlat=%%s&dlon=%%s" %% (cid, FLAT, FLON)).get_json()
    if "error" in it:
        continue
    wo_scan_cells += 1
    for a in it.get("alts", []):
        if _alt_has_transit(a):
            transit_alts += 1
        else:
            wo_alts += 1
            if len(wo_examples) < 5:
                wo_examples.append([cid, a.get("line")])

# Concrete regression for a short-hop route that used to fold its transit leg into walking and
# reconcile the whole scheduled value into one fake walk leg. It must expose the transit leg, and
# its alternatives must retain a genuinely different boarding corridor even when the primary is
# faster. Family identity comes from server-discovered structure, never a concrete line label.
try:
    server.limiter.reset()
except Exception:
    pass
M_OLAT, M_OLON = 37.7664, -122.4267
M_DLAT, M_DLON = 37.7750, -122.4194
m_compute = c.get("/compute?lat=%%s&lon=%%s" %% (M_DLAT, M_DLON))
m_var = c.get("/variance?dlat=%%s&dlon=%%s" %% (M_DLAT, M_DLON))
m_it = c.get("/itinerary?olat=%%s&olon=%%s&dlat=%%s&dlon=%%s&pin=1" %%
             (M_OLAT, M_OLON, M_DLAT, M_DLON))
mission = {"compute_status": m_compute.status_code, "var_status": m_var.status_code,
           "it_status": m_it.status_code}
if m_it.status_code == 200:
    mj = m_it.get_json()
    m_geom = mj.get("geom", [])
    m_alts = mj.get("alts", [])
    m_psig = route_sig(m_geom)
    m_family_key = (mj.get("family") or {}).get("key")
    m_alt_family_keys = sorted({(a.get("family") or {}).get("key") for a in m_alts
                                if (a.get("family") or {}).get("key")})
    mission.update({
        "total": mj.get("total"),
        "legs": mj.get("legs", []),
        "has_primary_transit": any(g.get("mode") == "transit" and g.get("name") for g in m_geom),
        "single_walk": (len(mj.get("legs", [])) == 1 and mj.get("legs", [{}])[0].get("mode") == "walk"),
        "alt_labels": [a.get("line") for a in m_alts],
        "primary_family_key": m_family_key,
        "alt_family_keys": m_alt_family_keys,
        "structure_complete": has_route_structure(mj)
                              and bool(m_alts)
                              and all(has_route_structure(a) for a in m_alts),
        "has_distinct_boarding_family": bool(
            m_family_key and any(key != m_family_key for key in m_alt_family_keys)),
        "alt_dup_primary": sum(1 for a in m_alts
                               if route_sig((a.get("typical") or {}).get("legs") or a.get("legs"))
                               == m_psig),
    })

# Screenshot regression: this origin/destination used to serve the exact primary journey again as
# alt #1, wasting one capped route slot and showing duplicate compare rows. Its useful optionality
# is structural: one boarding-corridor family offers both walk-finish and transit-tail branches.
T_DLAT, T_DLON = 37.7714154, -122.4030885
t_compute = c.get("/compute?lat=%%s&lon=%%s" %% (T_DLAT, T_DLON))
t_var = c.get("/variance?dlat=%%s&dlon=%%s" %% (T_DLAT, T_DLON))
t_it = c.get("/itinerary?olat=%%s&olon=%%s&dlat=%%s&dlon=%%s&pin=1" %%
             (M_OLAT, M_OLON, T_DLAT, T_DLON))
townsend = {"compute_status": t_compute.status_code, "var_status": t_var.status_code,
            "it_status": t_it.status_code}
if t_it.status_code == 200:
    tj = t_it.get_json()
    t_alts = tj.get("alts", [])
    t_psig = route_sig(tj.get("geom", []))
    t_alt_totals = [x for x in (alt_total(a) for a in t_alts) if x is not None]
    t_family_branches = family_branch_kinds(t_alts)
    townsend.update({
        "total": tj.get("total"),
        "alt_labels": [a.get("line") for a in t_alts],
        "min_alt_total": min(t_alt_totals) if t_alt_totals else None,
        "structure_complete": bool(t_alts) and all(has_route_structure(a) for a in t_alts),
        "family_branch_kinds": t_family_branches,
        "has_walk_transit_siblings": any(
            set(kinds) == {"walk", "transit"} for kinds in t_family_branches.values()),
        "alt_dup_primary": sum(1 for a in t_alts
                               if route_sig((a.get("typical") or {}).get("legs") or a.get("legs"))
                               == t_psig),
    })

# Same cell at slow walk speed, without priming /variance. Planned branch expansion must still
# recover both structural siblings: stay on the boarding corridor and walk, or transfer to a
# transit tail toward the destination. This protects the walk-speed-dependent branch-loss failure
# without assigning special policy to either observed line name.
t_slow_it = c.get("/itinerary?olat=%%s&olon=%%s&dlat=%%s&dlon=%%s&speed=slow&pin=1" %%
                  (M_OLAT, M_OLON, T_DLAT, T_DLON))
townsend_slow = {"it_status": t_slow_it.status_code}
if t_slow_it.status_code == 200:
    tsj = t_slow_it.get_json()
    ts_alts = tsj.get("alts", [])
    ts_alt_totals = [x for x in (alt_total(a) for a in ts_alts) if x is not None]
    ts_family_branches = family_branch_kinds(ts_alts)
    townsend_slow.update({
        "total": tsj.get("total"),
        "alt_labels": [a.get("line") for a in ts_alts],
        "min_alt_total": min(ts_alt_totals) if ts_alt_totals else None,
        "structure_complete": bool(ts_alts) and all(has_route_structure(a) for a in ts_alts),
        "family_branch_kinds": ts_family_branches,
        "has_walk_transit_siblings": any(
            set(kinds) == {"walk", "transit"} for kinds in ts_family_branches.values()),
    })

t_fast_it = c.get("/itinerary?olat=%%s&olon=%%s&dlat=%%s&dlon=%%s&speed=fast&pin=1" %%
                  (M_OLAT, M_OLON, T_DLAT, T_DLON))
townsend_fast = {"it_status": t_fast_it.status_code}
if t_fast_it.status_code == 200:
    tfj = t_fast_it.get_json()
    tf_alts = tfj.get("alts", [])
    tf_alt_totals = [x for x in (alt_total(a) for a in tf_alts) if x is not None]
    tf_family_branches = family_branch_kinds(tf_alts)
    townsend_fast.update({
        "total": tfj.get("total"),
        "alt_labels": [a.get("line") for a in tf_alts],
        "min_alt_total": min(tf_alt_totals) if tf_alt_totals else None,
        "structure_complete": bool(tf_alts) and all(has_route_structure(a) for a in tf_alts),
        "family_branch_kinds": tf_family_branches,
        "has_walk_branch": any("walk" in kinds for kinds in tf_family_branches.values()),
    })

print(json.dumps({
    "health": {k: health.get(k) for k in ("engine", "semantic", "walk")},
    "n_cells": len(cells), "n_reach": len(reach), "n_sample": len(sample),
    "shape_bad": shape_bad[:5], "same_bad": same_bad[:5],
    "p50_match": p50_match, "p50_total": p50_total,
    "p5_match": p5_match, "p5_total": p5_total, "best_present": best_present,
    "mismatches": mismatches[:5], "legs50_bad": legs50_bad, "legs5_bad": legs5_bad,
    "first_wait_bad": first_wait_bad,
    "transit": transit, "geom_checked": geom_checked, "geom_bad": geom_bad,
    "attr_cells": len(attr), "attr_bad": attr_bad,
    "var_status": var.status_code, "var_keys": sorted(vb.keys()), "has_realistic": has_realistic,
    "n_variance": len(variance), "v_frag": v_frag,
    "v_altcap": v_altcap, "v_altdom": v_altdom, "alt_cells": alt_cells,
    "frag_pos": frag_pos, "var_cache_ok": var_cache_ok, "pin": pin,
    "wo_scan_cells": wo_scan_cells, "wo_alts": wo_alts, "wo_examples": wo_examples,
    "transit_alts": transit_alts, "mission": mission, "townsend": townsend,
    "townsend_slow": townsend_slow, "townsend_fast": townsend_fast,
}))
'''


def test_itinerary_equals_map_departafter():
    """Stage 2 regression: under RAPTOR_SEMANTIC=departafter the server serves the map, hover
    breakdown, color-by-line, and route geometry from the JVM-free DepartAfterJourneyTree — no
    R5 fallback. Over many reachable cells the /itinerary total must equal the cell's /compute p50
    (itinerary == map by construction), its legs (min + wait) must reconcile to that total, the
    geom legs must mirror the breakdown 1:1, /attribution must return a non-empty dominant line
    per reachable cell, and the boot must be JVM-free.

    Stage 3 (the metric-contract fix): best-case (p5) and typical (p50) are DIFFERENT percentiles
    of the departure window -> DIFFERENT journeys, so /itinerary returns BOTH (root/.typical = the
    p50 journey, .best = the p5 journey) and hover==map holds for BOTH (p50 journey total ==
    cells[c][1], p5 journey total == cells[c][0]). /variance carries the service-noise overlay
    {frag, stuck, alt} ONLY — NO "realistic" headline (the typical headline is the bare p50 the map
    paints, never the MC committed value). /itinerary?pin=1 gives each route (primary + each alt) its
    OWN p5 + p50 journey + its OWN frag (p90-p50); every route's p5 <= p50, frag >= 0, and the
    primary's p50 == the cell's served p50."""
    import subprocess
    import sys as _sys

    scripts = os.path.join(_HERE, "..", "scripts")
    driver = _DEPARTAFTER_DRIVER % {"scripts": scripts, "flat": FERRY_LAT, "flon": FERRY_LON}
    env = dict(os.environ)
    env.update(USE_RAPTOR="1", USE_WALK_GRAPH="1", RAPTOR_SEMANTIC="departafter", RAPTOR_MC="1")
    # Keep the child off the repo-default numba cache (the suite's gotcha): inherit an explicit
    # NUMBA_CACHE_DIR if the parent has one, else give the child a scratch dir of its own.
    env.setdefault("NUMBA_CACHE_DIR", os.path.join(_HERE, ".nbcache_departafter"))
    proc = subprocess.run([_sys.executable, "-c", driver], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"depart-after server boot/driver failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    out = proc.stdout.strip().splitlines()[-1]
    res = json.loads(out)

    assert res["health"]["semantic"] == "departafter", res["health"]
    assert res["health"]["engine"] == "raptor" and res["health"]["walk"] == "graph", res["health"]
    assert _approx_cell_count(res["n_cells"]), f"got {res['n_cells']} cells"
    assert res["n_reach"] >= 100, f"only {res['n_reach']} reachable cells (depart-after)"
    assert res["n_sample"] >= 15, f"only {res['n_sample']} sampled cells"
    assert not res["shape_bad"], f"/compute not [best<=scheduled]: {res['shape_bad']}"
    assert not res["same_bad"], f"/compute depart-after should be [scheduled, scheduled]: {res['same_bad']}"
    # hover == map for BOTH compatibility slots: root/.typical == cells[c][1] and .best == cells[c][0].
    assert res["p50_match"] == res["p50_total"] == res["n_sample"], (
        res["p50_match"], res["p50_total"], res["n_sample"], res["mismatches"])
    assert res["best_present"] == res["n_sample"], (
        f"{res['best_present']}/{res['n_sample']} sampled cells carried a .best (p5) journey")
    assert res["p5_match"] == res["p5_total"] == res["n_sample"], (
        res["p5_match"], res["p5_total"], res["n_sample"], res["mismatches"])
    assert res["legs50_bad"] == 0, f"{res['legs50_bad']} cells whose p50 legs don't sum to the total"
    assert res["legs5_bad"] == 0, f"{res['legs5_bad']} cells whose p5 legs don't sum to the total"
    assert res["first_wait_bad"] == 0, (
        f"{res['first_wait_bad']} sampled depart-after journeys showed a first transit wait")
    assert res["transit"] > 0, "no transit legs seen across the depart-after sample"
    # geom (the drawn hover route) mirrors the breakdown 1:1 on each journey that returned it.
    assert res["geom_checked"] > 0, "no geom returned on any sampled depart-after journey"
    assert res["geom_bad"] == 0, f"{res['geom_bad']} journeys whose geom != breakdown legs"
    # color-by-line: a non-empty dominant line per reachable cell.
    assert res["attr_cells"] >= 100 and res["attr_bad"] == 0, (res["attr_cells"], res["attr_bad"])
    # ---- Stage 3: the service-noise MC overlay {frag, stuck, alt} ONLY (NO realistic) -------------
    assert res["var_status"] == 200, res["var_status"]
    assert "variance" in res["var_keys"], res["var_keys"]
    # the depart-after typical headline is the bare p50 (cells[c][1]) the map paints — /variance must
    # NOT surface a "realistic" number that would override it.
    assert res["has_realistic"] is False, "depart-after /variance surfaced a 'realistic' headline"
    assert res["n_variance"] >= 100, f"depart-after /variance returned only {res['n_variance']} cells"
    # frag >= 0 (= p90 - the served p50, by the p50-floored draws) on every reachable cell.
    assert res["v_frag"] == 0, f"{res['v_frag']} cells with negative fragility"
    assert res["frag_pos"] > 0, "depart-after MC produced zero fragility everywhere (delays not applied)"
    # alt-lines: capped by the configured display cap, never the cell's own dominant line, present on
    # a meaningful set of cells.
    assert res["v_altcap"] == 0, f"{res['v_altcap']} cells exceeded the configured alt-line cap"
    assert res["v_altdom"] == 0, f"{res['v_altdom']} cells whose alt includes their own dominant line"
    assert res["alt_cells"] > 0, "no cell carried alt-lines under depart-after"
    # deterministic + cached: a second identical /variance GET is byte-identical.
    assert res["var_cache_ok"], "depart-after /variance not deterministic across two GETs"
    # /itinerary?pin=1 per-route p5/p50/frag (the compare-list metric-consistency invariant).
    pin = res["pin"]
    assert pin, "no depart-after cell carried alt chips -> pin per-route untested"
    assert pin["plain_no_frag"], "a plain hover computed per-route fragility (must be pin=1 only)"
    # the PRIMARY's p50 == the cell's served p50 (cells[c][1]); its p5 <= p50; its frag >= 0.
    assert pin["prim_p50_ok"], "primary p50 journey total != the cell's served p50 (headline drift)"
    assert pin["prim_p5_le_p50"], "primary p5 (best-case) > p50 (typical)"
    assert pin["prim_frag_ok"], f"primary fragility missing/negative ({pin['prim_frag']})"
    # RECONCILIATION (the metric-contract bug): the primary pinned strip's frag must equal the cell's
    # served /variance frag (both = committed_p90 - the bare served p50, same per-workplace seed) so
    # `served_p50 + frag == committed_p90` consistently. The OLD bug used committed_p90 - committed_p50
    # for /variance, which drifted from the per-route number; this asserts they now agree.
    assert pin["prim_frag_matches_variance"], (
        f"primary pinned frag {pin['prim_frag']} != served /variance frag {pin['served_frag']} "
        f"(depart-after frag derivations disagree -> headline won't reconcile)")
    assert pin["alt_reconcile_bad"] == 0, (
        f"{pin['alt_reconcile_bad']} alts whose served-p50 + frag bad-day is inconsistent")
    # every alt carries BOTH percentiles, p5 <= p50 PER alt, and a non-negative frag.
    assert pin["alt_both"] > 0, "no alt carried both a best-case and a typical journey under pin=1"
    assert pin["alt_p5_gt_p50"] == 0, f"{pin['alt_p5_gt_p50']} alts with p5 (best-case) > p50 (typical)"
    assert pin["alt_with_frag"] > 0, "no alt carried a per-route fragility under depart-after pin=1"
    assert pin["alt_frag_bad"] == 0, f"{pin['alt_frag_bad']} alts with negative fragility"
    # BUG 2: walk-only "alternatives" are pointless and must never be served. Sweep the walkable
    # alt-cells (closest by best-case minutes) and assert ZERO transit-less alts, while a transit
    # alt is still served (the filter didn't nuke every alternative).
    assert res["wo_scan_cells"] > 0, "BUG2 scan found no alt-cells to check under depart-after"
    assert res["wo_alts"] == 0, (
        f"BUG2: {res['wo_alts']} walk-only (transit-less) alts served across "
        f"{res['wo_scan_cells']} depart-after cells, e.g. {res['wo_examples']}")
    assert res["transit_alts"] > 0, (
        "BUG2: no transit alternatives served under depart-after — filter is over-aggressive")
    mission = res["mission"]
    assert mission["compute_status"] == 200 and mission["var_status"] == 200, mission
    assert mission["it_status"] == 200, mission
    assert mission["has_primary_transit"], (
        f"Mission Dolores route collapsed to fake walking: {mission}")
    assert not mission["single_walk"], (
        f"Mission Dolores route served as one walk leg again: {mission}")
    assert mission["structure_complete"], (
        f"Mission Dolores route omitted structural family/branch metadata: {mission}")
    assert mission["has_distinct_boarding_family"], (
        f"Mission Dolores route lost its distinct boarding-corridor alternative: {mission}")
    assert mission["alt_dup_primary"] == 0, (
        f"Mission Dolores served primary route again as an alt: {mission}")
    townsend = res["townsend"]
    assert townsend["compute_status"] == 200 and townsend["var_status"] == 200, townsend
    assert townsend["it_status"] == 200, townsend
    assert townsend["alt_dup_primary"] == 0, (
        f"15th/Dolores -> 650 Townsend served primary route again as an alt: {townsend}")
    # The tree's primary is the planner's committed route, not a promise that every separately
    # traced structural alternative is slower. A valid faster alternative must survive dominance;
    # the frontend labels the anchor "Planner route" instead of falsely calling it recommended.
    assert townsend["structure_complete"], (
        f"15th/Dolores -> 650 Townsend omitted structural route metadata: {townsend}")
    assert townsend["has_walk_transit_siblings"], (
        "15th/Dolores -> 650 Townsend lost walk-vs-transit-tail sibling optionality: "
        f"{townsend}")
    townsend_slow = res["townsend_slow"]
    assert townsend_slow["it_status"] == 200, townsend_slow
    assert townsend_slow["structure_complete"], (
        "slow-walk Townsend response omitted structural route metadata: "
        f"{townsend_slow}")
    assert townsend_slow["has_walk_transit_siblings"], (
        "15th/Dolores -> 650 Townsend slow walk lost deterministic walk-vs-transit-tail "
        f"branch optionality without /variance priming: {townsend_slow}")
    townsend_fast = res["townsend_fast"]
    assert townsend_fast["it_status"] == 200, townsend_fast
    assert townsend_fast["structure_complete"], (
        "fast-walk Townsend response omitted structural route metadata: "
        f"{townsend_fast}")
    assert townsend_fast["has_walk_branch"], (
        "15th/Dolores -> 650 Townsend fast walk lost its boarding-corridor walk branch: "
        f"{townsend_fast}")


# --------------------------------------------------------------------------------------
# /attribution — dominant-line-per-cell, cached + deterministic
# --------------------------------------------------------------------------------------
@pytest.mark.slow
def test_attribution_shape_caching_and_count(client, server):
    # Use a fresh destination key (Twin Peaks) to avoid colliding with any cached Ferry result,
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
    0 <= stuck <= 1, alt-lines respect the configured cap and never include the cell's own dominant
    line, and a second identical GET returns the identical payload (LRU cache + the
    deterministic per-workplace sha256 seed)."""
    _skip_unless_mc(server)
    alt_cap = getattr(server.sr, "RAPTOR_ALT_CHIP_CAP", 6)
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
            assert len(alt) <= alt_cap, f"{cid}: alt has {len(alt)} entries (> {alt_cap} cap)"
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


def test_missing_walk_speed_uses_calibrated_medium_scalar(server):
    """The product default is Medium, not the graph's 4.8 km/h bake reference."""
    with server.app.test_request_context("/compute"):
        speed, scalar = server._req_speed()
    expected = server.config.WALK_KMH / server.WALK_SPEEDS[server.DEFAULT_SPEED]
    assert speed == server.DEFAULT_SPEED == "med"
    assert scalar == pytest.approx(expected)
    assert scalar != 1.0

    with server.app.test_request_context("/compute?speed=not-a-preset"):
        invalid_speed, invalid_scalar = server._req_speed()
    assert invalid_speed == speed
    assert invalid_scalar == pytest.approx(scalar)


def test_omitted_walk_speed_matches_explicit_medium_across_product_endpoints(client, server):
    """The HTTP product default and ``speed=med`` are one routing/cache identity end to end."""
    try:
        server.limiter.reset()
    except Exception:
        pass

    def get(path, *, explicit=False):
        separator = "&" if "?" in path else "?"
        suffix = f"{separator}speed=med" if explicit else ""
        response = client.get(path + suffix)
        assert response.status_code == 200, response.get_data(as_text=True)
        body = response.get_json()
        if isinstance(body, dict):
            body.pop("ms", None)  # cache-hit timing is intentionally not response identity
        return body

    compute_path = f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}"
    exact_path = f"/compute_exact?lat={FERRY_LAT}&lon={FERRY_LON}"
    attribution_path = f"/attribution?dlat={FERRY_LAT}&dlon={FERRY_LON}"
    variance_path = f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}"

    default_compute = get(compute_path)
    assert get(compute_path, explicit=True) == default_compute
    assert get(exact_path, explicit=True) == get(exact_path)
    assert get(attribution_path, explicit=True) == get(attribution_path)
    default_variance = get(variance_path)
    assert get(variance_path, explicit=True) == default_variance

    variance = default_variance.get("variance") or {}
    candidate = next((cid for cid, value in variance.items() if (value or {}).get("alt")), None)
    if candidate is None:
        candidate = next(
            cid for cid, pair in default_compute["cells"].items()
            if isinstance(pair, list) and len(pair) == 2 and pair[1] is not None
        )
    itinerary_path = (
        f"/itinerary?id={candidate}&dlat={FERRY_LAT}&dlon={FERRY_LON}&pin=1"
    )
    default_itinerary = get(itinerary_path)
    explicit_itinerary = get(itinerary_path, explicit=True)
    assert explicit_itinerary == default_itinerary
    options = [default_itinerary, *(default_itinerary.get("alts") or [])]
    choice_keys = [option.get("choice_key") for option in options]
    assert all(choice_keys) and len(choice_keys) == len(set(choice_keys))


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
    alt_cap = getattr(server.sr, "RAPTOR_ALT_CHIP_CAP", 6)
    assert len(alts) <= alt_cap, f"{cid}: {len(alts)} alts (> {alt_cap} cap)"

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


# --------------------------------------------------------------------------------------
# BUG 2 regression: WALK-ONLY "alternatives" are pointless — never serve a transit-less alt.
# --------------------------------------------------------------------------------------
# A user reported that a walkable cell's compare card showed multiple pure-walk "routes" (e.g.
# walk 26/27/27/30m) — different walking paths labeled with a bus line you don't actually ride. They
# come from alt_lines_window enumerating a line by an access stop whose journey degenerates to pure
# walking (a sub-2-min ride folded into walk by _TINY_HOP_MIN, or a longer via-stop walk). Such an
# alt has no transit leg → no information. The fix (server_raptor._legs_have_transit) drops any alt
# whose traced journey carries no ride, in BOTH semantics. This is a SCANNING test (user policy):
# it sweeps EVERY cell the MC surfaced alt chips for and asserts NO served alt is walk-only, AND that
# a transit cell still gets its real (transit) alternatives.
@pytest.mark.slow
def test_no_walk_only_alternatives(client, server):
    _skip_unless_mc(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    rv = client.get(f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert rv.status_code == 200, rv.get_data(as_text=True)
    mc = server._mc_peek(FERRY_LAT, FERRY_LON, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED)
    assert mc is not None, "MC not built"

    def _legs_transit(legs):
        return any((l or {}).get("mode") == "transit" for l in (legs or ()))

    # Sweep cells with alt chips and assert each SERVED alt rides a real transit leg. The walk-only-
    # alt bug lives on WALKABLE (low-time) cells near the workplace, so bias the sample toward those:
    # take the closest 90 chip-cells by best-case minutes (then bounded under the /itinerary 120/min
    # rate limit). The whole grid's chip cells number ~1k+, so this is a broad-but-bounded scan.
    cells = client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}").get_json()["cells"]
    chip_cids = [server._RAPTOR.cell_ids[ci]
                 for ci, lines in mc.get("alt_chips", {}).items() if lines]
    assert chip_cids, "no cell surfaced alt chips — can't exercise the alt path"

    def _best(cid):
        v = cells.get(str(cid)) or cells.get(cid)
        return v[0] if (v and v[0] is not None) else 10 ** 6
    sample = sorted(chip_cids, key=_best)[:90]             # walkable-first, bounded for the rate limit
    checked_cells = 0
    walk_only_alts = []                                   # (cell, line, legs) offenders, if any
    transit_alts_seen = 0
    for cid in sample:
        resp = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}")
        assert resp.status_code == 200, resp.get_data(as_text=True)
        for a in resp.get_json().get("alts", []):
            legs = a.get("legs", [])
            if _legs_transit(legs):
                transit_alts_seen += 1
            else:
                walk_only_alts.append((cid, a.get("line"), legs))
        checked_cells += 1
    assert not walk_only_alts, (
        f"served {len(walk_only_alts)} walk-only (transit-less) alts across {checked_cells} cells: "
        f"{walk_only_alts[:5]}")
    # A transit cell must still get its real alts (the filter didn't nuke every alternative).
    assert transit_alts_seen > 0, (
        "no transit alternatives served across any cell — the filter is over-aggressive")


@pytest.mark.slow
def test_itinerary_pin_per_route_typicals(client, server, monkeypatch):
    """/itinerary?pin=1 for a cell with alts carries a per-ROUTE committed-plan TYPICAL: the
    PRIMARY gains ``real``/``frag`` and EACH alt gains its OWN ``real``/``frag`` (the consistency
    fix so the compare card can show every strip on the same metric). Asserts:
      * the primary's ``real`` == that cell's served /variance ``realistic`` (same committed MC,
        same per-workplace seed -> the primary strip matches the headline);
      * every route honors ``perfect <= committed`` (``real >= best-case``) PER route, primary + alts;
      * frag >= 0 and the typicals are sane ints;
      * a PLAIN hover (no pin=1) does NOT compute them (alts carry no ``real``), and the pin typicals
        are cached (a second pin=1 request is byte-identical + the typ cache is populated)."""
    _skip_unless_mc(server)
    try:
        server.limiter.reset()
    except Exception:
        pass
    # Other tests may have left this bounded MC result cached after its deliberately single retained
    # lossless scenario was replaced by a newer workplace. Force this test's /variance call through
    # the capture path; a stale token is expected to fall back exactly, but is not what this test is
    # exercising.
    mc_key = server._coarse_key(
        FERRY_LAT, FERRY_LON, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED)
    server._RAPTOR_MC_CACHE.pop(mc_key)
    client.get(f"/compute?lat={FERRY_LAT}&lon={FERRY_LON}")
    rv = client.get(f"/variance?dlat={FERRY_LAT}&dlon={FERRY_LON}")
    assert rv.status_code == 200, rv.get_data(as_text=True)
    realistic = rv.get_json()["realistic"]
    assert "scenario" not in rv.get_data(as_text=True).lower(), \
        "private MC scenario/token leaked through the /variance JSON boundary"
    cid, _chips = _cell_with_alt_chips(server, FERRY_LAT, FERRY_LON)
    assert cid is not None, "MC surfaced no alt chips on any cell"

    # PLAIN hover first: alts must NOT carry per-route typicals (the gate keeps them off hover).
    hov = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}").get_json()
    assert hov["alts"], f"{cid}: expected alts on the cell"
    assert all("real" not in a for a in hov["alts"]), (
        f"{cid}: a plain hover computed per-alt typicals (should be pin=1 only): {hov['alts']}")
    # And the typ cache is still empty for this cell (the hover never touched route_typicals).
    mc = server._mc_peek(FERRY_LAT, FERRY_LON, server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED)
    ci = server._RAPTOR.cell_index[cid]
    assert ci not in mc["typ"], f"{cid}: plain hover populated the typ cache"
    token = mc.get("_scenario_token")
    assert token, "production /variance failed to retain its lossless pin accelerator"
    scenario = server.sr._mc_scenario_for(
        token, server._coarse_key(FERRY_LAT, FERRY_LON,
                                  server.DEFAULT_MAX_RIDES, server.DEFAULT_SPEED))
    assert scenario is not None and scenario.nbytes <= server.sr._MC_SCENARIO_MAX_BYTES

    # PINNED: every route gains its own typical + fragility.
    # A warm pin must consume the retained tail: rebuilding even one reverse profile is a failure.
    from core import raptor as _raptor
    monkeypatch.setattr(
        _raptor, "montecarlo_commute_committed",
        lambda *a, **k: pytest.fail("warm pin rebuilt the full committed reverse profiles"))
    t0 = time.time()
    body = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}&pin=1").get_json()
    dt_ms = (time.time() - t0) * 1000.0
    alts = body["alts"]
    assert alts, f"{cid}: pinned itinerary lost its alts"

    # Primary's real == the served realistic for this cell (same committed MC + seed).
    assert "real" in body and "frag" in body, f"{cid}: pinned primary lacks real/frag: {body.keys()}"
    assert isinstance(body["real"], int) and isinstance(body["frag"], int)
    assert body["frag"] >= 0, f"{cid}: negative primary fragility {body['frag']}"
    assert body["real"] == realistic[cid], (
        f"{cid}: primary typical {body['real']} != served realistic {realistic[cid]} "
        f"(seed/model drift between route_typicals and /variance)")
    assert body["real"] >= body["total"], (
        f"{cid}: primary typical {body['real']} < best-case {body['total']} (perfect<=committed lost)")

    # Each alt: its OWN typical, honoring perfect<=committed per route.
    saw = 0
    for a in alts:
        if "real" not in a:                          # an unreachable-via-its-stop alt: allowed None
            continue
        assert isinstance(a["real"], int) and isinstance(a.get("frag"), int), a
        assert a["frag"] >= 0, f"{cid}/{a['line']}: negative fragility {a['frag']}"
        assert a["real"] >= a["min"], (
            f"{cid}/{a['line']}: alt typical {a['real']} < its best-case {a['min']} "
            f"(perfect<=committed lost per route)")
        saw += 1
    assert saw > 0, f"{cid}: no alt carried a per-route typical"

    # Cached per pinned cell: the typ cache is now populated, and a repeat pin=1 is byte-identical.
    assert ci in mc["typ"], f"{cid}: pin=1 did not cache the per-route typicals"
    body2 = client.get(f"/itinerary?id={cid}&dlat={FERRY_LAT}&dlon={FERRY_LON}&pin=1").get_json()
    assert body2["alts"] == alts and body2.get("real") == body["real"], (
        f"{cid}: pinned typicals not stable across two identical pin=1 requests")
    # Timing: a pinned cell with a few alts resolves its typicals well under a second.
    assert dt_ms < 900, f"{cid}: pin=1 typicals took {dt_ms:.0f}ms (> 900ms budget)"


def test_mc_scenario_token_eviction_and_budget_fall_back(server, monkeypatch):
    """Only one opaque scenario is retained; stale/wrong-key/oversize tokens resolve to None."""
    sr = server.sr

    class Scenario:
        def __init__(self, nbytes, name):
            self.nbytes = nbytes
            self.name = name

    monkeypatch.setattr(sr, "_MC_SCENARIO_ACTIVE", None)
    monkeypatch.setattr(sr, "_MC_SCENARIO_SEQ", 0)
    one = Scenario(1024, "one")
    two = Scenario(2048, "two")
    token1 = sr._retain_mc_scenario(("workplace-1",), one)
    assert sr._mc_scenario_for(token1, ("workplace-1",)) is one
    assert sr._mc_scenario_for(token1, ("wrong-key",)) is None

    token2 = sr._retain_mc_scenario(("workplace-2",), two)
    assert token2 != token1
    assert sr._mc_scenario_for(token1, ("workplace-1",)) is None   # evicted -> exact fallback
    assert sr._mc_scenario_for(token2, ("workplace-2",)) is two

    oversized = Scenario(sr._MC_SCENARIO_MAX_BYTES + 1, "too-large")
    assert sr._retain_mc_scenario(("workplace-3",), oversized) is None
    assert sr._mc_scenario_for(token2, ("workplace-2",)) is two    # rejected build didn't evict


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
def test_geocode_ferry_building(client, server, monkeypatch):
    monkeypatch.setattr(
        server.geo, "geocode",
        lambda q, cache=False: (37.7955, -122.3937, "Ferry Building, San Francisco"))
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
def test_autocomplete_results_are_sf_bounded(client, server, monkeypatch):
    monkeypatch.setattr(server.geo, "autocomplete", lambda q, limit=6: [
        {"label": "Ferry Building, San Francisco", "lat": 37.7955, "lon": -122.3937},
        {"label": "Ferry Plaza, San Francisco", "lat": 37.7951, "lon": -122.3935},
    ])
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
    breakdowns come from _RAPTOR_TREE_CACHE (a bounded LRU keyed by destination + rides
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
