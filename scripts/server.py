#!/usr/bin/env python
"""
Live commute-isochrone server: set ANY workplace address in the browser and the map
recomputes door-to-door times from a grid of SF origins.

Production uses one routing stack: RAPTOR transit plus the hill-aware walking graph. Run:

    .venv/bin/python scripts/server.py        # then open http://127.0.0.1:8000

Env: GRID_M (default 200) trades detail for speed; WINDOW_MIN, PORT.

The workplace address lives only in the browser — this process never hardcodes or
persists it. /compute and its compatibility alias /compute_exact both use the same
RAPTOR result; /itinerary, /attribution, and /variance consume the corresponding
RAPTOR tree and walking-graph caches.
"""
import os, json, math, time, datetime as dt
_BOOT_MONOTONIC = time.monotonic()
# Cap numba's thread pool BEFORE numba is imported (via core.raptor_*). Bounds the parallel MC
# kernel's worker threads — fewer idle threads (lower RSS) and no oversubscription on a small box.
# Overridable via NUMBA_NUM_THREADS. (The MC kernel is also serialized below: numba's workqueue
# threading layer is NOT threadsafe, so two concurrent parallel kernels would abort the process.)
os.environ.setdefault("NUMBA_NUM_THREADS", str(min(4, os.cpu_count() or 4)))
from pathlib import Path
from flask import Flask, request, jsonify
from flask_limiter import Limiter

from core import config, feeds, geo, raptor_build, readiness, static_bundle
from core import server_raptor as sr             # RAPTOR engine glue (state + builders, JVM-free)
config.load_dotenv()             # load .env (GEOCODER / GEOAPIFY_KEY) for the geocoder
# geopandas and the grid helper are imported only on the one-time static-bundle build path.

# ---- RAPTOR configuration --------------------------------------------------------------
RAPTOR_SEMANTIC = sr.RAPTOR_SEMANTIC
RAPTOR_MC = sr.RAPTOR_MC
# Validate the semantic loudly rather than silently serving the wrong branch.
if RAPTOR_SEMANTIC not in ("arriveby", "departafter"):
    raise SystemExit(f"RAPTOR_SEMANTIC must be 'arriveby' or 'departafter', "
                     f"got {RAPTOR_SEMANTIC!r}")
DEFAULT_MAX_RIDES = sr.DEFAULT_MAX_RIDES

HERE = Path(__file__).resolve().parent
GRID_M = int(os.environ.get("GRID_M", str(config.GRID_M)))

# ---- Boot: feeds, model date, grid ------------------------------------------------------
GTFS = config.gtfs_paths()

# ---- Static page data: cells (GeoJSON), origin coords, line shapes, service date --------
# These are workplace-INDEPENDENT (only the feeds + neighborhoods determine them). We cache them to
# data/server_static.json so normal server boots avoid geopandas/shapely/pandas.
_STATIC = config.DATA / "server_static.json"


_GTFS_SOURCES = raptor_build._source_mtimes(GTFS)
_GRID_SOURCE = config.neigh_path()


def _grid_source_metadata(path):
    """Return the readable direct source metadata stored in the static bundle."""
    return static_bundle.source_metadata(path)


_GRID_SOURCE_META = _grid_source_metadata(_GRID_SOURCE)
_bundle = None
if _STATIC.exists():
    try:
        _b = json.loads(_STATIC.read_text())
        _bundle_shape_ok = (
            isinstance(_b.get("cells"), dict)
            and isinstance(_b.get("lines"), dict)
            and isinstance(_b.get("origin_ll"), dict)
            and all(isinstance(v, (list, tuple)) and len(v) == 2
                    for v in _b["origin_ll"].values()))
        _bundle_grid_meta = {
            "name": _b.get("grid_source_name"),
            "size": _b.get("grid_source_size"),
            "mtime_ns": _b.get("grid_source_mtime_ns"),
        }
        if (config.normalize_source_mtimes(_b.get("source_mtimes", ())) == _GTFS_SOURCES
                and _b.get("grid_m") == GRID_M
                and _GRID_SOURCE_META is not None
                and _bundle_grid_meta.get("name") == _GRID_SOURCE_META.get("name")
                and _bundle_grid_meta.get("size") == _GRID_SOURCE_META.get("size")
                and config.normalize_mtime_ns(_bundle_grid_meta.get("mtime_ns"))
                    == _GRID_SOURCE_META.get("mtime_ns")
                and isinstance(_b.get("svc_date"), str) and len(_b["svc_date"]) == 8
                and _bundle_shape_ok):
            _bundle = _b
    except Exception:
        _bundle = None

if _bundle is not None:                          # LEAN boot — json only, no geopandas/shapely/pandas
    CELLS_GEOJSON = _bundle["cells"]; LINES = _bundle["lines"]
    ORIGIN_LL = {k: tuple(v) for k, v in _bundle["origin_ll"].items()}
    _SVC_DATE = dt.datetime.strptime(_bundle["svc_date"], "%Y%m%d").date()
    DEP = config.departure(_SVC_DATE)
    print(f"[boot] modeling weekday {_SVC_DATE} @ {DEP:%-I:%M%p}".lower())
    print(f"[boot] ready: {len(ORIGIN_LL)} origins (JVM-free, lean static bundle — no geopandas). "
          f"Open http://127.0.0.1:8000")
else:                                            # BUILD path — needs the geo/pandas stack (one-time)
    # Auto-pick a weekday with trips in ALL feeds (a hardcoded date silently breaks after a repull).
    _SVC_DATE = feeds.pick_service_date(GTFS)
    DEP = config.departure(_SVC_DATE)
    print(f"[boot] modeling weekday {_SVC_DATE} @ {DEP:%-I:%M%p}".lower())
    print(f"[boot] building grid @ {GRID_M}m (one-time geo build; caching static bundle)...")
    try:                                         # cache the bundle so the next JVM-free boot is lean
        _b = static_bundle.build_static_bundle(
            _STATIC, GTFS, grid_m=GRID_M, service_date=_SVC_DATE,
            source_mtimes=_GTFS_SOURCES)
        CELLS_GEOJSON = _b["cells"]
        LINES = _b["lines"]
        ORIGIN_LL = {k: tuple(v) for k, v in _b["origin_ll"].items()}
        print(f"[boot] ready: {len(ORIGIN_LL)} origins (static bundle cached). "
              "Open http://127.0.0.1:8000")
    except Exception as _e:
        raise RuntimeError(f"could not build static bundle: {_e}") from _e


# ---- Shared cache type ------------------------------------------------------------------
_BoundedLRU = sr.BoundedLRU


# ---- RAPTOR engine and graph-backed walking --------------------------------------------
# The engine state and builders live in core.server_raptor; this module resolves the one-time
# engine and graph snap tables and hands them over via sr.init().
_RAPTOR = None
_WG = None
_WG_STOP_NODES = _WG_STOP_CONN = _WG_CELL_NODES = _WG_CELL_CONN = None
_WG_STOP_GIDS = None

from core import raptor_engine
import numpy as _np
from core import walk as _walkmod

_acc_path = config.DATA / "raptor_cache" / f"access_walk_{GRID_M}m_{_SVC_DATE:%Y%m%d}.npz"
if not _acc_path.exists():
    raise FileNotFoundError(
        f"required RAPTOR walking bake {_acc_path.name} is missing "
        f"(run scripts/build_walk_graph.py and scripts/bake_walk_access.py"
        + (f" with GRID_M={GRID_M}" if GRID_M != config.GRID_M else "") + ")")

_RAPTOR = raptor_engine.RaptorEngine(GTFS, _SVC_DATE, access_path=_acc_path, verbose=True)
_gl, _go = _RAPTOR.data["stop_lat"], _RAPTOR.data["stop_lon"]
_gids = [g for g in range(_RAPTOR.data["n_stops"]) if not _np.isnan(_gl[g])]
_WG = _walkmod.WalkGraph.load()
_WG_STOP_GIDS = _np.asarray(_gids, dtype=_np.int32)
_WG_STOP_NODES, _WG_STOP_CONN = _WG.snap(
    _np.column_stack(([_go[g] for g in _gids], [_gl[g] for g in _gids])))
_cll = _np.array([[ORIGIN_LL[c][1], ORIGIN_LL[c][0]] for c in _RAPTOR.cell_ids])
_WG_CELL_NODES, _WG_CELL_CONN = _WG.snap(_cll)
print(f"[boot] RAPTOR engine ON (semantic={RAPTOR_SEMANTIC}); graph walking ON "
      f"({_acc_path.name})")

sr.init(raptor=_RAPTOR, wg=_WG, wg_stop_nodes=_WG_STOP_NODES, wg_stop_conn=_WG_STOP_CONN,
        wg_cell_nodes=_WG_CELL_NODES, wg_cell_conn=_WG_CELL_CONN,
        wg_stop_gids=_WG_STOP_GIDS, origin_ll=ORIGIN_LL)


def _required_feed_paths():
    """Return the stable role-to-archive mapping used by readiness checks.

    ``config.gtfs_paths()`` is intentionally a compact routing input and drops missing
    archives. Readiness must distinguish a missing current Muni, BART, or Caltrain archive,
    so keep the role mapping explicit.
    """
    return {
        "muni": config.DATA / config.MUNI_CURRENT,
        "bart": config.DATA / config.BART,
        "caltrain": config.DATA / config.CALTRAIN,
    }


def _boot_readiness():
    """Run expensive data/artifact checks once, after the live runtime is initialized."""
    # Pass the already-loaded RAPTOR mapping, while the canonical graph and artifacts remain
    # path-addressable for the structural validators.  check_readiness is deliberately called
    # only here; probes below read the frozen result and never touch the archives.
    try:
        data_check = readiness.check_readiness(
            _required_feed_paths(), _RAPTOR.data, config.DATA / "walk_graph.npz", _acc_path,
            _STATIC, grid_m=GRID_M, grid_source=_GRID_SOURCE,
        )
    except Exception:
        data_check = readiness.ReadinessResult(False, "runtime_load_failed")
    try:
        runtime_check = readiness.validate_runtime_state(
            engine_kind="raptor", graph_backed=True, initialized=True,
            semantic=RAPTOR_SEMANTIC, engine=_RAPTOR, walk_graph=_WG,
            service_date=_SVC_DATE,
        )
    except Exception:
        runtime_check = readiness.CheckResult(False, "runtime_load_failed")
    if not data_check.ready:
        return data_check
    if not runtime_check.ready:
        return readiness.ReadinessResult(
            False, runtime_check.reason_code, data_check.service_date,
            runtime_check.detail,
        )
    return data_check


# Immutable, safe-to-serialize boot state.  Do not replace this from a request handler: a
# controlled process restart is the refresh boundary for feed and artifact validation.
_READINESS = _boot_readiness()

# ---- Re-export the RAPTOR glue (state, caches, locks, key fns, builders) so the thin handlers
# below and the tests that reach into server.X keep working. These are the SAME objects the
# module owns (cache/lock identity preserved → tests' .clear()/.pop()/acquire() act on the live
# state).
WALK_SPEEDS = sr.WALK_SPEEDS
DEFAULT_SPEED = sr.DEFAULT_SPEED
_RAPTOR = sr._RAPTOR
_RAPTOR_EGRESS_CACHE = sr._RAPTOR_EGRESS_CACHE
_RAPTOR_TREE_CACHE = sr._RAPTOR_TREE_CACHE
_RAPTOR_MC_CACHE = sr._RAPTOR_MC_CACHE
_mc_peek = sr.mc_peek
_raptor_tree = sr.raptor_tree
_Busy = sr.Busy

def _req_max_rides():
    """Parse the ``maxrides`` query param into the model's ride cap. Absent/blank/invalid ->
    DEFAULT_MAX_RIDES (today's behavior). The frontend sends rides directly (transfers+1):
    0 transfers -> 1, 1 -> 2, 2 -> 3, "Any" -> 8. We clamp to [1, DEFAULT_MAX_RIDES] so a
    bogus value can't disable transit (rides must be >= 1) or exceed the model default."""
    raw = request.args.get("maxrides")
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_RIDES
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_RIDES
    return max(1, min(DEFAULT_MAX_RIDES, v))


# Walk-speed toggle (RAPTOR only): scalar = bake reference / selected pace. The engine multiplies
# every WALK reference-second by it. Medium is the calibrated typical pace; Fast is brisk.
# The presets live in core.config (WALK_SPEEDS / DEFAULT_SPEED), re-exported via core.server_raptor.
def _req_speed():
    """(speed_name, walk_scalar) from ?speed=slow|med|fast; missing/invalid uses Medium."""
    s = (request.args.get("speed") or "").lower()
    if s not in WALK_SPEEDS:
        s = DEFAULT_SPEED
    return s, config.WALK_KMH / WALK_SPEEDS[s]


app = Flask(__name__)

# ---- Rate limiting (Flask-Limiter, in-memory) -----------------------------------------
# Single-process app, so the default in-memory store is correct (no Redis needed). Per-IP
# limits are applied per endpoint below via @limiter.limit. Choosing the right IP source is
# load-bearing for the limits to be real:
#   - BOTH `CF-Connecting-IP` and `X-Forwarded-For` are forgeable by any client that reaches
#     the origin directly — a proxy header is only trustworthy when the proxy chain is
#     guaranteed. In production that guarantee comes from the cloudflare-ufw firewall
#     (80/443 restricted to CF CIDRs) plus TRUST_PROXY=1 in sfci.service; the headers
#     themselves prove nothing. So gate BOTH on TRUST_PROXY, preferring CF-Connecting-IP
#     (when traffic really transits Cloudflare it's the canonical real-client header,
#     whereas XFF can carry a client-supplied first hop that Caddy appends to rather than
#     replaces). Note: with TRUST_PROXY=1 a direct-to-origin attacker defeats per-IP limits
#     regardless of which header we read — the firewall, not this function, is the actual
#     abuse boundary.
#   - Without TRUST_PROXY, use the socket peer (correct for localhost dev / direct hits).
def _client_ip():
    if os.environ.get("TRUST_PROXY", "").lower() in ("1", "true", "yes"):
        cf = request.headers.get("CF-Connecting-IP", "").strip()
        if cf:
            return cf
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            first = fwd.split(",")[0].strip()
            if first:
                return first
    return request.remote_addr or "127.0.0.1"


def _parse_ll(lat_raw, lon_raw):
    """Parse + validate lat/lon from query args — used by EVERY coordinate-taking endpoint.
    Rejects missing/non-numeric/NaN/inf coords with a friendly JSON 400 instead of a 500
    (NaN is a valid float() but never compares equal, so it would bypass every coord-keyed
    cache), and rejects finite points outside the graph-backed supported area before a coordinate
    can burn a full compute for a garbage result.  The loose bbox below is only a cheap prefilter;
    ``WalkGraph.supports_point`` is the authoritative policy."""
    try:
        lat = float(lat_raw); lon = float(lon_raw)
    except (TypeError, ValueError):
        raise _BadRequest("lat/lon must be numeric")
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise _BadRequest("lat/lon must be finite")
    lo_min, la_min, lo_max, la_max = config.SF_VALID_BBOX   # loose box; see core/config.py
    # This box is deliberately only a cheap prefilter.  The walk graph is the authoritative
    # supported-area policy: its connector threshold is what prevents water/East Bay points
    # from being silently snapped to the nearest SF edge and sent through an expensive route.
    if not (la_min <= lat <= la_max and lo_min <= lon <= lo_max):
        raise _OutsideSupportedArea()
    if _WG is None or not _WG.supports_point(lon, lat, max_connector_m=300):
        raise _OutsideSupportedArea()
    return lat, lon


def _validate_geocoder_point(lat_raw, lon_raw):
    """Validate one provider result with the same coordinate policy as API inputs."""
    return _parse_ll(lat_raw, lon_raw)


class _BadRequest(Exception):
    pass


class _OutsideSupportedArea(Exception):
    pass


@app.errorhandler(_BadRequest)
def _bad_request(e):
    return jsonify({"error": "bad_request", "detail": str(e)}), 400


_OUTSIDE_SUPPORTED_DETAIL = "Location is outside the supported San Francisco walking area."


@app.errorhandler(_OutsideSupportedArea)
def _outside_supported_area(_e):
    return jsonify({"error": "outside_supported_area", "detail": _OUTSIDE_SUPPORTED_DETAIL}), 422


limiter = Limiter(key_func=_client_ip, app=app, storage_uri="memory://")


# Dynamic-data endpoints whose response depends on workplace/speed/transfer params. Geocode and
# autocomplete are pure functions of the query string -> let the browser cache them. The page
# bundle (GET /) is workplace-agnostic and shipped once at boot -> stays cacheable + bfcache-
# eligible so back-navigation doesn't refetch the network and rerun /compute + /variance.
_NO_STORE_PATHS = frozenset({"/compute", "/compute_exact", "/itinerary", "/attribution", "/variance",
                             "/livez", "/readyz", "/healthz"})


@app.after_request
def _no_cache(resp):
    """Disable browser/bf-cache for the dynamic API endpoints only. Without no-store, a heuristic
    cache hit on /itinerary or /variance (URL identical across runs) would silently hide a
    server-side fix and could serve a stale response after a workplace/speed change."""
    if request.path in _NO_STORE_PATHS:
        resp.headers["Cache-Control"] = "no-store"
    return resp


def _perf_benchmark_enabled():
    """Whether to expose request-local phase data for the offline benchmark harness only."""
    return os.environ.get("PERF_BENCHMARK_STATS", "").lower() in ("1", "true", "yes", "on")


def _phase_response(body, phases):
    """Attach opt-in standard timing headers without changing a product JSON payload.

    ``phases`` is created by a single handler and passed explicitly through the routing seams; it
    never outlives the request.  Keep counts in X-Perf-Phases, while Server-Timing receives only
    duration values so browser tooling can render it correctly.  Normal production requests take
    the plain jsonify path and have no timer/header work.
    """
    response = jsonify(body)
    if not phases:
        return response
    clean = {}
    for name, value in phases.items():
        if isinstance(name, str) and isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)) and float(value) >= 0:
                clean[name] = round(float(value), 3)
    if not clean:
        return response
    durations = [
        # A period is valid in an HTTP token, so retain the same names in both headers.  The
        # benchmark parser can then merge them without creating a misleading duplicate phase.
        f"{name};dur={value:.3f}"
        for name, value in clean.items() if name.endswith("_ms")
    ]
    if durations:
        response.headers["Server-Timing"] = ", ".join(durations)
    response.headers["X-Perf-Phases"] = json.dumps(clean, separators=(",", ":"), sort_keys=True)
    return response


@app.errorhandler(429)
def _ratelimit_json(e):
    """Flask-Limiter's default 429 page is HTML, so the frontend's r.json() throws and the user
    just sees 'error' in the tooltip. Return JSON so callers can handle it (retry / show toast)."""
    return jsonify({"error": "rate_limited", "detail": str(getattr(e, "description", e))}), 429


# ---- RAPTOR grid travel-times + breakdown/variance/geometry -------------------------
# The whole RAPTOR engine integration (state, caches, locks, the egress/tree/MC builders,
# the journey-geometry provider, and the /itinerary assemblers for both semantics) lives in
# core.server_raptor (re-exported above). The thin Flask handlers below call into sr.*.


@app.route("/compute")
@limiter.limit("60/minute")
def _compute():
    lat, lon = _parse_ll(request.args.get("lat"), request.args.get("lon"))
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    t0 = dt.datetime.now()
    cells = sr.compute_raptor(lat, lon, max_rides, speed, walk_scalar)
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[compute:raptor] rides={max_rides} speed={speed} {ms:.0f}ms")
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": round(ms)})


@app.route("/compute_exact")
@limiter.limit("12/minute")
def _compute_exact():
    lat, lon = _parse_ll(request.args.get("lat"), request.args.get("lon"))
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    cells = sr.compute_raptor(lat, lon, max_rides, speed, walk_scalar)
    return jsonify({"dest": [lat, lon], "cells": cells, "ms": 0})


# RAPTOR itinerary assemblers in core.server_raptor are now the sole source for /itinerary.

@app.route("/itinerary")
@limiter.limit("120/minute")
def _itinerary():
    cid = request.args.get("id")
    if cid is not None and cid in ORIGIN_LL:
        olat, olon = ORIGIN_LL[cid]
    else:  # olat/olon are only required when no (valid) cell id resolves the origin
        olat, olon = _parse_ll(request.args.get("olat"), request.args.get("olon"))
    dlat, dlon = _parse_ll(request.args.get("dlat"), request.args.get("dlon"))
    # Use the same transfer cap and walking settings as /compute so the breakdown matches the map.
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    # Both semantic assemblers trace the selected cell from the shared RAPTOR tree. Off-grid
    # points snap to the nearest cell inside the assembler.
    _pin = request.args.get("pin") == "1"
    _perf = {} if (_pin and _perf_benchmark_enabled()) else None
    assemble = (sr.itinerary_departafter if RAPTOR_SEMANTIC == "departafter"
                else sr.itinerary_arriveby)
    res = assemble(cid, olat, olon, dlat, dlon, max_rides, speed, walk_scalar, pin=_pin,
                   perf=_perf)
    res = dict(res)
    res["olat"], res["olon"] = round(olat, 5), round(olon, 5)
    return _phase_response(res, _perf) if _perf is not None else jsonify(res)
@app.route("/attribution")
@limiter.limit("12/minute")
def _attribution():
    """Return the dominant transit line per cell from the RAPTOR tree."""
    dlat, dlon = _parse_ll(request.args.get("dlat"), request.args.get("dlon"))
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    attr = sr.raptor_attribution(dlat, dlon, max_rides, speed, walk_scalar)
    print(f"[attr:raptor] rides={max_rides} speed={speed} -> {len(attr)} cells")
    return jsonify(attr)


@app.route("/variance")
@limiter.limit("20/minute")
def _variance():
    """Service-noise overlay (RAPTOR, both semantics). Lazy + cached; fetched by the frontend AFTER
    /compute paints the map (progressive refinement, like /compute_exact). Empty when service-noise
    mode is disabled
    so the frontend simply keeps the painted map.

    BOTH semantics build the SAME committed-plan MC off the workplace's cached traced tree, but serve
    DIFFERENT shapes because the headline TYPICAL differs:
      - ARRIVE-BY: ``{realistic, variance}`` (byte-unchanged). The headline typical IS the committed
        MC ``realistic`` (the arrive-by "perfect-timing" base is too rosy, so the realistic overlay
        replaces it as the typical color/number).
      - DEPART-AFTER: ``{variance}`` ONLY (NO ``realistic``). The headline typical is the bare
        depart-after p50 (served by /compute as cells[c][1]); the MC is used solely for the
        ``frag``/``stuck``/``alt`` overlay — surfacing a committed ``realistic`` here would override
        the p50 the map already paints, which the target model forbids. ``frag`` = p90-p50 with p90
        the committed-MC p90 and p50 the served depart-after p50 (the SAME floor _raptor_mc_build
        passes to montecarlo), so frag >= 0 by construction.
    Per-cell ``variance`` shape is identical for both: ``{frag, std, stuck[, alt:[lines]]}``."""
    dlat, dlon = _parse_ll(request.args.get("dlat"), request.args.get("dlon"))
    # Validate the destination even when the optional MC overlay is disabled.  All coordinate
    # endpoints share the supported-area contract; feature gating must not turn /variance into a
    # silent 200 for an invalid workplace.
    if not (RAPTOR_SEMANTIC in ("arriveby", "departafter") and RAPTOR_MC):
        return jsonify({"realistic": {}, "variance": {}})
    max_rides = _req_max_rides()
    speed, walk_scalar = _req_speed()
    t0 = dt.datetime.now()
    perf = {} if _perf_benchmark_enabled() else None
    # Non-blocking like /compute_exact: another workplace's MC running -> retryable 503
    # (the frontend's loadVariance retries once and otherwise keeps the perfect map).
    try:
        out = sr.raptor_mc(dlat, dlon, max_rides, speed, walk_scalar, perf=perf)
    except _Busy:
        print("[variance:raptor] busy -> 503")
        return jsonify({"busy": True}), 503, {"Retry-After": "4"}
    ms = (dt.datetime.now() - t0).total_seconds() * 1000
    print(f"[variance:raptor] rides={max_rides} speed={speed} {ms:.0f}ms "
          f"-> {len(out['variance'])} cells")
    # Pick the JSON-able keys explicitly: the MC entry also carries the internal alt-route
    # plumbing (alt_bundle's numpy arrays, the lazy JourneyTree cache) that backs /itinerary's
    # drawn alternatives — never serialize those.
    t_assembly = time.perf_counter() if perf is not None else None
    body = {"dest": [dlat, dlon], "variance": out["variance"], "ms": round(ms)}
    if RAPTOR_SEMANTIC == "arriveby":
        # arrive-by: the committed realistic IS the headline typical -> serve it (byte-unchanged).
        body["realistic"] = out["realistic"]
    # depart-after: NO realistic — the headline typical is the bare p50 the map paints (cells[c][1]).
    if perf is None:
        return jsonify(body)
    perf["variance.response_assembly_ms"] = round((time.perf_counter() - t_assembly) * 1000.0, 3)
    perf["variance.request_ms"] = round((dt.datetime.now() - t0).total_seconds() * 1000.0, 3)
    return _phase_response(body, perf)


@app.route("/geocode")
@limiter.limit("60/minute")
def _geocode():
    q = request.args.get("q", "")
    if not q.strip():
        return jsonify({"error": "empty"}), 400
    try:
        lat, lon, label = geo.geocode(q, cache=False)   # don't pollute the dest cache
    except LookupError:
        return jsonify({"error": "not found"}), 404
    except (OSError, ValueError, KeyError):
        return jsonify({"error": "geocoding failed"}), 502
    # Providers are SF-biased but not authoritative.  A valid upstream hit can still be in
    # Oakland, Berkeley, water, or otherwise beyond the graph connector policy; reject it before
    # the browser can submit it to /compute.  Malformed/nonfinite provider coordinates retain the
    # API's ordinary bad-request contract rather than becoming a routing failure.
    try:
        lat, lon = _validate_geocoder_point(lat, lon)
    except _OutsideSupportedArea as e:
        return _outside_supported_area(e)
    except _BadRequest as e:
        return _bad_request(e)
    return jsonify({"lat": lat, "lon": lon, "label": label})


@app.route("/autocomplete")
@limiter.limit("120/minute")
def _autocomplete():
    """Type-ahead suggestions for the workplace box. {"results":[{"label","lat","lon"}, ...]}
    with at most 6 entries. Blank/too-short q -> {"results":[]} (don't hit the geocoder for a
    single character). Upstream failures degrade to an empty list rather than an error so the
    box stays usable."""
    q = request.args.get("q", "")
    if len(q.strip()) < 2:
        return jsonify({"results": []})
    try:
        results = geo.autocomplete(q, limit=6)
    except (OSError, ValueError, KeyError):
        return jsonify({"results": []})
    # Keep upstream failures' existing empty-list behavior, but never present a suggestion that
    # the coordinate-taking endpoints would reject.  Filter in response order and preserve the
    # provider's labels/metadata for accepted entries.
    supported = []
    for result in results or ():
        try:
            _validate_geocoder_point(result["lat"], result["lon"])
        except (_BadRequest, _OutsideSupportedArea, KeyError, TypeError):
            continue
        supported.append(result)
        if len(supported) >= 6:
            break
    return jsonify({"results": supported})


# ---- Page (built once at boot from the template + shared viz.js) -----------------------
def _build_page():
    html = (HERE / "templates" / "index.html").read_text()
    viz = (HERE / "assets" / "viz.js").read_text()
    _arriveby = RAPTOR_SEMANTIC == "arriveby"
    _departafter = RAPTOR_SEMANTIC == "departafter"
    cfg = {"raptor": True, "arriveby": _arriveby, "departafter": _departafter,
           "speedtoggle": True,
           "timephrase": ("arriving by ~9:00am" if _arriveby
                          else (f"leaving ~{DEP:%-I:%M%p} — typical door-to-door".lower()
                                if _departafter else f"leaving ~{DEP:%-I:%M%p}".lower()))}
    # No default workplace is EVER injected into the page (privacy invariant): the .env
    # DEFAULT_ADDRESS / DEST_LAT/LON would otherwise put a real personal address into the
    # served HTML. A first-time visitor sees the onboarding prompt and types their own; a
    # returning visitor's own previously-typed workplace is restored from localStorage by
    # the frontend. DEFAULT_ADDRESS in .env/sfci.env no longer affects the public page.
    cfg["default_wp"] = None

    def _js(o):
        # json.dumps does NOT escape "<", so a "</script>" (or "<!--") inside third-party data
        # (geocoder label, GTFS route names, DataSF neighborhood names) would terminate the
        # inline <script> and inject markup. "<" is a valid JSON escape, so the parsed
        # values are byte-identical — only the embedding is hardened.
        return json.dumps(o).replace("<", "\\u003c")

    # __CFG__ is replaced LAST: it carries the only live string (the geocoder label), so if
    # that ever contained a literal "__CELLS__"/"__LINES__" token it stays literal instead of
    # expanding into the multi-MB JSON blobs.
    return (html.replace("/*__VIZ__*/", viz)
                .replace("__CELLS__", _js(CELLS_GEOJSON))
                .replace("__LINES__", _js(LINES))
                .replace("__CFG__", _js(cfg)))


PAGE_HTML = _build_page()


@app.route("/")
def _index():
    return PAGE_HTML


def _safe_readiness_payload():
    """Serialize only the immutable, API-safe readiness fields.

    The direct validators already redact parser exceptions and paths.  Keep a second narrow
    boundary here because the cached result is also an intentional test-injection seam.
    """
    result = _READINESS
    reason_code = getattr(result, "reason_code", "runtime_load_failed")
    if reason_code not in readiness.REASON_CODES:
        reason_code = "runtime_load_failed"
    body = {"ok": bool(getattr(result, "ready", False)), "reason_code": reason_code}
    body.update({
        "engine": "raptor",
        "semantic": RAPTOR_SEMANTIC,
        "walk": "graph",
        "uptime_s": round(max(0.0, time.monotonic() - _BOOT_MONOTONIC), 3),
    })
    value = getattr(result, "service_date", None)
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        body["service_date"] = value
        body["svc_date"] = dt.datetime.strptime(value, "%Y%m%d").date().isoformat()
    detail = getattr(result, "detail", None)
    if detail in readiness.DEFAULT_REQUIRED_FEEDS:
        body["detail"] = detail
    # Benchmark-only aggregate telemetry. This remains opt-in and is available only when the
    # runtime is actually ready; it contains aggregate counts only, never paths or cache keys.
    if body["ok"] and _perf_benchmark_enabled():
        body["benchmark"] = {
            "pid": os.getpid(),
            "rss_bytes": _benchmark_rss_bytes(),
            "cache_counts": _benchmark_cache_counts(),
            "cache_memory": _benchmark_cache_memory(),
        }
    return body


@app.route("/livez", methods=["GET", "HEAD"], provide_automatic_options=False)
def _livez():
    """Cheap process liveness; never reads feeds, artifacts, or runtime caches."""
    return jsonify({"ok": True, "reason_code": "process_alive"})


@app.route("/readyz", methods=["GET", "HEAD"], provide_automatic_options=False)
def _readyz():
    body = _safe_readiness_payload()
    return jsonify(body), (200 if body["ok"] else 503)


@app.route("/healthz", methods=["GET", "HEAD"], provide_automatic_options=False)
def _healthz():
    """Compatibility alias for /readyz; both paths expose the exact same readiness payload."""
    return _readyz()


@app.errorhandler(405)
def _method_not_allowed(e):
    if request.path in {"/livez", "/readyz", "/healthz"}:
        return jsonify({"ok": False, "reason_code": "method_not_allowed"}), 405
    return e


def _benchmark_lru_values(cache):
    """Snapshot an internal BoundedLRU's values for opt-in aggregate telemetry only.

    The cache lock makes the snapshot internally consistent. Values themselves are not copied:
    callers immediately count nested containers while the public health payload receives only
    integers. This helper must never return keys because cache keys contain workplace buckets.
    """
    with cache.lock:
        return list(cache._od.values())


def _benchmark_cache_counts():
    """Aggregate cache occupancy for PERF_BENCHMARK_STATS; never expose cache keys or values."""
    tree_entries = _benchmark_lru_values(sr._RAPTOR_TREE_CACHE)
    mc_entries = _benchmark_lru_values(sr._RAPTOR_MC_CACHE)

    def _nested_count(entries, name):
        return sum(len((entry or {}).get(name) or {}) for entry in entries)

    return {
        "raptor_egress_workplaces": len(sr._RAPTOR_EGRESS_CACHE),
        "raptor_tree_workplaces": len(tree_entries),
        "raptor_mc_workplaces": len(mc_entries),
        "walk_path_workplaces": len(sr._WALKPATH_TREE_CACHE),
        "egress_payload_bytes": sr._RAPTOR_EGRESS_CACHE.nbytes,
        "walk_path_payload_bytes": sr._WALKPATH_TREE_CACHE.nbytes,
        "route_geometry_cells": _nested_count(tree_entries, "geom"),
        "best_route_geometry_cells": _nested_count(tree_entries, "geom5"),
        "planned_branch_cells": _nested_count(tree_entries, "branch_geom"),
        # Depart-after trees populate their child-deadline mapping lazily.  Take each tree's
        # own snapshot instead of iterating the live mapping while a pin request extends it.
        "deadline_trees": sum(len(_benchmark_deadline_tree_snapshot(
            (entry or {}).get("tree"))) for entry in tree_entries),
        "mc_alt_geometry_cells": _nested_count(mc_entries, "alt_geom"),
        "mc_typical_cells": _nested_count(mc_entries, "typ"),
    }


def _benchmark_cell_cache_snapshot(entries, name, lock):
    """Snapshot nested ``BoundedCellCache`` payload roots without measuring under ``lock``."""
    snapshots = {}
    cache_instances = entries_count = max_entries = 0
    with lock:
        for entry in entries:
            cache = (entry or {}).get(name)
            if cache is None:
                continue
            values = dict(cache)
            snapshots[id(cache)] = values
            cache_instances += 1
            entries_count += len(values)
            max_entries += int(getattr(cache, "maxsize", len(values)))
    return snapshots, {
        "cache_instances": cache_instances,
        "entries": entries_count,
        "max_entries": max_entries,
        "max_bytes": None,                         # shadow phase: no nested byte cap is enforced
    }


def _benchmark_deadline_tree_snapshot(tree):
    """Return a lock-consistent tuple of a JourneyTree's lazy deadline children.

    This deliberately retains values only: cache keys can encode a workplace and must never
    reach benchmark telemetry.  A few lightweight test doubles do not expose the private lock;
    for those, a best-effort tuple is still safer than leaking a live mapping into the estimator.
    """
    if tree is None:
        return ()
    trees = getattr(tree, "_trees", None)
    if trees is None:
        return ()
    trees_lock = getattr(tree, "_trees_lock", None)
    try:
        if trees_lock is not None:
            with trees_lock:
                return tuple(trees.values())
        return tuple(trees.values())
    except RuntimeError:
        # Health telemetry is best-effort; a concurrent mutation can be measured on the next
        # poll without holding a request-owned tree lock across a deep traversal.
        return ()


def _benchmark_tree_shadow(tree, memo):
    """Stable-ish attribute snapshot for a JourneyTree used only by shadow byte telemetry.

    Depart-after deadline trees mutate behind their own lock. Copy that mapping's values while the
    lock is held, then traverse the ordinary snapshot after release. Other lazy tree fields are
    assigned atomically; a health poll may observe either the before or after value, both safe.
    """
    if tree is None:
        return None
    ident = id(tree)
    if ident in memo:
        return memo[ident]
    try:
        shadow = dict(vars(tree))
    except (TypeError, RuntimeError):
        return tree
    memo[ident] = shadow
    # Do not leave the live deadline mapping inside the outer shadow.  It mutates behind a
    # per-tree lock, whereas the expensive ownership traversal deliberately happens unlocked.
    if getattr(tree, "_trees", None) is not None:
        shadow["_trees"] = _benchmark_deadline_tree_snapshot(tree)
    return shadow


def _benchmark_outer_shadow(entries, nested, *, trees=False):
    """Replace live nested mappings/tree LRUs with snapshots before a deep estimate."""
    out = []
    tree_memo = {}
    for entry in entries:
        shadow = dict(entry or {})
        for name, cache_snapshots in nested.items():
            cache = shadow.get(name)
            if cache is not None:
                shadow[name] = cache_snapshots.get(id(cache), ())
        if trees:
            shadow["tree"] = _benchmark_tree_shadow(shadow.get("tree"), tree_memo)
            shadow["tree5"] = _benchmark_tree_shadow(shadow.get("tree5"), tree_memo)
        out.append(shadow)
    return tuple(out)


def _benchmark_cache_memory():
    """Opt-in shadow byte accounting for request-owned RAPTOR cache payloads.

    This deliberately does not change eviction. It supplies the evidence needed to choose safe
    byte caps later: outer tree/MC estimates include their nested payloads; nested categories are
    also broken out so lazy geometry/pin growth is visible. All figures are aggregate integers.
    """
    tree_entries = _benchmark_lru_values(sr._RAPTOR_TREE_CACHE)
    mc_entries = _benchmark_lru_values(sr._RAPTOR_MC_CACHE)
    borrowed = sr._benchmark_borrowed_root_ids()

    tree_nested = {}
    tree_nested_meta = {}
    for name, lock in (("geom", sr._GEOM_LOCK), ("geom5", sr._GEOM_LOCK),
                       ("branch_geom", sr._ALT_LOCK)):
        snapshots, meta = _benchmark_cell_cache_snapshot(tree_entries, name, lock)
        tree_nested[name] = snapshots
        tree_nested_meta[name] = meta
    mc_nested = {}
    mc_nested_meta = {}
    for name, lock in (("alt_geom", sr._ALT_LOCK), ("typ", sr._TYP_LOCK)):
        snapshots, meta = _benchmark_cell_cache_snapshot(mc_entries, name, lock)
        mc_nested[name] = snapshots
        mc_nested_meta[name] = meta

    tree_shadow = _benchmark_outer_shadow(tree_entries, tree_nested, trees=True)
    mc_shadow = _benchmark_outer_shadow(mc_entries, mc_nested)
    tree_bytes = sr._owned_payload_nbytes(tree_shadow, borrowed_root_ids=borrowed)
    mc_bytes = sr._owned_payload_nbytes(mc_shadow, borrowed_root_ids=borrowed)

    for name, snapshots in tree_nested.items():
        roots = tuple(snapshots.values())
        tree_nested_meta[name]["estimated_owned_bytes"] = sr._owned_payload_nbytes(
            roots, borrowed_root_ids=borrowed)
    for name, snapshots in mc_nested.items():
        roots = tuple(snapshots.values())
        mc_nested_meta[name]["estimated_owned_bytes"] = sr._owned_payload_nbytes(
            roots, borrowed_root_ids=borrowed)

    with sr._MC_SCENARIO_LOCK:
        active = sr._MC_SCENARIO_ACTIVE
        scenario = active[2] if active is not None else None
    scenario_bytes = sr._owned_payload_nbytes(scenario, borrowed_root_ids=borrowed)

    # These roots are retained by independent caches, not just bookkeeping weights.  Snapshot
    # their stored values before measuring so a single aggregate can identity-de-duplicate aliases
    # shared with RAPTOR entries or one another.  The LRU helper never exposes cache keys.
    egress_values = _benchmark_lru_values(sr._RAPTOR_EGRESS_CACHE)
    workplace_walk_values = _benchmark_lru_values(sr._WALKPATH_TREE_CACHE)
    cell_walk_values = _benchmark_lru_values(sr._CELL_WALKPATH_TREE_CACHE)

    retained_roots = (
        tree_shadow, mc_shadow, scenario,
        egress_values, workplace_walk_values, cell_walk_values,
    )
    request_owned_retained_bytes = sr._owned_payload_nbytes(
        retained_roots, borrowed_root_ids=borrowed)

    def weighted(cache, values):
        return {
            "entries": len(values),
            "max_entries": int(cache.maxsize),
            "accounted_payload_bytes": int(cache.nbytes),
            "max_bytes": cache.maxbytes,
            "estimated_owned_bytes": sr._owned_payload_nbytes(
                values, borrowed_root_ids=borrowed),
        }

    return {
        # This is the only additive retained-memory number.  Every category below is diagnostic
        # attribution and can share object identities (for example, an egress/path object can be
        # reachable through a tree), so consumers must not sum category estimates.
        "request_owned_retained_bytes": request_owned_retained_bytes,
        "category_estimates_non_additive": 1,
        "raptor_tree": {
            "entries": len(tree_entries),
            "max_entries": int(sr._RAPTOR_TREE_CACHE.maxsize),
            "estimated_owned_bytes": tree_bytes,
            "max_bytes": sr._RAPTOR_TREE_CACHE.maxbytes,
            "nested": tree_nested_meta,
        },
        "raptor_mc": {
            "entries": len(mc_entries),
            "max_entries": int(sr._RAPTOR_MC_CACHE.maxsize),
            "estimated_owned_bytes": mc_bytes,
            "max_bytes": sr._RAPTOR_MC_CACHE.maxbytes,
            "nested": mc_nested_meta,
        },
        "mc_scenario": {
            "retained": scenario is not None,
            "estimated_owned_bytes": scenario_bytes,
            "max_bytes": int(sr._MC_SCENARIO_MAX_BYTES),
        },
        "known_weighted": {
            "raptor_egress": weighted(sr._RAPTOR_EGRESS_CACHE, egress_values),
            "workplace_walk_path": weighted(sr._WALKPATH_TREE_CACHE, workplace_walk_values),
            "cell_walk_path": weighted(sr._CELL_WALKPATH_TREE_CACHE, cell_walk_values),
        },
        # Tree/MC estimates include their nested payloads. Scenario is retained out-of-band and
        # therefore added separately. Weighted-cache totals stay separate to avoid implying that
        # this first shadow slice is already a global enforced budget.
        "estimated_tree_mc_scenario_bytes": tree_bytes + mc_bytes + scenario_bytes,
    }


def _benchmark_rss_bytes():
    """Best-effort current resident bytes for opt-in benchmark telemetry."""
    try:                                             # Linux: current RSS, no subprocess/dependency
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:                                             # macOS and other POSIX hosts
        import subprocess
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return int(out) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


if __name__ == "__main__":
    # In production, prefer waitress (waitress-serve) over Flask's dev server: clean SIGTERM,
    # no "WARNING: development server" noise, no surprise per-request thread spawn. Falls back
    # to Flask's threaded dev server if waitress isn't installed (so local dev still works).
    PORT = int(os.environ.get("PORT", "8000"))
    try:
        from waitress import serve
        print(f"[boot] waitress serving on 127.0.0.1:{PORT}")
        serve(app, host="127.0.0.1", port=PORT, threads=8, _quiet=False, channel_timeout=120)
    except ImportError:
        print(f"[boot] waitress not installed — falling back to Flask dev server on 127.0.0.1:{PORT}")
        app.run(host="127.0.0.1", port=PORT, threaded=True)
