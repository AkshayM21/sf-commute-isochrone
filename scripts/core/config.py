"""Paths, the canonical commute model, and feed locations — the single definition.

Every runtime and build helper reads its constants from here, so there is exactly ONE GTFS
feed list, ONE departure window, and ONE walk-speed model shared by the graph-native server.
"""
import os
import datetime as dt
from pathlib import Path


def normalize_mtime_ns(value: int) -> int:
    """Canonicalize non-negative nanosecond mtimes to portable whole-second precision."""
    result = int(value)
    return result if result < 0 else result - result % 1_000_000_000


def portable_mtime_ns(stat_result: os.stat_result) -> int:
    """Return whole-second mtime in nanosecond units for cross-host artifact metadata.

    The deployment path includes rsync protocol 29, which preserves mtimes only to whole seconds.
    Keeping nanosecond units avoids an artifact schema migration while making local and deployed
    metadata compare identically. Size remains the paired direct-source discriminator.
    """
    return normalize_mtime_ns(stat_result.st_mtime_ns)


def normalize_source_mtimes(values) -> tuple:
    """Normalize the mtime member of readable ``(name, size, mtime_ns)`` metadata."""
    return tuple(
        (item[0], item[1], None if item[2] is None else normalize_mtime_ns(item[2]))
        for item in values
    )

ROOT = Path(__file__).resolve().parents[2]   # scripts/core/config.py -> repo root


def _data_root() -> Path:
    """Resolve the runtime data root.

    Normal startup remains exactly ``<repo>/data``.  Refresh/build jobs can point a
    subprocess at an inactive generation with ``SFCI_DATA_DIR``; resolving here keeps
    every downstream path helper on the same tree and makes relative overrides explicit
    (relative to the repository, never to the caller's working directory).
    """
    raw = os.environ.get("SFCI_DATA_DIR")
    if raw is None or not raw.strip():
        return (ROOT / "data").resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError("SFCI_DATA_DIR must not be a filesystem root")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"SFCI_DATA_DIR is not a directory: {resolved}")
    return resolved


DATA = _data_root()
OUT = ROOT / "out"

# Coordinate reference systems
WGS = "EPSG:4326"          # lon/lat
UTM = "EPSG:32610"         # UTM 10N — metric grid for SF

# --- Canonical commute model (these values define the live server's behavior) ---------
GRID_M = 200               # default grid cell size (m); 200 = finest (~3000 cells).
DEP_HM = (8, 35)           # model departure time of day (hour, minute), local
WINDOW_MIN = 30            # departure-time window (min) the percentiles span
MAX_MIN = 75               # routing cap (min) — trips longer than this are unreachable
PERCENTILES = [5, 50]      # 5th = best-case (you time it well), 50th = realistic (median)
# The OSM walk graph and access bake are stored in reference seconds at 4.8 km/h. This is a
# *bake reference*, not the product's default walking pace: the engine rescales every walk leg by
# ``WALK_KMH / WALK_SPEEDS[speed]`` at request time.
WALK_KMH = 4.8
# Product walking presets, calibrated against the bounded SF Google-walking corpus documented in
# WALK_SPEED_CALIBRATION_2026-08-09.md. Medium reflects a typical SF city pace; Fast is genuinely
# brisk rather than merely the graph's bake reference. Fixed values keep results deterministic and
# monotonic (slow time >= medium time >= fast time).
WALK_SPEEDS = {"slow": 3.4, "med": 4.2, "fast": 5.2}   # km/h
DEFAULT_SPEED = "med"

# --- Mild walk-reluctance prior ("transit slightly preferred over walking") -----------
# A multiplier on WALK time in the access-stop DECISION/argmin ONLY — the reported door-to-door
# minutes stay TRUE clock time (the penalty steers the *choice*, the reported number is the chosen
# journey's true clock time). It fixes the user's anomaly ("fast walk -> walk to a FARTHER station
# AT THE SAME TIME") by preferring the closer stop among options that reach work at ~the same time.
#
# Two knobs work together so the prior is a TIE-BREAK, never a multi-minute degradation:
#   * WALK_RELUCTANCE (beta, default 1.15) — weights walk seconds in the within-window ranking.
#   * WALK_PRIOR_EPS_SEC (default 60s) — a HARD CAP: the prior may only re-select among access stops
#     whose TRUE travel time is within this window of the time-optimal stop, so the reported time
#     can change by at most ~1 min (the rounding minute) and a genuinely-faster farther stop (>eps)
#     is NEVER traded away. Without this cap, a raw beta multiplier on the full (<=25 min) access
#     walk could flip a stop up to ~3.75 min slower — over-correcting the user's "slight" preference.
# beta=1.0 reproduces the no-prior behavior exactly. The engine threads both through assemble +
# JourneyTree._select_arrays.
WALK_RELUCTANCE = float(os.environ.get("RAPTOR_WALK_RELUCTANCE", "1.15"))
WALK_PRIOR_EPS_SEC = float(os.environ.get("RAPTOR_WALK_PRIOR_EPS", "60"))

# --- Geographic bounds (minLon, minLat, maxLon, maxLat) -------------------------------
# Two DIFFERENT boxes — don't conflate them:
#   SF_BBOX       — TIGHT box around San Francisco proper; biases/filters the geocoder
#                   (core/geo.py) so providers agree on "SF". scripts/setup.sh (OSM clip)
#                   and scripts/fetch_dem.sh (DEM extent) carry their own slightly
#                   padded/clipped variants of this box — keep them roughly in sync if
#                   this ever changes.
#   SF_VALID_BBOX — LOOSE input-validation box (server _parse_ll): wide enough to accept
#                   near-SF workplaces (Daly City, East Bay edge) the grid can still
#                   route toward, while rejecting garbage coordinates that would burn a
#                   full compute for an unusable result.
SF_BBOX = (-122.55, 37.70, -122.34, 37.84)
SF_VALID_BBOX = (-123.1, 37.3, -122.0, 38.1)

# --- Data files -----------------------------------------------------------------------
OSM_FILE = "osm_sf.pbf"
NEIGH_FILE = "sf_neighborhoods.geojson"
# Transit feeds, in network order: current Muni 511 feed + BART + Caltrain.
MUNI_CURRENT = "muni_current.zip"
BART = "bart_gtfs.zip"
CALTRAIN = "caltrain.zip"


def osm_path() -> Path:
    return DATA / OSM_FILE


def neigh_path() -> Path:
    return DATA / NEIGH_FILE


def gtfs_paths(extra=None, *, replace=False):
    """The transit feeds to route on, as existing Paths in network order.

    Current Muni is mandatory for the production readiness contract. Non-existent feeds
    are dropped here so lightweight feed/build callers can inspect partial input; the
    server readiness check rejects missing required archives. `extra` adds caller feeds
    (names resolve under data/, absolute paths pass through); `replace=True` uses only
    those caller feeds, which is useful for explicit comparison exports."""
    muni = DATA / MUNI_CURRENT
    paths = [] if replace else [muni, DATA / BART, DATA / CALTRAIN]
    if extra:
        paths += [Path(e) if Path(e).is_absolute() else DATA / e for e in extra]
    return [p for p in paths if p.exists()]


def departure(service_date) -> dt.datetime:
    """Model departure datetime: DEP_HM on the given service date."""
    return dt.datetime(service_date.year, service_date.month, service_date.day, *DEP_HM)


def window() -> dt.timedelta:
    return dt.timedelta(minutes=int(os.environ.get("WINDOW_MIN", WINDOW_MIN)))


def load_dotenv():
    """Load KEY=VALUE pairs from the repo-root .env into os.environ without overriding
    values already present in the environment. Ignores blank lines, comments, and an
    optional `export ` prefix; strips surrounding quotes. Inline comments are NOT stripped
    (an address may legitimately contain '#'). The shell scripts parse .env separately."""
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
