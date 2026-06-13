"""Paths, the canonical commute model, and feed locations — the single definition.

Every script reads its constants from here, so there is exactly ONE GTFS feed list,
ONE departure window, ONE walk speed, etc. This is what makes the offline analyses
(isochrone.py, route_map.py, ...) actually comparable to the live server instead of
each re-deriving slightly different values.
"""
import os
import datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # scripts/core/config.py -> repo root
DATA = ROOT / "data"
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
WALK_KMH = 4.8             # walking speed (r5py's 3.6 default is too slow)
# Walk-speed toggle presets (the server's ?speed= param; the RAPTOR engine applies the
# scalar WALK_KMH / WALK_SPEEDS[speed] to its reference-second walk tables). 'med' IS the
# canonical WALK_KMH (scalar 1.0), so the baked access/egress tables need no rescaling at
# the default.
WALK_SPEEDS = {"slow": 4.0, "med": WALK_KMH, "fast": 5.6}   # km/h
DEFAULT_SPEED = "med"

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
# Transit feeds, in network order: Muni (current 511 feed, else a stale 2022 fallback)
# + BART + Caltrain.
MUNI_CURRENT = "muni_current.zip"
MUNI_FALLBACK = "muni_gtfs_2022_fallback.zip"
BART = "bart_gtfs.zip"
CALTRAIN = "caltrain.zip"


def osm_path() -> Path:
    return DATA / OSM_FILE


def neigh_path() -> Path:
    return DATA / NEIGH_FILE


def gtfs_paths(extra=None):
    """The transit feeds to route on, as existing Paths in network order.

    Muni resolves to the current 511 feed, or the stale 2022 fallback with a warning if
    the current one hasn't been fetched. Non-existent feeds are dropped so a partial data
    download runs (degraded) rather than crashing at network build. `extra` adds caller
    feeds (names resolve under data/, absolute paths pass through)."""
    muni = DATA / MUNI_CURRENT
    if not muni.exists():
        fb = DATA / MUNI_FALLBACK
        if fb.exists():
            print(f"!! WARNING: using STALE Muni feed {fb.name} (no current feed found)")
            muni = fb
    paths = [muni, DATA / BART, DATA / CALTRAIN]
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
