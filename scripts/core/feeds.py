"""GTFS feed helpers: route-name resolution, route shapes, and service-date picking.

No r5py here, so importing this does not start the JVM.
"""
import io
import zipfile
import datetime as dt
from pathlib import Path

# pandas is imported LAZILY (it's ~44 MB): the JVM-free server precomputes this module's outputs
# (line shapes + service date) into a static artifact and never calls these functions at runtime,
# so importing `feeds` must not pull pandas. Call _pd() at the top of any function that needs it.
pd = None


def _pd():
    global pd
    if pd is None:
        import pandas as _p
        pd = _p
    return pd


# route_type -> overlay mode bucket (Caltrain is route_type 2 = rail, drawn as "bart")
_MODE = {"0": "metro", "1": "bart", "2": "bart", "5": "cable", "3": "bus"}


def _route_display_name(short, long_, rid):
    """short -> long -> id, the ONE naming policy (load_routes + route_shapes).
    Treats None/NaN/blank as missing: pandas leaves blank cells as float NaN even
    under dtype=str, and NaN is TRUTHY, so a bare or-chain would name a route "nan"."""
    for v in (short, long_):
        if v is None or (isinstance(v, float) and v != v):  # None / NaN
            continue
        s = str(v).strip()
        if s:
            return s
    return str(rid)


def load_routes(gtfs_paths):
    """{feed_stem: {route_id: name}} so colliding ids across feeds (Muni '8' vs
    BART '8'=Red-N) resolve correctly via the leg's own feed."""
    _pd()
    m = {}
    for p in gtfs_paths:
        d = {}
        with zipfile.ZipFile(p) as z:
            r = pd.read_csv(io.BytesIO(z.read("routes.txt")), dtype=str)
            for _, row in r.iterrows():
                rid = str(row["route_id"])
                d[rid] = _route_display_name(row.get("route_short_name"),
                                             row.get("route_long_name"), rid)
        m[Path(p).stem] = d
    return m


def route_name(route_id, feed, routes):
    """Feed-aware route name. `routes` is the map from load_routes(). Normalizes the
    float-ish ids r5py hands back ('8.0' -> '8') and returns None for missing ids."""
    if route_id is None or (isinstance(route_id, float) and route_id != route_id):  # None / NaN
        return None
    try:
        key = str(int(float(route_id)))
    except (ValueError, TypeError):
        key = str(route_id)
    return routes.get(str(feed), {}).get(key, key)


def route_shapes(gtfs_paths):
    """One representative (longest) shape per route, classified by mode, as a GeoJSON
    FeatureCollection — used for the transit-line overlay."""
    _pd()
    feats = []
    for p in gtfs_paths:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if "shapes.txt" not in names:
                continue
            shapes = pd.read_csv(io.BytesIO(z.read("shapes.txt")), dtype={"shape_id": str})
            trips = pd.read_csv(io.BytesIO(z.read("trips.txt")), dtype=str).dropna(subset=["shape_id"])
            routes = pd.read_csv(io.BytesIO(z.read("routes.txt")), dtype=str)
            rtype = dict(zip(routes.route_id, routes.route_type))
            # Missing columns fall through as None (NOT route_id) so a long-names-only
            # feed still gets its long names via _route_display_name.
            n = len(routes)
            short = routes["route_short_name"] if "route_short_name" in routes.columns else [None] * n
            long_ = routes["route_long_name"] if "route_long_name" in routes.columns else [None] * n
            rname = {rid: _route_display_name(s, l, rid)
                     for rid, s, l in zip(routes["route_id"], short, long_)}
            cnt = shapes.groupby("shape_id").size()
            best = {}
            for sid, rid in zip(trips.shape_id, trips.route_id):
                if rid not in best or cnt.get(sid, 0) > cnt.get(best[rid], 0):
                    best[rid] = sid
            for rid, sid in best.items():
                pts = shapes[shapes.shape_id == sid].sort_values("shape_pt_sequence")
                coords = pts[["shape_pt_lon", "shape_pt_lat"]].astype(float).values.tolist()
                if len(coords) < 2:
                    continue
                feats.append({"type": "Feature",
                              "geometry": {"type": "LineString", "coordinates": coords},
                              "properties": {"mode": _MODE.get(str(rtype.get(rid, "3")), "bus"),
                                             "name": str(rname.get(rid, rid))}})
    return {"type": "FeatureCollection", "features": feats}


def active_service_ids(z, names, date):
    """service_ids running on YYYYMMDD ``date`` per calendar.txt + calendar_dates.txt.

    ``z`` is an open ZipFile, ``names`` its namelist(). The ONE place this GTFS calendar
    logic lives — used by _feed_has_trips here and by raptor_build.build()."""
    _pd()
    wd = dt.datetime.strptime(date, "%Y%m%d").strftime("%A").lower()
    sids = set()
    if "calendar.txt" in names:
        cal = pd.read_csv(io.BytesIO(z.read("calendar.txt")), dtype=str)
        m = cal[(cal[wd] == "1") & (cal["start_date"] <= date) & (cal["end_date"] >= date)]
        sids = set(m["service_id"])
    if "calendar_dates.txt" in names:
        cd = pd.read_csv(io.BytesIO(z.read("calendar_dates.txt")), dtype=str)
        sids |= set(cd[(cd["date"] == date) & (cd["exception_type"] == "1")]["service_id"])
        sids -= set(cd[(cd["date"] == date) & (cd["exception_type"] == "2")]["service_id"])
    return sids


def _feed_has_trips(path, date):
    """True if feed `path` actually runs trips on YYYYMMDD `date`."""
    _pd()
    with zipfile.ZipFile(path) as z:
        sids = active_service_ids(z, z.namelist(), date)
        if not sids:
            return False
        trips = pd.read_csv(io.BytesIO(z.read("trips.txt")), dtype=str)
        return bool(trips["service_id"].isin(sids).any())


def pick_service_date(gtfs_paths):
    """First Wednesday on which EVERY feed actually runs trips (verified, in range).

    GTFS validity windows are short (Muni's are ~4-week signups), so a hardcoded date
    silently yields no service after a data repull. This walks the common calendar window
    and verifies trips per feed instead."""
    _pd()
    starts, ends = [], []
    for p in gtfs_paths:
        with zipfile.ZipFile(p) as z:
            if "calendar.txt" not in z.namelist():
                continue  # calendar_dates-only feed: still verified per-date by _feed_has_trips
            cal = pd.read_csv(io.BytesIO(z.read("calendar.txt")), dtype=str)
            w = cal[cal["wednesday"] == "1"]
            if len(w):
                starts.append(int(w["start_date"].min()))
                ends.append(int(w["end_date"].max()))
    if not starts:
        raise RuntimeError("no feed contributes Wednesday calendar.txt rows: "
                           + ", ".join(Path(p).name for p in gtfs_paths))
    lo = dt.datetime.strptime(str(max(starts)), "%Y%m%d").date()
    hi = dt.datetime.strptime(str(min(ends)), "%Y%m%d").date()
    d = lo + dt.timedelta(days=7)              # a week in, past signup-boundary swaps
    while d.weekday() != 2:                    # 2 == Wednesday
        d += dt.timedelta(days=1)
    while d <= hi:
        ds = d.strftime("%Y%m%d")
        if all(_feed_has_trips(p, ds) for p in gtfs_paths):
            return d
        d += dt.timedelta(days=7)
    raise RuntimeError("No common service Wednesday found across feeds "
                       "(window %s..%s)" % (lo, hi))
