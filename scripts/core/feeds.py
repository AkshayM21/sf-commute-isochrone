"""GTFS feed helpers: route-name resolution, route shapes, and service-date picking.

No r5py here, so importing this does not start the JVM.
"""
import io
import zipfile
import datetime as dt
from pathlib import Path
import pandas as pd

# route_type -> overlay mode bucket (Caltrain is route_type 2 = rail, drawn as "bart")
_MODE = {"0": "metro", "1": "bart", "2": "bart", "5": "cable", "3": "bus"}


def load_routes(gtfs_paths):
    """{feed_stem: {route_id: name}} so colliding ids across feeds (Muni '8' vs
    BART '8'=Red-N) resolve correctly via the leg's own feed."""
    m = {}
    for p in gtfs_paths:
        d = {}
        with zipfile.ZipFile(p) as z:
            r = pd.read_csv(io.BytesIO(z.read("routes.txt")), dtype=str)
            for _, row in r.iterrows():
                name = (row.get("route_short_name") or row.get("route_long_name")
                        or row.get("route_id"))
                d[str(row["route_id"])] = str(name).strip()
        m[Path(p).stem] = d
    return m


def route_name(route_id, feed, routes):
    """Feed-aware route name. `routes` is the map from load_routes(). Normalizes the
    float-ish ids r5py hands back ('8.0' -> '8') and returns None for missing ids."""
    if pd.isna(route_id):
        return None
    try:
        key = str(int(float(route_id)))
    except (ValueError, TypeError):
        key = str(route_id)
    return routes.get(str(feed), {}).get(key, key)


def route_shapes(gtfs_paths):
    """One representative (longest) shape per route, classified by mode, as a GeoJSON
    FeatureCollection — used for the transit-line overlay."""
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
            rname = dict(zip(routes.route_id,
                             routes.get("route_short_name", routes["route_id"]).fillna(routes["route_id"])))
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


def _feed_has_trips(path, date):
    """True if feed `path` actually runs trips on YYYYMMDD `date`."""
    wd = dt.datetime.strptime(date, "%Y%m%d").strftime("%A").lower()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        sids = set()
        if "calendar.txt" in names:
            cal = pd.read_csv(io.BytesIO(z.read("calendar.txt")), dtype=str)
            m = cal[(cal[wd] == "1") & (cal["start_date"] <= date) & (cal["end_date"] >= date)]
            sids = set(m["service_id"])
        if "calendar_dates.txt" in names:
            cd = pd.read_csv(io.BytesIO(z.read("calendar_dates.txt")), dtype=str)
            sids |= set(cd[(cd["date"] == date) & (cd["exception_type"] == "1")]["service_id"])
            sids -= set(cd[(cd["date"] == date) & (cd["exception_type"] == "2")]["service_id"])
        if not sids:
            return False
        trips = pd.read_csv(io.BytesIO(z.read("trips.txt")), dtype=str)
        return bool(trips["service_id"].isin(sids).any())


def pick_service_date(gtfs_paths):
    """First Wednesday on which EVERY feed actually runs trips (verified, in range).

    GTFS validity windows are short (Muni's are ~4-week signups), so a hardcoded date
    silently yields no service after a data repull. This walks the common calendar window
    and verifies trips per feed instead."""
    starts, ends = [], []
    for p in gtfs_paths:
        with zipfile.ZipFile(p) as z:
            cal = pd.read_csv(io.BytesIO(z.read("calendar.txt")), dtype=str)
            w = cal[cal["wednesday"] == "1"]
            if len(w):
                starts.append(int(w["start_date"].min()))
                ends.append(int(w["end_date"].max()))
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
    raise SystemExit("No common service Wednesday found across feeds")
