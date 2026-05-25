"""GTFS -> in-memory RAPTOR structures (route/pattern-based), JVM-free.

This is the productionized successor to ``prototypes/spike_raptor/gtfs_load.py``. It
parses the canonical feeds (Muni + BART + Caltrain) for one service date into FLAT,
contiguous numpy arrays (CSR-style) so the hot routing loop can run in pure numpy
today and drop into numba/Rust unchanged later (no Python objects in the inner loop).

Data model (standard "trip-pattern" RAPTOR):
  * Global stops: each (feed, gtfs_stop_id) -> a single int ``gid`` in [0, n_stops).
  * Pattern: an ordered stop sequence served the same way by >= 1 trip, with all its
    trips stored as two (ntrips x nstops) time matrices (arr, dep), seconds-after-midnight.
    Trips are sorted by first departure; a pattern with OVERTAKING (a later-departing trip
    that arrives earlier at some stop) is SPLIT into FIFO sub-patterns so the per-position
    binary search used by RAPTOR stays valid.
  * routes_at_stop: which (pattern, position) pairs serve each stop (CSR).
  * transfers: synthesized walk footpaths between stops within ``FOOTPATH_M`` great-circle
    metres (cost = distance / walk speed), bidirectional, spanning ALL feeds (so a Muni ->
    BART street transfer exists). GTFS transfers.txt is intentionally NOT used yet (the
    spike hit MAE 1.0 without it); see NOTES.

Only trips whose service runs on the model date (calendar + calendar_dates exceptions) AND
whose times intersect the morning band are kept. After-midnight trips (HH >= 24) fall
outside the band and are dropped, which is correct for an arrive-by-09:00 model.

The build is deterministic and cached to disk (see ``load_or_build``), keyed by the feed
file fingerprints + service date + band + footpath radius, so the server boots warm.
"""
import io
import os
import time
import pickle
import zipfile
import hashlib
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, feeds

FOOTPATH_M = float(os.environ.get("RAPTOR_FOOTPATH_M", "250"))  # synthesized-transfer radius (m)
BAND_START_H = 5.0           # keep trips active from 05:00 ...
# ... to the latest arrival deadline we will ever sweep: the arrive-by target plus the
# routing cap (a trip boarded right at the cap could still be in service). MAX_MIN past
# the window end is a safe upper bound for the morning model.

CACHE_DIR = config.DATA / "raptor_cache"


def _log(*a):
    print(*a, flush=True)


def _hms_to_sec(s):
    """'HH:MM:SS' (HH may exceed 24 for after-midnight trips) -> int seconds, or -1 if NaN."""
    if pd.isna(s):
        return -1
    p = str(s).split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])


def _active_service_ids(z, names, date):
    """service_ids running on YYYYMMDD ``date`` per calendar.txt + calendar_dates.txt."""
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


def _split_fifo(dep, arr):
    """Split a stop-sequence group's trips into FIFO sub-patterns (no overtaking).

    ``dep``/``arr`` are (ntrips x nstops) int64, already sorted by dep[:,0]. A new trip can
    join an existing sub-pattern only if, at EVERY position, its departure is >= the last
    trip's departure (so the column stays sorted and per-position binary search is valid).
    Returns a list of (dep_sub, arr_sub) arrays. Almost always one group; splits are rare
    (express overtaking a local sharing the identical stop list)."""
    groups = []            # each: [dep_rows list, arr_rows list, last_dep_vector]
    for t in range(dep.shape[0]):
        d, a = dep[t], arr[t]
        placed = False
        for g in groups:
            if np.all(d >= g[2]):          # does not overtake this group's latest trip
                g[0].append(d); g[1].append(a); g[2] = d
                placed = True
                break
        if not placed:
            groups.append([[d], [a], d])
    return [(np.array(g[0], dtype=np.int64), np.array(g[1], dtype=np.int64)) for g in groups]


def build(gtfs_paths, date_str, band_start_sec, band_end_sec, footpath_m=FOOTPATH_M):
    """Parse feeds into flat RAPTOR arrays for ``date_str`` (YYYYMMDD), trips intersecting
    [band_start_sec, band_end_sec]. Returns a dict of numpy arrays (see module docstring)."""
    stop_key_to_gid = {}
    stop_lat, stop_lon = [], []

    def get_gid(feed, sid, lat, lon):
        key = (feed, sid)
        g = stop_key_to_gid.get(key)
        if g is None:
            g = len(stop_lat)
            stop_key_to_gid[key] = g
            stop_lat.append(lat); stop_lon.append(lon)
        return g

    # group trips by identical (feed, stop-sequence); each group -> >=1 FIFO pattern
    seqs = {}                  # (feed, gids tuple) -> list[(dep array, arr array)]
    feed_trip_counts = {}
    for p in gtfs_paths:
        feed = Path(p).stem
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            sids = _active_service_ids(z, names, date_str)
            sdf = pd.read_csv(io.BytesIO(z.read("stops.txt")), dtype=str)
            slat = dict(zip(sdf.stop_id, pd.to_numeric(sdf.stop_lat, errors="coerce")))
            slon = dict(zip(sdf.stop_id, pd.to_numeric(sdf.stop_lon, errors="coerce")))
            trips = pd.read_csv(io.BytesIO(z.read("trips.txt")), dtype=str)
            trips = trips[trips.service_id.isin(sids)]
            active = set(trips.trip_id)
            st = pd.read_csv(io.BytesIO(z.read("stop_times.txt")), dtype=str)
            st = st[st.trip_id.isin(active)].copy()
            st["seq"] = pd.to_numeric(st.stop_sequence, errors="coerce")
            st["arr"] = st.arrival_time.map(_hms_to_sec)
            st["dep"] = st.departure_time.map(_hms_to_sec)
            st.loc[st["arr"] < 0, "arr"] = st["dep"]      # blank intermediate times: use the other
            st.loc[st["dep"] < 0, "dep"] = st["arr"]
            kept = 0
            for tid, g in st.sort_values(["trip_id", "seq"]).groupby("trip_id", sort=False):
                gids = tuple(get_gid(feed, sid, slat.get(sid, np.nan), slon.get(sid, np.nan))
                             for sid in g.stop_id)
                deps = g["dep"].to_numpy(); arrs = g["arr"].to_numpy()
                if deps[0] > band_end_sec or arrs[-1] < band_start_sec:
                    continue                              # outside the morning band
                seqs.setdefault((feed, gids), []).append((deps, arrs))
                kept += 1
            feed_trip_counts[feed] = kept
            _log(f"  {feed}: {len(sids)} services, {kept} trips in band")

    n_stops = len(stop_lat)
    stop_lat = np.asarray(stop_lat, dtype=np.float64)
    stop_lon = np.asarray(stop_lon, dtype=np.float64)

    # ---- finalize patterns (FIFO-split), then flatten to CSR arrays
    P_stops, P_dep, P_arr = [], [], []        # ragged, per pattern
    n_split = 0
    for (feed, gids), trips in seqs.items():
        dep = np.array([t[0] for t in trips], dtype=np.int64)
        arr = np.array([t[1] for t in trips], dtype=np.int64)
        order = np.argsort(dep[:, 0], kind="stable")
        dep, arr = dep[order], arr[order]
        subs = _split_fifo(dep, arr)
        if len(subs) > 1:
            n_split += len(subs) - 1
        garr = np.asarray(gids, dtype=np.int32)
        for dsub, asub in subs:
            P_stops.append(garr); P_dep.append(dsub); P_arr.append(asub)
    n_pat = len(P_stops)
    if n_split:
        _log(f"  overtaking: split into {n_split} extra FIFO sub-patterns")

    # CSR for pattern stops + (trip x stop) time matrices
    pat_nstops = np.array([len(s) for s in P_stops], dtype=np.int32)
    pat_ntrips = np.array([d.shape[0] for d in P_dep], dtype=np.int32)
    pat_stop_off = np.zeros(n_pat + 1, dtype=np.int64)
    pat_stop_off[1:] = np.cumsum(pat_nstops)
    pat_mat_off = np.zeros(n_pat + 1, dtype=np.int64)       # base into the flat dep/arr arrays
    pat_mat_off[1:] = np.cumsum(pat_ntrips.astype(np.int64) * pat_nstops.astype(np.int64))
    pat_stops = np.concatenate(P_stops).astype(np.int32) if n_pat else np.zeros(0, np.int32)
    pat_dep = (np.concatenate([d.ravel() for d in P_dep]).astype(np.int32)
               if n_pat else np.zeros(0, np.int32))
    pat_arr = (np.concatenate([a.ravel() for a in P_arr]).astype(np.int32)
               if n_pat else np.zeros(0, np.int32))

    # routes_at_stop CSR: stop -> list of (pattern, position)
    counts = np.zeros(n_stops, dtype=np.int64)
    for s in pat_stops:
        counts[s] += 1
    ras_off = np.zeros(n_stops + 1, dtype=np.int64)
    ras_off[1:] = np.cumsum(counts)
    ras_pat = np.empty(int(ras_off[-1]), dtype=np.int32)
    ras_pos = np.empty(int(ras_off[-1]), dtype=np.int32)
    cur = ras_off[:-1].copy()
    for pi in range(n_pat):
        base = pat_stop_off[pi]
        for pos in range(pat_nstops[pi]):
            s = pat_stops[base + pos]
            j = cur[s]; ras_pat[j] = pi; ras_pos[j] = pos; cur[s] = j + 1

    # transfers: synthesized footpaths within footpath_m, bidirectional, all feeds
    tr_off, tr_to, tr_time = _footpaths(stop_lat, stop_lon, footpath_m)

    _log(f"stops {n_stops}  patterns {n_pat}  ras {len(ras_pat)}  footpaths {len(tr_to)}")
    return dict(
        n_stops=n_stops, stop_lat=stop_lat, stop_lon=stop_lon,
        pat_nstops=pat_nstops, pat_ntrips=pat_ntrips,
        pat_stop_off=pat_stop_off, pat_mat_off=pat_mat_off,
        pat_stops=pat_stops, pat_dep=pat_dep, pat_arr=pat_arr,
        ras_off=ras_off, ras_pat=ras_pat, ras_pos=ras_pos,
        tr_off=tr_off, tr_to=tr_to, tr_time=tr_time,
        feed_trip_counts=feed_trip_counts, date=date_str,
        band=(band_start_sec, band_end_sec), footpath_m=footpath_m,
    )


def _footpaths(stop_lat, stop_lon, footpath_m):
    """Bidirectional walk footpaths between stops within ``footpath_m`` (great-circle),
    as CSR (tr_off, tr_to, tr_time[sec]). Grid-bucketed to avoid O(n^2)."""
    n = len(stop_lat)
    walk_mps = config.WALK_KMH * 1000.0 / 3600.0
    lat0 = float(np.nanmean(stop_lat)) if n else 0.0
    mlat, mlon = 111320.0, 111320.0 * np.cos(np.radians(lat0))
    xs, ys = stop_lon * mlon, stop_lat * mlat
    cell = footpath_m
    buckets = {}
    for i in range(n):
        if np.isnan(xs[i]):
            continue
        buckets.setdefault((int(xs[i] // cell), int(ys[i] // cell)), []).append(i)
    rows = [[] for _ in range(n)]
    for i in range(n):
        if np.isnan(xs[i]):
            continue
        bx, by = int(xs[i] // cell), int(ys[i] // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((bx + dx, by + dy), ()):
                    if j == i or np.isnan(xs[j]):
                        continue
                    d = ((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) ** 0.5
                    if d <= footpath_m:
                        rows[i].append((j, d / walk_mps))
    tr_off = np.zeros(n + 1, dtype=np.int64)
    tr_off[1:] = np.cumsum([len(r) for r in rows])
    tot = int(tr_off[-1])
    tr_to = np.empty(tot, dtype=np.int32)
    tr_time = np.empty(tot, dtype=np.int32)
    k = 0
    for r in rows:
        for (j, w) in r:
            tr_to[k] = j; tr_time[k] = int(round(w)); k += 1
    return tr_off, tr_to, tr_time


# ----------------------------------------------------------------- disk cache
def band_seconds():
    """(band_start, band_end) seconds for the canonical morning model."""
    start = int(BAND_START_H * 3600)
    end = (config.DEP_HM[0] * 3600 + config.DEP_HM[1] * 60
           + int(config.window().total_seconds()) + config.MAX_MIN * 60)
    return start, end


def _fingerprint(gtfs_paths, date_str, band, footpath_m):
    h = hashlib.sha256()
    for p in gtfs_paths:
        st = Path(p).stat()
        h.update(f"{Path(p).name}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(f"{date_str}:{band}:{footpath_m}".encode())
    return h.hexdigest()[:16]


def load_or_build(gtfs_paths=None, service_date=None, footpath_m=FOOTPATH_M, verbose=True):
    """Return RAPTOR structures for the canonical model, building + caching on first use.

    The cache key folds in each feed's size+mtime, so a GTFS repull (or a footpath-radius
    change) invalidates it automatically. Cheap to rebuild (~1-2s, no JVM)."""
    gtfs_paths = gtfs_paths or config.gtfs_paths()
    service_date = service_date or feeds.pick_service_date(gtfs_paths)
    date_str = service_date.strftime("%Y%m%d")
    band = band_seconds()
    fp = _fingerprint(gtfs_paths, date_str, band, footpath_m)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"raptor_{date_str}_{fp}.pkl"
    if cache.exists():
        if verbose:
            _log(f"[raptor] loading cached structures {cache.name}")
        with open(cache, "rb") as f:
            return pickle.load(f)
    if verbose:
        _log(f"[raptor] building structures for {date_str} band {band} ...")
    t = time.time()
    data = build(gtfs_paths, date_str, band[0], band[1], footpath_m)
    if verbose:
        _log(f"[raptor] built in {time.time()-t:.1f}s; trip counts {data['feed_trip_counts']}")
    assert all(v > 0 for v in data["feed_trip_counts"].values()), "a feed has 0 trips in band!"
    with open(cache, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return data


if __name__ == "__main__":
    d = load_or_build()
    _log("n_stops", d["n_stops"], "patterns", len(d["pat_nstops"]))
