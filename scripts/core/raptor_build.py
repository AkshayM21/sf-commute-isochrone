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
  * transfers: a legacy synthesized walk-footpath CSR between stops within ``FOOTPATH_M``
    great-circle metres (cost = distance / walk speed), bidirectional, spanning ALL feeds.
    The graph-native access bake replaces this runtime view with its directed, hill-aware
    transfer CSR and parses ``transfers.txt``/``pathways.txt`` separately; this legacy view
    remains for cache compatibility and synthetic callers that do not load an access artifact.

Only trips whose service runs on the model date (calendar + calendar_dates exceptions) AND
whose times intersect the morning band are kept. After-midnight trips (HH >= 24) fall
outside the band and are dropped, which is correct for an arrive-by-09:00 model.

The build is deterministic and cached to disk (see ``load_or_build``), keyed by the explicit
service date, time band, and footpath radius. The cache records direct feed source mtimes and
rebuilds when those sources change, without content-addressed metadata.
"""
import io
import os
import time
import pickle
import tempfile
import threading
import weakref
import zipfile
from pathlib import Path

import numpy as np

from . import config, feeds

# pandas (~44 MB) is used ONLY by build() (GTFS parse), which runs on a cache MISS. On the normal
# cache-hit boot we never touch it, so import it lazily — keeps the JVM-free server's RSS down.
pd = None


def _pd():
    global pd
    if pd is None:
        import pandas as _p
        pd = _p
    return pd

BUILD_VERSION = 6            # bump when the struct schema/invariants change (v2: + pat_feed/
                             # line/mode; v3: FIFO split enforces ARR-column sortedness too;
                             # v4: retain GTFS stop names for journey-action copy; v5: explicit
                             # source freshness metadata replaces content-addressed cache identities;
                             # v6: canonical feed/GTFS stop identity aligned to gids);
                             # a cached pkl with a different version is rebuilt in place (the
                             # filename is unchanged, so the gid-keyed access table stays valid).
FOOTPATH_M = float(os.environ.get("RAPTOR_FOOTPATH_M", "250"))  # synthesized-transfer radius (m)
BAND_START_H = 5.0           # keep trips active from 05:00 ...
# ... to the latest arrival deadline we will ever sweep: the arrive-by target plus the
# routing cap (a trip boarded right at the cap could still be in service). MAX_MIN past
# the window end is a safe upper bound for the morning model.

CACHE_DIR = config.DATA / "raptor_cache"

# A process-local lock prevents two request/boot threads from rebuilding and publishing the
# same cache identity at once.  Cross-process readers/writers are safe because publication is a
# single os.replace() of a fully flushed file; a second process may do duplicate work, but it
# cannot observe a partially written pickle.
_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS = weakref.WeakValueDictionary()


def _log(*a):
    print(*a, flush=True)


def _hms_to_sec(s):
    """'HH:MM:SS' (HH may exceed 24 for after-midnight trips) -> int seconds, or -1 if NaN."""
    if pd.isna(s):
        return -1
    p = str(s).split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])


def _split_fifo(dep, arr):
    """Split a stop-sequence group's trips into FIFO sub-patterns (no overtaking).

    ``dep``/``arr`` are (ntrips x nstops) int64, already sorted by dep[:,0]. A new trip can
    join an existing sub-pattern only if, at EVERY position, BOTH its departure and its
    arrival are >= the last trip's (so both columns stay sorted; the hot-path binary search
    in raptor.py/raptor_numba.py runs on the ARRIVAL column, and dep-sortedness alone does
    not imply arr-sortedness — a later-departing but faster-running trip can arrive earlier).
    Returns a list of (dep_sub, arr_sub) arrays. Almost always one group; splits are rare
    (express overtaking a local sharing the identical stop list)."""
    groups = []            # each: [dep_rows list, arr_rows list, last_dep_vector, last_arr_vector]
    for t in range(dep.shape[0]):
        d, a = dep[t], arr[t]
        placed = False
        for g in groups:
            if np.all(d >= g[2]) and np.all(a >= g[3]):  # does not overtake this group's latest trip
                g[0].append(d); g[1].append(a); g[2] = d; g[3] = a
                placed = True
                break
        if not placed:
            groups.append([[d], [a], d, a])
    return [(np.array(g[0], dtype=np.int64), np.array(g[1], dtype=np.int64)) for g in groups]


def build(gtfs_paths, date_str, band_start_sec, band_end_sec, footpath_m=FOOTPATH_M):
    """Parse feeds into flat RAPTOR arrays for ``date_str`` (YYYYMMDD), trips intersecting
    [band_start_sec, band_end_sec]. Returns a dict of numpy arrays (see module docstring)."""
    _pd()
    stop_key_to_gid = {}
    stop_feed, stop_id = [], []
    stop_lat, stop_lon, stop_name = [], [], []

    def get_gid(feed, sid, lat, lon, name):
        key = (feed, sid)
        g = stop_key_to_gid.get(key)
        if g is None:
            g = len(stop_lat)
            stop_key_to_gid[key] = g
            stop_feed.append(str(feed)); stop_id.append(str(sid))
            stop_lat.append(lat); stop_lon.append(lon); stop_name.append(str(name or "").strip())
        return g

    # group trips by (feed, route_id, stop-sequence) -> >=1 FIFO pattern. Keying on route_id
    # keeps a pattern intrinsically single-route (so it can be named feed-aware in Phase 2) and
    # fixes a latent merge of two routes that happen to share an identical stop list.
    seqs = {}                  # (feed_idx, route_id, gids tuple) -> list[(dep, arr)]
    feeds_list = [Path(p).stem for p in gtfs_paths]
    route_name = {}            # (feed_idx, route_id) -> display name
    route_mode = {}            # (feed_idx, route_id) -> overlay mode bucket ("bus"/"bart"/...)
    feed_trip_counts = {}
    for fi, p in enumerate(gtfs_paths):
        feed = feeds_list[fi]
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            sids = feeds.active_service_ids(z, names, date_str)
            sdf = pd.read_csv(io.BytesIO(z.read("stops.txt")), dtype=str)
            slat = dict(zip(sdf.stop_id, pd.to_numeric(sdf.stop_lat, errors="coerce")))
            slon = dict(zip(sdf.stop_id, pd.to_numeric(sdf.stop_lon, errors="coerce")))
            sname = dict(zip(sdf.stop_id, sdf.stop_name.fillna("") if "stop_name" in sdf else [""] * len(sdf)))
            rdf = pd.read_csv(io.BytesIO(z.read("routes.txt")), dtype=str)
            for _, r in rdf.iterrows():
                rid = str(r["route_id"])
                # the ONE naming policy (handles None/NaN/blank/whitespace)
                route_name[(fi, rid)] = feeds._route_display_name(
                    r.get("route_short_name"), r.get("route_long_name"), rid)
                route_mode[(fi, rid)] = feeds._MODE.get(str(r.get("route_type", "3")), "bus")
            trips = pd.read_csv(io.BytesIO(z.read("trips.txt")), dtype=str)
            trips = trips[trips.service_id.isin(sids)]
            trip_route = dict(zip(trips.trip_id, trips.route_id))
            active = set(trips.trip_id)
            st = pd.read_csv(io.BytesIO(z.read("stop_times.txt")), dtype=str)
            st = st[st.trip_id.isin(active)].copy()
            st["seq"] = pd.to_numeric(st.stop_sequence, errors="coerce")
            st["arr"] = st.arrival_time.map(_hms_to_sec)
            st["dep"] = st.departure_time.map(_hms_to_sec)
            st.loc[st["arr"] < 0, "arr"] = st["dep"]      # blank intermediate times: use the other
            st.loc[st["dep"] < 0, "dep"] = st["arr"]
            # rows with BOTH times blank survive the cross-patch as -1; a -1 in pat_arr/pat_dep
            # silently corrupts the reverse search (searchsorted treats the trip as arriving
            # before t=0), so drop the whole trip and say so. (Interpolating would guess.)
            bad = st.loc[(st["arr"] < 0) | (st["dep"] < 0), "trip_id"].unique()
            if len(bad):
                _log(f"  {feed}: dropping {len(bad)} trips with stop_times rows "
                     f"missing BOTH arrival and departure")
                st = st[~st.trip_id.isin(bad)]
            kept = 0
            for tid, g in st.sort_values(["trip_id", "seq"]).groupby("trip_id", sort=False):
                gids = tuple(get_gid(feed, sid, slat.get(sid, np.nan), slon.get(sid, np.nan),
                                     sname.get(sid, ""))
                             for sid in g.stop_id)
                deps = g["dep"].to_numpy(); arrs = g["arr"].to_numpy()
                if deps[0] > band_end_sec or arrs[-1] < band_start_sec:
                    continue                              # outside the morning band
                rid = str(trip_route.get(tid, ""))
                seqs.setdefault((fi, rid, gids), []).append((deps, arrs))
                kept += 1
            feed_trip_counts[feed] = kept
            _log(f"  {feed}: {len(sids)} services, {kept} trips in band")

    n_stops = len(stop_lat)
    stop_lat = np.asarray(stop_lat, dtype=np.float64)
    stop_lon = np.asarray(stop_lon, dtype=np.float64)

    # ---- finalize patterns (FIFO-split), then flatten to CSR arrays.
    # Intern (feed_idx, route_id) -> line_idx; line_table rows = (feed_stem, route_id, name, mode).
    # Iterate seqs in sorted key order so pattern + line indices are DETERMINISTIC run-to-run
    # (Phase 2 color-by-line stability depends on this).
    P_stops, P_dep, P_arr = [], [], []        # ragged, per pattern
    pat_feed_l, pat_line_l, pat_mode_l = [], [], []
    line_index = {}                           # (feed_idx, route_id) -> line_idx
    line_table = []                           # [(feed_stem, route_id, name, mode)]
    _MODE_CODE = {"metro": 0, "bart": 1, "cable": 2, "bus": 3}
    n_split = 0
    for (fi, rid, gids) in sorted(seqs.keys()):
        trips = seqs[(fi, rid, gids)]
        dep = np.array([t[0] for t in trips], dtype=np.int64)
        arr = np.array([t[1] for t in trips], dtype=np.int64)
        order = np.argsort(dep[:, 0], kind="stable")
        dep, arr = dep[order], arr[order]
        subs = _split_fifo(dep, arr)
        if len(subs) > 1:
            n_split += len(subs) - 1
        key = (fi, rid)
        li = line_index.get(key)
        if li is None:
            li = len(line_table)
            line_index[key] = li
            line_table.append((feeds_list[fi], rid, route_name.get(key, rid),
                               route_mode.get(key, "bus")))
        mode_code = _MODE_CODE.get(route_mode.get(key, "bus"), 3)
        garr = np.asarray(gids, dtype=np.int32)
        for dsub, asub in subs:
            # both time columns must be sorted per position — the reverse search
            # binary-searches the ARRIVAL column, so fail the bake loudly if not.
            assert (np.diff(dsub, axis=0) >= 0).all() and (np.diff(asub, axis=0) >= 0).all(), \
                f"FIFO violation after split: feed {feeds_list[fi]} route {rid}"
            P_stops.append(garr); P_dep.append(dsub); P_arr.append(asub)
            pat_feed_l.append(fi); pat_line_l.append(li); pat_mode_l.append(mode_code)
    n_pat = len(P_stops)
    pat_feed = np.asarray(pat_feed_l, dtype=np.int16)
    pat_line = np.asarray(pat_line_l, dtype=np.int32)
    pat_mode = np.asarray(pat_mode_l, dtype=np.int8)
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
    if n_pat:   # belt-and-braces: the blank-row drop above must leave no negative times
        assert pat_dep.min() >= 0 and pat_arr.min() >= 0, \
            "negative stop time leaked into pattern matrices (blank-row drop failed)"

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

    _log(f"stops {n_stops}  patterns {n_pat}  lines {len(line_table)}  "
         f"ras {len(ras_pat)}  footpaths {len(tr_to)}")
    return dict(
        build_version=BUILD_VERSION,
        n_stops=n_stops, stop_feed=np.asarray(stop_feed, dtype="U"),
        stop_id=np.asarray(stop_id, dtype="U"),
        stop_lat=stop_lat, stop_lon=stop_lon, stop_name=stop_name,
        pat_nstops=pat_nstops, pat_ntrips=pat_ntrips,
        pat_stop_off=pat_stop_off, pat_mat_off=pat_mat_off,
        pat_stops=pat_stops, pat_dep=pat_dep, pat_arr=pat_arr,
        pat_feed=pat_feed, pat_line=pat_line, pat_mode=pat_mode,
        feeds=feeds_list, line_table=line_table,            # pat_line -> (feed, route_id, name, mode)
        ras_off=ras_off, ras_pat=ras_pat, ras_pos=ras_pos,
        tr_off=tr_off, tr_to=tr_to, tr_time=tr_time,
        feed_trip_counts=feed_trip_counts, date=date_str,
        band=(band_start_sec, band_end_sec), footpath_m=footpath_m, source_mtimes=(),
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


def _source_mtimes(gtfs_paths):
    """Return stable direct source freshness metadata for a cache payload.

    The name/size/mtime tuple is deliberately human-readable and does not digest feed contents.
    Missing paths are represented explicitly so a deleted feed invalidates an old
    cache instead of silently looking fresh.
    """
    result = []
    for path in gtfs_paths:
        p = Path(path)
        try:
            st = p.stat()
            result.append((p.name, int(st.st_size), config.portable_mtime_ns(st)))
        except OSError:
            result.append((p.name, None, None))
    return tuple(result)


def _cache_path(gtfs_paths, date_str, band, footpath_m):
    """Return the readable canonical cache path for explicit routing parameters.

    ``gtfs_paths`` remains in the signature for compatibility with existing callers, but source
    identity belongs in the payload's direct mtime metadata rather than in the filename.
    """
    del gtfs_paths
    start, end = (int(band[0]), int(band[1]))
    footpath = format(float(footpath_m), ".6g")
    return CACHE_DIR / f"raptor_{date_str}_{start}-{end}_footpath{footpath}m.pkl"


_CACHE_REQUIRED_KEYS = frozenset({
    "build_version", "n_stops", "stop_feed", "stop_id", "stop_lat", "stop_lon", "stop_name",
    "pat_nstops", "pat_ntrips", "pat_stop_off", "pat_mat_off", "pat_stops",
    "pat_dep", "pat_arr", "pat_feed", "pat_line", "pat_mode", "feeds",
    "line_table", "ras_off", "ras_pat", "ras_pos", "tr_off", "tr_to",
    "tr_time", "feed_trip_counts", "date", "band", "footpath_m", "source_mtimes",
})


def _cache_lock(cache):
    """Return the stable process-local lock for one cache path."""
    key = str(Path(cache))
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = _CACHE_LOCKS[key] = threading.RLock()
        return lock


def _validate_cache_data(data, expected=None):
    """Return whether *data* is a complete, current RAPTOR cache payload.

    This is intentionally structural rather than a content digest: pickle truncation,
    unrelated pickles, and older schemas must all be treated as cache misses without risking a
    partially usable engine.  The builder's result is the source of truth for the values.
    """
    if not isinstance(data, dict) or data.get("build_version") != BUILD_VERSION:
        return False
    if not _CACHE_REQUIRED_KEYS.issubset(data):
        return False
    try:
        n_stops = int(data["n_stops"])
        pat_nstops = np.asarray(data["pat_nstops"])
        pat_ntrips = np.asarray(data["pat_ntrips"])
        pat_stop_off = np.asarray(data["pat_stop_off"])
        pat_mat_off = np.asarray(data["pat_mat_off"])
        pat_stops = np.asarray(data["pat_stops"])
        pat_dep = np.asarray(data["pat_dep"])
        pat_arr = np.asarray(data["pat_arr"])
        pat_feed = np.asarray(data["pat_feed"])
        pat_line = np.asarray(data["pat_line"])
        pat_mode = np.asarray(data["pat_mode"])
        stop_feed = np.asarray(data["stop_feed"])
        stop_id = np.asarray(data["stop_id"])
        ras_off = np.asarray(data["ras_off"])
        ras_pat = np.asarray(data["ras_pat"])
        ras_pos = np.asarray(data["ras_pos"])
        tr_off = np.asarray(data["tr_off"])
        tr_to = np.asarray(data["tr_to"])
        tr_time = np.asarray(data["tr_time"])
        arrays = (pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_stops,
                  pat_dep, pat_arr, pat_feed, pat_line, pat_mode, ras_off, ras_pat,
                  ras_pos, tr_off, tr_to, tr_time)
        if n_stops < 0 or any(a.ndim != 1 for a in arrays):
            return False
        if any(a.dtype.kind not in "iu" for a in arrays):
            return False
        n_patterns = len(pat_nstops)
        if len(pat_ntrips) != n_patterns:
            return False
        if any(len(a) != n_patterns for a in (pat_feed, pat_line, pat_mode)):
            return False
        if (stop_feed.ndim != 1 or stop_id.ndim != 1
                or stop_feed.dtype.kind not in "OUS" or stop_id.dtype.kind not in "OUS"
                or len(stop_feed) != n_stops or len(stop_id) != n_stops
                or len(set(zip(stop_feed.astype(str), stop_id.astype(str)))) != n_stops
                or any(not str(feed).strip() or not str(sid).strip()
                       for feed, sid in zip(stop_feed, stop_id))):
            return False
        stop_lat = np.asarray(data["stop_lat"])
        stop_lon = np.asarray(data["stop_lon"])
        stop_name = np.asarray(data["stop_name"])
        if (stop_lat.ndim != 1 or stop_lon.ndim != 1 or stop_name.ndim != 1
                or stop_lat.dtype.kind not in "fiu" or stop_lon.dtype.kind not in "fiu"
                or stop_name.dtype.kind not in "OUS"
                or len(stop_lat) != n_stops or len(stop_lon) != n_stops
                or len(stop_name) != n_stops):
            return False
        if np.any(np.isinf(stop_lat)) or np.any(np.isinf(stop_lon)):
            return False
        if len(ras_off) != n_stops + 1 or len(tr_off) != n_stops + 1:
            return False
        if len(pat_stop_off) != n_patterns + 1 or len(pat_mat_off) != n_patterns + 1:
            return False
        if any(off[0] != 0 or np.any(np.diff(off) < 0)
               for off in (pat_stop_off, pat_mat_off, ras_off, tr_off)):
            return False
        if (n_patterns and (np.any(pat_nstops <= 0) or np.any(pat_ntrips <= 0))):
            return False
        if not np.array_equal(np.diff(pat_stop_off), pat_nstops):
            return False
        matrix_sizes = pat_nstops.astype(np.int64) * pat_ntrips.astype(np.int64)
        if not np.array_equal(np.diff(pat_mat_off), matrix_sizes):
            return False
        if len(pat_stops) != int(pat_stop_off[-1]):
            return False
        if len(pat_dep) != len(pat_arr) or len(pat_dep) != int(pat_mat_off[-1]):
            return False
        if len(pat_dep) and (np.any(pat_dep < 0) or np.any(pat_arr < 0)):
            return False
        if len(ras_pat) != len(ras_pos) or len(ras_pat) != int(ras_off[-1]):
            return False
        if len(tr_to) != len(tr_time) or len(tr_to) != int(tr_off[-1]):
            return False
        if len(pat_stops) and (np.any(pat_stops < 0) or np.any(pat_stops >= n_stops)):
            return False
        if len(pat_feed) and (np.any(pat_feed < 0) or np.any(pat_feed >= len(data["feeds"]))):
            return False
        if len(pat_line) and (np.any(pat_line < 0) or np.any(pat_line >= len(data["line_table"]))):
            return False
        if len(pat_mode) and (np.any(pat_mode < 0) or np.any(pat_mode > 3)):
            return False
        if len(ras_pat):
            if np.any(ras_pat < 0) or np.any(ras_pat >= n_patterns) or np.any(ras_pos < 0):
                return False
            if np.any(ras_pos >= pat_nstops[ras_pat]):
                return False
        if len(tr_to) and (np.any(tr_to < 0) or np.any(tr_to >= n_stops)):
            return False
        if len(tr_time) and np.any(tr_time < 0):
            return False
        if len(data["feed_trip_counts"]) == 0 or any(
                int(v) <= 0 for v in data["feed_trip_counts"].values()):
            return False
        sources = data["source_mtimes"]
        if not isinstance(sources, (tuple, list)):
            return False
        for source in sources:
            if not isinstance(source, (tuple, list)) or len(source) != 3:
                return False
            if not isinstance(source[0], str):
                return False
            for value in source[1:]:
                if value is not None and (not isinstance(value, (int, np.integer)) or value < 0):
                    return False
        if expected is not None:
            if str(data["date"]) != str(expected["date"]):
                return False
            if tuple(data["band"]) != tuple(expected["band"]):
                return False
            if float(data["footpath_m"]) != float(expected["footpath_m"]):
                return False
            if config.normalize_source_mtimes(data["source_mtimes"]) != \
                    config.normalize_source_mtimes(expected["source_mtimes"]):
                return False
        return True
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError):
        return False


def _read_cache(cache, verbose=True, expected=None):
    """Read and validate one cache, returning None for every cache-miss condition."""
    try:
        with open(cache, "rb") as f:
            data = pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError,
            AttributeError, IndexError, KeyError, ImportError, UnicodeDecodeError,
            OverflowError):
        if verbose:
            _log(f"[raptor] cache {cache.name} is unreadable; rebuilding")
        return None
    if not _validate_cache_data(data, expected=expected):
        if verbose:
            version = data.get("build_version") if isinstance(data, dict) else "unknown"
            _log(f"[raptor] cache {cache.name} is invalid (build v{version}); rebuilding")
        return None
    return data


def _fsync_directory(directory):
    """Best-effort durability barrier for a completed atomic rename."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some filesystems/platforms do not support directory fsync
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _publish_cache(cache, data, verbose=True, expected=None):
    """Publish a validated payload atomically, returning False if persistence fails.

    The temporary file lives beside the target so os.replace remains atomic on the same
    filesystem.  Reopening and validating the temporary pickle catches short writes and schema
    mistakes before the old target is touched.  Any failure leaves an existing target intact.
    """
    tmp_name = None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{cache.name}.", suffix=".tmp",
                dir=str(cache.parent), delete=False) as f:
            tmp_name = f.name
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp_name, "rb") as f:
            published = pickle.load(f)
        if not _validate_cache_data(published, expected=expected):
            raise ValueError("serialized RAPTOR cache failed structural validation")
        os.replace(tmp_name, cache)
        tmp_name = None
        _fsync_directory(cache.parent)
        return True
    except (OSError, EOFError, pickle.UnpicklingError, ValueError, TypeError,
            AttributeError, IndexError, KeyError, ImportError, UnicodeDecodeError,
            OverflowError) as exc:
        if verbose:
            _log(f"[raptor] could not publish cache {cache.name}: {exc}")
        return False
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def load_or_build(gtfs_paths=None, service_date=None, footpath_m=FOOTPATH_M, verbose=True):
    """Return RAPTOR structures for the canonical model, building + caching on first use.

    The canonical filename contains only explicit routing parameters. A GTFS repull invalidates
    the payload through its direct source size/mtime metadata. Cheap to rebuild (~1-2s, no JVM).
    """
    gtfs_paths = gtfs_paths or config.gtfs_paths()
    service_date = service_date or feeds.pick_service_date(gtfs_paths)
    date_str = service_date.strftime("%Y%m%d")
    band = band_seconds()
    source_mtimes = _source_mtimes(gtfs_paths)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(gtfs_paths, date_str, band, footpath_m)
    expected = {"date": date_str, "band": band, "footpath_m": footpath_m,
                "source_mtimes": source_mtimes}
    with _cache_lock(cache):
        if cache.exists():
            data = _read_cache(cache, verbose=verbose, expected=expected)
            if data is not None:
                if verbose:
                    _log(f"[raptor] loading cached structures {cache.name}")
                return data
        if verbose:
            _log(f"[raptor] building structures for {date_str} band {band} ...")
        t = time.time()
        data = build(gtfs_paths, date_str, band[0], band[1], footpath_m)
        data["source_mtimes"] = source_mtimes
        if verbose:
            _log(f"[raptor] built in {time.time()-t:.1f}s; trip counts {data['feed_trip_counts']}")
        assert all(v > 0 for v in data["feed_trip_counts"].values()), "a feed has 0 trips in band!"
        if not _validate_cache_data(data, expected=expected):
            raise ValueError("RAPTOR build produced structurally invalid cache data")
        if not _publish_cache(cache, data, verbose=verbose, expected=expected):
            # Disk persistence is an optimization; a valid in-memory build remains usable and an
            # existing cache (if any) was deliberately left untouched.
            pass
        return data


if __name__ == "__main__":
    d = load_or_build()
    _log("n_stops", d["n_stops"], "patterns", len(d["pat_nstops"]))
