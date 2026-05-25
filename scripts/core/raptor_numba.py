"""Numba-compiled reverse range-RAPTOR sweep — the production hot path.

A faithful, byte-equivalent port of the validated pure-python ``raptor.reverse_raptor``
(same seeding, same one-hop transfer policy, same FIFO per-position binary search), compiled
to a single ``nogil`` kernel that sweeps ALL arrival deadlines in one call. ``nogil`` lets the
server run concurrent requests on real threads without GIL contention; ``cache=True`` persists
the compiled object so warmup is one-time.

Falls back is unnecessary: ``raptor.reverse_profile`` selects this kernel only when numba
imports cleanly, else uses the pure-python path (kept as the reference + test oracle).
"""
import numpy as np
from numba import njit, prange

NEG = -(1 << 60)


@njit(cache=True, nogil=True)
def _profile(n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off,
             pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos,
             tr_off, tr_to, tr_time, egress_g, egress_w, deadlines,
             board_slack, max_rounds):
    nd = deadlines.shape[0]
    n_pat = pat_nstops.shape[0]
    latest = np.full((n_stops, nd), NEG, dtype=np.int64)
    best = np.empty(n_stops, dtype=np.int64)
    prev = np.empty(n_stops, dtype=np.int64)
    flag = np.zeros(n_stops, dtype=np.uint8)
    q_pos = np.full(n_pat, -1, dtype=np.int64)
    cur = np.empty(n_stops, dtype=np.int64)        # marked stops to scan this round
    new = np.empty(n_stops, dtype=np.int64)        # marked this round (pattern + transfer)
    act = np.empty(n_pat, dtype=np.int64)          # patterns to scan this round

    for ti in range(nd):
        T = deadlines[ti]
        best[:] = NEG
        prev[:] = NEG
        # --- seed egress, then a one-hop transfer from the egress stops
        cn = 0
        for e in range(egress_g.shape[0]):
            g = egress_g[e]
            t = T - egress_w[e]
            if t > best[g]:
                best[g] = t
                prev[g] = t
                if flag[g] == 0:
                    flag[g] = 1
                    cur[cn] = g
                    cn += 1
        seed_n = cn
        for ii in range(seed_n):
            s = cur[ii]
            for k in range(tr_off[s], tr_off[s + 1]):
                j = tr_to[k]
                cand = best[s] - tr_time[k]
                if cand > best[j]:
                    best[j] = cand
                    prev[j] = cand
                    if flag[j] == 0:
                        flag[j] = 1
                        cur[cn] = j
                        cn += 1

        # --- rounds
        for _rnd in range(max_rounds):
            if cn == 0:
                break
            # build pattern queue from the marked stops
            an = 0
            for ii in range(cn):
                s = cur[ii]
                for k in range(ras_off[s], ras_off[s + 1]):
                    pi = ras_pat[k]
                    pos = ras_pos[k]
                    if q_pos[pi] < 0:
                        act[an] = pi
                        an += 1
                    if pos > q_pos[pi]:
                        q_pos[pi] = pos
            # start a fresh marked set
            for ii in range(cn):
                flag[cur[ii]] = 0
            nn = 0
            for pp in range(an):
                pi = act[pp]
                end_pos = q_pos[pi]
                q_pos[pi] = -1
                ns = pat_nstops[pi]
                nt = pat_ntrips[pi]
                sbase = pat_stop_off[pi]
                mbase = pat_mat_off[pi]
                trip = -1
                for pos in range(end_pos, -1, -1):
                    s = pat_stops[sbase + pos]
                    if trip >= 0:
                        d = pat_dep[mbase + trip * ns + pos]
                        if d > best[s]:
                            best[s] = d
                            if flag[s] == 0:
                                flag[s] = 1
                                new[nn] = s
                                nn += 1
                    pa = prev[s]
                    if pa > NEG:
                        key = pa - board_slack
                        lo = 0
                        hi = nt
                        while lo < hi:                  # searchsorted(arr_col, key, 'right')
                            mid = (lo + hi) >> 1
                            if pat_arr[mbase + mid * ns + pos] <= key:
                                lo = mid + 1
                            else:
                                hi = mid
                        idx = lo - 1
                        if idx >= 0 and (trip < 0 or idx > trip):
                            trip = idx
            # commit prev for pattern-marked stops
            for ii in range(nn):
                prev[new[ii]] = best[new[ii]]
            # one-hop transfers from the pattern-marked stops (mirror the python policy)
            tn = nn
            for ii in range(nn):
                s = new[ii]
                for k in range(tr_off[s], tr_off[s + 1]):
                    j = tr_to[k]
                    cand = best[s] - tr_time[k]
                    if cand > best[j]:
                        best[j] = cand
                        prev[j] = cand
                        if flag[j] == 0:
                            flag[j] = 1
                            new[tn] = j
                            tn += 1
            # next round scans the stops marked this round
            for ii in range(tn):
                cur[ii] = new[ii]
            cn = tn
        # clear flags left set by the final round before the next deadline
        for ii in range(cn):
            flag[cur[ii]] = 0
        latest[:, ti] = best
    return latest


INF = 1 << 60


@njit(cache=True, nogil=True)
def stop_arrival_profile(latest, deadlines, dep_grid):
    """Invert latest-departure profiles -> arrival-at-workplace profiles (see raptor.py)."""
    n = latest.shape[0]
    nT = deadlines.shape[0]
    ndg = dep_grid.shape[0]
    out = np.full((n, ndg), INF, dtype=np.int64)
    for s in range(n):
        if latest[s, nT - 1] < dep_grid[0]:
            continue
        for k in range(ndg):
            D = dep_grid[k]
            lo = 0
            hi = nT
            while lo < hi:                       # searchsorted(latest[s], D, 'left')
                mid = (lo + hi) >> 1
                if latest[s, mid] < D:
                    lo = mid + 1
                else:
                    hi = mid
            if lo < nT:
                out[s, k] = deadlines[lo]
    return out


@njit(cache=True, nogil=True)
def _pct_lower(srt, p):
    return srt[int(np.floor((srt.shape[0] - 1) * p / 100.0))]


@njit(cache=True, nogil=True)
def assemble_departafter(access_off, access_to, access_w, purewalk, arrivalW,
                         dep_grid, cell_deps, max_min, pct):
    n_cells = access_off.shape[0] - 1
    ndg = dep_grid.shape[0]
    nd = cell_deps.shape[0]
    npct = pct.shape[0]
    out = np.full((n_cells, npct), -1, dtype=np.int32)
    ttm = np.empty(nd, dtype=np.float64)
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        pw = purewalk[ci]
        for di in range(nd):
            D = cell_deps[di]
            best = INF
            for a in range(a0, a1):
                board = D + access_w[a]
                lo = 0
                hi = ndg
                while lo < hi:                   # searchsorted(dep_grid, board, 'left')
                    mid = (lo + hi) >> 1
                    if dep_grid[mid] < board:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < ndg:
                    v = arrivalW[access_to[a], lo]
                    if v < INF and v < best:
                        best = v
            tt = (best - D) / 60.0 if best < INF else 1e18
            if pw >= 0 and pw / 60.0 < tt:
                tt = pw / 60.0
            if tt > max_min:
                tt = max_min
            ttm[di] = np.ceil(tt)
        srt = np.sort(ttm)
        for p in range(npct):
            v = _pct_lower(srt, pct[p])
            out[ci, p] = -1 if v >= max_min else np.int32(v)
    return out


@njit(cache=True, nogil=True)
def assemble_arriveby(access_off, access_to, access_w, purewalk, latest, deadlines,
                      max_min, pct):
    n_cells = access_off.shape[0] - 1
    nd = deadlines.shape[0]
    npct = pct.shape[0]
    out = np.full((n_cells, npct), -1, dtype=np.int32)
    ttm = np.empty(nd, dtype=np.float64)
    NEGH = -(1 << 60) // 2
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        pw = purewalk[ci]
        for di in range(nd):
            T = deadlines[di]
            home = -(1 << 60)
            for a in range(a0, a1):
                h = latest[access_to[a], di] - access_w[a]
                if h > home:
                    home = h
            tt = (T - home) / 60.0 if home > NEGH else 1e18
            if pw >= 0 and (T - pw) >= 0 and pw / 60.0 < tt:
                tt = pw / 60.0
            if tt < 0:
                tt = 0.0
            if tt > max_min:
                tt = max_min
            ttm[di] = np.ceil(tt)
        srt = np.sort(ttm)
        for p in range(npct):
            v = _pct_lower(srt, pct[p])
            out[ci, p] = -1 if v >= max_min else np.int32(v)
    return out


def reverse_profile(data, egress_g, egress_w, deadlines, board_slack, max_rounds):
    """latest[n_stops, n_deadlines] via the compiled kernel."""
    return _profile(
        np.int64(data["n_stops"]),
        data["pat_nstops"].astype(np.int64), data["pat_ntrips"].astype(np.int64),
        data["pat_stop_off"].astype(np.int64), data["pat_mat_off"].astype(np.int64),
        data["pat_stops"].astype(np.int64), data["pat_dep"].astype(np.int64),
        data["pat_arr"].astype(np.int64),
        data["ras_off"].astype(np.int64), data["ras_pat"].astype(np.int64),
        data["ras_pos"].astype(np.int64),
        data["tr_off"].astype(np.int64), data["tr_to"].astype(np.int64),
        data["tr_time"].astype(np.int64),
        np.asarray(egress_g, dtype=np.int64), np.asarray(egress_w, dtype=np.int64),
        np.asarray(deadlines, dtype=np.int64),
        np.int64(board_slack), np.int64(max_rounds))


# ============================================================== service-noise Monte-Carlo
# Perturb the schedule (per-trip cumulative delay) and re-run the SAME validated reverse sweep
# per draw, streaming each draw's per-cell median-departure commute into an accumulator. This
# re-routes OPTIMALLY per scenario (miss an express -> the min over patterns lands on the next
# local), which a naive same-line headway score cannot do. The cell loop's cost is folded into
# the per-draw work; draws run in `prange` across cores (nogil), each thread holding only ONE
# perturbed schedule + ONE latest profile at a time (never R x n_stops).

@njit(cache=True, nogil=True)
def _perturb(pat_nstops, pat_ntrips, pat_mat_off, pat_trip_off, pat_dep, pat_arr,
             delta0, slope, dep_out, arr_out):
    """Apply a per-trip delay delta(pos) = delta0[trip] + slope[trip]*scheduled_elapsed(pos)
    to dep & arr (preserving dwell), then a per-pattern, per-position FIFO cumulative-max clamp
    across trips so the arrival column stays sorted (the per-position binary search in _profile
    requires it). delta0 / slope are indexed by GLOBAL trip id (pat_trip_off[pi] + trip)."""
    n_pat = pat_nstops.shape[0]
    for pi in range(n_pat):
        ns = pat_nstops[pi]
        nt = pat_ntrips[pi]
        mbase = pat_mat_off[pi]
        tbase = pat_trip_off[pi]
        for trip in range(nt):
            g = tbase + trip
            d0 = delta0[g]
            sl = slope[g]
            base = mbase + trip * ns
            dep0 = pat_dep[base]
            for pos in range(ns):
                off = base + pos
                elapsed = pat_dep[off] - dep0
                if elapsed < 0:
                    elapsed = 0
                inc = np.int64(d0 + sl * elapsed)
                dep_out[off] = pat_dep[off] + inc
                arr_out[off] = pat_arr[off] + inc
        # FIFO cumulative-max clamp per position across trips (no overtaking on dep or arr)
        for pos in range(ns):
            for trip in range(1, nt):
                a = mbase + trip * ns + pos
                b = mbase + (trip - 1) * ns + pos
                if dep_out[a] < dep_out[b]:
                    dep_out[a] = dep_out[b]
                if arr_out[a] < arr_out[b]:
                    arr_out[a] = arr_out[b]


@njit(parallel=True, nogil=True, cache=True)
def montecarlo(n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_trip_off,
               pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time,
               egress_g, egress_w, deadlines, board_slack, max_rounds,
               access_off, access_to, access_w, purewalk, d_med, max_min,
               delta0_all, slope_all):
    """For R draws (rows of delta0_all/slope_all), perturb -> reverse sweep -> per-cell commute
    at the single median departure ``d_med``. Returns commute_all[n_cells, R] in float minutes,
    capped at max_min (unreachable -> max_min, so it counts as 'stuck'). Cross-draw spread is
    pure SERVICE noise (the departure is fixed), which is exactly the variance we want."""
    R = delta0_all.shape[0]
    n_cells = access_off.shape[0] - 1
    nd = deadlines.shape[0]
    np_len = pat_dep.shape[0]
    commute_all = np.empty((n_cells, R), dtype=np.float64)
    capf = np.float64(max_min)
    for r in prange(R):
        dep_r = np.empty(np_len, dtype=np.int64)
        arr_r = np.empty(np_len, dtype=np.int64)
        _perturb(pat_nstops, pat_ntrips, pat_mat_off, pat_trip_off, pat_dep, pat_arr,
                 delta0_all[r], slope_all[r], dep_r, arr_r)
        latest = _profile(n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off,
                          pat_stops, dep_r, arr_r, ras_off, ras_pat, ras_pos,
                          tr_off, tr_to, tr_time, egress_g, egress_w, deadlines,
                          board_slack, max_rounds)
        for ci in range(n_cells):
            a0 = access_off[ci]
            a1 = access_off[ci + 1]
            best = INF
            for a in range(a0, a1):
                s = access_to[a]
                board = d_med + access_w[a]
                lo = 0
                hi = nd
                while lo < hi:                        # first deadline T with latest[s,T] >= board
                    mid = (lo + hi) >> 1
                    if latest[s, mid] < board:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < nd:
                    tt = deadlines[lo] - d_med
                    if tt < best:
                        best = tt
            pw = purewalk[ci]
            ttv = best
            if pw >= 0 and pw < ttv:
                ttv = pw
            m = ttv / 60.0 if ttv < INF else 1e18
            if m > capf:
                m = capf
            commute_all[ci, r] = m
    return commute_all
