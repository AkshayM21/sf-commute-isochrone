"""Reverse range-RAPTOR over the flat structures from ``raptor_build`` (numpy, JVM-free).

Productionized from ``prototypes/spike_raptor/raptor.py`` (validated MAE ~1.0 vs R5). The
single reverse search is rooted at the WORKPLACE: given a set of egress stops (each with a
latest feasible alight time for a target arrival deadline), it returns the LATEST departure
time from every stop such that you still reach the workplace by that deadline. Travel time to
work from a stop = deadline - latest_departure[stop].

Range over a sweep of deadlines (``reverse_profile``) gives a per-stop latest-departure
profile; inverting it (``stop_arrival_profile``) recovers the depart-after arrival profile
that R5's departure-window percentile model uses, which is what we validate against. The
SAME profile, read at a single arrival deadline, gives the arrive-by-09:00 product semantic
(``assemble_arriveby``).

HARD-WON (see prototypes/spike_raptor/NOTES.md), preserved here:
  * 60s BOARD SLACK — you must reach a stop >= 60s before a vehicle departs to catch it;
    without it the router runs uniformly ~2.7 min too fast.
  * The departure-from-stop grid must extend past the cell window by the access-walk cap.
  * Pattern columns are FIFO (raptor_build splits overtaking), so per-position binary search
    on the arrival column is valid.

The hot ``reverse_raptor`` is a tight loop over contiguous int arrays with no Python objects
in the inner body, so it can be JIT-compiled (numba) or ported (Rust) unchanged; an optional
numba kernel is used automatically when available (see ``_NUMBA``).
"""
import numpy as np

INF = np.int64(1 << 60)
NEG = np.int64(-(1 << 60))


def reverse_raptor(data, egress_g, egress_t, max_rounds=8, board_slack=60):
    """Latest-departure reverse (arrive-by) RAPTOR for ONE arrival deadline.

    egress_g : int32[k]  egress stop gids (stops within walking distance of the workplace)
    egress_t : int64[k]  latest time you may ALIGHT at that stop (= deadline - egress walk)
    Returns best : int64[n_stops] latest departure time from each stop that still reaches the
    workplace by the deadline (NEG if unreachable)."""
    n = data["n_stops"]
    pat_nstops = data["pat_nstops"]; pat_ntrips = data["pat_ntrips"]
    pat_stop_off = data["pat_stop_off"]; pat_mat_off = data["pat_mat_off"]
    pat_stops = data["pat_stops"]; pat_dep = data["pat_dep"]; pat_arr = data["pat_arr"]
    ras_off = data["ras_off"]; ras_pat = data["ras_pat"]; ras_pos = data["ras_pos"]
    tr_off = data["tr_off"]; tr_to = data["tr_to"]; tr_time = data["tr_time"]

    best = np.full(n, NEG, dtype=np.int64)
    prev = np.full(n, NEG, dtype=np.int64)
    marked = set()
    for g, t in zip(egress_g, egress_t):
        g = int(g)
        if t > best[g]:
            best[g] = t; prev[g] = t; marked.add(g)
    for g in list(marked):                       # seed footpath transfers from egress stops
        for k in range(int(tr_off[g]), int(tr_off[g + 1])):
            j = int(tr_to[k]); cand = best[g] - tr_time[k]
            if cand > best[j]:
                best[j] = cand; prev[j] = cand; marked.add(j)

    for _ in range(max_rounds):
        if not marked:
            break
        q = {}                                   # pattern -> latest marked position (scan backward)
        for s in marked:
            for k in range(int(ras_off[s]), int(ras_off[s + 1])):
                pi = int(ras_pat[k]); pos = int(ras_pos[k])
                if pi not in q or pos > q[pi]:
                    q[pi] = pos
        marked = set()
        for pi, end_pos in q.items():
            ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi])
            sbase = int(pat_stop_off[pi]); mbase = int(pat_mat_off[pi])
            trip = -1
            for pos in range(end_pos, -1, -1):
                s = int(pat_stops[sbase + pos])
                if trip >= 0:
                    d = int(pat_dep[mbase + trip * ns + pos])
                    if d > best[s]:
                        best[s] = d; marked.add(s)
                pa = prev[s]
                if pa > NEG:
                    col = pat_arr[mbase + pos: mbase + pos + nt * ns: ns]   # arrivals at pos, sorted
                    idx = int(np.searchsorted(col, pa - board_slack, side="right")) - 1
                    if idx >= 0 and (trip < 0 or idx > trip):
                        trip = idx
        newly = list(marked)
        for s in newly:
            prev[s] = best[s]
        tr_marked = set()
        for s in newly:
            for k in range(int(tr_off[s]), int(tr_off[s + 1])):
                j = int(tr_to[k]); cand = best[s] - tr_time[k]
                if cand > best[j]:
                    best[j] = cand; prev[j] = cand; tr_marked.add(j)
        marked |= tr_marked
    return best


def reverse_raptor_traced(data, egress_g, egress_t, egress_w, max_rounds=8, board_slack=60):
    """Single-deadline reverse RAPTOR that records BACK-POINTERS for journey reconstruction.

    Same algorithm as ``reverse_raptor`` but, for each stop, remembers HOW its latest-departure
    was achieved so a cell's journey can be traced cell -> access stop -> transit legs -> W (see
    ``raptor_journey``). Used once per workplace (the path tree), not in the range sweep, so it
    stays pure-python. Determinism (color-by-line stability): patterns are scanned in SORTED
    order and a tie (`d == best[s]`) keeps the FIRST writer, so the chosen parent is reproducible
    run-to-run (earlier rounds write first -> fewer-ride journeys preferred; a strictly-faster
    later-round journey still wins). Returns a dict of parent arrays + ``best``.

    Parent encoding per stop s (all sized n_stops):
      par_kind   int8   -1 unreachable, 0 egress-seed (terminal), 1 transit board, 2 footpath
      par_pat/par_trip/par_board/par_alight  int32  transit leg: board at par_board on par_pat/trip,
                                                     forward-ALIGHT at par_alight (a LATER position)
      par_from   int32  footpath: the stop you walk TO (toward W)
      par_nxfer  int16  reverse round it was set in (= rides from here to W; tie-break + xfer count)
      egress_sec int32  egress WALK seconds at seed stops (-1 otherwise)
    """
    n = data["n_stops"]
    pat_nstops = data["pat_nstops"]; pat_ntrips = data["pat_ntrips"]
    pat_stop_off = data["pat_stop_off"]; pat_mat_off = data["pat_mat_off"]
    pat_stops = data["pat_stops"]; pat_dep = data["pat_dep"]; pat_arr = data["pat_arr"]
    ras_off = data["ras_off"]; ras_pat = data["ras_pat"]; ras_pos = data["ras_pos"]
    tr_off = data["tr_off"]; tr_to = data["tr_to"]; tr_time = data["tr_time"]

    best = np.full(n, NEG, dtype=np.int64)
    prev = np.full(n, NEG, dtype=np.int64)
    par_kind = np.full(n, -1, dtype=np.int8)
    par_pat = np.full(n, -1, dtype=np.int32); par_trip = np.full(n, -1, dtype=np.int32)
    par_board = np.full(n, -1, dtype=np.int32); par_alight = np.full(n, -1, dtype=np.int32)
    par_from = np.full(n, -1, dtype=np.int32); par_nxfer = np.zeros(n, dtype=np.int16)
    egress_sec = np.full(n, -1, dtype=np.int32)

    def set_transit(s, d, pi, trip, board_pos, alight_pos, rnd):
        best[s] = d
        par_kind[s] = 1; par_pat[s] = pi; par_trip[s] = trip
        par_board[s] = board_pos; par_alight[s] = alight_pos; par_nxfer[s] = rnd
        par_from[s] = -1

    def set_foot(j, cand, frm, nxfer):
        best[j] = cand; prev[j] = cand
        par_kind[j] = 2; par_from[j] = frm; par_nxfer[j] = nxfer
        par_pat[j] = par_trip[j] = par_board[j] = par_alight[j] = -1

    marked = set()
    for g, t, w in zip(egress_g, egress_t, egress_w):
        g = int(g)
        if t > best[g]:
            best[g] = t; prev[g] = t; par_kind[g] = 0; egress_sec[g] = int(w); marked.add(g)
    for g in sorted(marked):                     # initial one-hop footpaths from egress stops
        for k in range(int(tr_off[g]), int(tr_off[g + 1])):
            j = int(tr_to[k]); cand = best[g] - tr_time[k]
            if cand > best[j]:
                set_foot(j, cand, g, 0); marked.add(j)

    for rnd in range(1, max_rounds + 1):
        if not marked:
            break
        q = {}
        for s in marked:
            for k in range(int(ras_off[s]), int(ras_off[s + 1])):
                pi = int(ras_pat[k]); pos = int(ras_pos[k])
                if pi not in q or pos > q[pi]:
                    q[pi] = pos
        marked = set()
        for pi in sorted(q):                     # deterministic pattern order
            end_pos = q[pi]
            ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi])
            sbase = int(pat_stop_off[pi]); mbase = int(pat_mat_off[pi])
            trip = -1; cur_alight = -1
            for pos in range(end_pos, -1, -1):
                s = int(pat_stops[sbase + pos])
                if trip >= 0:
                    d = int(pat_dep[mbase + trip * ns + pos])
                    if d > best[s]:              # strict improvement; ties keep first writer
                        set_transit(s, d, pi, trip, pos, cur_alight, rnd)
                        marked.add(s)
                pa = prev[s]
                if pa > NEG:
                    col = pat_arr[mbase + pos: mbase + pos + nt * ns: ns]
                    idx = int(np.searchsorted(col, pa - board_slack, side="right")) - 1
                    if idx >= 0 and (trip < 0 or idx > trip):
                        trip = idx; cur_alight = pos
        newly = list(marked)
        for s in newly:
            prev[s] = best[s]
        tr_marked = set()
        for s in sorted(newly):
            for k in range(int(tr_off[s]), int(tr_off[s + 1])):
                j = int(tr_to[k]); cand = best[s] - tr_time[k]
                if cand > best[j]:
                    set_foot(j, cand, s, int(par_nxfer[s])); tr_marked.add(j)
        marked |= tr_marked
    return dict(best=best, par_kind=par_kind, par_pat=par_pat, par_trip=par_trip,
                par_board=par_board, par_alight=par_alight, par_from=par_from,
                par_nxfer=par_nxfer, egress_sec=egress_sec)


def reverse_profile(data, egress_g, egress_w, deadlines, board_slack=60, max_rounds=8,
                    kernel=None):
    """Run reverse RAPTOR for each arrival deadline in ``deadlines`` (seconds).

    egress_w : int64[k] egress WALK seconds (stop -> workplace) for the stops egress_g.
    Returns latest : int64[n_stops, n_deadlines] latest departure per stop per deadline."""
    n = data["n_stops"]
    deadlines = np.asarray(deadlines, dtype=np.int64)
    egress_g = np.asarray(egress_g, dtype=np.int32)
    egress_w = np.asarray(egress_w, dtype=np.int64)
    if kernel is None:
        kernel = _select_kernel()
    if kernel == "numba":
        return _reverse_profile_numba(data, egress_g, egress_w, deadlines,
                                      board_slack, max_rounds)
    latest = np.empty((n, len(deadlines)), dtype=np.int64)
    for ti, T in enumerate(deadlines):
        latest[:, ti] = reverse_raptor(data, egress_g, T - egress_w,
                                       max_rounds=max_rounds, board_slack=board_slack)
    return latest


def stop_arrival_profile(latest, deadlines, dep_grid):
    """Invert latest-departure profiles into arrival-at-workplace profiles.

    arrivalW[s, k] = earliest workplace-arrival deadline reachable if you depart stop s no
    earlier than dep_grid[k] = min{ T in deadlines : latest[s, T] >= dep_grid[k] }.
    ``latest[s, :]`` is non-decreasing in the deadline, so this is a per-row searchsorted.
    Returns arrivalW : int64[n_stops, len(dep_grid)] (INF where never feasible)."""
    deadlines = np.asarray(deadlines, dtype=np.int64)
    dep_grid = np.asarray(dep_grid, dtype=np.int64)
    if _select_kernel() == "numba":
        from . import raptor_numba
        return raptor_numba.stop_arrival_profile(latest, deadlines, dep_grid)
    n = latest.shape[0]
    arrivalW = np.full((n, len(dep_grid)), INF, dtype=np.int64)
    nT = len(deadlines)
    for s in range(n):
        row = latest[s]
        if row[-1] < dep_grid[0]:
            continue
        ti = np.searchsorted(row, dep_grid, side="left")     # vectorized over dep_grid
        ok = ti < nT
        arrivalW[s, ok] = deadlines[ti[ok]]
    return arrivalW


# --------------------------------------------------------------- assembly (cells)
def assemble_departafter(access_off, access_to, access_w, purewalk, arrivalW, dep_grid,
                         cell_deps, max_min, percentiles=(5, 50)):
    """Depart-after percentile minutes per cell (the R5-comparable map value).

    For each cell and each CELL departure D in ``cell_deps``:
        tt(D) = min( pure_walk, min over access stops s [ arrivalW(s, D + access_walk) - D ] )
    then ceil to minutes; take each requested percentile over the window (R5 counts
    unreachable draws as the cap). ``access_*`` is the cell->stop walk table in CSR (seconds);
    ``purewalk`` is cell->W walk seconds (-1 if > cap). Returns int32[n_cells, n_pct]
    (-1 where that percentile is unreachable)."""
    n_cells = len(access_off) - 1
    pct = np.asarray(percentiles, dtype=np.float64)
    dep_grid = np.asarray(dep_grid, dtype=np.int64)
    cell_deps = np.asarray(cell_deps, dtype=np.int64)
    if _select_kernel() == "numba":
        from . import raptor_numba
        return raptor_numba.assemble_departafter(
            np.asarray(access_off, np.int64), np.asarray(access_to, np.int64),
            np.asarray(access_w, np.int64), np.asarray(purewalk, np.int64),
            arrivalW, dep_grid, cell_deps, np.int64(max_min), pct)
    out = np.full((n_cells, len(pct)), -1, dtype=np.int32)
    nd = len(cell_deps)
    ndg = len(dep_grid)
    for ci in range(n_cells):
        a0, a1 = int(access_off[ci]), int(access_off[ci + 1])
        gids = access_to[a0:a1]
        awalk = access_w[a0:a1].astype(np.int64)
        tt = np.full(nd, np.iinfo(np.int64).max, dtype=np.int64)
        if len(gids):
            sub = arrivalW[gids]                              # (nstops_cell, len(dep_grid))
            for di in range(nd):
                D = cell_deps[di]
                kk = np.searchsorted(dep_grid, D + awalk, side="left")
                ok = kk < ndg
                if not ok.any():
                    continue
                rr = np.where(ok)[0]
                vals = sub[rr, kk[rr]]
                vals = vals[vals < INF]
                if vals.size:
                    tt[di] = int(vals.min()) - D
        ttm = tt.astype(np.float64) / 60.0
        pw = purewalk[ci]
        if pw >= 0:
            ttm = np.minimum(ttm, pw / 60.0)
        ttm = np.ceil(np.where(ttm > max_min, max_min, ttm))
        vals = np.percentile(ttm, pct, method="lower")
        out[ci] = [(-1 if v >= max_min else int(v)) for v in np.atleast_1d(vals)]
    return out


def assemble_arriveby(access_off, access_to, access_w, purewalk, latest_at_deadline,
                      deadline, max_min):
    """Arrive-by minutes per cell for a single arrival ``deadline`` (seconds).

    latest_home_dep(cell) = max( deadline - pure_walk,
                                 max over access stops s [ latest_at_deadline[s] - access_walk ] )
    travel = deadline - latest_home_dep. ``latest_at_deadline`` is the reverse-RAPTOR
    latest-departure column at ``deadline``. Returns int32[n_cells] minutes (-1 unreachable)."""
    n_cells = len(access_off) - 1
    out = np.full(n_cells, -1, dtype=np.int32)
    cap_sec = max_min * 60
    for ci in range(n_cells):
        a0, a1 = int(access_off[ci]), int(access_off[ci + 1])
        gids = access_to[a0:a1]
        latest_home = NEG
        if len(gids):
            cand = latest_at_deadline[gids] - access_w[a0:a1].astype(np.int64)
            m = int(cand.max())
            if m > latest_home:
                latest_home = m
        pw = purewalk[ci]
        if pw >= 0 and deadline - pw > latest_home:
            latest_home = deadline - int(pw)
        if latest_home <= NEG:
            continue
        tt = (deadline - latest_home) / 60.0
        if tt < 0:
            tt = 0.0
        if tt >= max_min:
            continue
        out[ci] = int(np.ceil(tt))
    return out


# --------------------------------------------------------------- kernel selection
_NUMBA = None


def _select_kernel():
    global _NUMBA
    if _NUMBA is None:
        try:
            import numba  # noqa
            from . import raptor_numba  # noqa
            _NUMBA = True
        except Exception:
            _NUMBA = False
    return "numba" if _NUMBA else "python"


def _reverse_profile_numba(data, egress_g, egress_w, deadlines, board_slack, max_rounds):
    from . import raptor_numba
    return raptor_numba.reverse_profile(data, egress_g, egress_w, deadlines,
                                        board_slack, max_rounds)
