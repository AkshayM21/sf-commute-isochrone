"""Reverse range-RAPTOR over the flat structures from ``raptor_build`` (numpy, JVM-free).

Productionized from ``prototypes/spike_raptor/raptor.py`` (validated MAE ~1.0 vs R5). The
single reverse search is rooted at the WORKPLACE: given a set of egress stops (each with a
latest feasible alight time for a target arrival deadline), it returns the LATEST departure
time from every stop such that you still reach the workplace by that deadline. Travel time to
work from a stop = deadline - latest_departure[stop].

Range over a sweep of deadlines (``reverse_profile``) gives a per-stop latest-departure
profile; inverting it (``stop_arrival_profile``) recovers the depart-after arrival profile
that R5's departure-window percentile model uses, which is what we validate against. The
SAME profile, read at arrival deadlines, gives the arrive-by-09:00 product semantic
(``raptor_engine._assemble_arriveby_window``; single-deadline = that path with one deadline).

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
import threading

import numpy as np

INF = np.int64(1 << 60)
NEG = np.int64(-(1 << 60))

# numba's `workqueue` threading layer (the only one available in many envs — TBB/OMP absent) is
# NOT threadsafe: two `parallel=True` kernels running on different Python threads abort the whole
# process. Serialize the one parallel kernel we have (the MC) so concurrent callers are safe.
_MC_KERNEL_LOCK = threading.Lock()


def reverse_raptor(data, egress_g, egress_t, max_rounds=8, board_slack=60):
    """Latest-departure reverse (arrive-by) RAPTOR for ONE arrival deadline.

    egress_g : int32[k]  egress stop gids (stops within walking distance of the workplace)
    egress_t : int64[k]  latest time you may ALIGHT at that stop (= deadline - egress walk)
    Returns best : int64[n_stops] latest departure time from each stop that still reaches the
    workplace by the deadline (NEG if unreachable).

    LOCKSTEP: THREE implementations of this algorithm must stay byte-equivalent in routing
    semantics (seeding, one-hop transfer policy, board slack, FIFO per-position binary search
    on the ARRIVAL column): (1) this function — the pure-python reference/test oracle;
    (2) ``reverse_raptor_traced`` below — same algorithm plus back-pointers (and deterministic
    sorted scan order); (3) ``raptor_numba._profile`` — the compiled production hot path.
    Any semantic change (e.g. board-slack handling, transfer seeding) must land in all three.

    SNAPSHOT FOOTPATH RELAX (the lockstep-exactness fix, 2026-06-10): every footpath
    relaxation pass (the egress-seed pass AND each round's pass) computes its candidates from
    the SOURCE stops' values FROZEN at the start of the pass (``snap``), never from live
    ``best[]``. The result is order-independent — best_after[j] = max(best_before[j],
    max over sources s of snap[s] - tr_time) — so all three siblings produce BYTE-EQUAL
    profiles regardless of scan order (hash-set here, sorted in traced, index order in numba).
    The old live-read let a multi-hop walk cascade fire or not depending on iteration order
    (measured on the ferry workplace: 7.8% of reachable stops diverged — the live-read up to
    ~1183 s better, the snapshot up to ~263 s better; cell-level ~50/2999 served cells moved
    −2..+6 min, 0 reachability flips); transfers are one-hop per pass by policy, so the
    snapshot read is the correct semantics, not a restriction (tested byte-equal in
    tests/test_raptor.py::test_lockstep_byte_equal)."""
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
    snap = best.copy()                           # SNAPSHOT: relax from frozen source values
    for g in list(marked):                       # seed footpath transfers from egress stops
        for k in range(int(tr_off[g]), int(tr_off[g + 1])):
            j = int(tr_to[k]); cand = snap[g] - tr_time[k]
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
        snap = best.copy()                       # SNAPSHOT: relax from frozen source values
        tr_marked = set()
        for s in newly:
            for k in range(int(tr_off[s]), int(tr_off[s + 1])):
                j = int(tr_to[k]); cand = snap[s] - tr_time[k]
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

    LOCKSTEP: keep the routing semantics byte-equivalent with its two siblings —
    ``reverse_raptor`` above (the pure-python reference; this function only ADDS back-pointers
    and sorted/deterministic scan order, never different times) and ``raptor_numba._profile``
    (the compiled hot path). A semantic change to any one must land in all three.
    Footpath relaxation is SNAPSHOT-BASED like both siblings (candidates read the sources'
    values frozen at the start of each relax pass — see ``reverse_raptor``), which is what
    makes ``best`` BYTE-EQUAL across the three despite the different scan orders; only the
    PARENT choice on equal-value ties depends on the (sorted, deterministic) order here.

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
    snap = best.copy()                           # SNAPSHOT: relax from frozen source values
    for g in sorted(marked):                     # initial one-hop footpaths from egress stops
        for k in range(int(tr_off[g]), int(tr_off[g + 1])):
            j = int(tr_to[k]); cand = snap[g] - tr_time[k]
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
        snap = best.copy()                       # SNAPSHOT: relax from frozen source values
        tr_marked = set()
        for s in sorted(newly):
            for k in range(int(tr_off[s]), int(tr_off[s + 1])):
                j = int(tr_to[k]); cand = snap[s] - tr_time[k]
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
    Returns latest : int64[n_stops, n_deadlines] latest departure per stop per deadline.

    MONOTONE ROWS (2026-06-10): every row is non-decreasing in the deadline. Exact semantics
    guarantee it (a journey arriving by T also arrives by T' > T), but the marked-stop pruning
    + one-hop footpath policy can drop a propagation at a higher deadline that fired at a
    lower one (~58 of 3406 rows, worst 958 s, on the ferry workplace), and the binary
    searches downstream (``stop_arrival_profile``, the MC tail readout) require sorted rows.
    A row-wise running max over the (strictly ascending, asserted) deadline axis restores
    exactly those provably-feasible journeys — applied HERE for the python fallback and
    in-kernel at the end of ``raptor_numba._profile`` for the compiled path (the numba
    chokepoint also covers the MC ``_draw_profile``), so both kernels stay byte-equal."""
    n = data["n_stops"]
    deadlines = np.asarray(deadlines, dtype=np.int64)
    assert deadlines.size < 2 or bool(np.all(np.diff(deadlines) > 0)), \
        "reverse_profile: deadlines must be strictly ascending (the row cummax depends on it)"
    egress_g = np.asarray(egress_g, dtype=np.int32)
    egress_w = np.asarray(egress_w, dtype=np.int64)
    if kernel is None:
        kernel = _select_kernel()
    if kernel == "numba":
        from . import raptor_numba
        return raptor_numba.reverse_profile(data, egress_g, egress_w, deadlines,
                                            board_slack, max_rounds)
    latest = np.empty((n, len(deadlines)), dtype=np.int64)
    for ti, T in enumerate(deadlines):
        latest[:, ti] = reverse_raptor(data, egress_g, T - egress_w,
                                       max_rounds=max_rounds, board_slack=board_slack)
    np.maximum.accumulate(latest, axis=1, out=latest)   # monotone rows (see docstring)
    return latest


def stop_arrival_profile(latest, deadlines, dep_grid):
    """Invert latest-departure profiles into arrival-at-workplace profiles.

    arrivalW[s, k] = earliest workplace-arrival deadline reachable if you depart stop s no
    earlier than dep_grid[k] = min{ T in deadlines : latest[s, T] >= dep_grid[k] }.
    ``latest`` rows are GUARANTEED monotone non-decreasing since the row-wise cummax in
    ``reverse_profile`` (2026-06-10 — the old matched-UB caveat about binary-searching
    ~58 non-monotone rows is moot), so the per-row binary search is well-defined. The
    python fallback still implements the SAME explicit (lo+hi)>>1 binary search as
    ``raptor_numba.stop_arrival_profile`` so the two stay byte-equal on ANY input.
    Returns arrivalW : int64[n_stops, len(dep_grid)] (INF where never feasible)."""
    deadlines = np.asarray(deadlines, dtype=np.int64)
    dep_grid = np.asarray(dep_grid, dtype=np.int64)
    if _select_kernel() == "numba":
        from . import raptor_numba
        return raptor_numba.stop_arrival_profile(latest, deadlines, dep_grid)
    n = latest.shape[0]
    ndg = len(dep_grid)
    arrivalW = np.full((n, ndg), INF, dtype=np.int64)
    nT = len(deadlines)
    for s in range(n):
        row = latest[s]
        if row[-1] < dep_grid[0]:
            continue
        # masked vectorized binary search over the needles — each element follows exactly
        # the numba kernel's loop: while lo < hi: mid=(lo+hi)>>1; row[mid] < D ? lo : hi.
        lo = np.zeros(ndg, dtype=np.int64)
        hi = np.full(ndg, nT, dtype=np.int64)
        active = lo < hi
        while active.any():
            mid = (lo + hi) >> 1
            less = np.zeros(ndg, dtype=bool)
            less[active] = row[mid[active]] < dep_grid[active]
            lo = np.where(active & less, mid + 1, lo)
            hi = np.where(active & ~less, mid, hi)
            active = lo < hi
        ok = lo < nT
        arrivalW[s, ok] = deadlines[lo[ok]]
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
        args = (np.asarray(access_off, np.int64), np.asarray(access_to, np.int64),
                np.asarray(access_w, np.int64), np.asarray(purewalk, np.int64), arrivalW)
        # Arithmetic-indexing fast path (~7x): searchsorted(dep_grid, D + w) collapses to
        # di + ceil(w/step) when dep_grid is uniform AND shares origin + step with cell_deps.
        # The ENGINE's grids always qualify (DEP_STEP literal), but this is a public API with
        # caller-supplied grids (e.g. prototypes/spike_raptor/validate_core.py), so guard and
        # fall back to the general binsearch kernel when the assumption doesn't hold.
        uniform = False
        if len(dep_grid) >= 2 and len(cell_deps) >= 1:
            step = int(dep_grid[1] - dep_grid[0])
            uniform = (step > 0
                       and bool(np.all(np.diff(dep_grid) == step))
                       and int(dep_grid[0]) == int(cell_deps[0])
                       and bool(np.all(np.diff(cell_deps) == step)))
        if uniform:
            # arith-kernel guard: kw = ceil(w/step) assumes non-negative walk seconds — a
            # negative access_w would make k = di + kw index BEFORE the arrivalW row origin
            # (numba does no bounds checking, so that's a silent out-of-range read, not an
            # IndexError). Negative walk seconds are invalid input everywhere; fail loudly.
            assert args[2].size == 0 or int(args[2].min()) >= 0, \
                "assemble_departafter: access_w contains negative walk seconds"
            return raptor_numba.assemble_departafter_arith(
                *args, np.int64(dep_grid[0]), np.int64(step), np.int64(len(dep_grid)),
                cell_deps, np.int64(max_min), pct)
        return raptor_numba.assemble_departafter(
            *args, dep_grid, cell_deps, np.int64(max_min), pct)
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


# --------------------------------------------------------------- service-noise Monte-Carlo
def pat_trip_off(data):
    """CSR-style base into a per-(global)-trip array: trip g of pattern pi = off[pi] + trip."""
    off = np.zeros(len(data["pat_ntrips"]) + 1, dtype=np.int64)
    off[1:] = np.cumsum(np.asarray(data["pat_ntrips"], dtype=np.int64))
    return off


def apply_delays(data, delta0, slope, off=None):
    """Perturb the schedule by per-trip delay delta(pos)=delta0[g]+slope[g]*scheduled_elapsed(pos)
    (applied to BOTH dep & arr so dwell is preserved), then a per-pattern FIFO cumulative-max
    clamp per position across trips. Vectorized reference; MUST match raptor_numba._perturb.
    delta0/slope are per global trip (see ``pat_trip_off``). Returns (dep, arr) int32."""
    if off is None:
        off = pat_trip_off(data)
    pat_nstops = data["pat_nstops"]; pat_ntrips = data["pat_ntrips"]; pat_mat_off = data["pat_mat_off"]
    dep = np.asarray(data["pat_dep"], dtype=np.int64).copy()
    arr = np.asarray(data["pat_arr"], dtype=np.int64).copy()
    for pi in range(len(pat_nstops)):
        ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi]); mb = int(pat_mat_off[pi])
        if nt == 0 or ns == 0:
            continue
        D = dep[mb:mb + nt * ns].reshape(nt, ns)
        A = arr[mb:mb + nt * ns].reshape(nt, ns)
        tb = int(off[pi])
        d0 = delta0[tb:tb + nt][:, None]
        sl = slope[tb:tb + nt][:, None]
        elapsed = np.maximum(D - D[:, :1], 0)
        inc = (d0 + sl * elapsed).astype(np.int64)
        D = D + inc; A = A + inc
        np.maximum.accumulate(D, axis=0, out=D)        # no overtaking on dep ...
        np.maximum.accumulate(A, axis=0, out=A)        # ... or arr (keeps the arr column sorted)
        dep[mb:mb + nt * ns] = D.ravel(); arr[mb:mb + nt * ns] = A.ravel()
    return dep.astype(np.int32), arr.astype(np.int32)


def perturbed_data(data, delta0, slope, off=None):
    """A shallow copy of ``data`` with pat_dep/pat_arr replaced by one perturbed draw (the MC
    python-reference + alt-lines build-block). Returns (pdata, dep, arr)."""
    dep, arr = apply_delays(data, delta0, slope, off)
    pdata = dict(data); pdata["pat_dep"] = dep; pdata["pat_arr"] = arr
    return pdata, dep, arr


def _mc_flat_args(data):
    """The schedule CSR as the int64 tuple the MC kernel takes positionally: n_stops, the pattern
    arrays (incl. ``pat_trip_off`` for per-trip delay indexing), the routes-at-stop CSR, and the
    footpath CSR. One source of truth for the kernel's flat-arg contract (kept in sync with
    ``raptor_numba.montecarlo_committed`` parameter order; ``_draw_profile`` consumes the same
    prefix)."""
    return (np.int64(data["n_stops"]),
            data["pat_nstops"].astype(np.int64), data["pat_ntrips"].astype(np.int64),
            data["pat_stop_off"].astype(np.int64), data["pat_mat_off"].astype(np.int64),
            pat_trip_off(data),
            data["pat_stops"].astype(np.int64), data["pat_dep"].astype(np.int64),
            data["pat_arr"].astype(np.int64),
            data["ras_off"].astype(np.int64), data["ras_pat"].astype(np.int64),
            data["ras_pos"].astype(np.int64),
            data["tr_off"].astype(np.int64), data["tr_to"].astype(np.int64),
            data["tr_time"].astype(np.int64))


def montecarlo_commute_committed(data, egress_g, egress_w, deadlines, legs, perfect, max_min,
                                 delta0_all, slope_all, board_slack=60, max_rounds=8, kernel=None):
    """COMMITTED-PLAN commute_all[n_cells, R] in float minutes. ``legs`` is the per-cell committed
    first leg from ``raptor_journey.JourneyTree.committed_first_legs`` (departure + first board fixed
    from the unperturbed plan). Each draw perturbs the schedule, then per transit cell: board the
    next trip on the committed line, ride to the committed alight, and re-optimize the tail from the
    ACTUAL late arrival (so a missed transfer eats a real headway). Numba when available; the
    parallel kernel is serialized (workqueue not threadsafe)."""
    if kernel is None:
        kernel = _select_kernel()
    if kernel == "numba":
        from . import raptor_numba
        with _MC_KERNEL_LOCK:                          # serialize the non-threadsafe parallel kernel
            return raptor_numba.montecarlo_committed(
                *_mc_flat_args(data),
                np.asarray(egress_g, np.int64), np.asarray(egress_w, np.int64),
                np.asarray(deadlines, np.int64), np.int64(board_slack), np.int64(max_rounds),
                np.asarray(legs["commit_home"], np.int64), np.asarray(legs["commit_kind"], np.int64),
                np.asarray(legs["commit_walk0"], np.int64), np.asarray(legs["commit_pi"], np.int64),
                np.asarray(legs["commit_bpos"], np.int64), np.asarray(legs["commit_apos"], np.int64),
                np.asarray(legs["commit_as"], np.int64), np.asarray(perfect, np.int64),
                np.int64(max_min),
                np.ascontiguousarray(delta0_all, np.float64),
                np.ascontiguousarray(slope_all, np.float64))
    return _montecarlo_committed_python(data, egress_g, egress_w, deadlines, legs, perfect,
                                        max_min, delta0_all, slope_all, board_slack, max_rounds)


def _montecarlo_committed_python(data, egress_g, egress_w, deadlines, legs, perfect, max_min,
                                 delta0_all, slope_all, board_slack, max_rounds):
    """Slow pure-numpy reference for ``montecarlo_commute_committed`` (test/no-numba path)."""
    off = pat_trip_off(data)
    deadlines = np.asarray(deadlines, np.int64)
    nd = len(deadlines)
    home = np.asarray(legs["commit_home"]); kind = np.asarray(legs["commit_kind"])
    walk0 = np.asarray(legs["commit_walk0"]); pic = np.asarray(legs["commit_pi"])
    bpos = np.asarray(legs["commit_bpos"]); apos = np.asarray(legs["commit_apos"])
    as_ = np.asarray(legs["commit_as"]); perfect = np.asarray(perfect)
    pat_nstops = data["pat_nstops"]; pat_ntrips = data["pat_ntrips"]; pat_mat_off = data["pat_mat_off"]
    n_cells = len(home)
    Rn = delta0_all.shape[0]
    cm = np.empty((n_cells, Rn), dtype=np.float64)
    capf = float(max_min)
    for r in range(Rn):
        pdata, dep, arr = perturbed_data(data, delta0_all[r], slope_all[r], off)
        latest = reverse_profile(pdata, egress_g, egress_w, deadlines,
                                 board_slack=board_slack, max_rounds=max_rounds, kernel="python")
        for ci in range(n_cells):
            k = int(kind[ci])
            if k != 2:
                p = int(perfect[ci])
                cm[ci, r] = capf if (k == 0 or p < 0) else min(float(p), capf)
                continue
            pi = int(pic[ci]); ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi])
            mb = int(pat_mat_off[pi]); bp = int(bpos[ci]); ap = int(apos[ci])
            key = int(home[ci]) + int(walk0[ci])      # = your committed trip's departure (no extra slack)
            depcol = dep[mb + bp: mb + nt * ns: ns]
            lo = int(np.searchsorted(depcol, key, side="left"))
            if lo >= nt:
                cm[ci, r] = capf; continue
            arr_as = int(arr[mb + lo * ns + ap])
            # + board_slack: the sweep charges it at every transit-alight -> onward seam
            lo2 = int(np.searchsorted(latest[int(as_[ci])], arr_as + board_slack, side="left"))
            if lo2 >= nd:
                cm[ci, r] = capf; continue
            tt = (int(deadlines[lo2]) - int(home[ci])) / 60.0
            cm[ci, r] = min(max(tt, 0.0), capf)
    return cm
