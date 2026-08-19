"""Numba-compiled reverse range-RAPTOR sweep — the production hot path.

``_profile`` is a faithful, byte-equivalent port of the validated pure-python
``raptor.reverse_raptor`` (same seeding, same one-hop transfer policy, same FIFO per-position
binary search), compiled to a single ``nogil`` kernel that sweeps ALL arrival deadlines in one
call. ``nogil`` lets the server run concurrent requests on real threads without GIL contention;
``cache=True`` persists the compiled object so warmup is one-time.

LOCKSTEP: ``_profile`` has TWO siblings whose routing semantics must stay byte-equivalent —
``raptor.reverse_raptor`` (pure-python reference/test oracle) and ``raptor.reverse_raptor_traced``
(reference + back-pointers for journey reconstruction). A semantic change to any one (board
slack, transfer seeding, binary-search column, ...) must land in all three.

SNAPSHOT FOOTPATH RELAX (2026-06-10): every footpath relaxation pass (the egress-seed pass and
each round's pass) reads its SOURCE values from a snapshot frozen at the start of the pass
(``srcv``), never live ``best[]`` — so the relax is order-independent and the three siblings
are BYTE-EQUAL despite their different scan orders (index order here, hash-set in the
reference, sorted in traced). Enforced by tests/test_raptor.py::test_lockstep_byte_equal.

MONOTONE PROFILE ROWS (2026-06-10): ``_profile`` finishes with a row-wise running max across
the (ascending) deadline axis — the pruned sweep can drop a feasible propagation at a higher
deadline, and the cummax restores exactly those provably-feasible journeys, so every
downstream binary search over a profile row (``stop_arrival_profile``, the MC tail readout
``_first_ge``) operates on genuinely sorted data. The single-deadline siblings are unaffected
(one column); lockstep is asserted per-deadline on the raw columns, cummaxed for the profile.

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
    srcv = np.empty(n_stops, dtype=np.int64)       # snapshot of relax-source values per pass

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
        for ii in range(seed_n):                   # SNAPSHOT source values (order-independence)
            srcv[ii] = best[cur[ii]]
        for ii in range(seed_n):
            s = cur[ii]
            for k in range(tr_off[s], tr_off[s + 1]):
                j = tr_to[k]
                cand = srcv[ii] - tr_time[k]
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
                        # `trip` only moves upward as this pattern is scanned backwards.
                        # Anything at or below it can therefore never improve the selected
                        # vehicle.  The FIFO arrival column is sorted, so one successor probe
                        # proves that no later trip can win; otherwise search only the suffix.
                        # This is exactly the old right-side search followed by max(trip, idx),
                        # including the case where the old trip is not boardable at this stop.
                        if trip < nt - 1:
                            first = trip + 1
                            if pat_arr[mbase + first * ns + pos] <= key:
                                lo = first + 1
                                hi = nt
                                while lo < hi:          # upper_bound(arr_col, key), suffix only
                                    mid = (lo + hi) >> 1
                                    if pat_arr[mbase + mid * ns + pos] <= key:
                                        lo = mid + 1
                                    else:
                                        hi = mid
                                trip = lo - 1
            # commit prev for pattern-marked stops
            for ii in range(nn):
                prev[new[ii]] = best[new[ii]]
            # SNAPSHOT source values: the relax below must never read live best[] (a target
            # that is also a source would propagate a 2-hop cascade in scan order)
            for ii in range(nn):
                srcv[ii] = best[new[ii]]
            # one-hop transfers from the pattern-marked stops (mirror the python policy)
            tn = nn
            for ii in range(nn):
                s = new[ii]
                for k in range(tr_off[s], tr_off[s + 1]):
                    j = tr_to[k]
                    cand = srcv[ii] - tr_time[k]
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
    # MONOTONE CUMMAX (2026-06-10): a journey arriving by deadline T also arrives by any
    # T' > T, so every stop's latest-departure row is non-decreasing in the deadline in
    # EXACT semantics — but the marked-stop pruning + one-hop footpath policy can drop a
    # propagation at a higher deadline that fired at a lower one (~58 rows, worst 958 s on
    # the ferry workplace). A row-wise running max restores exactly those provably-feasible
    # journeys, making every downstream inversion/binary search (stop_arrival_profile,
    # the MC tail readout `_first_ge`) well-defined. REQUIRES ascending ``deadlines``
    # (asserted in both reverse_profile wrappers; every caller uses np.arange grids).
    # This is the numba-side chokepoint: both reverse_profile and the MC _draw_profile
    # consume `latest` straight from this kernel. Mirrored by np.maximum.accumulate in
    # the python fallback (raptor.reverse_profile).
    for s in range(n_stops):
        run = np.int64(NEG)
        for ti in range(nd):
            v = latest[s, ti]
            if v > run:
                run = v
            else:
                latest[s, ti] = run
    return latest


@njit(cache=True, nogil=True)
def _traced_compact(n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off,
                    pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos,
                    tr_off, tr_to, tr_time, egress_g, egress_t, egress_w,
                    board_slack, max_rounds):
    """Compiled single-deadline traced RAPTOR with deferred winning-parent materialization.

    Times and tie order mirror ``raptor.reverse_raptor_traced``.  During one pattern or footpath
    pass, a stop's intermediate strict improvements cannot be consumed: boards read the frozen
    ``prev`` label and footpaths read the frozen source snapshot.  Materialize only the final
    winning node per stop/pass, reducing the append-only table from hundreds of thousands of
    Python objects to at most ``(2 + 2*rounds) * n_stops`` compact numeric rows.
    """
    n_pat = pat_nstops.shape[0]
    best = np.full(n_stops, NEG, dtype=np.int64)
    prev = np.full(n_stops, NEG, dtype=np.int64)
    par_kind = np.full(n_stops, -1, dtype=np.int8)
    par_pat = np.full(n_stops, -1, dtype=np.int32)
    par_trip = np.full(n_stops, -1, dtype=np.int32)
    par_board = np.full(n_stops, -1, dtype=np.int32)
    par_alight = np.full(n_stops, -1, dtype=np.int32)
    par_from = np.full(n_stops, -1, dtype=np.int32)
    par_nxfer = np.zeros(n_stops, dtype=np.int16)
    egress_sec = np.full(n_stops, -1, dtype=np.int32)
    best_node = np.full(n_stops, -1, dtype=np.int32)
    prev_node = np.full(n_stops, -1, dtype=np.int32)

    cap = (2 + 2 * max_rounds) * n_stops + egress_g.shape[0] + 1
    nd_kind = np.empty(cap, dtype=np.int8)
    nd_stop = np.empty(cap, dtype=np.int32)
    nd_pat = np.empty(cap, dtype=np.int32)
    nd_trip = np.empty(cap, dtype=np.int32)
    nd_board = np.empty(cap, dtype=np.int32)
    nd_alight = np.empty(cap, dtype=np.int32)
    nd_to = np.empty(cap, dtype=np.int32)
    nd_egress = np.empty(cap, dtype=np.int32)
    nd_depth = np.empty(cap, dtype=np.int16)
    nd_next = np.empty(cap, dtype=np.int32)
    node_n = 0

    flag = np.zeros(n_stops, dtype=np.uint8)
    cur = np.empty(n_stops, dtype=np.int64)
    new = np.empty(n_stops, dtype=np.int64)
    q_pos = np.full(n_pat, -1, dtype=np.int64)
    srcv = np.empty(n_stops, dtype=np.int64)
    srcnode = np.empty(n_stops, dtype=np.int32)

    tr_pending = np.zeros(n_stops, dtype=np.uint8)
    tr_pi = np.empty(n_stops, dtype=np.int32)
    tr_trip_v = np.empty(n_stops, dtype=np.int32)
    tr_board_v = np.empty(n_stops, dtype=np.int32)
    tr_alight_v = np.empty(n_stops, dtype=np.int32)
    tr_next_v = np.empty(n_stops, dtype=np.int32)
    tr_depth_v = np.empty(n_stops, dtype=np.int16)

    foot_pending = np.zeros(n_stops, dtype=np.uint8)
    foot_from_v = np.empty(n_stops, dtype=np.int32)
    foot_next_v = np.empty(n_stops, dtype=np.int32)
    foot_depth_v = np.empty(n_stops, dtype=np.int16)
    foot_round_v = np.empty(n_stops, dtype=np.int16)

    # Egress seeds are terminal nodes and may be consumed by the frozen initial footpath pass.
    for e in range(egress_g.shape[0]):
        g = int(egress_g[e]); t = int(egress_t[e])
        if t > best[g]:
            best[g] = t; prev[g] = t; par_kind[g] = 0; egress_sec[g] = int(egress_w[e])
            nd_kind[node_n] = 0; nd_stop[node_n] = g; nd_depth[node_n] = 0
            nd_next[node_n] = -1; nd_pat[node_n] = -1; nd_trip[node_n] = -1
            nd_board[node_n] = -1; nd_alight[node_n] = -1; nd_to[node_n] = -1
            nd_egress[node_n] = int(egress_w[e])
            best_node[g] = node_n; prev_node[g] = node_n; node_n += 1
            flag[g] = 1

    cn = 0
    for s in range(n_stops):
        if flag[s] != 0:
            cur[cn] = s; srcv[cn] = best[s]; srcnode[cn] = best_node[s]; cn += 1
    seed_n = cn
    for ii in range(seed_n):
        s = int(cur[ii])
        source_depth = int(nd_depth[int(srcnode[ii])])
        for k in range(int(tr_off[s]), int(tr_off[s + 1])):
            j = int(tr_to[k]); cand = int(srcv[ii]) - int(tr_time[k])
            if cand > best[j]:
                best[j] = cand; prev[j] = cand
                par_kind[j] = 2; par_from[j] = s; par_nxfer[j] = 0
                par_pat[j] = -1; par_trip[j] = -1; par_board[j] = -1; par_alight[j] = -1
                foot_pending[j] = 1; foot_from_v[j] = s; foot_next_v[j] = srcnode[ii]
                foot_depth_v[j] = source_depth; foot_round_v[j] = 0; flag[j] = 1
    for j in range(n_stops):
        if foot_pending[j] != 0:
            nd_kind[node_n] = 2; nd_stop[node_n] = j; nd_depth[node_n] = foot_depth_v[j]
            nd_next[node_n] = foot_next_v[j]; nd_pat[node_n] = -1; nd_trip[node_n] = -1
            nd_board[node_n] = -1; nd_alight[node_n] = -1; nd_to[node_n] = foot_from_v[j]
            nd_egress[node_n] = -1
            best_node[j] = node_n; prev_node[j] = node_n; node_n += 1
            foot_pending[j] = 0
    cn = 0
    for s in range(n_stops):
        if flag[s] != 0:
            cur[cn] = s; cn += 1

    for rnd in range(1, max_rounds + 1):
        if cn == 0:
            break
        for ii in range(cn):
            s = int(cur[ii])
            for k in range(int(ras_off[s]), int(ras_off[s + 1])):
                pi = int(ras_pat[k]); pos = int(ras_pos[k])
                if pos > q_pos[pi]:
                    q_pos[pi] = pos
            flag[s] = 0

        for pi in range(n_pat):
            end_pos = int(q_pos[pi])
            if end_pos < 0:
                continue
            q_pos[pi] = -1
            ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi])
            sbase = int(pat_stop_off[pi]); mbase = int(pat_mat_off[pi])
            trip = -1; cur_alight = -1; cont_node = -1; cont_depth = 0
            for pos in range(end_pos, -1, -1):
                s = int(pat_stops[sbase + pos])
                if trip >= 0:
                    d = int(pat_dep[mbase + trip * ns + pos])
                    if d > best[s]:
                        best[s] = d
                        par_kind[s] = 1; par_pat[s] = pi; par_trip[s] = trip
                        par_board[s] = pos; par_alight[s] = cur_alight
                        par_nxfer[s] = rnd; par_from[s] = -1
                        tr_pending[s] = 1; tr_pi[s] = pi; tr_trip_v[s] = trip
                        tr_board_v[s] = pos; tr_alight_v[s] = cur_alight
                        tr_next_v[s] = cont_node; tr_depth_v[s] = cont_depth; flag[s] = 1
                pa = int(prev[s])
                if pa > NEG:
                    key = pa - int(board_slack)
                    lo = 0; hi = nt
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if int(pat_arr[mbase + mid * ns + pos]) <= key:
                            lo = mid + 1
                        else:
                            hi = mid
                    idx = lo - 1
                    if idx >= 0 and (trip < 0 or idx > trip):
                        trip = idx; cur_alight = pos; cont_node = int(prev_node[s])
                        cont_depth = (int(nd_depth[cont_node]) if cont_node >= 0 else 0)

        nn = 0
        for s in range(n_stops):
            if tr_pending[s] != 0:
                nd_kind[node_n] = 1; nd_stop[node_n] = s
                nd_depth[node_n] = tr_depth_v[s] + 1; nd_next[node_n] = tr_next_v[s]
                nd_pat[node_n] = tr_pi[s]; nd_trip[node_n] = tr_trip_v[s]
                nd_board[node_n] = tr_board_v[s]; nd_alight[node_n] = tr_alight_v[s]
                nd_to[node_n] = -1; nd_egress[node_n] = -1
                best_node[s] = node_n; node_n += 1; tr_pending[s] = 0
            if flag[s] != 0:
                new[nn] = s; nn += 1
        for ii in range(nn):
            s = int(new[ii]); prev[s] = best[s]; prev_node[s] = best_node[s]
            srcv[ii] = best[s]; srcnode[ii] = best_node[s]

        for ii in range(nn):
            s = int(new[ii]); source_node = int(srcnode[ii])
            source_depth = int(nd_depth[source_node])
            for k in range(int(tr_off[s]), int(tr_off[s + 1])):
                j = int(tr_to[k]); cand = int(srcv[ii]) - int(tr_time[k])
                if cand > best[j]:
                    best[j] = cand; prev[j] = cand
                    par_kind[j] = 2; par_from[j] = s; par_nxfer[j] = par_nxfer[s]
                    par_pat[j] = -1; par_trip[j] = -1; par_board[j] = -1; par_alight[j] = -1
                    foot_pending[j] = 1; foot_from_v[j] = s; foot_next_v[j] = source_node
                    foot_depth_v[j] = source_depth; foot_round_v[j] = par_nxfer[s]; flag[j] = 1
        for j in range(n_stops):
            if foot_pending[j] != 0:
                nd_kind[node_n] = 2; nd_stop[node_n] = j; nd_depth[node_n] = foot_depth_v[j]
                nd_next[node_n] = foot_next_v[j]; nd_pat[node_n] = -1; nd_trip[node_n] = -1
                nd_board[node_n] = -1; nd_alight[node_n] = -1; nd_to[node_n] = foot_from_v[j]
                nd_egress[node_n] = -1
                best_node[j] = node_n; prev_node[j] = node_n; node_n += 1
                foot_pending[j] = 0
        cn = 0
        for s in range(n_stops):
            if flag[s] != 0:
                cur[cn] = s; cn += 1

    return (best, par_kind, par_pat, par_trip, par_board, par_alight, par_from,
            par_nxfer, egress_sec, best_node,
            nd_kind[:node_n], nd_stop[:node_n], nd_pat[:node_n], nd_trip[:node_n],
            nd_board[:node_n], nd_alight[:node_n], nd_to[:node_n], nd_egress[:node_n],
            nd_depth[:node_n], nd_next[:node_n])


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
                         dep_grid, cell_deps, max_min, pct, beta, eps):
    """``beta`` = walk-reluctance multiplier (config.WALK_RELUCTANCE; 1.0 = no prior), ``eps`` = the
    hard cap (sec) on how much true time the prior may give up. Among access stops whose TRUE
    travel is within ``eps`` of the time-optimal (so the reported time changes by <= ~1 min and a
    genuinely-faster stop is never traded away), the stop minimizing ``true_time + (beta-1)*aw`` is
    chosen, but the TRUE ``true_time`` is reported. Pure walk competes on ``pw*beta`` but reports
    true ``pw``. ``beta == 1.0`` -> plain min-true-time."""
    n_cells = access_off.shape[0] - 1
    ndg = dep_grid.shape[0]
    nd = cell_deps.shape[0]
    npct = pct.shape[0]
    out = np.full((n_cells, npct), -1, dtype=np.int32)
    ttm = np.empty(nd, dtype=np.float64)
    bw = beta - 1.0
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        pw = purewalk[ci]
        for di in range(nd):
            D = cell_deps[di]
            opt = INF                             # pass 1: the time-optimal TRUE arrival
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
                    if v < opt:
                        opt = v
            best = INF                            # pass 2: penalized winner WITHIN the eps window
            bestsc = 1e18
            if opt < INF:
                for a in range(a0, a1):
                    board = D + access_w[a]
                    lo = 0
                    hi = ndg
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if dep_grid[mid] < board:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo < ndg:
                        v = arrivalW[access_to[a], lo]
                        if v < INF and v <= opt + eps:       # eps window: ~same time only
                            sc = (v - D) + bw * access_w[a]  # penalized score (decision only)
                            if sc < bestsc:
                                bestsc = sc; best = v
            tt = (best - D) / 60.0 if best < INF else 1e18
            # pure walk competes on its PENALIZED seconds (pw*beta) vs the chosen transit's
            # penalized score (bestsc), reporting the TRUE pw seconds if it wins.
            if pw >= 0 and pw * beta < (bestsc if best < INF else 1e18):
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
def assemble_departafter_arith(access_off, access_to, access_w, purewalk, arrivalW,
                               dep0, step, ndg, cell_deps, max_min, pct, beta, eps):
    """``assemble_departafter`` specialized for a UNIFORM dep_grid sharing origin+step with
    cell_deps (the engine's grids always do; ``raptor.assemble_departafter`` guards and falls
    back to the binsearch kernel otherwise). Then
        searchsorted(dep_grid, cell_deps[di] + w, 'left') == di + ceil(w / step)
    so the ~access_pairs x deadlines binary searches become direct reads, with ceil(w/step)
    hoisted per access pair and the arrivalW row hoisted out of the deadline loop.
    Byte-equal to the binsearch kernel on the engine grids (gated on all 5 golden workplaces).
    ``beta``/``eps`` = the walk-reluctance multiplier + true-time cap (decision-only; reports true
    clock time — see ``assemble_departafter``). Two passes per cell: pass 1 the time-optimal true
    arrival per di, pass 2 the penalized winner within the eps window."""
    n_cells = access_off.shape[0] - 1
    nd = cell_deps.shape[0]
    npct = pct.shape[0]
    out = np.full((n_cells, npct), -1, dtype=np.int32)
    ttm = np.empty(nd, dtype=np.float64)          # TRUE time (sec) of the chosen access stop per di
    scm = np.empty(nd, dtype=np.float64)          # penalized score (sec) that drove the choice
    optv = np.empty(nd, dtype=np.int64)           # time-optimal TRUE arrival per di
    bw = beta - 1.0
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        pw = purewalk[ci]
        for di in range(nd):
            ttm[di] = 1e18; scm[di] = 1e18; optv[di] = INF
        for a in range(a0, a1):                      # pass 1: time-optimal arrival per di
            w = access_w[a]
            kw = (w + step - 1) // step
            row = arrivalW[access_to[a]]
            for di in range(nd):
                k = di + kw
                if k < ndg:
                    v = row[k]
                    if v < optv[di]:
                        optv[di] = v
        for a in range(a0, a1):                      # pass 2: penalized winner within eps window
            w = access_w[a]
            kw = (w + step - 1) // step
            pen = bw * w                             # walk penalty for this pair (sec), hoisted
            row = arrivalW[access_to[a]]
            for di in range(nd):
                k = di + kw
                if k < ndg:
                    v = row[k]
                    if v < INF and v <= optv[di] + eps:    # eps window: ~same time only
                        tt = (v - cell_deps[di])     # TRUE seconds to work
                        sc = tt + pen                # penalized score (decision only)
                        if sc < scm[di]:
                            scm[di] = sc; ttm[di] = tt / 60.0
        for di in range(nd):
            tt = ttm[di]
            # pure walk competes on its penalized seconds vs the chosen transit's penalized score
            if pw >= 0 and pw * beta < scm[di]:
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
def select_departafter(access_off, access_to, access_w, purewalk, arrivalW,
                       dep_grid, cell_deps, max_min, pct1, beta, eps):
    """Per-cell SERVED-PERCENTILE selection (the tracer's anchor), lockstep with
    ``assemble_departafter`` for the SINGLE percentile ``pct1``.

    Re-runs the IDENTICAL per-departure penalized eps-window pick ``assemble_departafter`` does
    (two passes: time-optimal ``opt`` first, then the walk-reluctance argmin within ``arr<=opt+eps``,
    then pure-walk competition on ``pw*beta``), records per departure the chosen stop / its TRUE
    arrival / whether walk won, computes the SAME ``_pct_lower`` percentile minute, then picks the
    representative departure D* = the LATEST departure whose rounded minute equals that painted value
    and emits its (s*, D*, T* = chosen arrivalW, aw, is_walk). Returns ``painted`` (the same minute
    the value kernel returns for this percentile) plus the cell-aligned selection arrays. The minute
    is byte-identical to ``assemble_departafter``'s column for ``pct1`` (same arithmetic + percentile),
    so the value the map paints is UNCHANGED — this only ADDS the discarded selection."""
    n_cells = access_off.shape[0] - 1
    ndg = dep_grid.shape[0]
    nd = cell_deps.shape[0]
    painted = np.full(n_cells, -1, dtype=np.int32)
    s_star = np.full(n_cells, -1, dtype=np.int64)
    aw_sel = np.zeros(n_cells, dtype=np.int64)
    Dstar = np.full(n_cells, NEG, dtype=np.int64)
    Tstar = np.full(n_cells, -1, dtype=np.int64)
    is_walk = np.zeros(n_cells, dtype=np.bool_)
    ttm = np.empty(nd, dtype=np.float64)          # rounded minutes (percentile input)
    d_stop = np.empty(nd, dtype=np.int64)         # chosen transit stop per di (-1 walk/unreach)
    d_arr = np.empty(nd, dtype=np.int64)          # chosen stop's TRUE arrivalW per di
    d_walk = np.empty(nd, dtype=np.bool_)         # pure walk won this di
    bw = beta - 1.0
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        pw = purewalk[ci]
        for di in range(nd):
            D = cell_deps[di]
            opt = INF                             # pass 1: the time-optimal TRUE arrival
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
                    if v < opt:
                        opt = v
            best = INF                            # pass 2: penalized winner WITHIN the eps window
            bestsc = 1e18
            bstop = -1                            # chosen stop id for the winner
            if opt < INF:
                for a in range(a0, a1):
                    board = D + access_w[a]
                    lo = 0
                    hi = ndg
                    while lo < hi:
                        mid = (lo + hi) >> 1
                        if dep_grid[mid] < board:
                            lo = mid + 1
                        else:
                            hi = mid
                    if lo < ndg:
                        v = arrivalW[access_to[a], lo]
                        if v < INF and v <= opt + eps:
                            sc = (v - D) + bw * access_w[a]
                            if sc < bestsc:
                                bestsc = sc; best = v; bstop = access_to[a]
            tt = (best - D) / 60.0 if best < INF else 1e18
            walk_won = False
            if pw >= 0 and pw * beta < (bestsc if best < INF else 1e18):
                tt = pw / 60.0; walk_won = True; bstop = -1; best = INF
            if tt > max_min:
                tt = max_min
            ttm[di] = np.ceil(tt)
            d_stop[di] = bstop
            d_arr[di] = best if best < INF else -1
            d_walk[di] = walk_won
        srt = np.sort(ttm)
        v = _pct_lower(srt, pct1)
        if v >= max_min:
            continue                              # unreachable (painted stays -1)
        pm = np.int32(v)
        painted[ci] = pm
        # representative departure = LATEST di achieving the painted minute (stable single journey)
        Di = -1
        for di in range(nd - 1, -1, -1):
            if np.int32(ttm[di]) == pm:
                Di = di; break
        if Di < 0:
            continue
        Dstar[ci] = cell_deps[Di]
        if d_walk[Di] or d_stop[Di] < 0:
            is_walk[ci] = True
            aw_sel[ci] = pw if pw >= 0 else 0
        else:
            ss = d_stop[Di]
            s_star[ci] = ss
            Tstar[ci] = d_arr[Di]
            for a in range(a0, a1):               # access-walk sec of the chosen stop
                if access_to[a] == ss:
                    aw_sel[ci] = access_w[a]; break
    return painted, s_star, aw_sel, Dstar, Tstar, is_walk


@njit(cache=True, nogil=True)
def select_departafter_arith(access_off, access_to, access_w, purewalk, arrivalW,
                             dep0, step, ndg, cell_deps, max_min, pct1, beta, eps):
    """``select_departafter`` specialized for a UNIFORM dep_grid (origin+step == cell_deps), lockstep
    with ``assemble_departafter_arith``. Same kw=ceil(w/step) arithmetic indexing; emits the served-
    percentile selection (s*, D*, T*, aw, is_walk) + the painted minute byte-identical to the arith
    value kernel's ``pct1`` column."""
    n_cells = access_off.shape[0] - 1
    nd = cell_deps.shape[0]
    painted = np.full(n_cells, -1, dtype=np.int32)
    s_star = np.full(n_cells, -1, dtype=np.int64)
    aw_sel = np.zeros(n_cells, dtype=np.int64)
    Dstar = np.full(n_cells, NEG, dtype=np.int64)
    Tstar = np.full(n_cells, -1, dtype=np.int64)
    is_walk = np.zeros(n_cells, dtype=np.bool_)
    ttm = np.empty(nd, dtype=np.float64)          # TRUE time (sec) of chosen stop per di
    scm = np.empty(nd, dtype=np.float64)          # penalized score that drove the choice
    optv = np.empty(nd, dtype=np.int64)           # time-optimal TRUE arrival per di
    bstopv = np.empty(nd, dtype=np.int64)         # chosen stop per di (-1 if none)
    barrv = np.empty(nd, dtype=np.int64)          # chosen stop's TRUE arrivalW per di
    bw = beta - 1.0
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        pw = purewalk[ci]
        for di in range(nd):
            ttm[di] = 1e18; scm[di] = 1e18; optv[di] = INF; bstopv[di] = -1; barrv[di] = -1
        for a in range(a0, a1):                      # pass 1: time-optimal arrival per di
            w = access_w[a]
            kw = (w + step - 1) // step
            row = arrivalW[access_to[a]]
            for di in range(nd):
                k = di + kw
                if k < ndg:
                    v = row[k]
                    if v < optv[di]:
                        optv[di] = v
        for a in range(a0, a1):                      # pass 2: penalized winner within eps window
            w = access_w[a]
            kw = (w + step - 1) // step
            pen = bw * w
            stop = access_to[a]
            row = arrivalW[stop]
            for di in range(nd):
                k = di + kw
                if k < ndg:
                    v = row[k]
                    if v < INF and v <= optv[di] + eps:
                        tt = (v - cell_deps[di])
                        sc = tt + pen
                        if sc < scm[di]:
                            scm[di] = sc; ttm[di] = tt / 60.0; bstopv[di] = stop; barrv[di] = v
        for di in range(nd):
            tt = ttm[di]
            walk_won = False
            if pw >= 0 and pw * beta < scm[di]:
                tt = pw / 60.0; walk_won = True; bstopv[di] = -1; barrv[di] = -1
            if tt > max_min:
                tt = max_min
            ttm[di] = np.ceil(tt)
            if walk_won:
                bstopv[di] = -1
        srt = np.sort(ttm)
        v = _pct_lower(srt, pct1)
        if v >= max_min:
            continue
        pm = np.int32(v)
        painted[ci] = pm
        Di = -1
        for di in range(nd - 1, -1, -1):
            if np.int32(ttm[di]) == pm:
                Di = di; break
        if Di < 0:
            continue
        Dstar[ci] = cell_deps[Di]
        if bstopv[Di] < 0:
            is_walk[ci] = True
            aw_sel[ci] = pw if pw >= 0 else 0
        else:
            ss = bstopv[Di]
            s_star[ci] = ss
            Tstar[ci] = barrv[Di]
            for a in range(a0, a1):
                if access_to[a] == ss:
                    aw_sel[ci] = access_w[a]; break
    return painted, s_star, aw_sel, Dstar, Tstar, is_walk


@njit(cache=True, nogil=True)
def select_planned_departafter_arith(access_off, access_to, access_w, purewalk, arrivalW,
                                     dep0, step, ndg, cell_deps, max_min):
    """First-boarding-anchored scheduled commute.

    For each access stop and each first-boarding minute B in ``cell_deps``, score the journey as
    ``access_walk + arrivalW[stop, B] - B``. That treats the home departure as derived from the
    selected first vehicle (home = B - access_walk), so the controllable pre-board wait is not part
    of the displayed commute. The candidate set of B minutes is identical across walk speeds; faster
    walking can only reduce walk terms / unlock feasibility, so the value is monotone.

    LOCKSTEP SIBLING of ``raptor.select_planned_departafter``'s python fallback: byte-equal on the
    shared uniform-grid domain (this kernel gates on uniformity; every B is a dep_grid point here).
    Like the legacy kernels, ``is_walk``/the synthetic walk ``Dstar`` are written ONLY when the
    cell is actually painted (``pm < max_min``); unreachable cells keep the untouched sentinels.
    """
    n_cells = access_off.shape[0] - 1
    nd = cell_deps.shape[0]
    painted = np.full(n_cells, -1, dtype=np.int32)
    s_star = np.full(n_cells, -1, dtype=np.int64)
    aw_sel = np.zeros(n_cells, dtype=np.int64)
    Dstar = np.full(n_cells, NEG, dtype=np.int64)      # home departure = selected B - access walk
    Tstar = np.full(n_cells, -1, dtype=np.int64)       # workplace arrival from arrivalW
    is_walk = np.zeros(n_cells, dtype=np.bool_)
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        best = INF
        best_s = -1
        best_aw = 0
        best_B = NEG
        best_T = -1
        for a in range(a0, a1):
            w = access_w[a]
            row = arrivalW[access_to[a]]
            for di in range(nd):
                # cell_deps and dep_grid share origin/step in the engine, so B's row index is di.
                if di >= ndg:
                    continue
                T = row[di]
                if T >= INF:
                    continue
                B = cell_deps[di]
                cost = T - B + w
                # Determinism/tie-break: same displayed time -> later first boarding (less platform
                # slack under minute discretization), then shorter access walk, then lower stop id.
                if (cost < best
                        or (cost == best and (B > best_B
                            or (B == best_B and (w < best_aw
                                or (w == best_aw and access_to[a] < best_s)))))):
                    best = cost
                    best_s = access_to[a]
                    best_aw = w
                    best_B = B
                    best_T = T
        pw = purewalk[ci]
        walk_won = False
        if pw >= 0 and pw < best:
            best = pw
            best_s = -1
            best_aw = pw
            best_B = cell_deps[nd - 1] if nd > 0 else dep0
            best_T = -1
            walk_won = True
        pm = (best + 59) // 60
        if pm < max_min:
            painted[ci] = np.int32(pm)
            s_star[ci] = best_s
            aw_sel[ci] = best_aw
            Dstar[ci] = best_B - best_aw
            Tstar[ci] = best_T
            is_walk[ci] = walk_won
    return painted, s_star, aw_sel, Dstar, Tstar, is_walk


@njit(cache=True, nogil=True)
def assemble_arriveby(access_off, access_to, access_w, purewalk, latest, deadlines,
                      max_min, pct, beta, eps):
    """``beta``/``eps`` = the walk-reluctance multiplier + the true-time cap (decision-only). Among
    access stops whose TRUE home departure is within ``eps`` of the latest (time-optimal) one, the
    stop maximizing the PENALIZED home ``home - (beta-1)*access_w`` is chosen, but the reported time
    uses the TRUE chosen ``home`` so the map minutes stay exact clock time. Pure walk competes on
    ``pw*beta``. ``beta == 1.0`` -> plain max-true-home."""
    n_cells = access_off.shape[0] - 1
    nd = deadlines.shape[0]
    npct = pct.shape[0]
    out = np.full((n_cells, npct), -1, dtype=np.int32)
    ttm = np.empty(nd, dtype=np.float64)
    NEGH = -(1 << 60) // 2
    bw = beta - 1.0
    for ci in range(n_cells):
        a0 = access_off[ci]
        a1 = access_off[ci + 1]
        pw = purewalk[ci]
        for di in range(nd):
            T = deadlines[di]
            opt = -(1 << 60)                      # pass 1: time-optimal (latest) TRUE home
            for a in range(a0, a1):
                h = latest[access_to[a], di] - access_w[a]
                if h > opt:
                    opt = h
            home = -(1 << 60)                     # pass 2: penalized winner WITHIN the eps window
            bestph = -1e18
            if opt > NEGH:
                for a in range(a0, a1):
                    h = latest[access_to[a], di] - access_w[a]
                    if h > NEGH and h >= opt - eps:           # eps window: ~same time only
                        ph = h - bw * access_w[a]             # penalized home (decision only)
                        if ph > bestph:
                            bestph = ph; home = h
            tt = (T - home) / 60.0 if home > NEGH else 1e18
            # pure walk competes on penalized travel pw*beta vs the chosen transit's penalized
            # travel (T - bestph); reports the true pw minutes if it wins.
            if pw >= 0 and (T - pw) >= 0 and pw * beta < (T - bestph if home > NEGH else 1e18):
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
    """latest[n_stops, n_deadlines] via the compiled kernel. Rows are MONOTONE non-decreasing
    in the deadline (the in-kernel cummax — see _profile), which requires ascending deadlines."""
    deadlines = np.asarray(deadlines, dtype=np.int64)
    assert deadlines.size < 2 or bool(np.all(np.diff(deadlines) > 0)), \
        "reverse_profile: deadlines must be strictly ascending (the row cummax depends on it)"
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


def reverse_raptor_traced(data, egress_g, egress_t, egress_w, board_slack, max_rounds):
    """Expected traced-parent dict via the compiled compact single-deadline kernel."""
    values = _traced_compact(
        np.int64(data["n_stops"]),
        np.asarray(data["pat_nstops"]), np.asarray(data["pat_ntrips"]),
        np.asarray(data["pat_stop_off"]), np.asarray(data["pat_mat_off"]),
        np.asarray(data["pat_stops"]), np.asarray(data["pat_dep"]),
        np.asarray(data["pat_arr"]), np.asarray(data["ras_off"]),
        np.asarray(data["ras_pat"]), np.asarray(data["ras_pos"]),
        np.asarray(data["tr_off"]), np.asarray(data["tr_to"]),
        np.asarray(data["tr_time"]), np.asarray(egress_g, dtype=np.int64),
        np.asarray(egress_t, dtype=np.int64), np.asarray(egress_w, dtype=np.int64),
        np.int64(board_slack), np.int64(max_rounds))
    keys = ("best", "par_kind", "par_pat", "par_trip", "par_board", "par_alight",
            "par_from", "par_nxfer", "egress_sec", "best_node", "nd_kind", "nd_stop",
            "nd_pat", "nd_trip", "nd_board", "nd_alight", "nd_to", "nd_egress",
            "nd_depth", "nd_next")
    return dict(zip(keys, values))


# ============================================================== service-noise Monte-Carlo
# Perturb the schedule (per-trip cumulative delay) and re-run the SAME validated reverse sweep
# per draw. Each draw fills its column of commute_all[n_cells, R] under COMMITTED-PLAN semantics:
# the first leg is fixed to the published plan (board the next trip on the committed pattern, no
# optimal re-route), and only the TAIL re-optimizes from the actual late arrival — see the
# committed banner below. Draws run in `prange` across cores (nogil), each thread holding only
# ONE perturbed schedule + ONE working latest profile at a time.  The optional pin accelerator
# additionally writes a compact uint16 R x n_stops x deadlines snapshot, bounded out-of-band by
# the server; the normal path allocates its capture array with zero dimensions.

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


@njit(nogil=True, cache=True)
def _draw_profile(n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_trip_off,
                  pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time,
                  egress_g, egress_w, deadlines, board_slack, max_rounds, delta0, slope):
    """One Monte-Carlo draw: perturb the schedule then run the reverse profile. Returns
    (latest[n_stops, nd], dep_r, arr_r) — the committed kernel reads the latest profile for the
    re-optimized tail and the perturbed dep/arr to ride the fixed first leg."""
    np_len = pat_dep.shape[0]
    dep_r = np.empty(np_len, dtype=np.int64)
    arr_r = np.empty(np_len, dtype=np.int64)
    _perturb(pat_nstops, pat_ntrips, pat_mat_off, pat_trip_off, pat_dep, pat_arr,
             delta0, slope, dep_r, arr_r)
    latest = _profile(n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off,
                      pat_stops, dep_r, arr_r, ras_off, ras_pat, ras_pos,
                      tr_off, tr_to, tr_time, egress_g, egress_w, deadlines,
                      board_slack, max_rounds)
    return latest, dep_r, arr_r


@njit(nogil=True, cache=True)
def _first_ge(row, n, key):
    """Index of the first entry in row[0:n] that is >= key (n if none). Binary search on a
    non-decreasing row — the committed kernel's tail readout (earliest reachable deadline
    from the committed alight)."""
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1
        if row[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


# The two helpers below deliberately mirror pieces of ``montecarlo_committed`` rather than
# being called by it.  They exist solely for the offline stage profiler
# (scripts/mc_kernel_stage_benchmark.py): splitting the production parallel loop to collect
# timings would itself change the thing being measured.  Keeping the production kernel's hot
# path untouched is also useful as a direct array-equality oracle for that profiler.
@njit(nogil=True, cache=True)
def _mc_stage_score_committed_draw(pat_nstops, pat_ntrips, pat_mat_off, deadlines,
                                   board_slack, commit_home, commit_kind, commit_walk0,
                                   commit_pi, commit_bpos, commit_apos, commit_as, perfect,
                                   max_min, latest, dep_r, arr_r, commute_out):
    """Score one already-profiled MC draw into ``commute_out``.

    Offline stage-profiling seam only.  It must remain lockstep with the committed scoring
    block in :func:`montecarlo_committed`; production intentionally keeps that block inlined in
    its parallel draw loop.
    """
    n_cells = commit_home.shape[0]
    nd = deadlines.shape[0]
    capf = np.float64(max_min)
    for ci in range(n_cells):
        k = commit_kind[ci]
        if k != 2:
            p = perfect[ci]
            if k == 0 or p < 0:
                commute_out[ci] = capf
            else:
                commute_out[ci] = capf if p > capf else np.float64(p)
            continue
        pi = commit_pi[ci]
        ns = pat_nstops[pi]
        nt = pat_ntrips[pi]
        mbase = pat_mat_off[pi]
        bpos = commit_bpos[ci]
        apos = commit_apos[ci]
        key = commit_home[ci] + commit_walk0[ci]
        lo = 0
        hi = nt
        while lo < hi:
            mid = (lo + hi) >> 1
            off = mbase + mid * ns + bpos
            if dep_r[off] < key:
                lo = mid + 1
            else:
                hi = mid
        if lo >= nt:
            commute_out[ci] = capf
            continue
        off = mbase + lo * ns + apos
        arr_as = arr_r[off]
        d = _first_ge(latest[commit_as[ci]], nd, arr_as + board_slack)
        if d >= nd:
            commute_out[ci] = capf
            continue
        tt = (deadlines[d] - commit_home[ci]) / 60.0
        if tt < 0.0:
            tt = 0.0
        if tt > capf:
            tt = capf
        commute_out[ci] = tt


@njit(nogil=True, cache=True)
def _mc_stage_encode_tail(latest, deadlines, tail_out):
    """Encode one profiled draw exactly as ``capture_tail=True`` does.

    Offline stage-profiling seam only; returns the same per-draw validity byte the production
    capture loop writes.  ``tail_out`` is caller-owned to keep allocation out of the stage.
    """
    valid = np.uint8(1)
    n_stops = latest.shape[0]
    nd = deadlines.shape[0]
    for s in range(n_stops):
        for d in range(nd):
            value = latest[s, d]
            if value == NEG:
                tail_out[s, d] = np.uint16(65535)
            else:
                lag = deadlines[d] - value
                if lag < 0 or lag >= 65535:
                    valid = np.uint8(0)
                    tail_out[s, d] = np.uint16(65535)
                else:
                    tail_out[s, d] = np.uint16(lag)
    return valid


# ----------------------------------------------- committed-plan Monte-Carlo (forward sim)
# You commit the FIRST leg from the unperturbed plan (departure + line + board stop, no
# foreknowledge of delays), then per draw: board the next available trip on the committed line,
# ride to the committed alight, and re-optimize the TAIL from the ACTUAL (late) arrival via the
# perturbed reverse profile. So a late first leg that blows the transfer eats a real headway ->
# the honest realistic + fragility. The tail still re-optimizes with the perturbed schedule (you
# have real-time info en route), so it's a tight lower bound on the fully-committed truth.

@njit(parallel=True, nogil=True, cache=True)
def montecarlo_committed(n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_trip_off,
                         pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos,
                         tr_off, tr_to, tr_time, egress_g, egress_w, deadlines, board_slack,
                         max_rounds, commit_home, commit_kind, commit_walk0, commit_pi,
                         commit_bpos, commit_apos, commit_as, perfect, max_min,
                         delta0_all, slope_all, capture_tail=False):
    """commute_all[n_cells, R] float minutes for R draws, COMMITTED-PLAN semantics. Per draw:
    perturb -> reverse profile; per transit cell: catch the next trip on the committed pattern at
    the committed board position, ride to the committed alight, then read the earliest workplace
    arrival from that alight at the ACTUAL arrival time + board_slack (the sweep's alight->onward
    convention) off the perturbed profile. Deterministic (walk-only) cells take ``perfect`` every
    draw; unreachable -> cap."""
    R = delta0_all.shape[0]
    n_cells = commit_home.shape[0]
    nd = deadlines.shape[0]
    commute_all = np.empty((n_cells, R), dtype=np.float64)
    # Optional accelerator capture.  uint16 stores an exact deadline-relative lag; 65535 is the
    # unreachable sentinel.  A per-draw validity byte prevents any overflow from being retained —
    # commute_all remains authoritative and byte-unchanged even when capture is rejected.
    tail_lag = (np.empty((R, n_stops, nd), dtype=np.uint16) if capture_tail
                else np.empty((0, 0, 0), dtype=np.uint16))
    tail_valid = np.ones(R, dtype=np.uint8)
    capf = np.float64(max_min)
    for r in prange(R):
        latest, dep_r, arr_r = _draw_profile(
            n_stops, pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_trip_off,
            pat_stops, pat_dep, pat_arr, ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time,
            egress_g, egress_w, deadlines, board_slack, max_rounds,
            delta0_all[r], slope_all[r])
        for ci in range(n_cells):
            k = commit_kind[ci]
            if k != 2:                                   # unreachable (0) or deterministic walk (1)
                p = perfect[ci]
                if k == 0 or p < 0:
                    commute_all[ci, r] = capf
                else:
                    commute_all[ci, r] = capf if p > capf else np.float64(p)
                continue
            pi = commit_pi[ci]
            ns = pat_nstops[pi]
            nt = pat_ntrips[pi]
            mbase = pat_mat_off[pi]
            bpos = commit_bpos[ci]
            apos = commit_apos[ci]
            # You arrive at the committed board stop at commit_home + walk0, which is EXACTLY your
            # committed trip's departure (the perfect-timing plan), so board the first trip departing
            # at/after that (no extra slack — adding board_slack would skip your own planned trip).
            # The dep column at bpos is FIFO-sorted but STRIDED by ns across trips -> search in place.
            key = commit_home[ci] + commit_walk0[ci]
            lo = 0
            hi = nt
            while lo < hi:                               # first trip with perturbed dep >= key
                mid = (lo + hi) >> 1
                off = mbase + mid * ns + bpos
                if dep_r[off] < key:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= nt:                                 # missed the last trip on the committed line
                commute_all[ci, r] = capf
                continue
            off = mbase + lo * ns + apos
            arr_as = arr_r[off]                          # ACTUAL (late) arrival at the transfer stop
            # Consume the tail at arr_as + board_slack: the sweep charges board_slack at every
            # transit-alight -> onward seam (`pa - board_slack` in _profile), so the stressed
            # transfer must pay the same 60s the perfect-timing sweep paid for it.
            d = _first_ge(latest[commit_as[ci]], nd, arr_as + board_slack)
            if d >= nd:                                  # can't reach work within the horizon
                commute_all[ci, r] = capf
                continue
            tt = (deadlines[d] - commit_home[ci]) / 60.0
            if tt < 0.0:
                tt = 0.0
            if tt > capf:
                tt = capf
            commute_all[ci, r] = tt
        if capture_tail:
            valid = np.uint8(1)
            for s in range(n_stops):
                for d in range(nd):
                    value = latest[s, d]
                    if value == NEG:
                        tail_lag[r, s, d] = np.uint16(65535)
                    else:
                        lag = deadlines[d] - value
                        if lag < 0 or lag >= 65535:
                            valid = np.uint8(0)
                            tail_lag[r, s, d] = np.uint16(65535)
                        else:
                            tail_lag[r, s, d] = np.uint16(lag)
            tail_valid[r] = valid
    return commute_all, tail_lag, tail_valid


@njit(nogil=True, cache=True)
def montecarlo_committed_from_tail(pat_nstops, pat_ntrips, pat_mat_off, pat_trip_off,
                                   pat_dep, pat_arr, deadlines, tail_lag, board_slack,
                                   commit_home, commit_kind, commit_walk0, commit_pi,
                                   commit_bpos, commit_apos, commit_as, perfect, max_min,
                                   delta0_all, slope_all):
    """Score a small committed-route batch from retained per-draw tail profiles.

    Only the requested pattern's board and alight columns are reconstructed.  `_perturb`'s FIFO
    clamp is independent per pattern position, so the two running maxima below are exactly the
    corresponding columns of the full perturbed schedule used by ``montecarlo_committed``.
    """
    R = delta0_all.shape[0]
    n_rows = commit_home.shape[0]
    nd = deadlines.shape[0]
    out = np.empty((n_rows, R), dtype=np.float64)
    capf = np.float64(max_min)
    for row in range(n_rows):
        kind = commit_kind[row]
        for r in range(R):
            if kind != 2:
                p = perfect[row]
                if kind == 0 or p < 0:
                    out[row, r] = capf
                else:
                    out[row, r] = capf if p > capf else np.float64(p)
                continue
            pi = commit_pi[row]
            ns = pat_nstops[pi]
            nt = pat_ntrips[pi]
            mbase = pat_mat_off[pi]
            tbase = pat_trip_off[pi]
            bpos = commit_bpos[row]
            apos = commit_apos[row]
            key = commit_home[row] + commit_walk0[row]
            prev_dep = np.int64(NEG)
            prev_arr = np.int64(NEG)
            arr_as = np.int64(NEG)
            found = False
            for trip in range(nt):
                g = tbase + trip
                base = mbase + trip * ns
                dep0 = pat_dep[base]

                boff = base + bpos
                belapsed = pat_dep[boff] - dep0
                if belapsed < 0:
                    belapsed = 0
                binc = np.int64(delta0_all[r, g] + slope_all[r, g] * belapsed)
                dep_value = pat_dep[boff] + binc
                if dep_value < prev_dep:
                    dep_value = prev_dep
                prev_dep = dep_value

                aoff = base + apos
                aelapsed = pat_dep[aoff] - dep0
                if aelapsed < 0:
                    aelapsed = 0
                ainc = np.int64(delta0_all[r, g] + slope_all[r, g] * aelapsed)
                arr_value = pat_arr[aoff] + ainc
                if arr_value < prev_arr:
                    arr_value = prev_arr
                prev_arr = arr_value

                if dep_value >= key:
                    arr_as = arr_value
                    found = True
                    break
            if not found:
                out[row, r] = capf
                continue

            tail_key = arr_as + board_slack
            lo = 0
            hi = nd
            s = commit_as[row]
            while lo < hi:
                mid = (lo + hi) >> 1
                code = tail_lag[r, s, mid]
                if code == np.uint16(65535):
                    before = True
                else:
                    before = deadlines[mid] - np.int64(code) < tail_key
                if before:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= nd:
                out[row, r] = capf
                continue
            tt = (deadlines[lo] - commit_home[row]) / 60.0
            if tt < 0.0:
                tt = 0.0
            if tt > capf:
                tt = capf
            out[row, r] = tt
    return out
