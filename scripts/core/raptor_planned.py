"""Exact planned-branch schedule discovery with a Python oracle and optional compiled fast path."""
import numpy as np


# Compact planned-template trace outcomes.  Values are part of the Python/Numba differential
# contract; callers must treat MALFORMED as "use the established scalar tracer", not as a route
# rejection.  UNREACHABLE and INVALID_BOARD are exact negative answers for the requested tree.
PLANNED_TRACE_OK = np.uint8(1)
PLANNED_TRACE_REANCHORED = np.uint8(2)
PLANNED_TRACE_UNREACHABLE = np.uint8(3)
PLANNED_TRACE_INVALID_BOARD = np.uint8(4)
PLANNED_TRACE_MALFORMED = np.uint8(5)


def _compact_node_shape_valid(nid, nd_kind, nd_stop, nd_pat, nd_trip, nd_board,
                              nd_alight, nd_to, nd_egress, nd_next):
    return (0 <= int(nid) < min(len(nd_kind), len(nd_stop), len(nd_pat), len(nd_trip),
                                len(nd_board), len(nd_alight), len(nd_to), len(nd_egress),
                                len(nd_next)))


def _compact_footpath_sec_python(tr_off, tr_to, tr_time, stop, target):
    """Return the tracer's exact CSR footpath duration, including its missing-edge zero rule."""
    stop = int(stop); target = int(target)
    if stop < 0 or stop + 1 >= len(tr_off):
        return None
    start = int(tr_off[stop]); end = int(tr_off[stop + 1])
    if start < 0 or end < start or end > min(len(tr_to), len(tr_time)):
        return None
    for q in range(start, end):
        if int(tr_to[q]) == target:
            return int(tr_time[q])
    return 0


def _compact_ride_python(pi, trip, bpos, apos, pat_nstops, pat_ntrips, pat_mat_off,
                         pat_stop_off, pat_stops, pat_dep, pat_arr):
    """Validate one compact board node and return its schedule/stop-table offsets."""
    pi = int(pi); trip = int(trip); bpos = int(bpos); apos = int(apos)
    if (pi < 0 or pi >= len(pat_nstops) or pi >= len(pat_ntrips)
            or pi >= len(pat_mat_off)
            or pi >= len(pat_stop_off)):
        return None
    ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi])
    mat = int(pat_mat_off[pi]); stop_base = int(pat_stop_off[pi])
    if (ns <= 0 or nt < 0 or trip < 0 or trip >= nt or bpos < 0 or apos < 0
            or bpos >= ns or apos >= ns
            or mat < 0 or stop_base < 0):
        return None
    row = mat + trip * ns
    if (row < 0 or row + ns > min(len(pat_dep), len(pat_arr))
            or stop_base + ns > len(pat_stops)):
        return None
    return ns, row, stop_base


def _compact_boardability_python(root, start_at_stop,
                                 nd_kind, nd_stop, nd_pat, nd_trip, nd_board,
                                 nd_alight, nd_to, nd_egress, nd_next,
                                 pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off, pat_stops,
                                 pat_dep, pat_arr, tr_off, tr_to, tr_time):
    """Return 1 boardable, 0 unboardable, or -1 malformed for one immutable chain."""
    nid = int(root); clock = int(start_at_stop)
    for _ in range(64):
        if not _compact_node_shape_valid(
                nid, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                nd_to, nd_egress, nd_next):
            return -1
        kind = int(nd_kind[nid])
        if kind == 0:
            return 1
        if kind == 2:
            walk = _compact_footpath_sec_python(
                tr_off, tr_to, tr_time, nd_stop[nid], nd_to[nid])
            if walk is None:
                return -1
            clock += int(walk)
        elif kind == 1:
            shape = _compact_ride_python(
                nd_pat[nid], nd_trip[nid], nd_board[nid], nd_alight[nid],
                pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off, pat_stops,
                pat_dep, pat_arr)
            if shape is None:
                return -1
            _ns, row, _stop_base = shape
            bpos = int(nd_board[nid]); apos = int(nd_alight[nid])
            dep = int(pat_dep[row + bpos])
            if dep < clock:
                return 0
            clock = int(pat_arr[row + apos])
        else:
            return -1
        nid = int(nd_next[nid])
    return -1


def trace_planned_stop_templates_python(
        stops, requested_board, window_floor, tree_best,
        best_node, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
        nd_to, nd_egress, nd_next,
        pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off, pat_stops, pat_dep, pat_arr,
        tr_off, tr_to, tr_time, egress_sec, pattern_structural_rank,
        egress_inf, max_rides=8):
    """Exact Python oracle for one deadline's compact planned stop-template batch.

    The fixed-width ride tape stores ``(pattern, trip, board position, resolved alight
    position)`` per ride occurrence.  Structural equivalence is intentionally *not* decided here:
    callers must translate every tape row through the existing
    ``_planned_pattern_identity`` topology key before comparing routes.  Pattern ids, labels and
    hashes are never route-equivalence shortcuts.

    ``window_floor`` implements the overlay's B3 policy.  An unboardable requested B may re-anchor
    only to ``tree_best[stop] >= window_floor``; otherwise the row is INVALID_BOARD.  MALFORMED
    includes the 64-node guard and a ride-tape overflow and always means scalar fallback.

    Returns parallel arrays ``(status, effective_B, root_node, tail_sec, transit_sec,
    final_walk_sec, ride_count, ride_pi, ride_trip, ride_bpos, ride_apos,
    representative_pi)``.  ``tail_sec`` is the actual compact-chain clock from effective B to the
    workplace, not the planned overlay's deadline-anchor score ``T-B``; ranking must keep using
    the latter.  ``representative_pi`` uses the supplied structural rank only to resolve equal
    ride-duration ties and is not an equivalence key.
    """
    stops = np.asarray(stops, np.int64)
    requested_board = np.asarray(requested_board, np.int64)
    if len(stops) != len(requested_board):
        raise ValueError("stops and requested_board must have equal length")
    n = len(stops); width = int(max_rides)
    status = np.full(n, PLANNED_TRACE_MALFORMED, np.uint8)
    effective = np.asarray(requested_board, np.int64).copy()
    root_out = np.full(n, -1, np.int32)
    tail_sec = np.zeros(n, np.int64)
    transit_sec = np.zeros(n, np.int64)
    final_walk = np.zeros(n, np.int64)
    ride_count = np.zeros(n, np.int16)
    ride_pi = np.full((n, width), -1, np.int32)
    ride_trip = np.full((n, width), -1, np.int32)
    ride_bpos = np.full((n, width), -1, np.int32)
    ride_apos = np.full((n, width), -1, np.int32)
    representative = np.full(n, -1, np.int32)

    for row in range(n):
        stop = int(stops[row]); requested = int(requested_board[row])
        if stop < 0 or stop >= len(best_node) or stop >= len(tree_best):
            continue
        root = int(best_node[stop]); root_out[row] = root
        if root < 0:
            status[row] = PLANNED_TRACE_UNREACHABLE
            continue
        valid = _compact_boardability_python(
            root, requested, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
            nd_to, nd_egress, nd_next, pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off,
            pat_stops, pat_dep, pat_arr, tr_off, tr_to, tr_time)
        if valid < 0:
            continue
        outcome = PLANNED_TRACE_OK
        board = requested
        if valid == 0:
            board = int(tree_best[stop])
            if board < int(window_floor):
                status[row] = PLANNED_TRACE_INVALID_BOARD
                continue
            valid = _compact_boardability_python(
                root, board, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                nd_to, nd_egress, nd_next, pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off,
                pat_stops, pat_dep, pat_arr, tr_off, tr_to, tr_time)
            if valid < 0:
                continue
            if valid == 0:
                status[row] = PLANNED_TRACE_INVALID_BOARD
                continue
            outcome = PLANNED_TRACE_REANCHORED
        effective[row] = board

        nid = root; clock = board; rides = 0; transit = 0
        best_ride = -1; best_rank = None; best_pi = -1
        completed = False; malformed = False
        for _ in range(64):
            if not _compact_node_shape_valid(
                    nid, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                    nd_to, nd_egress, nd_next):
                malformed = True; break
            kind = int(nd_kind[nid])
            if kind == 0:
                walk = int(nd_egress[nid])
                if walk < 0:
                    walk = 0
                final_walk[row] = walk
                clock += walk
                completed = True
                break
            if kind == 2:
                walk = _compact_footpath_sec_python(
                    tr_off, tr_to, tr_time, nd_stop[nid], nd_to[nid])
                if walk is None:
                    malformed = True; break
                clock += int(walk)
                nid = int(nd_next[nid])
                continue
            if kind != 1 or rides >= width:
                malformed = True; break
            pi = int(nd_pat[nid]); trip = int(nd_trip[nid])
            bpos = int(nd_board[nid]); apos = int(nd_alight[nid])
            shape = _compact_ride_python(
                pi, trip, bpos, apos, pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off,
                pat_stops, pat_dep, pat_arr)
            if shape is None or pi >= len(pattern_structural_rank):
                malformed = True; break
            ns, schedule_row, stop_base = shape
            nxt = int(nd_next[nid])
            if (_compact_node_shape_valid(
                    nxt, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                    nd_to, nd_egress, nd_next) and int(nd_kind[nxt]) == 0):
                best_ap = apos
                best_stop = int(pat_stops[stop_base + best_ap])
                if best_stop < 0 or best_stop >= len(egress_sec):
                    malformed = True; break
                best_walk = int(egress_sec[best_stop])
                if best_walk >= int(egress_inf):
                    best_walk = int(nd_egress[nxt])
                    if best_walk < 0:
                        best_walk = 0
                best_finish = int(pat_arr[schedule_row + best_ap]) + best_walk
                for p in range(bpos + 1, ns):
                    candidate_stop = int(pat_stops[stop_base + p])
                    if candidate_stop < 0 or candidate_stop >= len(egress_sec):
                        malformed = True; break
                    candidate_walk = int(egress_sec[candidate_stop])
                    if candidate_walk >= int(egress_inf):
                        continue
                    candidate_finish = int(pat_arr[schedule_row + p]) + candidate_walk
                    if (candidate_finish < best_finish
                            or (candidate_finish == best_finish
                                and (candidate_walk < best_walk
                                     or (candidate_walk == best_walk and p > best_ap)))):
                        best_finish = candidate_finish
                        best_ap = p
                        best_walk = candidate_walk
                if malformed:
                    break
                apos = best_ap
                final_walk[row] = best_walk
            dep = int(pat_dep[schedule_row + bpos])
            arr = int(pat_arr[schedule_row + apos])
            if dep < clock:
                malformed = True; break  # validated above: compact arrays changed or disagree
            clock = arr
            duration = arr - dep
            transit += duration
            rank = int(pattern_structural_rank[pi])
            if duration > best_ride or (duration == best_ride
                                        and (best_rank is None or rank < best_rank)):
                best_ride = duration; best_rank = rank; best_pi = pi
            ride_pi[row, rides] = pi
            ride_trip[row, rides] = trip
            ride_bpos[row, rides] = bpos
            ride_apos[row, rides] = apos
            rides += 1
            if (_compact_node_shape_valid(
                    nxt, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                    nd_to, nd_egress, nd_next) and int(nd_kind[nxt]) == 0):
                clock += int(final_walk[row])
                completed = True
                break
            nid = nxt
        if malformed or not completed:
            status[row] = PLANNED_TRACE_MALFORMED
            continue
        ride_count[row] = rides
        transit_sec[row] = transit
        tail_sec[row] = clock - board
        representative[row] = best_pi
        status[row] = outcome

    return (status, effective, root_out, tail_sec, transit_sec, final_walk,
            ride_count, ride_pi, ride_trip, ride_bpos, ride_apos, representative)


def _raw_chain_valid_after_start(legs_raw, start_at_stop):
    """Standalone B3 validity oracle used by the committed batch differential test.

    Keep this here rather than importing ``DepartAfterJourneyTree``: the latter imports this
    module for the production fast path, while this function must remain a simple, cycle-free
    Python oracle.
    """
    t = int(start_at_stop)
    for leg in legs_raw:
        kind = leg[0]
        if kind == "access":
            continue
        if kind in ("walk", "walk_t", "egress"):
            t += int(leg[1])
            continue
        if kind == "ride":
            if int(leg[2]) < t:
                return False
            t = int(leg[3])
    return True


def extract_planned_committed_group_python(cells, tree, s_star, access_w, dstar, cell_deps,
                                           pat_dominant_rank):
    """Pure-Python oracle for one planned committed-extraction deadline group.

    Production uses ``raptor_planned_numba.extract_planned_committed_group`` to walk compact
    node tables without allocating thousands of raw leg lists.  This oracle intentionally uses
    the public, battle-tested ``JourneyTree._trace_from`` and the same B3 procedure so tests can
    differential-check every field before a compact row is trusted.  It returns the identical
    local-row tuple as the compiled kernel.
    """
    cells = np.asarray(cells, np.int64)
    n = len(cells)
    home = np.zeros(n, np.int64)
    kind = np.zeros(n, np.int8)
    walk0 = np.zeros(n, np.int64)
    pi_out = np.full(n, -1, np.int32)
    bpos_out = np.full(n, -1, np.int32)
    apos_out = np.full(n, -1, np.int32)
    as_out = np.full(n, -1, np.int32)
    dom_pi = np.full(n, -1, np.int32)
    wait_clamped = np.zeros(n, np.uint8)
    status = np.zeros(n, np.uint8)
    for row_i, ci_raw in enumerate(cells):
        ci = int(ci_raw)
        stop = int(s_star[ci])
        aw = int(access_w[ci])
        original_home = int(dstar[ci])
        if stop < 0:
            continue
        traced = tree._trace_from(stop, aw, original_home)
        if traced is None:
            continue
        legs_raw, _ignored = traced
        board_at_stop = original_home + aw
        latest_home = original_home
        if not _raw_chain_valid_after_start(legs_raw, board_at_stop):
            found = False
            for board in reversed(np.asarray(cell_deps, np.int64)):
                board = int(board)
                if board >= board_at_stop:
                    continue
                if _raw_chain_valid_after_start(legs_raw, board):
                    latest_home = board - aw
                    found = True
                    break
            if not found:
                wait_clamped[row_i] = 1

        first = None
        best = None
        for leg in legs_raw:
            if leg[0] != "ride":
                continue
            # Planned display does not fold tiny rides; this is deliberately unlike the legacy
            # arrive-by committed extractor.
            pi = int(leg[1])
            dep = int(leg[2])
            arr = int(leg[3])
            if first is None:
                first = leg
            rank = (-(arr - dep), int(pat_dominant_rank[pi]))
            if best is None or rank < best[0]:
                best = (rank, pi)

        home[row_i] = latest_home
        if first is None:
            kind[row_i] = 1
        else:
            _, pi, dep, _arr, bp, ap, alight_stop = first
            kind[row_i] = 2
            walk0[row_i] = int(dep) - latest_home
            pi_out[row_i] = int(pi)
            bpos_out[row_i] = int(bp)
            apos_out[row_i] = int(ap)
            as_out[row_i] = int(alight_stop)
            dom_pi[row_i] = int(best[1])
        status[row_i] = 1
    return (home, kind, walk0, pi_out, bpos_out, apos_out, as_out, dom_pi,
            wait_clamped, status)


def evaluate_planned_branch_probes(probes, trace_at_stop, valid_after_start):
    """Yield valid planned branch probes while tracing each ``(stop, deadline)`` once.

    A planned branch probe is a four-tuple ``(stop, board, deadline, access_walk)``.
    ``JourneyTree._trace_from`` selects its immutable node chain solely from the stop and the
    deadline tree: its access-walk and home-departure arguments only supply the leading access
    tuple and the returned clock anchor.  Therefore all probes sharing ``(stop, deadline)`` can
    share one stop-to-work raw tail.  Their board times *cannot* be merged: B3 validation is
    deliberately applied to every probe, because a later requested board can be unboardable while
    an earlier one on the same tail remains valid.

    ``trace_at_stop(stop, deadline)`` must return an immutable trace payload (normally a raw
    stop-to-work tail without an ``access`` tuple), or ``None``.  Callers may carry the source
    ``JourneyTree`` alongside that tail so a subsequently evicted bounded tree cache cannot
    change which handle formats the candidate. ``valid_after_start(trace, board)`` is called in
    the input probe order.  The yielded values are ``(stop, board, deadline, access_walk, trace)``
    in exactly that order.

    This helper intentionally does *not* deduplicate equal probes or choose a winner.  Access
    walk, home anchor, route identity and final display/tie handling remain the caller's existing
    responsibility, so it is a lossless extraction optimization rather than a routing policy.
    """
    tails = {}
    for probe in probes:
        if len(probe) != 4:
            raise ValueError("planned branch probe must be (stop, board, deadline, access_walk)")
        stop, board, deadline, access_walk = (int(value) for value in probe)
        key = (stop, deadline)
        if key not in tails:
            tails[key] = trace_at_stop(stop, deadline)
        trace = tails[key]
        if trace is None:
            continue
        # Keep every board probe.  In particular, do not short-circuit after a latest-board
        # failure: validity is monotone in the other direction, so an earlier board may be the
        # only legitimate candidate for this same stop/deadline tree.
        if valid_after_start(trace, board):
            yield stop, board, deadline, access_walk, trace


def _lower_bound_column(values, base, stride, count, key):
    lo, hi = 0, int(count)
    while lo < hi:
        mid = (lo + hi) >> 1
        if int(values[int(base) + mid * int(stride)]) < int(key):
            lo = mid + 1
        else:
            hi = mid
    return lo


def discover_one_tail_variants_python(data, egress_sec, first_pi, first_dep, first_bpos,
                                      board_slack, egress_inf):
    """Reference sibling of the compiled kernel; row order and tie rules are load-bearing."""
    pat_nstops = data["pat_nstops"]; pat_ntrips = data["pat_ntrips"]
    pat_stop_off = data["pat_stop_off"]; pat_mat_off = data["pat_mat_off"]
    pat_stops = data["pat_stops"]; pat_dep = data["pat_dep"]; pat_arr = data["pat_arr"]
    ras_off = data["ras_off"]; ras_pat = data["ras_pat"]; ras_pos = data["ras_pos"]
    tr_off = data["tr_off"]; tr_to = data["tr_to"]; tr_time = data["tr_time"]
    first_pi = int(first_pi); first_dep = int(first_dep); first_bpos = int(first_bpos)
    ns0 = int(pat_nstops[first_pi]); nt0 = int(pat_ntrips[first_pi])
    mb0 = int(pat_mat_off[first_pi]); sb0 = int(pat_stop_off[first_pi])
    trip0 = _lower_bound_column(pat_dep, mb0 + first_bpos, ns0, nt0, first_dep)
    if trip0 >= nt0 or int(pat_dep[mb0 + trip0 * ns0 + first_bpos]) != first_dep:
        return np.empty((0, 12), dtype=np.int64)

    out = []
    for xapos in range(first_bpos + 1, ns0):
        alight_stop = int(pat_stops[sb0 + xapos])
        arr0 = int(pat_arr[mb0 + trip0 * ns0 + xapos])
        transfers = [(alight_stop, 0)]
        for p in range(int(tr_off[alight_stop]), int(tr_off[alight_stop + 1])):
            walk_sec = int(tr_time[p])
            if walk_sec <= 12 * 60:
                transfers.append((int(tr_to[p]), walk_sec))
        for stop, walk_sec in transfers:
            ready = arr0 + walk_sec + int(board_slack)
            for rp in range(int(ras_off[stop]), int(ras_off[stop + 1])):
                pi = int(ras_pat[rp]); bpos = int(ras_pos[rp])
                ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi])
                mb = int(pat_mat_off[pi]); sb = int(pat_stop_off[pi])
                trip = _lower_bound_column(pat_dep, mb + bpos, ns, nt, ready)
                if trip >= nt:
                    continue
                trow = mb + trip * ns
                best = None
                for apos in range(bpos + 1, ns):
                    st2 = int(pat_stops[sb + apos]); eg = int(egress_sec[st2])
                    if eg >= int(egress_inf):
                        continue
                    finish = int(pat_arr[trow + apos]) + eg
                    rank = (finish, eg, -apos)
                    if best is None or rank < best[0]:
                        best = (rank, apos, st2, eg)
                if best is None:
                    continue
                _rank, apos, st2, eg = best
                out.append((xapos, alight_stop, arr0, stop, walk_sec, pi, bpos,
                            int(pat_dep[trow + bpos]), int(pat_arr[trow + apos]),
                            apos, st2, eg))
    return np.asarray(out, dtype=np.int64).reshape((-1, 12))


def discover_one_tail_variants(data, egress_sec, first_pi, first_dep, first_bpos,
                               board_slack, egress_inf):
    """Use the isolated cached Numba kernel when available, otherwise the exact Python oracle."""
    try:
        from .raptor import _select_kernel
        if _select_kernel() == "numba":
            from .raptor_planned_numba import discover_one_tail_variants as compiled
            return compiled(
                data["pat_nstops"], data["pat_ntrips"], data["pat_stop_off"],
                data["pat_mat_off"], data["pat_stops"], data["pat_dep"], data["pat_arr"],
                data["ras_off"], data["ras_pat"], data["ras_pos"], data["tr_off"],
                data["tr_to"], data["tr_time"], np.asarray(egress_sec), int(first_pi),
                int(first_dep), int(first_bpos), int(board_slack), int(egress_inf))
    except (ImportError, OSError):
        pass
    return discover_one_tail_variants_python(
        data, egress_sec, first_pi, first_dep, first_bpos, board_slack, egress_inf)
