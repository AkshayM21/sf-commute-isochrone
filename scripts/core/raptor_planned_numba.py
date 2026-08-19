"""Small compiled kernels used only by planned-route branch discovery.

This lives outside :mod:`raptor_numba` deliberately: changing a route-card optimization must not
invalidate the cached production profile/Monte-Carlo kernels and make the next map request compile
them all again.  The pure-Python byte-for-byte oracle is in :mod:`raptor_planned`.
"""
import numpy as np
from numba import njit


# Keep these numeric values lockstep with raptor_planned.PLANNED_TRACE_* without importing the
# oracle module into this isolated compilation unit.
_TRACE_OK = np.uint8(1)
_TRACE_REANCHORED = np.uint8(2)
_TRACE_UNREACHABLE = np.uint8(3)
_TRACE_INVALID_BOARD = np.uint8(4)
_TRACE_MALFORMED = np.uint8(5)


@njit(cache=True, nogil=True)
def _compact_node_valid(nid, nd_kind, nd_stop, nd_pat, nd_trip, nd_board,
                        nd_alight, nd_to, nd_egress, nd_next):
    nid = int(nid)
    return (nid >= 0 and nid < len(nd_kind) and nid < len(nd_stop)
            and nid < len(nd_pat) and nid < len(nd_trip) and nid < len(nd_board)
            and nid < len(nd_alight) and nid < len(nd_to) and nid < len(nd_egress)
            and nid < len(nd_next))


@njit(cache=True, nogil=True)
def _compact_footpath_sec(tr_off, tr_to, tr_time, stop, target):
    stop = int(stop); target = int(target)
    if stop < 0 or stop + 1 >= len(tr_off):
        return 0, 0
    start = int(tr_off[stop]); end = int(tr_off[stop + 1])
    if start < 0 or end < start or end > len(tr_to) or end > len(tr_time):
        return 0, 0
    for q in range(start, end):
        if int(tr_to[q]) == target:
            return 1, int(tr_time[q])
    # This is JourneyTree._footpath_sec's deliberate same-stop/missing-edge behavior.
    return 1, 0


@njit(cache=True, nogil=True)
def _compact_ride_shape(pi, trip, bpos, apos, pat_nstops, pat_ntrips, pat_mat_off,
                        pat_stop_off, pat_stops, pat_dep, pat_arr):
    pi = int(pi); trip = int(trip); bpos = int(bpos); apos = int(apos)
    if (pi < 0 or pi >= len(pat_nstops) or pi >= len(pat_ntrips)
            or pi >= len(pat_mat_off)
            or pi >= len(pat_stop_off)):
        return 0, 0, 0, 0
    ns = int(pat_nstops[pi]); nt = int(pat_ntrips[pi])
    mat = int(pat_mat_off[pi]); stop_base = int(pat_stop_off[pi])
    if (ns <= 0 or nt < 0 or trip < 0 or trip >= nt or bpos < 0 or apos < 0
            or bpos >= ns or apos >= ns
            or mat < 0 or stop_base < 0):
        return 0, 0, 0, 0
    row = mat + trip * ns
    if (row < 0 or row + ns > len(pat_dep) or row + ns > len(pat_arr)
            or stop_base + ns > len(pat_stops)):
        return 0, 0, 0, 0
    return 1, ns, row, stop_base


@njit(cache=True, nogil=True)
def _compact_boardability(root, start_at_stop,
                          nd_kind, nd_stop, nd_pat, nd_trip, nd_board,
                          nd_alight, nd_to, nd_egress, nd_next,
                          pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off, pat_stops,
                          pat_dep, pat_arr, tr_off, tr_to, tr_time):
    """Return 1 boardable, 0 unboardable, or -1 malformed for one compact chain."""
    nid = int(root); clock = int(start_at_stop)
    for _ in range(64):
        if not _compact_node_valid(
                nid, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                nd_to, nd_egress, nd_next):
            return -1
        kind = int(nd_kind[nid])
        if kind == 0:
            return 1
        if kind == 2:
            valid, walk = _compact_footpath_sec(
                tr_off, tr_to, tr_time, nd_stop[nid], nd_to[nid])
            if valid == 0:
                return -1
            clock += int(walk)
        elif kind == 1:
            valid, _ns, row, _stop_base = _compact_ride_shape(
                nd_pat[nid], nd_trip[nid], nd_board[nid], nd_alight[nid],
                pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off, pat_stops,
                pat_dep, pat_arr)
            if valid == 0:
                return -1
            bpos = int(nd_board[nid]); apos = int(nd_alight[nid])
            dep = int(pat_dep[row + bpos])
            if dep < clock:
                return 0
            clock = int(pat_arr[row + apos])
        else:
            return -1
        nid = int(nd_next[nid])
    return -1


@njit(cache=True, nogil=True)
def trace_planned_stop_templates(
        stops, requested_board, window_floor, tree_best,
        best_node, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
        nd_to, nd_egress, nd_next,
        pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off, pat_stops, pat_dep, pat_arr,
        tr_off, tr_to, tr_time, egress_sec, pattern_structural_rank,
        egress_inf, max_rides=8):
    """Compiled sibling of ``trace_planned_stop_templates_python``.

    It validates and summarizes one deadline tree without allocating Python leg objects.  The
    output ride tape is structural input only: callers still map each pattern through the existing
    topology identity before route comparison.  MALFORMED—including a 64-node guard or tape
    overflow—requests exact scalar fallback.
    """
    if len(stops) != len(requested_board):
        raise ValueError("stops and requested_board must have equal length")
    n = len(stops); width = int(max_rides)
    status = np.full(n, _TRACE_MALFORMED, np.uint8)
    effective = np.empty(n, np.int64)
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
        effective[row] = int(requested_board[row])
        stop = int(stops[row]); requested = int(requested_board[row])
        if stop < 0 or stop >= len(best_node) or stop >= len(tree_best):
            continue
        root = int(best_node[stop]); root_out[row] = root
        if root < 0:
            status[row] = _TRACE_UNREACHABLE
            continue
        valid = _compact_boardability(
            root, requested, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
            nd_to, nd_egress, nd_next, pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off,
            pat_stops, pat_dep, pat_arr, tr_off, tr_to, tr_time)
        if valid < 0:
            continue
        outcome = _TRACE_OK
        board = requested
        if valid == 0:
            board = int(tree_best[stop])
            if board < int(window_floor):
                status[row] = _TRACE_INVALID_BOARD
                continue
            valid = _compact_boardability(
                root, board, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                nd_to, nd_egress, nd_next, pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off,
                pat_stops, pat_dep, pat_arr, tr_off, tr_to, tr_time)
            if valid < 0:
                continue
            if valid == 0:
                status[row] = _TRACE_INVALID_BOARD
                continue
            outcome = _TRACE_REANCHORED
        effective[row] = board

        nid = root; clock = board; rides = 0; transit = 0
        best_ride = -1; best_rank = 9223372036854775807; best_pi = -1
        completed = 0; malformed = 0
        for _ in range(64):
            if not _compact_node_valid(
                    nid, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                    nd_to, nd_egress, nd_next):
                malformed = 1; break
            kind = int(nd_kind[nid])
            if kind == 0:
                walk = int(nd_egress[nid])
                if walk < 0:
                    walk = 0
                final_walk[row] = walk
                clock += walk
                completed = 1
                break
            if kind == 2:
                fp_valid, walk = _compact_footpath_sec(
                    tr_off, tr_to, tr_time, nd_stop[nid], nd_to[nid])
                if fp_valid == 0:
                    malformed = 1; break
                clock += int(walk)
                nid = int(nd_next[nid])
                continue
            if kind != 1 or rides >= width:
                malformed = 1; break
            pi = int(nd_pat[nid]); trip = int(nd_trip[nid])
            bpos = int(nd_board[nid]); apos = int(nd_alight[nid])
            shape_valid, ns, schedule_row, stop_base = _compact_ride_shape(
                pi, trip, bpos, apos, pat_nstops, pat_ntrips, pat_mat_off, pat_stop_off,
                pat_stops, pat_dep, pat_arr)
            if shape_valid == 0 or pi >= len(pattern_structural_rank):
                malformed = 1; break
            nxt = int(nd_next[nid])
            nxt_valid = _compact_node_valid(
                nxt, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                nd_to, nd_egress, nd_next)
            if nxt_valid and int(nd_kind[nxt]) == 0:
                best_ap = apos
                best_stop = int(pat_stops[stop_base + best_ap])
                if best_stop < 0 or best_stop >= len(egress_sec):
                    malformed = 1; break
                best_walk = int(egress_sec[best_stop])
                if best_walk >= int(egress_inf):
                    best_walk = int(nd_egress[nxt])
                    if best_walk < 0:
                        best_walk = 0
                best_finish = int(pat_arr[schedule_row + best_ap]) + best_walk
                for p in range(bpos + 1, ns):
                    candidate_stop = int(pat_stops[stop_base + p])
                    if candidate_stop < 0 or candidate_stop >= len(egress_sec):
                        malformed = 1; break
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
                if malformed != 0:
                    break
                apos = best_ap
                final_walk[row] = best_walk
            dep = int(pat_dep[schedule_row + bpos])
            arr = int(pat_arr[schedule_row + apos])
            if dep < clock:
                malformed = 1; break
            clock = arr
            duration = arr - dep
            transit += duration
            rank = int(pattern_structural_rank[pi])
            if duration > best_ride or (duration == best_ride and rank < best_rank):
                best_ride = duration; best_rank = rank; best_pi = pi
            ride_pi[row, rides] = pi
            ride_trip[row, rides] = trip
            ride_bpos[row, rides] = bpos
            ride_apos[row, rides] = apos
            rides += 1
            if nxt_valid and int(nd_kind[nxt]) == 0:
                clock += int(final_walk[row])
                completed = 1
                break
            nid = nxt
        if malformed != 0 or completed == 0:
            status[row] = _TRACE_MALFORMED
            continue
        ride_count[row] = rides
        transit_sec[row] = transit
        tail_sec[row] = clock - board
        representative[row] = best_pi
        status[row] = outcome
    return (status, effective, root_out, tail_sec, transit_sec, final_walk,
            ride_count, ride_pi, ride_trip, ride_bpos, ride_apos, representative)


# ---------------------------------------------------------------------------
# Depart-after committed-plan extraction
#
# This intentionally lives in the small planned-only compilation unit rather
# than raptor_numba: it is a response-shaping optimization, and must not cause
# the main reverse-profile/Monte-Carlo kernels to recompile.  The sibling
# Python oracle is ``extract_planned_committed_group_python`` in
# :mod:`raptor_planned`.


@njit(cache=True, nogil=True)
def _planned_chain_valid(best_node, nd_kind, nd_stop, nd_pat, nd_trip, nd_board,
                         nd_alight, nd_to, nd_egress, nd_next, pat_nstops,
                         pat_mat_off, pat_dep, pat_arr, tr_off, tr_to, tr_time,
                         egress_sec, s_star, start_at_stop, egress_inf):
    """Whether the compact raw chain is boardable from ``start_at_stop``.

    This is the allocation-free counterpart of
    ``DepartAfterJourneyTree._raw_chain_valid_after_start``.  It walks exactly
    the same node table as ``JourneyTree._trace_from``.  A final ride's
    no-overshoot alight does not affect this predicate—the following egress
    terminates it, with no subsequent boarding check.  ``0`` also means a
    malformed / legacy node chain, so the caller can fall back to the
    established Python tracer.
    """
    nid = int(best_node[s_star])
    t = int(start_at_stop)
    for _ in range(64):
        if nid < 0:
            return 0
        kind = int(nd_kind[nid])
        if kind == 0:  # egress
            return 1
        if kind == 2:  # footpath
            s = int(nd_stop[nid])
            j = int(nd_to[nid])
            walk = 0
            for q in range(int(tr_off[s]), int(tr_off[s + 1])):
                if int(tr_to[q]) == j:
                    walk = int(tr_time[q])
                    break
            t += walk
        elif kind == 1:  # board
            pi = int(nd_pat[nid])
            trip = int(nd_trip[nid])
            bpos = int(nd_board[nid])
            apos = int(nd_alight[nid])
            ns = int(pat_nstops[pi])
            row = int(pat_mat_off[pi]) + trip * ns
            dep = int(pat_dep[row + bpos])
            if dep < t:
                return 0
            t = int(pat_arr[row + apos])
        else:
            return 0
        nid = int(nd_next[nid])
    return 0


@njit(cache=True, nogil=True)
def extract_planned_committed_group(
        cells, s_star, access_w, dstar, cell_deps,
        best_node, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
        nd_to, nd_egress, nd_next,
        pat_nstops, pat_mat_off, pat_stop_off, pat_stops, pat_dep, pat_arr,
        tr_off, tr_to, tr_time, egress_sec, pat_dominant_rank, egress_inf):
    """Extract planned committed rows for one representative-deadline tree.

    ``cells`` contains only transit cells selected on this tree.  The result is
    local to that list (not a full grid) so the wrapper can group one compact
    parent table per ``T*``.  Each successful row is semantically identical to
    tracing the displayed journey then calling ``JourneyTree._fill_committed_leg``:
    planned B3 anchor repair, the first *displayed* board (including tiny hops),
    final no-overshoot alight, and dominant route tie-break are all preserved.

    ``status=0`` deliberately requests the Python fallback.  That protects
    fabricated/sparse legacy trees and future parent encodings instead of
    quietly guessing a committed plan.
    """
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

    for row_i in range(n):
        ci = int(cells[row_i])
        stop = int(s_star[ci])
        aw = int(access_w[ci])
        original_home = int(dstar[ci])
        if stop < 0:
            continue
        nid = int(best_node[stop])
        if nid < 0:
            continue

        # B3 is deliberately evaluated before extraction.  A raw compact tree
        # can be unboardable at the profile's selected B due to row cummax; use
        # the latest earlier window board that validates, exactly as
        # _planned_primary_home does.
        latest_home = original_home
        board_at_stop = original_home + aw
        valid = _planned_chain_valid(
            best_node, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
            nd_to, nd_egress, nd_next, pat_nstops, pat_mat_off, pat_dep, pat_arr,
            tr_off, tr_to, tr_time, egress_sec, stop, board_at_stop, egress_inf)
        if valid == 0:
            found = 0
            for dep_i in range(len(cell_deps) - 1, -1, -1):
                bv = int(cell_deps[dep_i])
                if bv >= board_at_stop:
                    continue
                if _planned_chain_valid(
                        best_node, nd_kind, nd_stop, nd_pat, nd_trip, nd_board, nd_alight,
                        nd_to, nd_egress, nd_next, pat_nstops, pat_mat_off, pat_dep, pat_arr,
                        tr_off, tr_to, tr_time, egress_sec, stop, bv, egress_inf) != 0:
                    latest_home = bv - aw
                    found = 1
                    break
            if found == 0:
                # This is B3's documented last-resort clamp, not an invalid
                # route: keep D* and ask the wrapper to mark the cell.
                wait_clamped[row_i] = 1

        # Walk the immutable chain once.  Parent node numbers are descending
        # toward W, so a 64-node guard is the same malformed-loop protection as
        # JourneyTree._trace_from.
        first_found = 0
        any_ride = 0
        best_ride = -1
        best_rank = 1 << 60
        best_pi = -1
        terminal = 0
        malformed = 0
        for _ in range(64):
            if nid < 0:
                malformed = 1
                break
            node_kind = int(nd_kind[nid])
            if node_kind == 0:
                terminal = 1
                break
            if node_kind == 2:
                nid = int(nd_next[nid])
                continue
            if node_kind != 1:
                malformed = 1
                break

            pi = int(nd_pat[nid])
            trip = int(nd_trip[nid])
            bp = int(nd_board[nid])
            ap = int(nd_alight[nid])
            ns = int(pat_nstops[pi])
            mat = int(pat_mat_off[pi]) + trip * ns
            stop_base = int(pat_stop_off[pi])
            nxt = int(nd_next[nid])

            # The exact final no-overshoot adjustment shared by
            # JourneyTree._trace_from and _build_stop_dominant.
            if nxt >= 0 and int(nd_kind[nxt]) == 0:
                best_ap = ap
                best_stop = int(pat_stops[stop_base + ap])
                best_walk = int(egress_sec[best_stop])
                if best_walk >= int(egress_inf):
                    fallback = int(nd_egress[nxt])
                    best_walk = fallback if fallback >= 0 else 0
                best_finish = int(pat_arr[mat + ap]) + best_walk
                for p in range(bp + 1, ns):
                    candidate_stop = int(pat_stops[stop_base + p])
                    candidate_walk = int(egress_sec[candidate_stop])
                    if candidate_walk >= int(egress_inf):
                        continue
                    candidate_finish = int(pat_arr[mat + p]) + candidate_walk
                    if (candidate_finish < best_finish
                            or (candidate_finish == best_finish and
                                (candidate_walk < best_walk
                                 or (candidate_walk == best_walk and p > best_ap)))):
                        best_finish = candidate_finish
                        best_ap = p
                        best_stop = candidate_stop
                        best_walk = candidate_walk
                ap = best_ap

            dep = int(pat_dep[mat + bp])
            arr = int(pat_arr[mat + ap])
            if first_found == 0:
                # Planned display keeps tiny rides, hence no two-minute filter.
                first_found = 1
                pi_out[row_i] = pi
                bpos_out[row_i] = bp
                apos_out[row_i] = ap
                as_out[row_i] = int(pat_stops[stop_base + ap])
                walk0[row_i] = dep - latest_home
            any_ride = 1
            ride = arr - dep
            rank = int(pat_dominant_rank[pi])
            if (ride > best_ride) or (ride == best_ride and rank < best_rank):
                best_ride = ride
                best_rank = rank
                best_pi = pi
            nid = nxt

        if malformed != 0 or terminal == 0:
            continue
        home[row_i] = latest_home
        if any_ride != 0:
            kind[row_i] = 2
            dom_pi[row_i] = best_pi
        else:
            kind[row_i] = 1
        status[row_i] = 1
    return (home, kind, walk0, pi_out, bpos_out, apos_out, as_out, dom_pi,
            wait_clamped, status)


@njit(cache=True, nogil=True)
def discover_one_tail_variants(
        pat_nstops, pat_ntrips, pat_stop_off, pat_mat_off, pat_stops, pat_dep, pat_arr,
        ras_off, ras_pat, ras_pos, tr_off, tr_to, tr_time, egress_sec,
        first_pi, first_dep, first_bpos, board_slack, egress_inf):
    """Return the exact downstream one-transfer variants for one boarded first trip.

    Rows preserve the reference loop's order and contain:
      xapos, first-alight stop/time, transfer stop/walk, second pattern/board position,
      second departure/arrival/alight position/stop, and final egress seconds.
    """
    ns0 = int(pat_nstops[first_pi])
    nt0 = int(pat_ntrips[first_pi])
    mb0 = int(pat_mat_off[first_pi])
    sb0 = int(pat_stop_off[first_pi])

    lo = 0
    hi = nt0
    while lo < hi:
        mid = (lo + hi) >> 1
        if pat_dep[mb0 + mid * ns0 + first_bpos] < first_dep:
            lo = mid + 1
        else:
            hi = mid
    trip0 = lo
    if trip0 >= nt0 or pat_dep[mb0 + trip0 * ns0 + first_bpos] != first_dep:
        return np.empty((0, 12), dtype=np.int64)

    # One result at most for each routes-at-stop occurrence. Count a tight allocation bound first;
    # the second pass does the schedule work. Both passes are compiled and the count pass is tiny.
    cap = 0
    for xapos in range(first_bpos + 1, ns0):
        alight_stop = int(pat_stops[sb0 + xapos])
        cap += int(ras_off[alight_stop + 1]) - int(ras_off[alight_stop])
        for p in range(int(tr_off[alight_stop]), int(tr_off[alight_stop + 1])):
            if int(tr_time[p]) <= 12 * 60:
                stop = int(tr_to[p])
                cap += int(ras_off[stop + 1]) - int(ras_off[stop])
    out = np.empty((cap, 12), dtype=np.int64)
    nout = 0

    for xapos in range(first_bpos + 1, ns0):
        alight_stop = int(pat_stops[sb0 + xapos])
        arr0 = int(pat_arr[mb0 + trip0 * ns0 + xapos])

        # transfer_i == -1 is the zero-walk same-stop option, followed by CSR transfer order.
        p0 = int(tr_off[alight_stop])
        p1 = int(tr_off[alight_stop + 1])
        for transfer_i in range(-1, p1 - p0):
            if transfer_i < 0:
                stop = alight_stop
                walk_sec = 0
            else:
                p = p0 + transfer_i
                walk_sec = int(tr_time[p])
                if walk_sec > 12 * 60:
                    continue
                stop = int(tr_to[p])
            ready = arr0 + walk_sec + int(board_slack)

            for rp in range(int(ras_off[stop]), int(ras_off[stop + 1])):
                pi = int(ras_pat[rp])
                bpos = int(ras_pos[rp])
                ns = int(pat_nstops[pi])
                nt = int(pat_ntrips[pi])
                mb = int(pat_mat_off[pi])
                sb = int(pat_stop_off[pi])

                lo2 = 0
                hi2 = nt
                while lo2 < hi2:
                    mid = (lo2 + hi2) >> 1
                    if pat_dep[mb + mid * ns + bpos] < ready:
                        lo2 = mid + 1
                    else:
                        hi2 = mid
                trip = lo2
                if trip >= nt:
                    continue
                trow = mb + trip * ns

                best_finish = 1 << 62
                best_eg = 1 << 62
                best_apos = -1
                best_stop = -1
                for apos in range(bpos + 1, ns):
                    st2 = int(pat_stops[sb + apos])
                    eg = int(egress_sec[st2])
                    if eg >= egress_inf:
                        continue
                    finish = int(pat_arr[trow + apos]) + eg
                    if (finish < best_finish
                            or (finish == best_finish and
                                (eg < best_eg or (eg == best_eg and apos > best_apos)))):
                        best_finish = finish
                        best_eg = eg
                        best_apos = apos
                        best_stop = st2
                if best_apos < 0:
                    continue

                out[nout, 0] = xapos
                out[nout, 1] = alight_stop
                out[nout, 2] = arr0
                out[nout, 3] = stop
                out[nout, 4] = walk_sec
                out[nout, 5] = pi
                out[nout, 6] = bpos
                out[nout, 7] = int(pat_dep[trow + bpos])
                out[nout, 8] = int(pat_arr[trow + best_apos])
                out[nout, 9] = best_apos
                out[nout, 10] = best_stop
                out[nout, 11] = best_eg
                nout += 1
    return out[:nout]
