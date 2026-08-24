"""Journey reconstruction from the reverse-RAPTOR back-pointers (Phase 2, JVM-free).

From a single traced arrive-by tree (``raptor.reverse_raptor_traced`` at the target deadline),
recover, for any grid cell: the door-to-door leg breakdown (walk-access -> board route -> ride ->
wait -> transfer -> ... -> egress walk, feed-aware names) AND the color-by-line dominant route.

INVARIANT (the whole point of Phase 2): the breakdown legs sum to the cell's map value, because
both come from THIS engine. The arrive-by map value is the *actual commute* of the latest-feasible
journey = (workplace arrival) - (latest home departure), which equals the sum of the traced legs
exactly (pre-rounding). Rounding is distributed across leg components before a final reconciliation
guard, so the displayed integer legs sum to the integer map minutes without overstating the last
walk leg.

Determinism: the traced tree already breaks ties deterministically (sorted pattern scan, first
writer wins); the dominant-line pick breaks ride-time ties by (route name, feed, route_id) so
color-by-line is stable run-to-run.
"""
import numpy as np

from .raptor import NEG, INF as _INF
from . import route_response as _route_response

_TINY_HOP_MIN = _route_response._TINY_HOP_MIN  # fold sub-2-min hops into adjacent walk
EGRESS_INF = _route_response.EGRESS_INF       # per-stop egress-walk sentinel


def reconcile_legs(legs, total_min):
    """Compatibility wrapper for the historical public helper.

    Resolve the delegated implementation at call time so both the old
    ``raptor_journey.reconcile_legs`` and new ``route_response.reconcile_legs`` monkeypatch
    seams remain effective.
    """
    return _route_response.reconcile_legs(legs, total_min)


class JourneyTree:
    """Wraps one traced reverse tree + the workplace's access/pure-walk so every cell's journey
    can be reconstructed without re-running RAPTOR."""

    def __init__(self, data, par, access_off, access_to, access_w, purewalk, deadline, max_min,
                 walk_reluctance=1.0, walk_prior_eps=60.0, egress_g=None, egress_w=None):
        self.d = data
        self.par = par
        self.best = par["best"]
        self.access_off = access_off
        self.access_to = access_to
        self.access_w = np.asarray(access_w, np.int64)
        self.purewalk = np.asarray(purewalk, np.int64)
        self.deadline = int(deadline)
        self.max_min = max_min
        self.beta = float(walk_reluctance)   # mild walk prior, access-stop DECISION only
        self.eps = float(walk_prior_eps)     # hard cap (sec) on the prior's true-time degradation
        self.n_cells = len(access_off) - 1
        self.line_table = data["line_table"]
        self.pat_line = data["pat_line"]
        # Per-stop egress WALK seconds (gid -> sec, sentinel for non-egress-reachable stops): drives
        # the no-overshoot alight pick in ``_trace_from`` (the final ride alights at the forward stop
        # minimizing arr + egress_walk, never riding PAST the workplace and walking back). Passed
        # explicitly from ``raptor_engine.journey_tree`` (it has egress_g/egress_w); a legacy caller
        # without them keeps egress_sec = sentinel everywhere -> the final-ride scan finds only the
        # tree's own alight stop (egress-reachable), so behavior is unchanged.
        self.egress_sec = np.full(data["n_stops"], EGRESS_INF, np.int64)
        if egress_g is not None and egress_w is not None:
            eg = np.asarray(egress_g, np.int64); ew = np.asarray(egress_w, np.int64)
            # keep the MIN per gid (a stop may appear once, but be defensive); ignore negatives
            for g, w in zip(eg, ew):
                w = int(w)
                if w >= 0 and w < self.egress_sec[g]:
                    self.egress_sec[g] = w
        self._node_stats = None              # lazy (depth, jtime) per stop — walk-prior guards
        self._node_dom = None                # lazy per-stop dominant pattern — alt enumeration
        self._traces = None                  # lazy full-grid trace memo (tree is immutable)
        self._sel = None                     # lazy vectorized _select memo (cell-aligned arrays)

    def _build_node_stats(self):
        """Per-stop (displayed_ride_depth, journey_time_from_stop) for the walk prior, in ONE
        bottom-up pass over the immutable node table (``nd_next[nid] < nid``, so increasing id order
        resolves dependencies). For each stop's best journey:

          depth[s]  = number of DISPLAYED transit legs (= xfers + 1), counting only SIGNIFICANT
                      rides (>= _TINY_HOP_MIN) so it matches ``_clock``/``itinerary`` (sub-2-min
                      hops fold into walk). The transfer guard caps the prior at this.
          jtime[s]  = the journey's TRUE duration from STOP s to W (= W-arrival - best[s]). The
                      arrive-by tree's ``best[s]`` is the LATEST departure reaching W by the
                      deadline; the latest-departure journey can ARRIVE before the deadline, so the
                      door-to-door time is the TRACED arrival minus the departure, NOT
                      deadline - departure. Two stops with near-equal latest departures can have
                      very different journey durations — so the eps window must compare jtime, not
                      the home departure. Cell journey via s = jtime[s] + access_walk.

        Node recurrences (toward W): ``warr[node]`` = absolute W-arrival; ``tail[node]`` = trailing
        walk seconds from a node to W with NO board below it (the pure-walk tail). A board ANCHORS
        the absolute arrival (its alight arr_sec + the tail after it); the board closest to W wins.
        Unreachable stops -> depth sentinel + jtime sentinel so they never win a guarded tie. Falls
        back to zeros if a legacy ``par`` lacks the node table."""
        n = self.best.shape[0]
        bn = self.par.get("best_node")
        if bn is None:
            return np.zeros(n, np.int64), np.full(n, 1 << 40, np.int64)
        d = self.d
        nd_kind = self.par["nd_kind"]; nd_next = self.par["nd_next"]
        nd_pat = self.par["nd_pat"]; nd_trip = self.par["nd_trip"]
        nd_board = self.par["nd_board"]; nd_alight = self.par["nd_alight"]
        nd_egress = self.par["nd_egress"]; nd_stop = self.par["nd_stop"]
        nd_to = self.par["nd_to"]
        pat_nstops = d["pat_nstops"]; pat_mat_off = d["pat_mat_off"]; pat_stop_off = d["pat_stop_off"]
        pat_dep = d["pat_dep"]; pat_arr = d["pat_arr"]; pat_stops = d["pat_stops"]
        tr_off = d["tr_off"]; tr_to = d["tr_to"]; tr_time = d["tr_time"]
        eg_sec = self.egress_sec
        m = nd_kind.shape[0]
        sig = np.zeros(m, np.int64)            # displayed-ride depth from this node
        tail = np.zeros(m, np.int64)           # trailing walk sec to W (no board below)
        warr = np.full(m, -1, np.int64)        # absolute W-arrival (-1 if no board in chain)
        # ride seconds the board node actually contributes (post no-overshoot alight): used so the
        # depth's significant-ride test matches the TRACED ride, not the (possibly longer) node ride.
        tiny = _TINY_HOP_MIN * 60
        for nid in range(m):                   # increasing id: nd_next[nid] < nid resolved
            nxt = int(nd_next[nid])
            k = int(nd_kind[nid])
            if k == 0:                         # egress: trailing walk only
                tail[nid] = int(nd_egress[nid]) if nd_egress[nid] >= 0 else 0
                sig[nid] = 0; warr[nid] = -1
            elif k == 2:                       # footpath nd_stop -> nd_to, then continue
                fp = _footpath_sec(tr_off, tr_to, tr_time, int(nd_stop[nid]), int(nd_to[nid]))
                tail[nid] = fp + tail[nxt]; sig[nid] = sig[nxt]; warr[nid] = warr[nxt]
            else:                              # board: anchor absolute arrival
                pi = int(nd_pat[nid]); trip = int(nd_trip[nid])
                ns = int(pat_nstops[pi]); mb = int(pat_mat_off[pi]); sb = int(pat_stop_off[pi])
                bp = int(nd_board[nid]); ap = int(nd_alight[nid])
                trow = mb + trip * ns
                dep_sec = int(pat_dep[trow + bp])
                tail[nid] = 0                  # board resets the trailing-walk accumulator
                if nxt >= 0 and int(nd_kind[nxt]) == 0:
                    # FINAL ride (continuation = egress): mirror _trace_from's no-overshoot alight so
                    # jtime matches the TRACED (optimized) journey, not the overshoot node chain.
                    ap, aw = _min_overshoot_alight(pat_arr, pat_stops, eg_sec, trow, sb, bp, ns,
                                                   ap, int(nd_egress[nxt]))
                    arr_sec = int(pat_arr[trow + ap])
                    warr[nid] = arr_sec + aw
                else:
                    arr_sec = int(pat_arr[trow + ap])
                    # the board CLOSEST to W sets the arrival: if a deeper board exists use its warr,
                    # else this board's alight + the trailing walk after it.
                    warr[nid] = warr[nxt] if warr[nxt] >= 0 else arr_sec + tail[nxt]
                ride = arr_sec - dep_sec
                sig[nid] = sig[nxt] + (1 if ride >= tiny else 0)
        depth = np.zeros(n, np.int64); jtime = np.full(n, 1 << 40, np.int64)
        ok = bn >= 0
        nodes = bn[ok]
        depth[ok] = sig[nodes]
        wa = warr[nodes]
        # journey time from the stop to W = (W-arrival - latest departure) when the chain has a
        # board; for a pure-footpath chain (no board, warr<0) it's just the trailing walk ``tail``
        # (a valid walk-via-stop journey, not the unreachable sentinel).
        jtime[ok] = np.where(wa >= 0, wa - self.best[ok], tail[nodes])
        depth[~ok] = 1 << 30                   # unreachable: never wins a guarded tie
        return depth, jtime

    def _build_stop_dominant(self):
        """Per-stop DOMINANT pattern index (the pattern of the longest SIGNIFICANT ride along that
        stop's single traced journey) — the line that journey would be labeled by ``_dominant``,
        precomputed for the whole grid in ONE bottom-up node-table pass (``nd_next[nid] < nid``).
        Returns int64[n_stops]: the dominant pattern id, or -1 for a walk-only / unreachable chain.

        Mirrors ``_dominant``'s pick EXACTLY so the alt window's primary label agrees with the
        traced-journey label (closes the documented downtown-1601 KNOWN GAP):
          * FINAL-RIDE NO-OVERSHOOT (2026-06-16): for the board whose continuation is the egress seed
            (``nxt`` kind 0), measure the ride to the SAME no-overshoot alight ``_trace_from`` emits
            (``_min_overshoot_alight``), NOT the longer raw-node-chain alight. Without this the final
            ride is measured at its overshoot length, which can make it (wrongly) beat the true
            dominant — e.g. cell 1601's F leg measured 811s (overshoot) vs the traced 578s, so this
            pass labeled the journey "F" while ``_dominant`` (on the re-picked alight) labeled it "28".
          * TIE-BREAK by route NAME, then feed, then route_id — the SAME key ``_dominant`` uses
            (was: lower ``pat_line`` index, which could disagree on an exact ride-second tie).
        Used to GROUP access stops by line for the alt window; the displayed alt line/time is still
        re-derived by ``_dominant`` on the actually-traced legs, so it stays exact. Falls back to all
        -1 if a legacy ``par`` lacks the node table."""
        n = self.best.shape[0]
        bn = self.par.get("best_node")
        if bn is None:
            return np.full(n, -1, np.int64)
        d = self.d
        nd_kind = self.par["nd_kind"]; nd_next = self.par["nd_next"]
        nd_pat = self.par["nd_pat"]; nd_trip = self.par["nd_trip"]
        nd_board = self.par["nd_board"]; nd_alight = self.par["nd_alight"]
        nd_egress = self.par["nd_egress"]
        pat_nstops = d["pat_nstops"]; pat_mat_off = d["pat_mat_off"]; pat_stop_off = d["pat_stop_off"]
        pat_dep = d["pat_dep"]; pat_arr = d["pat_arr"]; pat_stops = d["pat_stops"]
        eg_sec = self.egress_sec
        lt = self.line_table; pl = self.pat_line
        m = nd_kind.shape[0]
        tiny = _TINY_HOP_MIN * 60
        best_ride = np.full(m, -1, np.int64)   # longest significant ride sec in this node's chain
        best_pat = np.full(m, -1, np.int64)    # the pattern achieving it

        def _dom_key(ride, pi):
            feed, rid, name, _mode = lt[int(pl[pi])]
            return (-int(ride), name, feed, rid)

        for nid in range(m):                   # increasing id resolves nd_next first
            nxt = int(nd_next[nid])
            br = best_ride[nxt] if nxt >= 0 else -1
            bp = best_pat[nxt] if nxt >= 0 else -1
            if int(nd_kind[nid]) == 1:         # board: compare its ride to the chain's current best
                pi = int(nd_pat[nid]); trip = int(nd_trip[nid])
                ns = int(pat_nstops[pi]); mb = int(pat_mat_off[pi]); sb = int(pat_stop_off[pi])
                bpos = int(nd_board[nid]); apos = int(nd_alight[nid])
                trow = mb + trip * ns
                if nxt >= 0 and int(nd_kind[nxt]) == 0:
                    # FINAL ride: re-pick the alight to the no-overshoot stop _trace_from emits, so
                    # the ride length (and thus the dominant pick) matches the displayed journey.
                    apos, _aw = _min_overshoot_alight(pat_arr, pat_stops, eg_sec, trow, sb, bpos,
                                                      ns, apos, int(nd_egress[nxt]))
                ride = int(pat_arr[trow + apos]) - int(pat_dep[trow + bpos])
                if ride >= tiny and (bp < 0 or _dom_key(ride, pi) < _dom_key(br, bp)):
                    br = ride; bp = pi
            best_ride[nid] = br; best_pat[nid] = bp
        out = np.full(n, -1, np.int64)
        ok = bn >= 0
        out[ok] = best_pat[bn[ok]]
        return out

    @property
    def _stop_dominant(self):
        if self._node_dom is None:
            self._node_dom = self._build_stop_dominant()
        return self._node_dom

    @property
    def _stop_depth(self):
        if self._node_stats is None:
            self._node_stats = self._build_node_stats()
        return self._node_stats[0]

    @property
    def _stop_jtime(self):
        if self._node_stats is None:
            self._node_stats = self._build_node_stats()
        return self._node_stats[1]

    # -- cell -> chosen access stop (MIN door-to-door journey + walk prior) ----------------
    def _select_arrays(self):
        """Vectorized per-cell access-stop selection over the whole grid, computed lazily ONCE
        per tree (the per-cell python loop was ~93% of _trace_all; the segmented-reduce version
        is ~30x faster). Returns cell-aligned arrays
        (s_star int64, aw int64, latest int64, is_walk bool, walk_home int64).

        OBJECTIVE = MIN DOOR-TO-DOOR JOURNEY (2026-06-16, output-changing on the served arrive-by
        map). The arrive-by reverse tree optimizes LATEST HOME DEPARTURE (leave as late as possible
        and still arrive by the deadline), NOT the shortest trip — two access stops with near-equal
        latest departures can have very different door-to-door durations, because a latest-departure
        journey may ARRIVE well before the deadline (it just couldn't have left later). Picking the
        access stop by max-latest-home therefore showed a SLOWER route than an available faster line
        on ~1774 cells (the faster line surfaced only as an "alt"). We now anchor the selection on
        the SAME quantity the alt window ranks by — the cell's MINIMUM ``cell_jt = jtime[stop] + aw``
        (true door-to-door seconds; ``jtime`` from the node table is the TRACED arrival minus the
        departure, see ``_build_node_stats``) — so the primary is always the fastest route and faster
        walking can never lengthen the commute. ``latest`` (the REPORTED home departure) stays
        ``best[chosen] - aw[chosen]``, so the reported clock time is the chosen journey's exact time.

        WALK-RELUCTANCE PRIOR (``self.beta`` + ``self.eps``, decision-only): a tie-break ON TOP of
        the min-journey anchor — among access stops whose journey is within ``eps`` of the cell's
        MIN journey and whose ride depth is ``<=`` the anchor's, prefer LESS walking. Two safeguards:
          * EPS BAND (``self.eps`` sec): only stops with ``cell_jt <= anchor_jt + eps`` are
            candidates (the anchor IS the min, so the band is naturally one-sided; the symmetric
            ``lo`` bound is harmless). A genuinely-faster farther stop can't exist — the anchor is
            already the fastest — so the reported time can rise by at most the rounding minute.
          * TRANSFER GUARD: candidates capped to ride depth ``<=`` the anchor's depth, so the prior
            can never add a transfer.
        Within that window the stop minimizing ``pen = cell_jt + (beta-1)*aw`` wins (then LESS walk,
        then first index). ``beta == 1.0`` reduces to "min-journey, then least aw, then first index"
        (NO LONGER the old max-latest-home byte-equal anchor — the objective changed).

        Edge cases replicated from the loop:
          * empty access segments (cells with zero access pairs): reduce only over nonempty segments
            (``nz = flatnonzero(diff(off) > 0)``) — empty cells keep (-1, 0, NEG).
          * all-unreachable segments: an unreachable pair has ``jtime`` at the sentinel, so its
            ``cell_jt`` exceeds the cap and it never wins; segments with no reachable pair keep
            (-1, 0, NEG) via the ``ok`` mask.
          * first-index tie: the segmented first-argmax (``minimum.reduceat`` over masked indices)."""
        if self._sel is None:
            off = np.asarray(self.access_off, np.int64)
            aw_all = self.access_w
            h = self.best[self.access_to] - aw_all              # int64 TRUE home; NEG - w if unreach
            bw = self.beta - 1.0
            n = self.n_cells
            seglen = np.diff(off)
            nz = np.flatnonzero(seglen > 0)
            IMIN = np.iinfo(np.int64).min

            def _seg_first_argmax(key):
                """Per nonempty segment: index of the FIRST pair achieving the max key (or -1 if no
                pair has key > IMIN). Returns (sel int64[len(nz)], ok bool[len(nz)])."""
                if not len(nz):
                    return np.zeros(0, np.int64), np.zeros(0, bool)
                starts = off[nz]
                segmax = np.maximum.reduceat(key, starts)
                inv = np.zeros(n, np.int64); inv[nz] = np.arange(len(nz))
                cellid = np.repeat(np.arange(n), seglen)
                ismax = key == segmax[inv[cellid]]
                idx = np.arange(len(key), dtype=np.int64)
                masked = np.where(ismax, idx, len(key))
                first = np.minimum.reduceat(masked, starts)
                ok = segmax > IMIN
                return first, ok

            # per-pair door-to-door journey seconds (the alt window's ranking key); unreachable pairs
            # carry the jtime sentinel (~1<<40) so cell_jt overflows the cap and never wins.
            jt_stop = self._stop_jtime[self.access_to]
            cell_jt = jt_stop + aw_all
            JBIG = np.int64(1 << 39)
            reach_pair = jt_stop < JBIG
            depth = self._stop_depth[self.access_to]
            cellid = np.repeat(np.arange(n), seglen)

            # ANCHOR = the cell's MIN journey (segmented argmin over cell_jt) — its journey time is
            # the eps reference and its ride depth the transfer ceiling. Implemented as a first-argmax
            # over the negated journey (so ties keep the FIRST index, identical reduce machinery).
            base_key = np.where(reach_pair, -cell_jt, IMIN)
            base_first, base_ok = _seg_first_argmax(base_key)
            if bw == 0.0:
                first, ok = base_first, base_ok
            else:
                base_jt = np.full(n, 1 << 40, np.int64); d_base = np.full(n, 1 << 30, np.int64)
                bsel = base_first[base_ok]
                base_jt[nz[base_ok]] = cell_jt[bsel]      # the anchor's (min) journey time
                d_base[nz[base_ok]] = depth[bsel]
                eps = np.int64(round(self.eps))
                # eps band around the MIN journey + transfer guard. The anchor IS the min, so
                # cell_jt >= base_jt always holds for reachable pairs; the lower ``lo`` bound is kept
                # (harmless: never excludes a candidate) so the form mirrors the symmetric original.
                lo = base_jt[cellid] - eps; hi = base_jt[cellid] + eps
                allowed = (reach_pair
                           & (cell_jt >= lo) & (cell_jt <= hi)       # within eps of the min journey
                           & (depth <= d_base[cellid]))             # transfer guard: <= anchor rides
                # within the band: MINIMIZE penalized journey (jtime + beta*aw), so a closer stop
                # (less aw) wins among ~equal-time options, then LESS walk, then first index.
                pen = cell_jt.astype(np.float64) + bw * aw_all.astype(np.float64)   # = jt + beta*aw
                _W = np.int64(10_000_000)
                key = np.where(allowed, -(np.rint(pen * 100.0).astype(np.int64) * _W + aw_all), IMIN)
                first, ok = _seg_first_argmax(key)
            latest = np.full(n, NEG, np.int64)
            s_star = np.full(n, -1, np.int64)
            aw = np.zeros(n, np.int64)
            anchor_jt = np.full(n, 1 << 40, np.int64)           # the min-journey anchor's true sec
            if len(nz):
                sel = first[ok]
                latest[nz[ok]] = h[sel]                          # TRUE home of the chosen stop
                s_star[nz[ok]] = self.access_to[sel]
                aw[nz[ok]] = aw_all[sel]
                if bw == 0.0:
                    anchor_jt[nz[ok]] = cell_jt[sel]
                else:
                    anchor_jt[nz[base_ok]] = cell_jt[bsel]
            pw = self.purewalk
            walk_home = np.where(pw >= 0, self.deadline - pw, NEG)
            # Walk-vs-transit decided against the MIN-JOURNEY transit anchor (durations, not home
            # departures): pure walk wins iff its duration <= the anchor transit journey's duration
            # (walk wins ties, matching the old ``walk_home >= anchor_home`` direction translated to
            # durations). The prior re-selects only AMONG transit access stops, never flips a cell
            # walk<->transit, so which cells are walk-only is set purely by this comparison.
            pw_dur = np.where(pw >= 0, pw.astype(np.int64), 1 << 40)
            is_walk = (pw >= 0) & (pw_dur <= anchor_jt)
            self._sel = (s_star, aw, latest, is_walk, walk_home)
        return self._sel

    def _select(self, ci):
        """Scalar lookup into the lazy vectorized memo — same tuple contract as the original
        per-cell loop: (s_star, aw, latest_home, False) for transit, (None, pw, walk_home, True)
        when the pure walk wins or ties. Kept as the scalar API for one-off hovers
        (_trace_uncached / _geometry) without forcing a full-grid trace fill."""
        s_star, aw, latest, is_walk, walk_home = self._select_arrays()
        if is_walk[ci]:
            return None, int(self.purewalk[ci]), int(walk_home[ci]), True
        return int(s_star[ci]), int(aw[ci]), int(latest[ci]), False

    # -- raw forward legs (exact seconds) -------------------------------------------------
    def _trace(self, ci):
        """Memo-aware trace: serve from the full-grid memo when it exists, else compute the
        single cell uncached (keeps a one-off hover cheap without forcing a full-grid fill)."""
        tr = self._traces
        if tr is not None:
            return tr[ci]
        return self._trace_uncached(ci)

    def _trace_all(self):
        """Trace every cell ONCE and memoize. The tree is immutable after construction, all
        readers (_clock/_dominant/committed_first_legs) are read-only on legs_raw, and the memo
        is assigned in a single statement (atomic under the GIL) — so a concurrent hover
        itinerary() is safe and full-grid callers (commute_and_dominant at tree build,
        committed_first_legs on a /variance miss) walk the ~3000 back-pointer chains only once
        per tree instead of once per caller."""
        if self._traces is None:
            self._traces = [self._trace_uncached(ci) for ci in range(self.n_cells)]
        return self._traces

    def _trace_uncached(self, ci):
        """Return (legs_raw, latest_home) in FORWARD order, or None if unreachable. legs_raw is a
        list of tuples: ("access",sec) ("ride",pi,dep_sec,arr_sec,board_pos,alight_pos,alight_stop)
        ("walk_t",sec,from_stop,to_stop) ("egress",sec,from_stop), or, for a walk-only journey,
        [("walk",sec)]. The ride's board/alight positions + alight stop feed the committed-plan MC
        (``committed_first_legs``); the walk legs' stop gids feed the route GEOMETRY (everything
        else reads positionally from the front, so the appends are backward-compatible); the
        clock/dominant readers use only pi/dep/arr.

        Walks the immutable ride-depth NODE CHAIN (``best_node`` -> ``nd_next``), not the per-stop
        ``par_*`` arrays: each node records the EXACT continuation the value consumed at relax time,
        so the reconstructed journey can never carry more transit legs than ``max_rounds`` (the
        per-stop chain could append a phantom ride — see ``raptor.reverse_raptor_traced``)."""
        s_star, aw, latest_home, is_walk = self._select(ci)
        if is_walk:
            if self.purewalk[ci] < 0:
                return None
            return [("walk", int(self.purewalk[ci]))], latest_home
        if s_star < 0:
            return None
        return self._trace_from(s_star, aw, latest_home)

    def _trace_from(self, s_star, aw, latest_home):
        """The shared node-chain walk: trace the tree's single journey FROM access stop ``s_star``
        (reached after ``aw`` access-walk seconds, departing home at ``latest_home``) FORWARD toward
        W. Returns (legs_raw, latest_home) — the same forward-order tuple list ``_trace_uncached``
        produces — or None if the stop is unreachable / the chain is malformed. Used both by the
        cell's SELECTED journey (``_trace_uncached``) and by the per-access-stop ALTERNATIVE
        enumeration (``alt_lines_window`` / ``itinerary_via_stop``), so an alt route is traced by the
        exact same machinery as the primary.

        NO-OVERSHOOT ALIGHT (2026-06-16): the reverse tree optimizes LATEST HOME DEPARTURE, so its
        node chain can ride PAST the stop closest to W and then walk back (a longer ride AND a longer
        egress walk — strictly dominated). For the FINAL ride (the board whose continuation is the
        egress seed), we keep the line/trip + board (preserving the access selection + reported
        departure) but re-pick the ALIGHT to MINIMIZE the downstream W-arrival = ``arr[p] +
        egress_walk(stop@p)`` over every forward, egress-reachable position ``p >= bpos`` on that
        trip, then emit the egress from that optimal stop. Intermediate rides (continuation is a
        footpath or another board) keep the tree's alight: the transfer stop is structurally fixed,
        and moving it would change the transfer sequence (out of scope). The board CLOSEST to W
        already sets the arrival, so this never rides slower — it only stops overshooting."""
        if s_star < 0:
            return None
        par = self.par
        d = self.d
        pat_nstops = d["pat_nstops"]; pat_mat_off = d["pat_mat_off"]; pat_stop_off = d["pat_stop_off"]
        pat_dep = d["pat_dep"]; pat_arr = d["pat_arr"]; pat_stops = d["pat_stops"]
        tr_off = d["tr_off"]; tr_to = d["tr_to"]; tr_time = d["tr_time"]
        nd_kind = par["nd_kind"]; nd_stop = par["nd_stop"]; nd_pat = par["nd_pat"]
        nd_trip = par["nd_trip"]; nd_board = par["nd_board"]; nd_alight = par["nd_alight"]
        nd_to = par["nd_to"]; nd_egress = par["nd_egress"]; nd_next = par["nd_next"]
        eg_sec = self.egress_sec

        # Trace from s* (home-side access stop) FORWARD toward W via the node chain: each node moves
        # W-ward (a board rides to its alight nearer W; a footpath walks toward W), so the list is
        # already in forward home->W order — no reversal.
        legs = [("access", int(aw))]
        nid = int(par["best_node"][s_star])
        for _ in range(64):                     # bounded (loop-route / cycle safety)
            if nid < 0:
                return None
            k = int(nd_kind[nid])
            if k == 1:                          # transit board, ride to nd_alight
                pi = int(nd_pat[nid]); trip = int(nd_trip[nid])
                bpos = int(nd_board[nid]); apos = int(nd_alight[nid])
                ns = int(pat_nstops[pi]); mbase = int(pat_mat_off[pi]); sbase = int(pat_stop_off[pi])
                trow = mbase + trip * ns
                nxt = int(nd_next[nid])
                if nxt >= 0 and int(nd_kind[nxt]) == 0:
                    # FINAL ride (continuation = egress): re-pick the alight to minimize the
                    # downstream W-arrival arr[p] + egress_walk(stop@p) over forward egress-reachable
                    # positions, then walk to W from that optimal stop (no overshoot).
                    apos, eg = _min_overshoot_alight(pat_arr, pat_stops, eg_sec, trow, sbase, bpos,
                                                     ns, apos, int(nd_egress[nxt]))
                    alight_stop = int(pat_stops[sbase + apos])
                    arr_sec = int(pat_arr[trow + apos])
                    dep_sec = int(pat_dep[trow + bpos])
                    legs.append(("ride", pi, dep_sec, arr_sec, bpos, apos, alight_stop))
                    legs.append(("egress", eg, alight_stop))
                    return legs, latest_home
                alight_stop = int(pat_stops[sbase + apos])
                dep_sec = int(pat_dep[trow + bpos])
                arr_sec = int(pat_arr[trow + apos])
                legs.append(("ride", pi, dep_sec, arr_sec, bpos, apos, alight_stop))
            elif k == 2:                        # footpath nd_stop -> nd_to (toward W)
                s = int(nd_stop[nid]); j = int(nd_to[nid])
                legs.append(("walk_t", _footpath_sec(tr_off, tr_to, tr_time, s, j), s, j))
            else:                               # egress seed: walk to W, done
                s = int(nd_stop[nid]); eg = int(nd_egress[nid])
                legs.append(("egress", eg if eg >= 0 else 0, s))
                return legs, latest_home
            nid = int(nd_next[nid])
        return None                             # exceeded guard (shouldn't happen)

    def _clock(self, legs_raw, latest_home, segs=False, fold_tiny=True):
        """Forward-simulate the clock through legs_raw -> (out, total_sec) where out is the
        folded walk/transit list with exact seconds and total_sec = sum of all leg durations
        (= the journey's actual commute, arrival - latest home departure). With ``segs=True``
        each out leg also carries a "segs" list of geometry-source descriptors (the raw legs
        folded into it), so the route GEOMETRY is derived from the SAME folding the displayed
        breakdown uses — geom legs match breakdown legs 1:1 by construction."""
        out = []
        t = latest_home
        for leg in legs_raw:
            k = leg[0]
            if k in ("access", "walk_t", "egress", "walk"):
                w = leg[1]; t += w
                sg = None
                if segs:
                    if k == "access":   sg = ("access",)
                    elif k == "walk":   sg = ("purewalk",)
                    elif k == "walk_t": sg = ("walkt", int(leg[2]), int(leg[3]))
                    else:               sg = ("egress", int(leg[2]))
                _push_walk(out, w, sg)
            else:                                        # ride
                pi, dep_sec, arr_sec = leg[1], leg[2], leg[3]
                wait = max(0, dep_sec - t)
                ride = arr_sec - dep_sec
                if fold_tiny and ride < _TINY_HOP_MIN * 60:  # fold a fluke 1-stop hop into walk
                    _push_walk(out, wait + ride,
                               ("ridefold", int(pi), int(leg[4]), int(leg[5])) if segs else None)
                else:
                    d = {"mode": "transit", "line": self._name(pi),
                         "sec": ride, "wait_sec": wait}
                    if segs:
                        d["segs"] = [("ride", int(pi), int(leg[4]), int(leg[5]))]
                    out.append(d)
                t = arr_sec
        return out, t - latest_home

    # -- public: full itinerary (hover) ---------------------------------------------------
    def itinerary(self, ci, geom_provider=None):
        """Breakdown dict for one cell ({"total","xfers","legs"}), or None if unreachable.
        With a ``geom_provider`` (see ``_geometry``) the dict also carries "geom": an ordered
        leg list aligned 1:1 with "legs" (mode+name+min match), each with the leg's real map
        polyline — ride legs from the pattern's stop coordinates, walk legs from the provider
        (walk-graph paths, or straight approx segments)."""
        tr = self._trace(ci)
        if tr is None:
            return None
        legs_raw, latest_home = tr
        out, total_sec = self._clock(legs_raw, latest_home, segs=geom_provider is not None)
        total_min = int(np.ceil(total_sec / 60.0))
        if total_min >= self.max_min:
            return None
        res = self._format(out, total_min)
        if geom_provider is not None:
            res["geom"] = self._geometry(ci, res["legs"], geom_provider)
            for l in res["legs"]:
                l.pop("segs", None)              # internal plumbing; not part of the response
        return res

    # -- alternatives: dominance window over the per-access-stop journeys ------------------
    def alt_lines_window(self, perfect, window_min, cells=None):
        """Per-cell ALTERNATIVE lines as a deterministic, walk-speed-STABLE dominance window over
        THIS (unperturbed) tree's per-access-stop journeys — the diag's recommended fix (option A,
        .plans/alt_walkspeed_diag.md). For each cell, every access stop offers exactly one journey
        (the tree's latest-departure chain from that stop); its door-to-door time = jtime[stop] +
        access_walk and its dominant line = ``_stop_dominant[stop]``. We keep, per distinct line, its
        FASTEST such time + the access stop achieving it, then keep every line whose best time is
        within ``window_min`` of the cell's best (= min over perfect + all access-stop journeys).

        Unlike the old K-draw MC vote (which dropped a within-1-2-min short-walk bus when faster
        walking made some OTHER journey the strict argmin in all K draws), this is K-free and stable:
        a bus within the window at slow walk stays within it at fast (its gap to best grows only by
        the walk-speed delta on the access leg). Each kept line carries a real access stop, so the
        route is traceable via ``itinerary_via_stop``.

        ``cells`` (optional iterable of cell indices) restricts the loop to a SUBSET — the per-stop
        node tables (``_stop_jtime``/``_stop_dominant``) are built once for the whole tree regardless,
        but only the listed cells are populated in the result (the rest stay None). The depart-after
        ``alt_lines_window`` uses this to run each representative T*-tree over ONLY its own cells (so
        the ~21 T*-trees don't each pay a full-grid pass).

        Returns list[dict|None]: per cell ``{line_name: (min_minutes, access_stop_gid)}`` sorted
        closest-first (the PRIMARY line is NOT excluded here — the caller drops it), or None when no
        transit line is within the window."""
        perfect = np.asarray(perfect, np.int64)
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        aw = self.access_w
        jtime = self._stop_jtime                       # per-stop journey sec to W (node table)
        sdom = self._stop_dominant                     # per-stop dominant pattern (node table)
        win = int(round(window_min))
        cap = self.max_min
        out = [None] * self.n_cells
        cell_iter = range(self.n_cells) if cells is None else cells
        for ci in cell_iter:
            a0, a1 = int(off[ci]), int(off[ci + 1])
            if a1 <= a0:
                continue
            cb = int(perfect[ci]) if perfect[ci] >= 0 else (1 << 30)
            best = {}                                  # line_name -> [min_minutes, access_stop]
            for k in range(a0, a1):
                s = int(to[k]); pi = int(sdom[s])
                if pi < 0:
                    continue                           # walk-only / unreachable chain from s
                jt = int(jtime[s])
                if jt >= (1 << 39):
                    continue                           # unreachable sentinel
                m = int(np.ceil((jt + int(aw[k])) / 60.0))
                if m >= cap:
                    continue
                if m < cb:
                    cb = m                             # an access-stop journey can beat perfect's round
                name = self._name(pi)
                cur = best.get(name)
                if cur is None or m < cur[0] or (m == cur[0] and s < cur[1]):
                    best[name] = [m, s]
            if not best:
                continue
            kept = {ln: (mk[0], mk[1]) for ln, mk in best.items() if mk[0] <= cb + win}
            if not kept:
                continue
            out[ci] = dict(sorted(kept.items(), key=lambda kv: (kv[1][0], kv[0])))
        return out

    def itinerary_via_stop(self, ci, s_star, geom_provider=None):
        """Like ``itinerary`` but for an EXPLICIT access stop (an alternative route surfaced by
        ``alt_lines_window``): trace the tree's journey from ``s_star`` to W for cell ``ci`` using
        the same machinery the primary route uses, so the alt's breakdown/geometry are byte-faithful
        to the tree. ``s_star`` must be one of cell ``ci``'s access stops (its access-walk seconds
        are looked up from the CSR). Returns the same dict ``itinerary`` returns, or None."""
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        a0, a1 = int(off[ci]), int(off[ci + 1])
        awk = None
        for k in range(a0, a1):
            if int(to[k]) == int(s_star):
                awk = int(self.access_w[k]); break
        if awk is None:
            return None
        latest_home = int(self.best[int(s_star)]) - awk
        tr = self._trace_from(int(s_star), awk, latest_home)
        if tr is None:
            return None
        legs_raw, latest_home = tr
        out, total_sec = self._clock(legs_raw, latest_home, segs=geom_provider is not None)
        total_min = int(np.ceil(total_sec / 60.0))
        if total_min >= self.max_min:
            return None
        res = self._format(out, total_min)
        if geom_provider is not None:
            res["geom"] = self._geometry(ci, res["legs"], geom_provider, s_star=int(s_star))
            for l in res["legs"]:
                l.pop("segs", None)
        return res

    # -- route geometry (hover map drawing) -----------------------------------------------
    def _ride_pts(self, pi, bpos, apos):
        """[[lat, lon], ...] stop-coordinate polyline for a ride from board to alight position
        (inclusive) along pattern ``pi``. NaN-coordinate stops (no coords in the feed) are
        skipped."""
        d = self.d
        sbase = int(d["pat_stop_off"][pi])
        lat, lon, stops = d["stop_lat"], d["stop_lon"], d["pat_stops"]
        pts = []
        for p in range(int(bpos), int(apos) + 1):
            g = int(stops[sbase + p])
            la, lo = float(lat[g]), float(lon[g])
            if np.isnan(la) or np.isnan(lo):
                continue
            pts.append([round(la, 6), round(lo, 6)])
        return pts

    def _geometry(self, ci, legs, provider, s_star=None):
        """Assemble the per-leg map geometry for an already-formatted leg list (each leg
        carrying its "segs" descriptors from ``_clock``). The provider supplies WALK paths:
          access(ci, stop_gid) / purewalk(ci) / transfer(s, j) / egress(s)
        each returning ([[lat,lon],...], approx_bool) — real walk-graph paths, or a straight
        2-point fallback with approx=True. Ride points come from the pattern stop sequence
        of the SAME traced legs the breakdown shows (the hover==map invariant extends to the
        drawn route). ``s_star`` overrides the access stop (for an ALTERNATIVE route via a
        different stop); None uses the cell's SELECTED stop. Returns
        [{mode, name, feed?, tmode?, min, wait?, pts, approx?}, ...] aligned 1:1 with ``legs``."""
        if s_star is None:
            s_star, _aw, _lh, _is_walk = self._select(ci)
        geom = []
        for l in legs:
            pts, approx = [], False
            feed = route_id = tmode = None
            ride_meta = []
            for sg in l.get("segs", ()):
                kind = sg[0]
                if kind in ("ride", "ridefold"):
                    seg_pts, seg_ap = self._ride_pts(sg[1], sg[2], sg[3]), False
                    if kind == "ride":
                        feed, route_id, _name, tmode = self.line_table[int(self.pat_line[sg[1]])]
                        pi, bpos, apos = (int(sg[1]), int(sg[2]), int(sg[3]))
                        base = int(self.d["pat_stop_off"][pi])
                        stops = self.d["pat_stops"]
                        board_gid = int(stops[base + bpos])
                        alight_gid = int(stops[base + apos])
                        pat_nstops = self.d.get("pat_nstops")
                        if pat_nstops is not None:
                            ns = int(pat_nstops[pi])
                        else:
                            offsets = self.d["pat_stop_off"]
                            ns = ((int(offsets[pi + 1]) if pi + 1 < len(offsets) else len(stops))
                                  - base)
                        terminal_gid = int(stops[base + ns - 1])
                        ride_meta.append((board_gid, alight_gid, terminal_gid))
                elif kind == "access":
                    seg_pts, seg_ap = provider.access(ci, s_star)
                elif kind == "purewalk":
                    seg_pts, seg_ap = provider.purewalk(ci)
                elif kind == "walkt":
                    seg_pts, seg_ap = provider.transfer(sg[1], sg[2])
                else:                                    # egress
                    seg_pts, seg_ap = provider.egress(sg[1])
                approx = approx or seg_ap
                for p in seg_pts:
                    if not pts or pts[-1] != p:          # dedup shared junction points
                        pts.append(p)
            g = {"mode": l["mode"], "name": l.get("line"), "min": l["min"], "pts": pts}
            if l["mode"] == "transit":
                g["feed"] = feed
                g["route_id"] = route_id
                g["tmode"] = tmode                       # bart|metro|bus|cable (color key)
                # GTFS-derived journey actions. Sparse fixtures and old callers may omit the v4
                # stop-name table; in that case the geometry remains valid and these optional
                # fields simply stay absent.
                names = self.d.get("stop_name")
                if names is None:
                    names = ()
                lat = self.d.get("stop_lat")
                lon = self.d.get("stop_lon")
                if ride_meta:
                    board_gid = ride_meta[0][0]
                    alight_gid = ride_meta[-1][1]
                    terminal_gid = ride_meta[0][2]

                    def stop_meta(gid):
                        name = str(names[gid] or "").strip() if gid < len(names) else ""
                        item = {"name": name} if name else {}
                        if lat is not None and lon is not None:
                            la, lo = float(lat[gid]), float(lon[gid])
                            if not np.isnan(la) and not np.isnan(lo):
                                item.update({"lat": round(la, 6), "lon": round(lo, 6)})
                        return item

                    board = stop_meta(board_gid)
                    alight = stop_meta(alight_gid)
                    toward = (str(names[terminal_gid] or "").strip()
                              if terminal_gid < len(names) else "")
                    if board:
                        g["board"] = board
                    if alight:
                        g["alight"] = alight
                    if toward:
                        g["toward"] = toward
                if l.get("wait"):
                    g["wait"] = l["wait"]
            if approx:
                g["approx"] = True
            geom.append(g)
        return geom

    # -- committed-plan MC: per-cell committed FIRST leg (departure + first board) ---------
    def committed_first_legs(self):
        """Per-cell committed plan extracted from THIS (unperturbed) arrive-by tree, for the
        committed-plan Monte-Carlo: you choose your departure and first train from the published
        schedule with no foreknowledge of delays. Returns cell-aligned int arrays:

          commit_home   int64  committed home departure (sec); NEG if unreachable
          commit_kind   int8   0 unreachable, 1 deterministic (walk-only / no transit), 2 transit
          commit_walk0  int64  walk seconds home -> first BOARD stop (access + any leading footpaths)
          commit_pi     int32  first committed transit pattern (-1 if not transit)
          commit_bpos   int32  board position within that pattern
          commit_apos   int32  committed alight position (a LATER position) within that pattern
          commit_as     int32  committed alight stop gid (you re-optimize the TAIL from here)

        The forward sim (raptor_numba.montecarlo_committed) then, per delayed draw, boards the next
        available trip on ``commit_pi`` at ``commit_bpos`` (the same line you planned; a late earlier
        trip you can also catch), rides to ``commit_apos``, and re-optimizes from ``commit_as`` at the
        ACTUAL (late) arrival — so a first-leg delay that blows the transfer costs a real headway.

        Reuses the memoized traces (``_trace_all`` — the same per-cell journeys the map/hover show)
        and just reads the FIRST ride + the walk leading up to it, so the committed plan is the
        displayed plan by construction."""
        out = self._empty_committed(self.n_cells)
        for ci, tr in enumerate(self._trace_all()):
            if tr is None:
                continue                                 # unreachable (kind stays 0)
            self._fill_committed_leg(out, ci, tr)
        return out

    @staticmethod
    def _fill_committed_leg(out, idx, tr, include_tiny=False):
        """Write the committed first leg of one traced journey ``tr`` (= (legs_raw, latest_home))
        into the cell-aligned committed-leg arrays ``out`` at row ``idx``. Shared by
        ``committed_first_legs`` (the full-grid primary plan) AND ``committed_legs_via_stops`` (the
        per-route ALT plans for one pinned cell), so an alt's committed plan is extracted by the EXACT
        same rule the primary uses.

        ``include_tiny`` selects the first-ride rule so it always matches the DISPLAY's:
          * False (arrive-by/legacy default, byte-identical to the old behavior): skip sub-2-min
            hops, mirroring ``_clock``'s ``fold_tiny=True`` fold — the displayed first ride is the
            first SIGNIFICANT ride.
          * True (planned depart-after): the planned display does NOT fold tiny hops
            (``fold_tiny=False``), so its first displayed ride can be a sub-2-min hop; committing to
            the first ride REGARDLESS of tininess keeps the MC's committed boarding identical to the
            boarding the breakdown shows (B4)."""
        legs_raw, latest_home = tr
        out["commit_home"][idx] = latest_home
        # Find the first ride under the DISPLAY's fold rule (see ``include_tiny`` above), so the
        # plan the MC scores is the plan the breakdown DISPLAYS. Under the legacy rule a sub-2-min
        # hop is shown as walk in the hover; treating it as transit here would attach
        # delay-variance to a chip the user can't see — and on cells whose displayed first ride is
        # a LATER, real leg, it would attribute fragility to the wrong line entirely.
        ride = None
        for leg in legs_raw:
            if leg[0] == "ride" and (include_tiny or (leg[3] - leg[2]) >= _TINY_HOP_MIN * 60):
                ride = leg; break
        if ride is None:                                 # walk-only (incl. all-tiny-rides) -> deterministic
            out["commit_kind"][idx] = 1
            return
        _, pi, dep_sec, _arr, bpos, apos, alight_stop = ride
        # walk0 = total seconds from home to the board stop on the unperturbed plan, which is
        # exactly dep_sec - latest_home (the perfect plan boards with 0 slack). Naturally
        # absorbs leading walks AND any tiny rides folded by the loop above — no manual sum.
        out["commit_kind"][idx] = 2; out["commit_pi"][idx] = pi
        out["commit_walk0"][idx] = int(dep_sec) - int(latest_home)
        out["commit_bpos"][idx] = bpos; out["commit_apos"][idx] = apos; out["commit_as"][idx] = alight_stop

    @staticmethod
    def _empty_committed(n):
        """A committed-leg dict (the arrays ``committed_first_legs`` returns) sized for ``n`` rows,
        all defaulting to "unreachable" (commit_kind 0). Filled by ``_fill_committed_leg``."""
        return dict(
            commit_home=np.full(n, NEG, np.int64), commit_kind=np.zeros(n, np.int8),
            commit_walk0=np.zeros(n, np.int64), commit_pi=np.full(n, -1, np.int32),
            commit_bpos=np.full(n, -1, np.int32), commit_apos=np.full(n, -1, np.int32),
            commit_as=np.full(n, -1, np.int32))

    def committed_legs_via_stops(self, ci, stops):
        """Per-ROUTE committed first legs for ONE cell ``ci``, one row per access stop in ``stops``
        (a list of stop gids surfaced by ``alt_lines_window`` for this cell). Each row is the
        committed plan of the journey traced FROM that access stop (``_trace_from`` via
        ``itinerary_via_stop``'s machinery) — the same extraction ``committed_first_legs`` applies to
        the cell's primary. Lets the committed-plan MC score each alternative route's typical with the
        SAME model as the primary, so the compare-list numbers are directly comparable.

        Returns the committed-leg dict (``_empty_committed`` arrays, len == len(stops)); a stop that
        is unreachable / off-cell keeps kind 0 (the kernel takes it as the cap)."""
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        a0, a1 = int(off[ci]), int(off[ci + 1])
        out = self._empty_committed(len(stops))
        for row, s_star in enumerate(stops):
            s_star = int(s_star)
            awk = None
            for k in range(a0, a1):
                if int(to[k]) == s_star:
                    awk = int(self.access_w[k]); break
            if awk is None or s_star < 0:
                continue                                 # off-cell / unreachable -> kind 0 (cap)
            latest_home = int(self.best[s_star]) - awk
            tr = self._trace_from(s_star, awk, latest_home)
            if tr is None:
                continue
            self._fill_committed_leg(out, row, tr)
        return out

    # -- public: per-cell commute minutes + dominant line (map + color-by-line) -----------
    def commute_and_dominant(self):
        commute = np.full(self.n_cells, -1, dtype=np.int32)
        dom = [None] * self.n_cells
        for ci, tr in enumerate(self._trace_all()):
            if tr is None:
                continue
            legs_raw, latest_home = tr
            _, total_sec = self._clock(legs_raw, latest_home)
            m = int(np.ceil(total_sec / 60.0))
            if m >= self.max_min:
                continue
            commute[ci] = m
            dom[ci] = self._dominant(legs_raw)
        return commute, dom

    # -- leg formatting + rounding reconciliation (hover == map) ---------------------------
    def _format(self, out, total_min):
        # ``reconcile_legs`` is passed explicitly to preserve the historical module-level
        # monkeypatch seam while the pure formatting implementation lives in route_response.
        return _route_response.format_legs(out, total_min, reconcile_fn=reconcile_legs)

    def _dominant(self, legs_raw):
        rides = [l for l in legs_raw if l[0] == "ride" and (l[3] - l[2]) >= _TINY_HOP_MIN * 60]
        if not rides:
            return "walk only"
        # longest ride wins; tie-break by (route name, feed, route_id) for run-to-run stability
        best = None; best_key = None
        for r in rides:
            pi, dep_sec, arr_sec = r[1], r[2], r[3]
            feed, rid, name, _mode = self.line_table[int(self.pat_line[pi])]
            key = (-(arr_sec - dep_sec), name, feed, rid)
            if best_key is None or key < best_key:
                best_key = key; best = name
        return best

    def _name(self, pi):
        return self.line_table[int(self.pat_line[pi])][2]


# DepartAfterJourneyTree was SPLIT OUT to core.raptor_journey_da (keeps this file under the
# 1000-line smell threshold). Re-imported here so `raptor_journey.DepartAfterJourneyTree` (used by
# raptor_engine + the tests) keeps resolving. It imports JourneyTree/reconcile_legs/_TINY_HOP_MIN
# from THIS module, so the import lands AFTER those are defined (here, at module end).
from .raptor_journey_da import DepartAfterJourneyTree  # noqa: E402  (cyclic-safe: end of module)


def _push_walk(out, sec, seg=None):
    return _route_response._push_walk(out, sec, seg)


def _footpath_sec(tr_off, tr_to, tr_time, s, j):
    return _route_response._footpath_sec(tr_off, tr_to, tr_time, s, j)


def _min_overshoot_alight(pat_arr, pat_stops, eg_sec, trow, sbase, bpos, ns, apos, nd_egress):
    return _route_response._min_overshoot_alight(
        pat_arr, pat_stops, eg_sec, trow, sbase, bpos, ns, apos, nd_egress)
