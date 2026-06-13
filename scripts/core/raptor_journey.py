"""Journey reconstruction from the reverse-RAPTOR back-pointers (Phase 2, JVM-free).

From a single traced arrive-by tree (``raptor.reverse_raptor_traced`` at the target deadline),
recover, for any grid cell: the door-to-door leg breakdown (walk-access -> board route -> ride ->
wait -> transfer -> ... -> egress walk, feed-aware names) AND the color-by-line dominant route.

INVARIANT (the whole point of Phase 2): the breakdown legs sum to the cell's map value, because
both come from THIS engine. The arrive-by map value is the *actual commute* of the latest-feasible
journey = (workplace arrival) - (latest home departure), which equals the sum of the traced legs
exactly (pre-rounding). Rounding is reconciled into the egress walk (mirroring server.py's
_build_itin) so the displayed integer legs sum to the integer map minutes.

Determinism: the traced tree already breaks ties deterministically (sorted pattern scan, first
writer wins); the dominant-line pick breaks ride-time ties by (route name, feed, route_id) so
color-by-line is stable run-to-run.
"""
import numpy as np

from .raptor import NEG

_TINY_HOP_MIN = 2.0          # fold sub-2-min transit hops into adjacent walk (matches server.py)


def reconcile_legs(legs, total_min):
    """Final rounding reconciliation for a leg breakdown — the ONE implementation shared by
    the RAPTOR hover path (``JourneyTree._format``) and the R5 recorded-path breakdown
    (``server._build_itin``), so the two displays can never drift.

    ``legs`` is the already-ROUNDED integer leg list ({"mode","line","min"[,"wait"]});
    ``total_min`` the integer map total the legs must sum to. Drops zero-minute walk legs,
    then patches the rounding residual into the LAST walk leg (clamped >= 0; a positive
    residual with no walk leg appends one) and re-filters, so the displayed legs + waits sum
    EXACTLY to the map color's minutes. Returns the response dict {"total","xfers","legs"}."""
    total_min = int(total_min)
    legs = [l for l in legs if not (l["mode"] == "walk" and l["min"] <= 0)]
    rides_kept = sum(1 for l in legs if l["mode"] != "walk")
    cur = sum(l["min"] for l in legs) + sum(l.get("wait", 0) for l in legs)
    diff = total_min - cur
    if diff != 0:
        walks = [l for l in legs if l["mode"] == "walk"]
        if walks:
            walks[-1]["min"] = max(0, walks[-1]["min"] + diff)
        elif diff > 0:
            legs.append({"mode": "walk", "line": None, "min": diff})
        legs = [l for l in legs if not (l["mode"] == "walk" and l["min"] <= 0)]
    return {"total": total_min, "xfers": max(0, rides_kept - 1), "legs": legs}


class JourneyTree:
    """Wraps one traced reverse tree + the workplace's access/pure-walk so every cell's journey
    can be reconstructed without re-running RAPTOR."""

    def __init__(self, data, par, access_off, access_to, access_w, purewalk, deadline, max_min):
        self.d = data
        self.par = par
        self.best = par["best"]
        self.access_off = access_off
        self.access_to = access_to
        self.access_w = np.asarray(access_w, np.int64)
        self.purewalk = np.asarray(purewalk, np.int64)
        self.deadline = int(deadline)
        self.max_min = max_min
        self.n_cells = len(access_off) - 1
        self.line_table = data["line_table"]
        self.pat_line = data["pat_line"]
        self._traces = None                  # lazy full-grid trace memo (tree is immutable)
        self._sel = None                     # lazy vectorized _select memo (cell-aligned arrays)

    # -- cell -> chosen access stop (must match assemble_arriveby's argmax) ----------------
    def _select_arrays(self):
        """Vectorized per-cell access-stop selection over the whole grid, computed lazily ONCE
        per tree (the per-cell python loop was ~93% of _trace_all; the segmented-reduce version
        is ~30x faster and tuple-identical). Returns cell-aligned arrays
        (s_star int64, aw int64, latest int64, is_walk bool, walk_home int64).

        Edge cases replicated from the loop:
          * empty access segments (cells with zero access pairs): raw ``reduceat`` would
            duplicate the NEXT segment's value on a zero-length segment (and a trailing empty
            segment indexes out of range), so reduce only over the nonempty segments
            (``nz = flatnonzero(diff(off) > 0)``) — empty cells keep (-1, 0, NEG).
          * all-unreachable segments: every h there is NEG - w != NEG, so the raw segment max
            is a contaminated sentinel; clamp with ``segmax > NEG // 2`` so those cells keep
            the loop's exact (-1, 0, NEG).
          * tie rule: the loop's strict ``>`` keeps the FIRST index achieving the segment max;
            replicated as a segmented first-argmax (``minimum.reduceat`` over masked indices)."""
        if self._sel is None:
            off = np.asarray(self.access_off, np.int64)
            h = self.best[self.access_to] - self.access_w        # int64; NEG - w if unreachable
            n = self.n_cells
            latest = np.full(n, NEG, np.int64)
            s_star = np.full(n, -1, np.int64)
            aw = np.zeros(n, np.int64)
            seglen = np.diff(off)
            nz = np.flatnonzero(seglen > 0)
            if len(nz):
                starts = off[nz]
                segmax = np.maximum.reduceat(h, starts)
                inv = np.zeros(n, np.int64); inv[nz] = np.arange(len(nz))
                cellid = np.repeat(np.arange(n), seglen)         # pair -> cell
                ismax = h == segmax[inv[cellid]]
                idx = np.arange(len(h), dtype=np.int64)
                masked = np.where(ismax, idx, len(h))
                first = np.minimum.reduceat(masked, starts)      # first index achieving seg max
                ok = segmax > NEG // 2
                latest[nz[ok]] = segmax[ok]
                s_star[nz[ok]] = self.access_to[first[ok]]
                aw[nz[ok]] = self.access_w[first[ok]]
            pw = self.purewalk
            walk_home = np.where(pw >= 0, self.deadline - pw, NEG)
            is_walk = walk_home >= latest      # pure walk wins (or ties) -> walk-only journey
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
        clock/dominant readers use only pi/dep/arr."""
        s_star, aw, latest_home, is_walk = self._select(ci)
        if is_walk:
            if self.purewalk[ci] < 0:
                return None
            return [("walk", int(self.purewalk[ci]))], latest_home
        if s_star < 0:
            return None
        par = self.par
        d = self.d
        pat_nstops = d["pat_nstops"]; pat_mat_off = d["pat_mat_off"]; pat_stop_off = d["pat_stop_off"]
        pat_dep = d["pat_dep"]; pat_arr = d["pat_arr"]; pat_stops = d["pat_stops"]
        tr_off = d["tr_off"]; tr_to = d["tr_to"]; tr_time = d["tr_time"]
        par_kind = par["par_kind"]; par_pat = par["par_pat"]; par_trip = par["par_trip"]
        par_board = par["par_board"]; par_alight = par["par_alight"]; par_from = par["par_from"]
        egress_sec = par["egress_sec"]

        # Trace from s* (home-side access stop) FORWARD toward W: each parent step moves W-ward
        # (a transit board rides to its alight nearer W; a footpath walks toward W), so the list
        # is already in forward home->W order — no reversal.
        legs = [("access", int(aw))]
        s = s_star
        for _ in range(64):                     # bounded (loop-route / cycle safety)
            k = int(par_kind[s])
            if k == 1:                          # transit board at s, ride to par_alight
                pi = int(par_pat[s]); trip = int(par_trip[s])
                bpos = int(par_board[s]); apos = int(par_alight[s])
                ns = int(pat_nstops[pi]); mbase = int(pat_mat_off[pi]); sbase = int(pat_stop_off[pi])
                alight_stop = int(pat_stops[sbase + apos])
                dep_sec = int(pat_dep[mbase + trip * ns + bpos])
                arr_sec = int(pat_arr[mbase + trip * ns + apos])
                legs.append(("ride", pi, dep_sec, arr_sec, bpos, apos, alight_stop))
                s = alight_stop
            elif k == 2:                        # footpath s -> par_from (toward W)
                j = int(par_from[s])
                legs.append(("walk_t", _footpath_sec(tr_off, tr_to, tr_time, s, j), s, j))
                s = j
            else:                               # egress seed: walk to W, done
                legs.append(("egress", int(egress_sec[s]) if egress_sec[s] >= 0 else 0, s))
                return legs, latest_home
        return None                             # exceeded guard (shouldn't happen)

    def _clock(self, legs_raw, latest_home, segs=False):
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
                if ride < _TINY_HOP_MIN * 60:            # fold a fluke 1-stop hop into walk
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

    def _geometry(self, ci, legs, provider):
        """Assemble the per-leg map geometry for an already-formatted leg list (each leg
        carrying its "segs" descriptors from ``_clock``). The provider supplies WALK paths:
          access(ci, stop_gid) / purewalk(ci) / transfer(s, j) / egress(s)
        each returning ([[lat,lon],...], approx_bool) — real walk-graph paths, or a straight
        2-point fallback with approx=True. Ride points come from the pattern stop sequence
        of the SAME traced legs the breakdown shows (the hover==map invariant extends to the
        drawn route). Returns [{mode, name, feed?, tmode?, min, wait?, pts, approx?}, ...]
        aligned 1:1 with ``legs``."""
        s_star, _aw, _lh, _is_walk = self._select(ci)
        geom = []
        for l in legs:
            pts, approx = [], False
            feed = tmode = None
            for sg in l.get("segs", ()):
                kind = sg[0]
                if kind in ("ride", "ridefold"):
                    seg_pts, seg_ap = self._ride_pts(sg[1], sg[2], sg[3]), False
                    if kind == "ride":
                        feed, _rid, _name, tmode = self.line_table[int(self.pat_line[sg[1]])]
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
                g["tmode"] = tmode                       # bart|metro|bus|cable (color key)
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
        n = self.n_cells
        out = dict(
            commit_home=np.full(n, NEG, np.int64), commit_kind=np.zeros(n, np.int8),
            commit_walk0=np.zeros(n, np.int64), commit_pi=np.full(n, -1, np.int32),
            commit_bpos=np.full(n, -1, np.int32), commit_apos=np.full(n, -1, np.int32),
            commit_as=np.full(n, -1, np.int32))
        for ci, tr in enumerate(self._trace_all()):
            if tr is None:
                continue                                 # unreachable (kind stays 0)
            legs_raw, latest_home = tr
            out["commit_home"][ci] = latest_home
            # Find the first SIGNIFICANT ride (mirroring _clock's _TINY_HOP_MIN fold), so the plan
            # the MC scores is the plan the breakdown DISPLAYS. A sub-2-min hop is shown as walk in
            # the hover; treating it as transit here would attach delay-variance to a chip the user
            # can't see — and on cells whose displayed first ride is a LATER, real leg, it would
            # attribute fragility to the wrong line entirely.
            ride = None
            for leg in legs_raw:
                if leg[0] == "ride" and (leg[3] - leg[2]) >= _TINY_HOP_MIN * 60:
                    ride = leg; break
            if ride is None:                             # walk-only (incl. all-tiny-rides) -> deterministic
                out["commit_kind"][ci] = 1
                continue
            _, pi, dep_sec, _arr, bpos, apos, alight_stop = ride
            # walk0 = total seconds from home to the board stop on the unperturbed plan, which is
            # exactly dep_sec - latest_home (the perfect plan boards with 0 slack). Naturally
            # absorbs leading walks AND any tiny rides folded by the loop above — no manual sum.
            out["commit_kind"][ci] = 2; out["commit_pi"][ci] = pi
            out["commit_walk0"][ci] = int(dep_sec) - int(latest_home)
            out["commit_bpos"][ci] = bpos; out["commit_apos"][ci] = apos; out["commit_as"][ci] = alight_stop
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
        # to minutes, then reconcile residual into the last walk leg so legs sum to total_min.
        # The geometry-source "segs" (when _clock collected them) ride along on each leg dict
        # — reconcile_legs filters/patches the SAME dicts, so a dropped zero-minute walk leg
        # drops its geometry too (geom stays 1:1 with the DISPLAYED legs).
        legs = []
        for l in out:
            if l["mode"] == "walk":
                d = {"mode": "walk", "line": None, "min": int(round(l["sec"] / 60.0))}
            else:
                d = {"mode": "transit", "line": l["line"],
                     "min": int(round(l["sec"] / 60.0)),
                     "wait": int(round(l["wait_sec"] / 60.0))}
            if "segs" in l:
                d["segs"] = l["segs"]
            legs.append(d)
        return reconcile_legs(legs, total_min)

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


def _push_walk(out, sec, seg=None):
    if sec <= 0:
        return
    if out and out[-1]["mode"] == "walk":
        out[-1]["sec"] += sec
        if seg is not None:
            out[-1].setdefault("segs", []).append(seg)
    else:
        d = {"mode": "walk", "line": None, "sec": sec}
        if seg is not None:
            d["segs"] = [seg]
        out.append(d)


def _footpath_sec(tr_off, tr_to, tr_time, s, j):
    for k in range(int(tr_off[s]), int(tr_off[s + 1])):
        if int(tr_to[k]) == j:
            return int(tr_time[k])
    return 0                                             # same-stop transfer (no footpath edge)
