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

    # -- cell -> chosen access stop (must match assemble_arriveby's argmax) ----------------
    def _select(self, ci):
        a0, a1 = int(self.access_off[ci]), int(self.access_off[ci + 1])
        s_star, aw, latest_home = -1, 0, NEG
        best = self.best
        for a in range(a0, a1):
            g = int(self.access_to[a]); h = int(best[g]) - int(self.access_w[a])
            if h > latest_home:
                latest_home = h; s_star = g; aw = int(self.access_w[a])
        pw = int(self.purewalk[ci])
        walk_home = (self.deadline - pw) if pw >= 0 else NEG
        if walk_home >= latest_home:        # pure walk wins (or ties) -> walk-only journey
            return None, pw, walk_home, True
        return s_star, aw, latest_home, False

    # -- raw forward legs (exact seconds) -------------------------------------------------
    def _trace(self, ci):
        """Return (legs_raw, latest_home) in FORWARD order, or None if unreachable. legs_raw is a
        list of tuples: ("access",sec) ("ride",pi,dep_sec,arr_sec,board_pos,alight_pos,alight_stop)
        ("walk_t",sec) ("egress",sec), or, for a walk-only journey, [("walk",sec)]. The ride's board/
        alight positions + alight stop feed the committed-plan MC (``committed_first_legs``); the
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
                legs.append(("walk_t", _footpath_sec(tr_off, tr_to, tr_time, s, j)))
                s = j
            else:                               # egress seed: walk to W, done
                legs.append(("egress", int(egress_sec[s]) if egress_sec[s] >= 0 else 0))
                return legs, latest_home
        return None                             # exceeded guard (shouldn't happen)

    def _clock(self, legs_raw, latest_home):
        """Forward-simulate the clock through legs_raw -> (out, total_sec) where out is the
        folded walk/transit list with exact seconds and total_sec = sum of all leg durations
        (= the journey's actual commute, arrival - latest home departure)."""
        out = []
        t = latest_home
        for leg in legs_raw:
            if leg[0] in ("access", "walk_t", "egress", "walk"):
                w = leg[1]; t += w; _push_walk(out, w)
            else:                                        # ride
                pi, dep_sec, arr_sec = leg[1], leg[2], leg[3]
                wait = max(0, dep_sec - t)
                ride = arr_sec - dep_sec
                if ride < _TINY_HOP_MIN * 60:            # fold a fluke 1-stop hop into walk
                    _push_walk(out, wait + ride)
                else:
                    out.append({"mode": "transit", "line": self._name(pi),
                                "sec": ride, "wait_sec": wait})
                t = arr_sec
        return out, t - latest_home

    # -- public: full itinerary (hover) ---------------------------------------------------
    def itinerary(self, ci):
        tr = self._trace(ci)
        if tr is None:
            return None
        legs_raw, latest_home = tr
        out, total_sec = self._clock(legs_raw, latest_home)
        total_min = int(np.ceil(total_sec / 60.0))
        if total_min >= self.max_min:
            return None
        return self._format(out, total_min)

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

        Reuses ``_trace`` (the same per-cell journey the map/hover show) and just reads the FIRST ride
        + the walk leading up to it, so the committed plan is the displayed plan by construction."""
        n = self.n_cells
        out = dict(
            commit_home=np.full(n, NEG, np.int64), commit_kind=np.zeros(n, np.int8),
            commit_walk0=np.zeros(n, np.int64), commit_pi=np.full(n, -1, np.int32),
            commit_bpos=np.full(n, -1, np.int32), commit_apos=np.full(n, -1, np.int32),
            commit_as=np.full(n, -1, np.int32))
        for ci in range(n):
            tr = self._trace(ci)
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
        for ci in range(self.n_cells):
            tr = self._trace(ci)
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
        # to minutes, then reconcile residual into the last walk leg so legs sum to total_min
        legs = []
        for l in out:
            if l["mode"] == "walk":
                legs.append({"mode": "walk", "line": None, "min": int(round(l["sec"] / 60.0))})
            else:
                legs.append({"mode": "transit", "line": l["line"],
                             "min": int(round(l["sec"] / 60.0)),
                             "wait": int(round(l["wait_sec"] / 60.0))})
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


def _push_walk(out, sec):
    if sec <= 0:
        return
    if out and out[-1]["mode"] == "walk":
        out[-1]["sec"] += sec
    else:
        out.append({"mode": "walk", "line": None, "sec": sec})


def _footpath_sec(tr_off, tr_to, tr_time, s, j):
    for k in range(int(tr_off[s]), int(tr_off[s + 1])):
        if int(tr_to[k]) == j:
            return int(tr_time[k])
    return 0                                             # same-stop transfer (no footpath edge)
