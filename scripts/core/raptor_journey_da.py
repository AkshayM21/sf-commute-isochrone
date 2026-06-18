"""Depart-after journey tracer — split out of raptor_journey.py (post-migration hygiene).

``DepartAfterJourneyTree`` is the depart-after sibling of ``raptor_journey.JourneyTree``: it
serves the depart-after map color, hover breakdown, color-by-line, and committed-MC support by
wrapping per-T* arrive-by ``JourneyTree`` instances (one per representative arrival deadline).
It reuses ``JourneyTree`` verbatim (``_trace_from`` / ``_clock`` / ``_format`` / ``_geometry`` /
the committed-leg helpers) plus the shared ``reconcile_legs`` rounding reconciliation, so this is
a CLEAN split: ``raptor_journey`` keeps the arrive-by tracer + the low-level helpers, this module
holds only the depart-after wrapper. It is re-exported from ``raptor_journey`` (``raptor_journey.
DepartAfterJourneyTree``) for backward compatibility with ``raptor_engine`` + the tests. JVM-free.
"""
import numpy as np

from .raptor import INF as _INF
from .raptor_journey import JourneyTree, reconcile_legs, _TINY_HOP_MIN


class DepartAfterJourneyTree:
    """Depart-after journey tracer (Stage 1 of the depart-after map migration, JVM-free).

    The depart-after map value is NOT a fresh forward earliest-arrival search (that traces a
    genuinely different journey — ~3% match). It is the percentile, over the [DEP, DEP+WINDOW]
    departure window, of ``tt(D) = arrivalW(s, D + access_walk) - D`` where ``arrivalW`` is the
    inverted reverse profile (``raptor.stop_arrival_profile``) — the EXACT quantity
    ``raptor.assemble_departafter`` paints. So hover==map holds only if the tracer anchors its
    total on the engine's own ``arrivalW``-derived value, then traces the SAME
    ``reverse_raptor_traced`` tree the arrive-by path builds, at the cell's representative
    arrival deadline T*.

    Algorithm (proto6 / verify_prior: 14626/14626 = 100% over the 5 golden workplaces):
      1. Per cell, over the window departures, compute ``tt(D)`` reading ``arrivalW`` with the SAME
         penalized eps-window access-stop selection ``assemble_departafter`` uses (time-optimal
         first, then a walk-reluctance tie-break inside the eps band — replicating it EXACTLY is
         what avoids proto3's -1 min offsets; a plain time-optimal pick misses 11/14626). Take the
         painted percentile (default p50). The painted value == ``engine.departafter`` by
         construction.
      2. Pick the representative departure ``D*`` = the LATEST departure whose ``tt(D*)`` equals the
         painted percentile (deterministic tie-break -> one stable drawn journey). Its penalized
         access stop ``s*`` and arrival deadline ``T* = arrivalW(s*, D*+aw)`` are exact byproducts.
      3. Build (lazily, cached) ONE ``reverse_raptor_traced`` tree at deadline T* — the IDENTICAL
         call ``raptor_engine.journey_tree`` makes for arrive-by, only the deadline differs — wrap
         it in a vanilla ``JourneyTree``, and read the journey from ``s*`` via
         ``JourneyTree._trace_from(s*, aw, latest_home=D*)`` (the existing no-overshoot final-alight
         + clock + format + dominant machinery, reused verbatim).
      4. Report ``total = ceil((T*-D*)/60)`` so legs reconcile to the painted minute (the same
         reconciliation the arrive-by path uses).

    Only ~15-17 distinct T* exist per workplace, so the per-T* trees are cheap and shared across
    every cell that resolves to the same deadline. Pure numpy/numba — NO r5py.

    Exposes the same caller contract the arrive-by ``JourneyTree`` does:
      ``commute_and_dominant()`` -> (commute int32[n_cells], dominant list) for the map + color-by-line;
      ``itinerary(ci, geom_provider=None)`` -> the per-cell breakdown (+ geom) for hover.
    """

    def __init__(self, data, access_off, access_to, access_w, purewalk, arrivalW, dep_grid,
                 cell_deps, max_min, egress_g, egress_w, percentile=50.0,
                 walk_reluctance=1.0, walk_prior_eps=60.0, max_rounds=8, board_slack=60):
        self.d = data
        self.access_off = np.asarray(access_off, np.int64)
        self.access_to = np.asarray(access_to, np.int64)
        self.access_w = np.asarray(access_w, np.int64)
        self.purewalk = np.asarray(purewalk, np.int64)
        self.arrivalW = arrivalW
        self.dep_grid = np.asarray(dep_grid, np.int64)
        self.cell_deps = np.asarray(cell_deps, np.int64)
        self.max_min = int(max_min)
        self.egress_g = np.asarray(egress_g, np.int32)
        self.egress_w = np.asarray(egress_w, np.int64)
        self.percentile = float(percentile)
        self.beta = float(walk_reluctance)
        self.eps = float(walk_prior_eps)
        self.max_rounds = int(max_rounds)
        self.board_slack = int(board_slack)
        self.n_cells = len(self.access_off) - 1
        self.line_table = data["line_table"]
        self.pat_line = data["pat_line"]
        self._sel = None             # lazy per-cell (s_star, aw, Dstar, Tstar, is_walk, painted_min)
        self._trees = {}             # T* (int sec) -> JourneyTree at deadline T*
        self._dom = None             # cached full-grid dominant list (color-by-line; ~20 traced trees)

    # -- per-cell painted selection: (s*, D*, T*) lockstep with assemble_departafter ------
    def _select_arrays(self):
        """Per-cell representative departure D*, its penalized access stop s*, arrival deadline T*,
        and the painted minute. Delegates to the SINGLE-SOURCE selection kernel
        ``raptor.select_departafter`` (the numba njit fast path on the engine grids; python
        reference otherwise) — the SAME per-departure penalized eps-window pick + ``method='lower'``
        percentile + LATEST-departure anchor that ``raptor.assemble_departafter`` runs for the VALUE,
        only here it ADDITIONALLY emits the selection it otherwise discards. So the painted minute is
        byte-identical to ``engine.departafter`` and the selection can never re-derive-drift from the
        value (the triplicated python loop is gone). Cell-aligned arrays
        (s_star int64[-1 walk/unreach], aw int64, Dstar int64, Tstar int64[-1 walk/unreach],
        is_walk bool, painted_min int32[-1 unreachable])."""
        if self._sel is not None:
            return self._sel
        from . import raptor as R
        painted, s_star, aw_sel, Dstar, Tstar, is_walk = R.select_departafter(
            self.access_off, self.access_to, self.access_w, self.purewalk, self.arrivalW,
            self.dep_grid, self.cell_deps, self.max_min, percentile=self.percentile,
            beta=self.beta, eps=self.eps)
        self._sel = (np.asarray(s_star, np.int64), np.asarray(aw_sel, np.int64),
                     np.asarray(Dstar, np.int64), np.asarray(Tstar, np.int64),
                     np.asarray(is_walk, bool), np.asarray(painted, np.int32))
        return self._sel

    # -- per-T* traced tree (lazy + cached) ----------------------------------------------
    def _tree_at(self, Tstar):
        """A ``JourneyTree`` for the single reverse-traced tree at arrival deadline T* (cached).
        Built with the SAME ``reverse_raptor_traced`` call the arrive-by path uses (only the
        deadline differs), so ``_trace_from`` / ``_clock`` / ``_dominant`` / the no-overshoot
        alight all behave identically. The access table + walk-prior knobs are passed through so
        any incidental ``_select``-based geometry fallback stays consistent; the depart-after total
        is anchored on T*-D*, not on the tree's own arrive-by selection."""
        T = int(Tstar)
        jt = self._trees.get(T)
        if jt is None:
            from . import raptor as R
            par = R.reverse_raptor_traced(self.d, self.egress_g, T - self.egress_w, self.egress_w,
                                          max_rounds=self.max_rounds, board_slack=self.board_slack)
            jt = JourneyTree(self.d, par, self.access_off, self.access_to, self.access_w,
                             self.purewalk, T, self.max_min,
                             walk_reluctance=self.beta, walk_prior_eps=self.eps,
                             egress_g=self.egress_g, egress_w=self.egress_w)
            self._trees[T] = jt
        return jt

    def _trace_raw(self, ci):
        """(legs_raw, latest_home=D*) for cell ``ci`` anchored on the painted (s*, D*, T*), or None
        if unreachable. Walk-only cells return a single pure-walk leg; transit cells trace the
        T*-deadline tree FROM s* via ``JourneyTree._trace_from`` (the existing machinery)."""
        s_star, aw_sel, Dstar, Tstar, is_walk, painted = self._select_arrays()
        if painted[ci] < 0:
            return None
        if is_walk[ci]:
            if self.purewalk[ci] < 0:
                return None
            return [("walk", int(self.purewalk[ci]))], int(Dstar[ci])
        ss = int(s_star[ci])
        if ss < 0:
            return None
        jt = self._tree_at(int(Tstar[ci]))
        return jt._trace_from(ss, int(aw_sel[ci]), int(Dstar[ci]))

    # -- public: per-cell commute minutes (MAP color) -------------------------------------
    def commute(self):
        """commute int32[n_cells] — the depart-after MAP color (the painted percentile minute,
        -1 unreachable), straight from the SINGLE-SOURCE selection kernel. NO trees are traced:
        this is the engine's ``departafter`` value EXACTLY (byte-identical painted minutes), so
        the map deliverable is served in the kernel's tens-of-ms, not seconds. The traced
        dominant line (color-by-line) is the only thing that needs the per-T* trees — see
        ``commute_and_dominant`` / ``dominant``."""
        return self._select_arrays()[5].copy()

    # -- public: per-cell commute minutes + dominant line (map + color-by-line) -----------
    def commute_and_dominant(self):
        """(commute int32[n_cells], dominant list[n_cells]) — the depart-after map color +
        color-by-line. ``commute[ci]`` is the painted percentile minute (instant, from the
        selection kernel); ``dominant[ci]`` the traced journey's dominant line (same ``_dominant``
        tie-break as arrive-by). -1 / None for unreachable cells. hover==map by construction (both
        anchor on the painted value).

        Color-by-line REQUIRES the per-T* reverse-traced trees (one per distinct representative
        deadline, ~15-21/workplace) — the same per-tree ``reverse_raptor_traced`` cost the arrive-by
        sibling pays once. The map color alone is free: use ``commute()`` when the dominant line is
        not wanted (the toggle is lazy in the server)."""
        commute = self.commute()
        return commute, self.dominant(commute)

    def dominant(self, commute=None):
        """dominant list[n_cells] — the traced journey's dominant line per reachable cell (None
        else), the color-by-line attribution. Anchored on the kernel-selected (s*, D*, T*) and read
        from the per-T* trees (built lazily + cached). ``commute`` (the painted array) is reused if
        supplied, else recomputed from the kernel. The full result is cached on the tree (tracing the
        ~20 per-T* trees is the expensive part, ~0.9s); since the tree is shared by reference across
        the server's shallow-copied cache entries, the first color-by-line read pays it and repeats
        are free — and it stays OFF the /compute + hover path (the map color uses ``commute()``)."""
        if self._dom is not None:
            return self._dom
        if commute is None:
            commute = self._select_arrays()[5]
        dom = [None] * self.n_cells
        for ci in range(self.n_cells):
            if commute[ci] < 0:
                continue
            tr = self._trace_raw(ci)
            if tr is None:
                continue
            legs_raw, _lh = tr
            dom[ci] = self._dominant(legs_raw)
        self._dom = dom
        return dom

    # -- public: full itinerary (hover) ---------------------------------------------------
    def itinerary(self, ci, geom_provider=None):
        """Breakdown dict ({"total","xfers","legs"[,"geom"]}) for cell ``ci``, or None if
        unreachable. ``total`` == the painted depart-after minute; the legs reconcile to it (the
        same reconciliation the arrive-by path uses). Geometry, when a ``geom_provider`` is given,
        is traced from the SAME T*-deadline tree as the breakdown (hover==map extends to drawing).

        Note the leg WALK seconds (access/egress/transfer) already carry the engine's walk-speed
        scaling because the access/egress/pure-walk arrays passed in are walk-scalar-scaled by the
        caller (``RaptorEngine``); the schedule legs are exact GTFS times."""
        tr = self._trace_raw(ci)
        if tr is None:
            return None
        _s, _aw, _D, _T, _w, painted = self._select_arrays()
        total_min = int(painted[ci])
        if total_min < 0 or total_min >= self.max_min:
            return None
        legs_raw, latest_home = tr
        # Walk-only cells have no T*-tree; clock + format via a tree built at the most convenient
        # deadline would be wasteful — fold them directly (a single walk leg reconciles trivially).
        if len(legs_raw) == 1 and legs_raw[0][0] == "walk":
            out = [{"mode": "walk", "line": None, "sec": int(legs_raw[0][1])}]
            if geom_provider is not None:
                out[0]["segs"] = [("purewalk",)]
            res = reconcile_legs(
                [{"mode": "walk", "line": None,
                  "min": int(round(out[0]["sec"] / 60.0)),
                  **({"segs": out[0]["segs"]} if geom_provider is not None else {})}],
                total_min)
            if geom_provider is not None:
                res["geom"] = self._walk_geom(ci, res["legs"], geom_provider)
                for l in res["legs"]:
                    l.pop("segs", None)
            return res
        jt = self._tree_at(int(_T[ci]))
        out, _total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None)
        res = jt._format(out, total_min)        # total anchored on the painted minute
        if geom_provider is not None:
            res["geom"] = jt._geometry(ci, res["legs"], geom_provider, s_star=int(_s[ci]))
            for l in res["legs"]:
                l.pop("segs", None)
        return res

    def _walk_geom(self, ci, legs, provider):
        geom = []
        for l in legs:
            pts, approx = [], False
            for sg in l.get("segs", ()):
                if sg[0] == "purewalk":
                    pts, approx = provider.purewalk(ci)
            g = {"mode": l["mode"], "name": l.get("line"), "min": l["min"], "pts": pts}
            if approx:
                g["approx"] = True
            geom.append(g)
        return geom

    def _dominant(self, legs_raw):
        rides = [l for l in legs_raw if l[0] == "ride" and (l[3] - l[2]) >= _TINY_HOP_MIN * 60]
        if not rides:
            return "walk only"
        best = None; best_key = None
        for r in rides:
            pi, dep_sec, arr_sec = r[1], r[2], r[3]
            feed, rid, name, _mode = self.line_table[int(self.pat_line[pi])]
            key = (-(arr_sec - dep_sec), name, feed, rid)
            if best_key is None or key < best_key:
                best_key = key; best = name
        return best

    # -- committed-plan MC support (Stage 3 of the depart-after map migration) -------------
    # The committed-plan Monte-Carlo (raptor_engine.montecarlo / route_typicals) is semantic-
    # agnostic at the KERNEL: it takes each cell's COMMITTED first leg (home departure + first
    # board, fixed from the unperturbed plan) and re-optimizes the tail from the actual late
    # arrival via a per-draw reverse profile. Under depart-after the home departure is ALREADY
    # committed — it's the representative window departure D* the painted p50 anchors on — so the
    # committed model maps onto depart-after directly: we just extract the committed first leg from
    # THIS depart-after tree's per-cell traced journey (the SAME journey the map/hover/color-by-line
    # show) instead of the arrive-by tree's. The extraction rule (``JourneyTree._fill_committed_leg``)
    # and the kernel are reused verbatim, so the depart-after realistic/fragility/per-route typical
    # are scored by the identical model as arrive-by and stay metric-consistent with the headline
    # (committed p50 floored at the cell's depart-after p5 best-case).

    def _select(self, ci):
        """Scalar access-stop lookup for cell ``ci`` mirroring ``JourneyTree._select``'s tuple
        contract: ``(s_star, aw, latest_home=D*, is_walk)``. Reads the painted depart-after
        selection (the kernel-chosen s*, D*, T*), so the committed-MC + per-route-typical paths
        (``server._itinerary_alt_typicals`` calls ``tree._select(ci)``) score the SAME journey the
        map paints. Walk-only / unreachable cells return ``(-1, purewalk, D*, True)``."""
        s_star, aw_sel, Dstar, _Tstar, is_walk, painted = self._select_arrays()
        if painted[ci] < 0 or is_walk[ci]:
            pw = int(self.purewalk[ci]) if self.purewalk[ci] >= 0 else -1
            return -1, pw, int(Dstar[ci]), True
        return int(s_star[ci]), int(aw_sel[ci]), int(Dstar[ci]), False

    def committed_first_legs(self):
        """Per-cell committed plan extracted from THIS depart-after tree's traced journeys, in the
        SAME cell-aligned array layout ``JourneyTree.committed_first_legs`` produces (so the committed
        kernel + ``raptor_engine.montecarlo`` consume it unchanged). Each cell's plan is read off the
        journey anchored on its painted (s*, D*, T*): the committed home departure is D* (the
        representative window departure), and the first significant board is found by the SAME
        ``_fill_committed_leg`` rule the arrive-by path uses. So the depart-after committed MC scores
        the displayed depart-after journey, by construction."""
        out = JourneyTree._empty_committed(self.n_cells)
        for ci in range(self.n_cells):
            tr = self._trace_raw(ci)
            if tr is None:
                continue                                 # unreachable (kind stays 0)
            JourneyTree._fill_committed_leg(out, ci, tr)
        return out

    def committed_legs_via_stops(self, ci, stops):
        """Per-ROUTE committed first legs for ONE cell ``ci``, one row per access stop in ``stops``
        (the depart-after analog of ``JourneyTree.committed_legs_via_stops``). Each route is traced
        from its access stop on the cell's representative T*-tree (the journey ``alt_lines_window``
        surfaced), departing at the depart-after committed home D*, and its committed first leg is
        extracted by the SAME ``_fill_committed_leg`` rule the primary uses — so each alt route's
        per-route typical is scored by the identical committed model as the primary, keeping the
        compare-list metric-consistent under depart-after.

        Returns the committed-leg dict (``_empty_committed`` arrays, len == len(stops)); an off-cell
        / unreachable stop keeps kind 0 (the kernel takes it as the cap)."""
        out = JourneyTree._empty_committed(len(stops))
        s_star, aw_sel, Dstar, Tstar, is_walk, painted = self._select_arrays()
        if painted[ci] < 0 or is_walk[ci]:
            return out                                   # walk-only / unreachable cell -> all kind 0
        T = int(Tstar[ci]); Dstar_ci = int(Dstar[ci])
        jt = self._tree_at(T)
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        a0, a1 = int(off[ci]), int(off[ci + 1])
        for row, s in enumerate(stops):
            s = int(s)
            if s < 0:
                continue                                 # off-cell -> kind 0 (cap)
            awk = None
            for k in range(a0, a1):
                if int(to[k]) == s:
                    awk = int(self.access_w[k]); break
            if awk is None:
                continue
            # The committed home departure for an alt is the SAME representative D* (you leave home
            # at the painted window departure); the alt route is the journey traced from s on the
            # cell's T*-tree. This mirrors the primary (D* + the s*-traced journey), so all routes
            # of the cell share one committed departure and are directly comparable.
            tr = jt._trace_from(s, awk, Dstar_ci)
            if tr is None:
                continue
            JourneyTree._fill_committed_leg(out, row, tr)
        return out

    def _stop_percentile_anchor(self, ci, s, awk, percentile):
        """The PER-STOP depart-after percentile anchor for a FIXED access stop ``s`` of cell ``ci``:
        over the [DEP, DEP+WINDOW] cell departures D, the door-to-door time riding ONLY this stop's
        route is ``tt_s(D) = arrivalW[s, D+awk] - D`` (ceil to minutes, capped). Returns
        ``(painted_min, Dstar, Tstar)`` where ``painted_min`` is the ``method='lower'`` percentile of
        that per-stop distribution, ``Dstar`` the LATEST departure achieving it, and ``Tstar`` the
        stop's TRUE workplace arrival at D* (== arrivalW[s, D*+awk]) — the SAME monotone-percentile
        construction ``select_departafter`` uses for the PRIMARY, restricted to one stop (no cross-stop
        min, no walk prior; a fixed route has neither). Anchoring an alt's p5/p50 on its OWN per-stop
        percentile guarantees ``alt p5 <= alt p50`` (the percentile is monotone in ``percentile`` over
        the same distribution), unlike the latest-departure best-case which mixes the CELL's p5/p50
        deadline trees. Returns ``(None, None, None)`` if the stop never reaches W within the cap."""
        dep_grid = self.dep_grid; cell_deps = self.cell_deps
        ndg = len(dep_grid)
        row = self.arrivalW[int(s)]                      # arrivalW[s, k] over the departure grid
        kk = np.searchsorted(dep_grid, cell_deps + int(awk), side="left")
        ttm = np.full(len(cell_deps), self.max_min, np.float64)   # unreachable draw -> cap
        ok = kk < ndg
        if ok.any():
            arr = np.full(len(cell_deps), _INF, np.int64)
            arr[ok] = row[kk[ok]]
            fin = arr < _INF
            tt = (arr[fin] - cell_deps[fin]).astype(np.float64) / 60.0
            ttm[fin] = np.ceil(np.minimum(tt, self.max_min))
        v = int(np.percentile(ttm, float(percentile), method="lower"))
        if v >= self.max_min:
            return None, None, None
        Dstar = None
        for di in range(len(cell_deps) - 1, -1, -1):     # LATEST departure achieving the percentile
            if int(ttm[di]) == v:
                Dstar = int(cell_deps[di]); break
        if Dstar is None:
            return None, None, None
        kd = int(np.searchsorted(dep_grid, Dstar + int(awk), side="left"))
        if kd >= ndg:
            return None, None, None
        Tstar = int(row[kd])
        if Tstar >= _INF:
            return None, None, None
        return v, Dstar, Tstar

    def itinerary_via_stop(self, ci, s_star, geom_provider=None, percentile=None):
        """Like ``itinerary`` but for an EXPLICIT access stop (an ALTERNATIVE route surfaced by
        ``alt_lines_window``), the depart-after analog of ``JourneyTree.itinerary_via_stop``: trace
        the alt's representative T*-tree from ``s_star`` to W via the SAME machinery the primary route
        uses, so the alt's breakdown/geometry are byte-faithful to a displayed depart-after journey.
        ``s_star`` must be one of cell ``ci``'s access stops. Returns the same dict ``itinerary``
        returns, or None.

        ``percentile`` (None | 5 | 50 | ...): the depart-after metric the alt journey represents.
          - None (legacy): the alt's BEST-CASE on the CELL's representative T*-tree, anchored on the
            T*-tree's LATEST departure from ``s_star`` (``best[s] - access_walk``) — the arrive-by alt
            contract + the alt-window's ``mk[0]``. Kept for callers/tests that want the chip number.
          - a percentile: the alt's OWN per-stop percentile journey (``_stop_percentile_anchor``), so
            ``itinerary_via_stop(ci,s,percentile=5).total <= ...percentile=50).total`` holds PER ALT
            (each alt's p5 <= its p50), and ``total`` == the alt's per-stop p5/p50 minute exactly.
            This is what the depart-after compare card uses for each alt's best-case/typical strips."""
        _s_arr, _aw_arr, D_arr, T_arr, is_walk_arr, painted_arr = self._select_arrays()
        if painted_arr[ci] < 0 or is_walk_arr[ci]:
            return None                                  # walk-only / unreachable cell -> no transit alt
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        a0, a1 = int(off[ci]), int(off[ci + 1])
        s = int(s_star)
        awk = None
        for k in range(a0, a1):
            if int(to[k]) == s:
                awk = int(self.access_w[k]); break
        if awk is None:
            return None
        if percentile is None:
            jt = self._tree_at(int(T_arr[ci]))
            latest_home = int(jt.best[s]) - awk          # T*-tree latest departure from s (== window)
            tr = jt._trace_from(s, awk, latest_home)
            if tr is None:
                return None
            legs_raw, latest_home = tr
            out, total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None)
            total_min = int(np.ceil(total_sec / 60.0))
        else:
            painted_min, Dstar, Tstar = self._stop_percentile_anchor(ci, s, awk, percentile)
            if painted_min is None:
                return None
            jt = self._tree_at(int(Tstar))               # tree at THIS alt's representative deadline
            tr = jt._trace_from(s, awk, int(Dstar))
            if tr is None:
                return None
            legs_raw, latest_home = tr
            out, _total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None)
            total_min = int(painted_min)                 # total anchored on the per-stop percentile
        if total_min >= self.max_min:
            return None
        res = jt._format(out, total_min)
        if geom_provider is not None:
            res["geom"] = jt._geometry(ci, res["legs"], geom_provider, s_star=s)
            for l in res["legs"]:
                l.pop("segs", None)
        return res

    def alt_lines_window(self, perfect, window_min):
        """Per-cell ALTERNATIVE lines as a deterministic dominance window, the depart-after analog of
        ``JourneyTree.alt_lines_window``. Each reachable transit cell resolves to a representative
        arrival deadline T* (only ~15-21 distinct per workplace), so we group cells by T*, run that
        T*-tree's ``alt_lines_window`` over the grid ONCE (reusing the arrive-by implementation
        verbatim), and keep each cell's row from ITS T*-tree. ``perfect`` is the cell's depart-after
        best-case (p5) floor, so the window is measured against the same baseline the headline uses.

        Returns list[dict|None]: per cell ``{line_name: (min_minutes, access_stop_gid)}`` sorted
        closest-first (the server drops the PRIMARY line + caps at 4), or None where no transit line
        is within the window. Walk-only / unreachable cells -> None."""
        perfect = np.asarray(perfect, np.int64)
        s_star, aw_sel, Dstar, Tstar, is_walk, painted = self._select_arrays()
        out = [None] * self.n_cells
        # distinct representative deadlines among reachable TRANSIT cells (walk-only/unreach skip)
        by_T = {}
        for ci in range(self.n_cells):
            if painted[ci] < 0 or is_walk[ci] or int(s_star[ci]) < 0:
                continue
            by_T.setdefault(int(Tstar[ci]), []).append(ci)
        for T, cis in by_T.items():
            jt = self._tree_at(T)
            win = jt.alt_lines_window(perfect, window_min, cells=cis)   # ONLY this T*'s cells
            for ci in cis:
                out[ci] = win[ci]
        return out
