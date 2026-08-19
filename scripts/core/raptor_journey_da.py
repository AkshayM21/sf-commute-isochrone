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
import math
import os
import threading
from collections import OrderedDict

import numpy as np

from .raptor import INF as _INF
from .raptor_journey import JourneyTree, reconcile_legs, _TINY_HOP_MIN, EGRESS_INF


_RAW_TAIL_UNSET = object()
_CACHE_MISSING = object()
_PLANNED_CACHE_LOCK_INIT = threading.Lock()
_PLANNED_TEMPLATE_MAX_RIDES = 8


class _CompactPlannedStopTemplate:
    """Stop-global overlay summary whose raw tuple chain is materialized only on demand.

    The citywide alternative overlay needs only ``B``, ``T``, display label and the exact
    structural route key.  Pin/itinerary consumers additionally need raw legs.  Keeping that
    distinction explicit prevents the overlay from allocating one Python tuple list per candidate
    stop while preserving the established scalar template at the boundary that truly needs it.
    """
    __slots__ = ("stop", "board", "deadline", "tail_sec", "label", "route_key",
                 "_raw_tail")

    def __init__(self, stop, board, deadline, tail_sec, label, route_key):
        self.stop = int(stop)
        self.board = int(board)
        self.deadline = int(deadline)
        self.tail_sec = int(tail_sec)
        self.label = label
        self.route_key = route_key
        self._raw_tail = _RAW_TAIL_UNSET

    def materialize_raw_tail(self, owner):
        if self._raw_tail is _RAW_TAIL_UNSET:
            jt = owner._tree_at(self.deadline)
            traced = jt._trace_from(self.stop, 0, self.board)
            raw_tail = None
            if traced is not None:
                legs_raw, _raw_home = traced
                # The compiled primitive already established this.  Rechecking only at the rare
                # materialization boundary protects exactness if a mutable/synthetic fixture
                # changed the parent arrays between overlay extraction and pin inspection.
                if owner._raw_chain_valid_after_start(legs_raw, self.board):
                    candidate = tuple(legs_raw[1:])
                    # The compact overlay stored both the structural route key and its public
                    # label.  Boardability alone is not enough at this lazy boundary: a synthetic
                    # fixture, future mutable parent table, or uncovered adapter drift could leave
                    # a still-boardable chain whose route no longer matches that summary.  Reject
                    # that split-brain payload so the caller re-enters the established scalar
                    # oracle and keeps route identity, label, and raw itinerary lockstep.
                    try:
                        summary_matches = (
                            owner._planned_route_identity(candidate) == self.route_key
                            and owner._planned_route_label(candidate) == self.label
                        )
                    except Exception:
                        summary_matches = False
                    if summary_matches:
                        raw_tail = candidate
            self._raw_tail = raw_tail
        return self._raw_tail


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
                 walk_reluctance=1.0, walk_prior_eps=60.0, max_rounds=8, board_slack=60,
                 planned=False):
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
        self.planned = bool(planned)
        self.n_cells = len(self.access_off) - 1
        self.line_table = data["line_table"]
        self.pat_line = data["pat_line"]
        self._sel = None             # lazy per-cell (s_star, aw, Dstar, Tstar, is_walk, painted_min)
        self._planned_target_sec = None  # exact raw primary seconds, cell-aligned (planned only)
        # A compact deadline tree is ~0.3-0.4 MiB. Planned branch enumeration can touch dozens of
        # distinct T* values, so retain a generous bounded working set; eviction is accuracy-neutral
        # because a missing tree is rebuilt by the exact same deterministic kernel.
        self._trees = OrderedDict()  # T* (int sec) -> JourneyTree at deadline T*
        self._tree_cache_max = 128
        self._trees_lock = threading.Lock()
        self._tree_flights = {}      # T* -> {event, result, error}; waiters survive LRU eviction
        self._dom = None             # cached full-grid dominant list (color-by-line; ~20 traced trees)
        self._planned_board_idx = None
        self._planned_stop_anchor = None
        self._planned_validated_stop = {}
        # Compact overlay extraction and lazy pin materialization share this cache across request
        # threads.  An RLock keeps a compact holder -> scalar normalization atomic and permits the
        # rare validation fallback to re-enter the scalar helper without a second lock order.
        self._planned_validated_stop_lock = threading.RLock()
        self._planned_pattern_identity_cache = {}
        self._planned_structural_rank_cache = None
        self._wait_clamped = set()   # cells whose planned trace needed the last-resort wait clamp (B3)

    # -- per-cell painted selection: (s*, D*, T*) lockstep with assemble_departafter ------
    def _select_arrays(self):
        """Per-cell representative departure D*, its penalized access stop s*, arrival deadline T*,
        and the painted minute. Delegates to the SINGLE-SOURCE selection kernel
        ``raptor.select_departafter`` (the numba njit fast path on the engine grids; python
        reference otherwise) — the SAME per-departure penalized eps-window pick + ``method='lower'``
        percentile + LATEST-departure anchor that ``raptor.assemble_departafter`` runs for the VALUE,
        only here it ADDITIONALLY emits the selection it otherwise discards. Planned rows are then
        raw-boardability-validated before publication; a stale profile anchor is atomically replaced
        by the best truthful candidate. Cell-aligned arrays
        (s_star int64[-1 walk/unreach], aw int64, Dstar int64, Tstar int64[-1 walk/unreach],
        is_walk bool, painted_min int32[-1 unreachable])."""
        if self._sel is not None:
            return self._sel
        from . import raptor as R
        if self.planned:
            painted, s_star, aw_sel, Dstar, Tstar, is_walk = R.select_planned_departafter(
                self.access_off, self.access_to, self.access_w, self.purewalk, self.arrivalW,
                self.dep_grid, self.cell_deps, self.max_min)
        else:
            painted, s_star, aw_sel, Dstar, Tstar, is_walk = R.select_departafter(
                self.access_off, self.access_to, self.access_w, self.purewalk, self.arrivalW,
                self.dep_grid, self.cell_deps, self.max_min, percentile=self.percentile,
                beta=self.beta, eps=self.eps)
        s_star = np.asarray(s_star, np.int64)
        aw_sel = np.asarray(aw_sel, np.int64)
        Dstar = np.asarray(Dstar, np.int64)
        Tstar = np.asarray(Tstar, np.int64)
        is_walk = np.asarray(is_walk, bool)
        painted = np.asarray(painted, np.int32)
        if self.planned:
            (s_star, aw_sel, Dstar, Tstar, is_walk, painted,
             self._planned_target_sec) = self._reselect_planned_primaries(
                s_star, aw_sel, Dstar, Tstar, is_walk, painted)
        self._sel = (s_star, aw_sel, Dstar, Tstar, is_walk, painted)
        return self._sel

    def _reselect_planned_primaries(self, s_star, aw_sel, Dstar, Tstar, is_walk, painted):
        """Replace stale planned primary anchors before their map minutes are published.

        ``arrivalW`` is a monotone profile summary, while a single T*-deadline parent chain is
        concrete.  A selected profile anchor can therefore point at a chain whose first board has
        already departed.  Correct this at the selection boundary, not in the formatter: retain a
        raw-boardable candidate with the smallest exact raw-clock metric, and atomically publish
        its stop/access/home/deadline/ceil-minute tuple.  No candidate means unreachable.  That
        keeps the map, hover, committed plan, and dominant-line trace on one truthful primary.
        """
        s_star = s_star.copy(); aw_sel = aw_sel.copy(); Dstar = Dstar.copy()
        Tstar = Tstar.copy(); is_walk = is_walk.copy(); painted = painted.copy()
        target_sec = np.full(self.n_cells, -1, np.int64)
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        aw = np.asarray(self.access_w, np.int64)
        board_grid = np.asarray(self.cell_deps, np.int64)

        # Resolve each accessible stop exactly once.  This is the same validated template universe
        # used by via-stop alternatives, including the tree-local ``best[s]`` re-anchor.  Compact
        # production templates carry raw tail seconds, so the citywide comparison remains one
        # compiled trace per unique stop plus a cheap pass over access pairs.
        unique_stops = np.unique(to[(to >= 0)])
        templates = self._planned_primary_templates(unique_stops)

        for ci in range(self.n_cells):
            best = None
            best_kind = None
            a0, a1 = int(off[ci]), int(off[ci + 1])
            for k in range(a0, a1):
                s = int(to[k]); access_sec = int(aw[k])
                if s < 0:
                    continue
                template = templates.get(s)
                if template is None:
                    continue
                board, deadline, _label, _route_key = self._planned_template_summary(template)
                raw_sec = self._planned_template_tail_sec(template) + access_sec
                if raw_sec < 0 or int(np.ceil(raw_sec / 60.0)) >= self.max_min:
                    continue
                home = int(board) - access_sec
                cand = (int(raw_sec), access_sec, -int(board), s, home, int(deadline))
                if best is None or cand[:4] < best[:4]:
                    best = cand; best_kind = "transit"
            walk_sec = int(self.purewalk[ci])
            if walk_sec >= 0 and int(np.ceil(walk_sec / 60.0)) < self.max_min:
                # A corrected transit candidate can be slower than the direct walk the provisional
                # profile row had hidden.  Rank it by the same exact-minute primary objective;
                # the negative stop sentinel gives deterministic walk preference on an exact tie.
                walk = (walk_sec, 0, 0, -1)
                if best is None or walk < best[:4]:
                    best = walk; best_kind = "walk"
            if best is None:
                s_star[ci] = -1; aw_sel[ci] = 0; Dstar[ci] = -1; Tstar[ci] = -1
                is_walk[ci] = False; painted[ci] = -1
                target_sec[ci] = -1
                self._wait_clamped.discard(ci)
                continue
            if best_kind == "walk":
                walk_home = (int(board_grid[-1]) if len(board_grid) else 0) - int(walk_sec)
                s_star[ci] = -1; aw_sel[ci] = 0; Dstar[ci] = walk_home; Tstar[ci] = -1
                is_walk[ci] = True; painted[ci] = int(np.ceil(walk_sec / 60.0)); target_sec[ci] = walk_sec
                self._wait_clamped.discard(ci)
                continue
            raw_sec, access_sec, _neg_board, s, home, deadline = best
            s_star[ci] = s; aw_sel[ci] = access_sec; Dstar[ci] = home; Tstar[ci] = deadline
            is_walk[ci] = False; painted[ci] = int(np.ceil(raw_sec / 60.0)); target_sec[ci] = raw_sec
            self._wait_clamped.discard(ci)
        return s_star, aw_sel, Dstar, Tstar, is_walk, painted, target_sec

    @staticmethod
    def _fold_first_visible_wait(out):
        """Keep controllable first-stop allowance out of the physical-walk description.

        A rider can leave home later and board the same first vehicle, so the first visible transit
        wait is schedule allowance rather than physical walking. It remains in ``sec`` to preserve
        the reconciled, backwards-compatible integer ``min`` total, while the source leg records
        an exact ``physical_sec`` / ``schedule_allowance_sec`` split. Public formatting converts
        those to decimal minute fields. When a zero-second access walk was dropped by
        ``_push_walk``, create an allowance-only access leg; in geometry mode it retains the
        ``("access",)`` tag so geometry and legs stay 1:1."""
        for i, leg in enumerate(out):
            if leg.get("mode") != "transit":
                continue
            wait = int(leg.get("wait_sec", 0))
            if wait <= 0:
                return
            leg["wait_sec"] = 0
            for j in range(i - 1, -1, -1):
                if out[j].get("mode") == "walk":
                    walk = out[j]
                    physical = int(walk.get("physical_sec", walk.get("sec", 0)))
                    allowance = int(walk.get("schedule_allowance_sec", 0)) + wait
                    walk["physical_sec"] = physical
                    walk["schedule_allowance_sec"] = allowance
                    walk["sec"] = physical + allowance
                    return
            first = {"mode": "walk", "line": None, "sec": wait,
                     "physical_sec": 0, "schedule_allowance_sec": wait}
            if "segs" in leg:                            # geometry mode: mirror the access-walk tag
                first["segs"] = [("access",)]
            out.insert(0, first)
            return

    @staticmethod
    def _attach_walk_truth(res, out, geom=None):
        """Mirror formatted walk truth fields onto the corresponding geometry leg.

        ``JourneyTree._format`` carries planned metadata before ``reconcile_legs`` filters
        zero-rounded walks. Thus the truth stays attached to the leg that actually survives,
        rather than to an ordinal walk position that could now refer to a later egress.

        The count-equal fallback only supports small direct callers that supply preformatted legs
        without those fields. It is deliberately disabled whenever reconciliation dropped a walk.
        """
        display_walks = [leg for leg in (res.get("legs") or ()) if leg.get("mode") == "walk"]
        if (display_walks
                and not any("schedule_allowance_min" in leg for leg in display_walks)):
            source_walks = [leg for leg in out
                            if leg.get("mode") == "walk" and int(leg.get("sec", 0)) > 0]
            if len(source_walks) == len(display_walks):
                for source, leg in zip(source_walks, display_walks):
                    if "schedule_allowance_sec" in source:
                        leg["physical_min"] = int(source.get("physical_sec", 0)) / 60.0
                        leg["schedule_allowance_min"] = int(source["schedule_allowance_sec"]) / 60.0
        for leg_i, leg in enumerate(res.get("legs") or ()):
            if leg.get("mode") != "walk":
                continue
            if "schedule_allowance_min" not in leg:
                continue
            if geom is not None and leg_i < len(geom) and geom[leg_i].get("mode") == "walk":
                geom[leg_i]["physical_min"] = leg["physical_min"]
                geom[leg_i]["schedule_allowance_min"] = leg["schedule_allowance_min"]

    @staticmethod
    def _reconcile_planned_target(out, total_sec, target_sec):
        """Apply an exact planned target's signed seconds residual before formatting.

        The generic ``reconcile_legs`` only knows rounded display components.  That is too late
        for a scheduled target that is longer than the raw trace: it would add the positive
        residual to the final walk and present schedule slack as physical egress.  Keep both
        directions in one place instead.  A negative residual may consume only explicit schedule
        allowance; it MUST NOT shorten physical access/transfer/egress truth.  A positive residual
        belongs to the first access walk as explicit schedule allowance (creating an allowance-only
        access leg when the raw chain has none).  Returns unabsorbed negative seconds, or zero.
        Callers must surface/drop an inconsistent anchor instead of asking ``reconcile_legs`` to
        fabricate a shorter physical walk.
        """
        residual = int(target_sec) - int(total_sec)
        if residual > 0:
            # The first walk before transit is the access leg.  It may already carry folded
            # first-board slack; preserve its physical truth and add only schedule allowance.
            for leg in out:
                if leg.get("mode") != "walk":
                    if leg.get("mode") == "transit":
                        break
                    continue
                physical = int(leg.get("physical_sec", leg.get("sec", 0)))
                allowance = int(leg.get("schedule_allowance_sec", 0)) + residual
                leg["physical_sec"] = physical
                leg["schedule_allowance_sec"] = allowance
                leg["sec"] = physical + allowance
                return 0

            # A zero-second/missing access raw leg is omitted by _clock.  Preserve geometry's
            # source truth exactly as _fold_first_visible_wait does for an allowance-only leg.
            access = {"mode": "walk", "line": None, "sec": residual,
                      "physical_sec": 0, "schedule_allowance_sec": residual}
            if any("segs" in leg for leg in out):
                access["segs"] = [("access",)]
            out.insert(0, access)
            return 0

        excess = -residual
        if excess <= 0:
            return 0
        for leg in out:
            if excess <= 0:
                return 0
            if leg.get("mode") != "walk":
                continue
            if "schedule_allowance_sec" in leg:
                # B3 excess is re-anchor slack. The fold placed that slack in schedule
                # allowance, so remove it without ever reducing genuine walking.
                allowance = int(leg["schedule_allowance_sec"])
                physical = int(leg.get("physical_sec", 0))
                take_allowance = min(excess, allowance)
                allowance -= take_allowance
                excess -= take_allowance
                leg["physical_sec"] = physical
                leg["schedule_allowance_sec"] = allowance
                leg["sec"] = physical + allowance
        return excess

    # -- per-T* traced tree (lazy + cached) ----------------------------------------------
    def _tree_at(self, Tstar):
        """A ``JourneyTree`` for the single reverse-traced tree at arrival deadline T* (cached).
        Built with the SAME ``reverse_raptor_traced`` call the arrive-by path uses (only the
        deadline differs), so ``_trace_from`` / ``_clock`` / ``_dominant`` / the no-overshoot
        alight all behave identically. The access table + walk-prior knobs are passed through so
        any incidental ``_select``-based geometry fallback stays consistent; the depart-after total
        is anchored on T*-D*, not on the tree's own arrive-by selection."""
        T = int(Tstar)
        with self._trees_lock:
            jt = self._trees.get(T)
            if jt is not None:
                self._trees.move_to_end(T)
                return jt
            flight = self._tree_flights.get(T)
            if flight is None:
                flight = {"event": threading.Event(), "result": None, "error": None}
                self._tree_flights[T] = flight
                owner = True
            else:
                owner = False

        if not owner:
            flight["event"].wait()
            if flight["error"] is not None:
                raise flight["error"]
            # Return the owner's strong result directly. A burst of other deadlines may have
            # evicted T from the bounded LRU before this waiter is scheduled; that must not turn a
            # successful same-key build into a spurious KeyError or a duplicate build.
            return flight["result"]

        try:
            from . import raptor as R
            par = R.reverse_raptor_traced_fast(
                self.d, self.egress_g, T - self.egress_w, self.egress_w,
                max_rounds=self.max_rounds, board_slack=self.board_slack)
            jt = JourneyTree(self.d, par, self.access_off, self.access_to, self.access_w,
                             self.purewalk, T, self.max_min,
                             walk_reluctance=self.beta, walk_prior_eps=self.eps,
                             egress_g=self.egress_g, egress_w=self.egress_w)
        except BaseException as exc:
            with self._trees_lock:
                flight["error"] = exc
                self._tree_flights.pop(T, None)
                flight["event"].set()
            raise
        else:
            with self._trees_lock:
                self._trees[T] = jt
                self._trees.move_to_end(T)
                while len(self._trees) > self._tree_cache_max:
                    self._trees.popitem(last=False)
                flight["result"] = jt
                self._tree_flights.pop(T, None)
                flight["event"].set()
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
        tr = jt._trace_from(ss, int(aw_sel[ci]), int(Dstar[ci]))
        if tr is None:
            return None
        return tr

    # -- public: per-cell commute minutes (MAP color) -------------------------------------
    def commute(self):
        """commute int32[n_cells] — the published depart-after map minute, -1 unreachable.

        Planned selections validate raw boardability before publication, so a stale profile row can
        be replaced by a truthful primary (or direct walk/unreachable). Nonplanned selections remain
        the direct kernel result.
        """
        return self._select_arrays()[5].copy()

    # -- public: per-cell commute minutes + dominant line (map + color-by-line) -----------
    def commute_and_dominant(self):
        """(commute int32[n_cells], dominant list[n_cells]) — the depart-after map color +
        color-by-line. ``commute[ci]`` is the truthful published planned minute; ``dominant[ci]``
        the traced journey's dominant line (same ``_dominant``
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
        out, total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None,
                                   fold_tiny=not self.planned)
        if self.planned:
            self._fold_first_visible_wait(out)
            planned_target = self._planned_target_sec
            target_sec = (int(planned_target[ci]) if planned_target is not None
                          and int(planned_target[ci]) >= 0 else int(total_sec))
            total_min = int(np.ceil(target_sec / 60.0))
            if self._reconcile_planned_target(out, total_sec, target_sec):
                return None                      # selection must never publish this stale anchor
        res = jt._format(out, total_min)        # total anchored on the truthful selected minute
        if self.planned:
            self._attach_walk_truth(res, out)
        if self.planned and ci in self._wait_clamped:
            res["wait_clamped"] = True          # B3 last resort: chain boardable only pre-window
        if geom_provider is not None:
            res["geom"] = jt._geometry(ci, res["legs"], geom_provider, s_star=int(_s[ci]))
            if self.planned:
                self._attach_walk_truth(res, out, res["geom"])
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
        if not rides and self.planned:
            rides = [l for l in legs_raw if l[0] == "ride"]
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

    def _planned_dominant_rank(self):
        """Numeric form of ``_dominant``'s route-name tie break for compact batching.

        The compiled extractor cannot compare the feed/name/route-id Python strings in
        ``line_table``.  Convert their existing deterministic ordering into a small integer rank
        once per immutable tree, while giving equal public keys the same rank.  Equal keys have the
        same public dominant label, so preserving the equivalence class (rather than an arbitrary
        pattern id) is the exact observable contract.
        """
        cached = getattr(self, "_planned_dom_rank", None)
        if cached is not None:
            return cached
        pat_line = np.asarray(self.pat_line, np.int64)
        rows = self.line_table
        order = list(range(len(pat_line)))
        order.sort(key=lambda pi: (rows[int(pat_line[pi])][2],
                                   rows[int(pat_line[pi])][0],
                                   rows[int(pat_line[pi])][1]))
        rank = np.empty(len(pat_line), np.int64)
        last_key = None
        current = -1
        for pi in order:
            line = rows[int(pat_line[pi])]
            key = (line[2], line[0], line[1])
            if key != last_key:
                current += 1
                last_key = key
            rank[pi] = current
        self._planned_dom_rank = rank
        return rank

    def _compact_planned_committed(self, out, build_dom):
        """Fill exact planned committed rows from compact parent chains when possible.

        Returns ``(dominant_or_none, fallback_cells)``.  A compact failure is intentionally local:
        synthetic/legacy parent shapes and a malformed chain use the established Python tracer for
        that cell instead of weakening the committed-plan contract.  This path is planned-only;
        legacy depart-after retains its unchanged trace implementation.
        """
        if not self.planned:
            return None, np.arange(self.n_cells, dtype=np.int64)
        try:
            from .raptor import _select_kernel
            if _select_kernel() != "numba":
                return None, np.arange(self.n_cells, dtype=np.int64)
            from .raptor_planned_numba import extract_planned_committed_group
        except (ImportError, AttributeError):
            return None, np.arange(self.n_cells, dtype=np.int64)

        s_star, aw_sel, Dstar, Tstar, is_walk, painted = self._select_arrays()
        # ``[]`` is a successful compact extraction with no requested dominant map; ``None`` is
        # reserved for "fast path unavailable" so the caller can distinguish it from that case.
        dom = [None] * self.n_cells if build_dom else []
        # Pure-walk rows have no parent tree.  Their committed row and dominant label are direct
        # consequences of _trace_raw + _fill_committed_leg, so never send them through Numba.
        walk = (painted >= 0) & np.asarray(is_walk, bool)
        if np.any(walk):
            out["commit_home"][walk] = np.asarray(Dstar, np.int64)[walk]
            out["commit_kind"][walk] = 1
            if build_dom:
                for ci in np.flatnonzero(walk):
                    dom[int(ci)] = "walk only"

        transit = ((painted >= 0) & ~np.asarray(is_walk, bool)
                   & (np.asarray(s_star, np.int64) >= 0))
        unresolved = set(int(ci) for ci in np.flatnonzero((painted >= 0) & ~walk & ~transit))
        if not np.any(transit):
            return dom, np.asarray(sorted(unresolved), np.int64)

        # Each representative deadline owns a distinct immutable compact parent table.  Grouping
        # lets the Numba loop walk it densely, rather than crossing Python 2,999 times through
        # _tree_at + _trace_from.
        all_cells = np.flatnonzero(transit).astype(np.int64, copy=False)
        deadlines, inverse = np.unique(np.asarray(Tstar, np.int64)[all_cells], return_inverse=True)
        required = ("best_node", "nd_kind", "nd_stop", "nd_pat", "nd_trip", "nd_board",
                    "nd_alight", "nd_to", "nd_egress", "nd_next")
        data = self.d
        rank = self._planned_dominant_rank()
        for group_i, deadline in enumerate(deadlines):
            cells = all_cells[inverse == group_i]
            jt = self._tree_at(int(deadline))
            par = getattr(jt, "par", None)
            if par is None or any(name not in par for name in required):
                unresolved.update(int(ci) for ci in cells)
                continue
            try:
                result = extract_planned_committed_group(
                    cells, s_star, aw_sel, Dstar, self.cell_deps,
                    par["best_node"], par["nd_kind"], par["nd_stop"], par["nd_pat"],
                    par["nd_trip"], par["nd_board"], par["nd_alight"], par["nd_to"],
                    par["nd_egress"], par["nd_next"],
                    data["pat_nstops"], data["pat_mat_off"], data["pat_stop_off"],
                    data["pat_stops"], data["pat_dep"], data["pat_arr"],
                    data["tr_off"], data["tr_to"], data["tr_time"], jt.egress_sec,
                    rank, EGRESS_INF)
            except Exception:
                # An unrecognized test fixture dtype/layout is not a production correctness
                # decision.  Preserve the prior behavior for this group.
                unresolved.update(int(ci) for ci in cells)
                continue
            (home, kind, walk0, pi, bpos, apos, alight, dom_pi,
             wait_clamped, status) = result
            for row_i, ci_raw in enumerate(cells):
                ci = int(ci_raw)
                if int(status[row_i]) == 0:
                    unresolved.add(ci)
                    continue
                out["commit_home"][ci] = home[row_i]
                out["commit_kind"][ci] = kind[row_i]
                out["commit_walk0"][ci] = walk0[row_i]
                out["commit_pi"][ci] = pi[row_i]
                out["commit_bpos"][ci] = bpos[row_i]
                out["commit_apos"][ci] = apos[row_i]
                out["commit_as"][ci] = alight[row_i]
                if int(wait_clamped[row_i]):
                    self._wait_clamped.add(ci)
                if build_dom:
                    dpi = int(dom_pi[row_i])
                    dom[ci] = (self.line_table[int(self.pat_line[dpi])][2]
                               if dpi >= 0 else "walk only")
        return dom, np.asarray(sorted(unresolved), np.int64)

    def committed_first_legs(self):
        """Per-cell committed plan extracted from THIS depart-after tree's traced journeys, in the
        SAME cell-aligned array layout ``JourneyTree.committed_first_legs`` produces (so the committed
        kernel + ``raptor_engine.montecarlo`` consume it unchanged). Each cell's plan is read off the
        journey anchored on its painted (s*, D*, T*): the committed home departure is D* (the
        representative window departure), and the first displayed board is found by the SAME
        ``_fill_committed_leg`` rule the arrive-by path uses (including planned tiny hops). So the
        depart-after committed MC scores the displayed depart-after journey, by construction."""
        out = JourneyTree._empty_committed(self.n_cells)
        # /variance needs both the committed first leg and the primary-line exclusion map. They
        # previously traced every cell twice (once here, once in ``dominant``). Derive dominance
        # from the exact same raw chain while it is already hot; ``dominant`` then becomes a cache
        # read. This is only a side-effect cache and does not alter the committed arrays.
        build_dom = self._dom is None
        dom, fallback = self._compact_planned_committed(out, build_dom)
        # ``fallback`` is normally empty on production compact trees.  It remains deliberately
        # narrow: rather than falling back for the whole grid when one sparse/synthetic parent is
        # encountered, trace only that exceptional cell with the pre-existing oracle path.
        if dom is None:
            dom = [None] * self.n_cells if build_dom else None
            fallback = np.arange(self.n_cells, dtype=np.int64)
        for ci_raw in fallback:
            ci = int(ci_raw)
            tr = self._trace_raw(ci)
            if tr is None:
                continue                                 # unreachable (kind stays 0)
            # include_tiny mirrors the planned display's fold_tiny=False (B4): the committed first
            # boarding is always the first ride the breakdown SHOWS. tr's latest_home is the
            # planned D* anchor (B1), so commit_home + walk0 == the displayed first departure.
            JourneyTree._fill_committed_leg(out, ci, tr, include_tiny=self.planned)
            if build_dom:
                dom[ci] = self._dominant(tr[0])
        if build_dom:
            self._dom = dom
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
        if painted[ci] < 0 or (is_walk[ci] and not self.planned):
            return out                                   # walk-only / unreachable cell -> all kind 0
        T = int(Tstar[ci]) if not self.planned else None
        Dstar_ci = int(Dstar[ci])
        jt = self._tree_at(T) if T is not None else None
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
            if self.planned:
                # Guard-validated anchor (B3) — the same anchor the alt's chip/strip use, so the
                # committed plan is extracted from the exact journey the compare card shows. The
                # anchor's D (= B - access_walk) is the committed home (B1); walk0 derives as
                # dep - D, preserving commit_home + walk0 == dep.
                va = self._validated_stop_anchor(ci, s, awk)
                if va is None:
                    continue
                _pm, alt_D, _alt_T, legs_raw, _jt_alt = va
                tr = (legs_raw, int(alt_D))
            else:
                # The committed home departure for an alt is the SAME representative D* (you leave
                # home at the painted window departure); the alt route is traced from s on the
                # cell's T*-tree.
                tr = jt._trace_from(s, awk, Dstar_ci)
                if tr is None:
                    continue
            JourneyTree._fill_committed_leg(out, row, tr, include_tiny=self.planned)
        return out

    def _stop_planned_anchor(self, ci, s, awk):
        """Best scheduled first-boarding anchor for one explicit access stop."""
        tail, best_B, best_T = self._planned_stop_anchors()
        s = int(s)
        if best_B[s] < 0:
            return None, None, None
        best = int(tail[s]) + int(awk)
        pm = int((best + 59) // 60)
        if pm >= self.max_min:
            return None, None, None
        return pm, int(best_B[s]) - int(awk), int(best_T[s])

    def _validated_stop_anchor(self, ci, s, awk):
        """The planned per-stop anchor, GUARD-VALIDATED against the raw T*-tree chain (B3).

        ``_stop_planned_anchor`` reads the (row-wise cummax-monotonized) inverted profile; the raw
        single-deadline tree it anchors on can disagree — the traced chain's first board may depart
        BEFORE the at-stop arrival ``B``, which would let ``_clock``'s wait clamp turn an
        unboardable ride into a zero-wait one. Deterministic policy: validate the traced chain at
        ``B``; if invalid, re-anchor to the tree's OWN latest at-stop time ``best[s]`` when that is
        still inside the boarding window (the chain is boardable from ``best[s]`` by tree
        construction; the anchored minute is recomputed from the SAME ``T`` readout, so the number
        can only grow), else DROP the candidate — alt paths must only surface real, boardable
        journeys (the primary is independently corrected at planned selection publication).

        Returns ``(pm, D, T, legs_raw, jt)`` — the validated anchored minute, home departure
        ``D`` (= boarding minute − access walk), deadline ``T``, the traced raw chain, and its
        ``JourneyTree`` — or None. Shared by ``itinerary_via_stop("planned")``,
        ``committed_legs_via_stops`` and ``_planned_alt_lines_window`` so the chip minute, the
        strip breakdown and the committed-MC plan all come from ONE anchor."""
        choice = self._validated_stop_choice(s, awk)
        if choice is None:
            return None
        pm, D, T, raw_tail, jt, _label, _route_key = choice
        # The cached trace starts at the stop.  Restore this cell's access walk exactly; every
        # downstream raw tuple is immutable/shared and independent of the access cell.
        legs_raw = [("access", int(awk)), *raw_tail]
        return int(pm), int(D), int(T), legs_raw, jt

    def _validated_stop_choice(self, s, awk):
        """Stop-global validated route plus this access pair's exact anchored minute.

        For planned semantics the requested at-stop boarding minute ``B`` and deadline ``T`` are
        independent of the cell's access walk; only home departure (``B - walk``) and total
        (``T - B + walk``) vary.  Cache the raw stop-to-work chain, validation fallback, public
        label, and opaque route identity once per stop instead of retracing/re-hashing them for
        every one of the ~500k cell/stop access pairs in the citywide alternative overlay.
        """
        s = int(s); awk = int(awk)
        template = self._validated_stop_template(s)
        if template is None:
            return None
        template = self._planned_template_with_raw(s, template)
        if template is None:
            return None
        B, T, raw_tail, label, route_key = template
        jt = self._tree_at(T)
        pm = int((self._planned_template_tail_sec(template) + awk + 59) // 60)
        if pm >= self.max_min:
            return None
        return pm, int(B) - awk, int(T), raw_tail, jt, label, route_key

    @staticmethod
    def _planned_template_summary(template):
        """Return the four overlay fields without forcing a compact template's raw chain."""
        if isinstance(template, _CompactPlannedStopTemplate):
            return (template.board, template.deadline, template.label, template.route_key)
        B, T, _raw_tail, label, route_key = template
        return int(B), int(T), label, route_key

    @staticmethod
    def _planned_template_tail_sec(template):
        """Exact elapsed seconds from the validated at-stop board anchor to the workplace."""
        if isinstance(template, _CompactPlannedStopTemplate):
            return int(template.tail_sec)
        B, _T, raw_tail, _label, _route_key = template
        return int(DepartAfterJourneyTree._planned_raw_total_sec(raw_tail, int(B)))

    def _planned_stop_cache_lock(self):
        """Return the planned-template cache lock, including for lightweight ``__new__`` tests."""
        lock = getattr(self, "_planned_validated_stop_lock", None)
        if lock is None:
            with _PLANNED_CACHE_LOCK_INIT:
                lock = getattr(self, "_planned_validated_stop_lock", None)
                if lock is None:
                    lock = threading.RLock()
                    self._planned_validated_stop_lock = lock
        return lock

    def _planned_template_with_raw(self, s, template):
        """Adapt a lazy compact template to the established scalar five-tuple contract."""
        if not isinstance(template, _CompactPlannedStopTemplate):
            return template
        s = int(s)
        cache = self._planned_validated_stop
        lock = self._planned_stop_cache_lock()
        with lock:
            current = cache.get(s, _CACHE_MISSING)
            if current is not _CACHE_MISSING:
                if not isinstance(current, _CompactPlannedStopTemplate):
                    return current
                # A stale caller always follows the cache-owned holder.  Keeping materialization
                # under this narrow cache lock makes holder -> scalar publication one transaction.
                template = current

            raw_tail = template.materialize_raw_tail(self)
            if raw_tail is not None:
                scalar = (template.board, template.deadline, raw_tail,
                          template.label, template.route_key)
                cache[s] = scalar
                return scalar

            # A production parent table is immutable, so this is expected only for synthetic or
            # mutated fixtures.  Replace the still-cache-owned compact row with the scalar oracle's
            # answer atomically; never expose an empty window for a late grouped publication.
            return self._validated_stop_template_from_anchor(
                s, template.board, template.deadline, self._tree_at(template.deadline),
                force_scalar=True)

    def _validated_stop_template_from_anchor(self, s, B, T, jt, force_scalar=False):
        """Build one validated planned template from a pre-resolved anchor/tree.

        ``_planned_alt_lines_window`` sees every candidate stop's immutable profile anchor up
        front.  Its old inner loop asked ``_tree_at(T)`` once per candidate merely to retrieve the
        same bounded-cache entry.  The grouped caller resolves that tree once per ``T`` and hands
        it here.  Crucially this is *not* a B3 shortcut: every stop still traces its own raw chain
        and, when needed, performs the same tree-local ``best[s]`` re-anchor validation.

        Templates retain only ``T`` rather than ``jt`` so this optimization cannot accidentally
        defeat the deadline-tree LRU's memory ceiling.
        """
        s = int(s); B = int(B); T = int(T)
        cache = self._planned_validated_stop
        lock = self._planned_stop_cache_lock()
        with lock:
            current = cache.get(s, _CACHE_MISSING)
            if (current is not _CACHE_MISSING
                    and not (force_scalar
                             and isinstance(current, _CompactPlannedStopTemplate))):
                return current

        template = None
        if B >= 0 and T >= 0:
            tr = jt._trace_from(s, 0, B)
            if tr is not None:
                legs_raw, _raw_home = tr
                if not self._raw_chain_valid_after_start(legs_raw, B):
                    Bs = int(jt.best[s])
                    if (len(self.cell_deps) == 0 or Bs < int(self.cell_deps[0])
                            or not self._raw_chain_valid_after_start(legs_raw, Bs)):
                        legs_raw = None
                    else:
                        B = Bs
                if legs_raw is not None:
                    raw_tail = tuple(legs_raw[1:])
                    label = self._planned_route_label(raw_tail)
                    route_key = self._planned_route_identity(raw_tail)
                    template = (B, T, raw_tail, label, route_key)
        with lock:
            current = cache.get(s, _CACHE_MISSING)
            if (current is _CACHE_MISSING
                    or (force_scalar and isinstance(current, _CompactPlannedStopTemplate))):
                cache[s] = template
                return template
            return current

    def _validated_stop_templates_grouped(self, stops, best_B, best_T):
        """Return candidate templates while resolving each representative deadline once.

        This is deliberately a narrow extraction optimization.  Grouping only shares the
        ``JourneyTree`` for equal ``T``; it neither coalesces stops nor changes their requested
        profile ``B``.  That preserves the B3 rule in :meth:`_validated_stop_template_from_anchor`
        even for stops whose raw parent chains diverge under the same deadline.
        """
        if self._planned_template_mode() == "compact":
            compact = self._validated_stop_templates_grouped_compact(stops, best_B, best_T)
            if compact is not None:
                return compact

        cache = self._planned_validated_stop
        lock = self._planned_stop_cache_lock()
        templates = {}
        by_T = {}
        for raw_s in np.asarray(stops, np.int64):
            s = int(raw_s)
            with lock:
                current = cache.get(s, _CACHE_MISSING)
                if current is not _CACHE_MISSING:
                    templates[s] = current
                    continue
            B = int(best_B[s]); T = int(best_T[s])
            if B < 0 or T < 0:
                with lock:
                    templates[s] = cache.setdefault(s, None)
                continue
            by_T.setdefault(T, []).append((s, B))

        for T, members in by_T.items():
            jt = self._tree_at(T)
            for s, B in members:
                templates[s] = self._validated_stop_template_from_anchor(s, B, T, jt)
        return templates

    @staticmethod
    def _planned_template_mode():
        """Select the planned-overlay extractor with a deliberately safe rollback contract.

        Compact extraction is the production default: it preserves the scalar template's route
        identity/ranking while avoiding raw Python parent-chain materialization for every overlay
        candidate.  Set ``RAPTOR_PLANNED_TEMPLATE_MODE=legacy`` to force the established scalar
        path during an incident or a differential investigation.  An unset environment selects the
        default compact path; an empty or unrecognized *set* value is normalized to the safe legacy
        fallback rather than accidentally enabling an experimental spelling.
        """
        mode = os.environ.get("RAPTOR_PLANNED_TEMPLATE_MODE")
        if mode is None:
            return "compact"
        return "compact" if mode.strip().lower() == "compact" else "legacy"

    def _validated_stop_templates_grouped_compact(self, stops, best_B, best_T):
        """Exact compact template extraction for the citywide planned overlay.

        Returns ``None`` when the compiled primitive is unavailable so the caller can execute the
        unchanged scalar grouping.  Successful rows cache a lazy structural summary; only a later
        itinerary/pin request materializes its raw Python chain.  MALFORMED rows fall back one stop
        at a time, while exact UNREACHABLE/INVALID_BOARD outcomes remain negative templates.
        """
        try:
            from .raptor import _select_kernel
            if _select_kernel() != "numba":
                return None
            from .raptor_planned import (
                PLANNED_TRACE_INVALID_BOARD,
                PLANNED_TRACE_MALFORMED,
                PLANNED_TRACE_OK,
                PLANNED_TRACE_REANCHORED,
                PLANNED_TRACE_UNREACHABLE,
            )
            from .raptor_planned_numba import trace_planned_stop_templates
        except (ImportError, OSError):
            return None

        cache = self._planned_validated_stop
        lock = self._planned_stop_cache_lock()
        templates = {}
        by_T = {}
        for raw_s in np.asarray(stops, np.int64):
            s = int(raw_s)
            with lock:
                current = cache.get(s, _CACHE_MISSING)
                if current is not _CACHE_MISSING:
                    templates[s] = current
                    continue
            B = int(best_B[s]); T = int(best_T[s])
            if B < 0 or T < 0:
                with lock:
                    templates[s] = cache.setdefault(s, None)
                continue
            by_T.setdefault(T, []).append((s, B))

        required_parent = ("best_node", "nd_kind", "nd_stop", "nd_pat", "nd_trip",
                           "nd_board", "nd_alight", "nd_to", "nd_egress", "nd_next")
        required_data = ("pat_nstops", "pat_ntrips", "pat_mat_off", "pat_stop_off",
                         "pat_stops", "pat_dep", "pat_arr", "tr_off", "tr_to", "tr_time")
        d = getattr(self, "d", None)
        if d is None:
            return None
        if any(name not in d for name in required_data):
            return None
        structural_rank = self._planned_structural_pattern_rank()
        window_floor = (int(self.cell_deps[0]) if len(self.cell_deps)
                        else int(np.iinfo(np.int64).max))

        for T, members in by_T.items():
            jt = self._tree_at(T)
            par = getattr(jt, "par", None)
            if par is None or any(name not in par for name in required_parent):
                for s, B in members:
                    templates[s] = self._validated_stop_template_from_anchor(s, B, T, jt)
                continue
            group_stops = np.asarray([s for s, _B in members], np.int64)
            group_boards = np.asarray([B for _s, B in members], np.int64)
            try:
                result = trace_planned_stop_templates(
                    group_stops, group_boards, window_floor, np.asarray(jt.best),
                    par["best_node"], par["nd_kind"], par["nd_stop"], par["nd_pat"],
                    par["nd_trip"], par["nd_board"], par["nd_alight"], par["nd_to"],
                    par["nd_egress"], par["nd_next"],
                    d["pat_nstops"], d["pat_ntrips"], d["pat_mat_off"],
                    d["pat_stop_off"], d["pat_stops"], d["pat_dep"], d["pat_arr"],
                    d["tr_off"], d["tr_to"], d["tr_time"], jt.egress_sec,
                    structural_rank, EGRESS_INF, _PLANNED_TEMPLATE_MAX_RIDES)
                (status, effective_B, _root, tail_sec, _transit, _final_walk, ride_count,
                 ride_pi, _ride_trip, ride_bpos, ride_apos, _representative) = result

                # Build the whole group's compact summaries before mutating the shared cache.  If
                # a future pattern-key shape surprises this adapter, every row cleanly returns to
                # the scalar oracle instead of leaving a half-compact group behind.
                resolved = {}
                fallback = []
                for row, (s, B) in enumerate(members):
                    row_status = int(status[row])
                    if row_status in (int(PLANNED_TRACE_OK), int(PLANNED_TRACE_REANCHORED)):
                        rides = int(ride_count[row])
                        route_key = tuple(
                            (self._planned_pattern_identity(int(ride_pi[row, k])),
                             int(ride_bpos[row, k]), int(ride_apos[row, k]))
                            for k in range(rides))
                        names = [self.line_table[int(self.pat_line[int(ride_pi[row, k])])][2]
                                 for k in range(rides)]
                        label = " > ".join(names) if names else "walk only"
                        resolved[s] = _CompactPlannedStopTemplate(
                            s, int(effective_B[row]), T, int(tail_sec[row]), label, route_key)
                    elif row_status in (int(PLANNED_TRACE_UNREACHABLE),
                                        int(PLANNED_TRACE_INVALID_BOARD)):
                        resolved[s] = None
                    elif row_status == int(PLANNED_TRACE_MALFORMED):
                        fallback.append((s, B))
                    else:
                        fallback.append((s, B))
            except Exception:
                for s, B in members:
                    templates[s] = self._validated_stop_template_from_anchor(s, B, T, jt)
                continue

            # Preserve a scalar (or negative result) that a concurrent lazy consumer published
            # while this group was being extracted.  A late bulk write must never regress the
            # shared row from its normalized five-tuple back to a compact holder.
            with lock:
                for s, candidate in resolved.items():
                    templates[s] = cache.setdefault(s, candidate)
            for s, B in fallback:
                templates[s] = self._validated_stop_template_from_anchor(s, B, T, jt)
        return templates

    def _planned_primary_templates(self, stops):
        """Exact best validated template per stop for planned primary publication.

        A minute profile can have several board anchors with the same minimum ``T-B``.  The old
        latest-board tie break was harmless while publication used that rounded profile value, but
        it is not authoritative once cards expose raw clock seconds: an earlier tied anchor can
        have less deadline slack and therefore a faster real journey.  Evaluate every tied profile
        minimum with the existing compact tracer and retain the smallest raw tail.  Access walking
        remains an additive per-cell term, so this stop-global result is the complete candidate
        universe shared by the primary and via-stop alternatives.

        The established grouped scalar path remains the safe fallback for synthetic fixtures,
        incidents that disable compact templates, or malformed parent/data arrays.
        """
        stops = np.unique(np.asarray(stops, np.int64))
        profile_tail, best_B, best_T = self._planned_stop_anchors()
        fallback = lambda: self._validated_stop_templates_grouped(stops, best_B, best_T)
        if stops.size == 0 or self._planned_template_mode() != "compact":
            return fallback()
        try:
            from .raptor import _select_kernel
            if _select_kernel() != "numba":
                return fallback()
            from .raptor_planned import (PLANNED_TRACE_OK, PLANNED_TRACE_REANCHORED)
            from .raptor_planned_numba import trace_planned_stop_templates
        except (ImportError, OSError):
            return fallback()

        required_parent = ("best_node", "nd_kind", "nd_stop", "nd_pat", "nd_trip",
                           "nd_board", "nd_alight", "nd_to", "nd_egress", "nd_next")
        required_data = ("pat_nstops", "pat_ntrips", "pat_mat_off", "pat_stop_off",
                         "pat_stops", "pat_dep", "pat_arr", "tr_off", "tr_to", "tr_time")
        d = getattr(self, "d", None)
        if d is None or any(name not in d for name in required_data):
            return fallback()

        if self._planned_board_idx is None:
            self._planned_board_idx = np.searchsorted(self.dep_grid, self.cell_deps, side="left")
        by_T = {}
        for s in stops:
            s = int(s)
            wanted = int(profile_tail[s])
            if wanted >= _INF:
                continue
            row = self.arrivalW[s]
            for di, col in enumerate(self._planned_board_idx):
                if int(col) >= len(row):
                    continue
                T = int(row[int(col)]); B = int(self.cell_deps[di])
                if T < _INF and T - B == wanted:
                    by_T.setdefault(T, []).append((s, B))

        structural_rank = self._planned_structural_pattern_rank()
        window_floor = (int(self.cell_deps[0]) if len(self.cell_deps)
                        else int(np.iinfo(np.int64).max))
        winners = {}
        try:
            for T, members in by_T.items():
                jt = self._tree_at(T)
                par = getattr(jt, "par", None)
                if par is None or any(name not in par for name in required_parent):
                    return fallback()
                group_stops = np.asarray([s for s, _B in members], np.int64)
                group_boards = np.asarray([B for _s, B in members], np.int64)
                result = trace_planned_stop_templates(
                    group_stops, group_boards, window_floor, np.asarray(jt.best),
                    par["best_node"], par["nd_kind"], par["nd_stop"], par["nd_pat"],
                    par["nd_trip"], par["nd_board"], par["nd_alight"], par["nd_to"],
                    par["nd_egress"], par["nd_next"],
                    d["pat_nstops"], d["pat_ntrips"], d["pat_mat_off"],
                    d["pat_stop_off"], d["pat_stops"], d["pat_dep"], d["pat_arr"],
                    d["tr_off"], d["tr_to"], d["tr_time"], jt.egress_sec,
                    structural_rank, EGRESS_INF, _PLANNED_TEMPLATE_MAX_RIDES)
                (status, effective_B, _root, tail_sec, _transit, _final_walk, ride_count,
                 ride_pi, _ride_trip, ride_bpos, ride_apos, _representative) = result
                for row_i, (s, _requested_B) in enumerate(members):
                    if int(status[row_i]) not in (int(PLANNED_TRACE_OK),
                                                  int(PLANNED_TRACE_REANCHORED)):
                        continue
                    rides = int(ride_count[row_i])
                    route_key = tuple(
                        (self._planned_pattern_identity(int(ride_pi[row_i, k])),
                         int(ride_bpos[row_i, k]), int(ride_apos[row_i, k]))
                        for k in range(rides))
                    names = [self.line_table[int(self.pat_line[int(ride_pi[row_i, k])])][2]
                             for k in range(rides)]
                    label = " > ".join(names) if names else "walk only"
                    candidate = _CompactPlannedStopTemplate(
                        s, int(effective_B[row_i]), T, int(tail_sec[row_i]), label, route_key)
                    rank = (candidate.tail_sec, -candidate.board, candidate.route_key)
                    current = winners.get(int(s))
                    if current is None or rank < current[0]:
                        winners[int(s)] = (rank, candidate)
        except Exception:
            return fallback()

        templates = {int(s): None for s in stops}
        templates.update({s: value[1] for s, value in winners.items()})
        lock = self._planned_stop_cache_lock()
        with lock:
            for s, template in templates.items():
                self._planned_validated_stop[s] = template
        return templates

    def _validated_stop_template(self, s):
        """Validated stop-to-work chain cached independently of any cell/access walk.

        This remains the public scalar/oracle path.  Bulk callers use
        :meth:`_validated_stop_templates_grouped` to avoid repeated deadline-cache lookups while
        executing the exact same per-stop B3 validation routine.
        """
        s = int(s)
        cache = self._planned_validated_stop
        lock = self._planned_stop_cache_lock()
        with lock:
            current = cache.get(s, _CACHE_MISSING)
            if current is not _CACHE_MISSING:
                return current
        _tail, best_B, best_T = self._planned_stop_anchors()
        B = int(best_B[s]); T = int(best_T[s])
        if B < 0 or T < 0:
            with lock:
                return cache.setdefault(s, None)
        return self._validated_stop_template_from_anchor(s, B, T, self._tree_at(T))

    @staticmethod
    def _planned_stop_anchors_oracle(arrivalW, kk, cell_deps):
        """Literal pre-vectorization reference for planned stop-anchor selection.

        This deliberately keeps the prior scalar rule available to narrow differential tests.  It
        is not called by production routing: one citywide alternative overlay used to execute this
        Python loop for every stop.  The important contract is the latest-``B`` tie break and the
        ``(_INF, -1, -1)`` sentinel for an unusable stop.
        """
        arrivalW = np.asarray(arrivalW)
        kk = np.asarray(kk, np.int64)
        cell_deps = np.asarray(cell_deps, np.int64)
        n = arrivalW.shape[0]
        tail = np.full(n, _INF, np.int64)
        best_B = np.full(n, -1, np.int64)
        best_T = np.full(n, -1, np.int64)
        for s in range(n):
            row = arrivalW[s]
            btail = _INF; bB = -1; bT = -1
            for di, k in enumerate(kk):
                if k >= len(row):
                    continue
                T = int(row[k])
                if T >= _INF:
                    continue
                B = int(cell_deps[di])
                cost = T - B
                if cost < btail or (cost == btail and B > bB):
                    btail = cost; bB = B; bT = T
            if bB >= 0:
                tail[s] = int(btail); best_B[s] = int(bB); best_T[s] = int(bT)
        return tail, best_B, best_T

    @staticmethod
    def _planned_stop_anchors_vectorized(arrivalW, kk, cell_deps):
        """Vectorized equivalent of :meth:`_planned_stop_anchors_oracle`.

        Each stop minimizes ``arrivalW[stop, B] - B`` over the window's grid probes.  ``kk`` is
        produced by ``searchsorted`` and is therefore non-negative; probes past the profile are
        ignored.  For an equal tail cost we must take the *latest* boarding minute, exactly as the
        scalar implementation did.  Selecting the first matching probe after that only resolves
        duplicate identical ``B`` values (which necessarily have the same ``T`` for an equal cost).
        """
        arrivalW = np.asarray(arrivalW)
        kk = np.asarray(kk, np.int64)
        cell_deps = np.asarray(cell_deps, np.int64)
        n = arrivalW.shape[0]
        tail = np.full(n, _INF, np.int64)
        best_B = np.full(n, -1, np.int64)
        best_T = np.full(n, -1, np.int64)
        if n == 0 or kk.size == 0 or arrivalW.shape[1] == 0:
            return tail, best_B, best_T

        in_profile = kk < arrivalW.shape[1]
        if not np.any(in_profile):
            return tail, best_B, best_T
        cols = kk[in_profile]
        board = cell_deps[in_profile]
        arrivals = np.asarray(arrivalW[:, cols], np.int64)
        costs = arrivals - board[None, :]

        # The scalar loop starts from ``btail = _INF``.  Thus a finite arrival with a cost greater
        # than that sentinel is not a contender; a cost exactly equal to it is a contender only
        # when its boarding minute improves the initial ``bB = -1``.  Normal schedule inputs have
        # non-negative board minutes and fall in the first branch, but retaining this edge behavior
        # makes this a true replacement rather than a merely equivalent common-case shortcut.
        contender = ((arrivals < _INF)
                     & ((costs < _INF) | ((costs == _INF) & (board[None, :] > -1))))
        if not np.any(contender):
            return tail, best_B, best_T
        masked_costs = np.where(contender, costs, _INF)
        chosen_cost = np.min(masked_costs, axis=1)
        has_choice = np.any(contender, axis=1)
        cost_tie = contender & (costs == chosen_cost[:, None])
        chosen_board = np.max(np.where(cost_tie, board[None, :], -1), axis=1)
        board_tie = cost_tie & (board[None, :] == chosen_board[:, None])
        chosen_probe = np.argmax(board_tie, axis=1)  # earliest duplicate = scalar loop behavior
        rows = np.arange(n)
        usable = has_choice & (chosen_board >= 0)
        if np.any(usable):
            tail[usable] = chosen_cost[usable]
            best_B[usable] = chosen_board[usable]
            best_T[usable] = arrivals[rows[usable], chosen_probe[usable]]
        return tail, best_B, best_T

    def _planned_stop_anchors(self):
        """Per-stop best scheduled tail for the planned first-board objective.

        For a fixed stop, the access walk is an additive constant, so the best first-board minute
        minimizes ``arrivalW[stop, B] - B`` independent of the cell. Cache that once and reuse it
        across every cell/stop access pair during alternative enumeration.  The batched numpy
        implementation removes the former Python stop × window loop while preserving its latest-B
        tie rule and sentinels (covered by the scalar oracle's differential tests).
        """
        if self._planned_stop_anchor is not None:
            return self._planned_stop_anchor
        if self._planned_board_idx is None:
            self._planned_board_idx = np.searchsorted(self.dep_grid, self.cell_deps, side="left")
        self._planned_stop_anchor = self._planned_stop_anchors_vectorized(
            self.arrivalW, self._planned_board_idx, self.cell_deps)
        return self._planned_stop_anchor

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
        if painted_arr[ci] < 0 or (is_walk_arr[ci] and percentile != "planned"):
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
        if percentile == "planned":
            va = self._validated_stop_anchor(ci, s, awk)  # B3 guard + re-anchor/drop
            if va is None:
                return None
            painted_min, Dstar, _Tstar, legs_raw, jt = va
            # latest_home = the validated D (= boarding minute - access walk), NOT the vehicle
            # departure (B1): the genuine pre-board wait stays in the clock and is exposed as
            # schedule allowance by the fold below, so the legs reconcile to the anchored minute with
            # only the sub-minute quantization residual.
            out, total_sec = jt._clock(legs_raw, int(Dstar), segs=geom_provider is not None,
                                       fold_tiny=False)
            self._fold_first_visible_wait(out)
            target_sec = int(total_sec)
            total_min = int(np.ceil(target_sec / 60.0))
        elif percentile is None:
            jt = self._tree_at(int(T_arr[ci]))
            latest_home = int(jt.best[s]) - awk          # T*-tree latest departure from s (== window)
            tr = jt._trace_from(s, awk, latest_home)
            if tr is None:
                return None
            legs_raw, latest_home = tr
            out, total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None)
            total_min = int(np.ceil(total_sec / 60.0))
            target_sec = int(total_sec)
        else:
            painted_min, Dstar, Tstar = self._stop_percentile_anchor(ci, s, awk, percentile)
            if painted_min is None:
                return None
            jt = self._tree_at(int(Tstar))               # tree at THIS alt's representative deadline
            tr = jt._trace_from(s, awk, int(Dstar))
            if tr is None:
                return None
            legs_raw, latest_home = tr
            out, total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None)
            total_min = int(painted_min)                 # total anchored on the per-stop percentile
            target_sec = int(Tstar) - int(Dstar)
        if total_min >= self.max_min:
            return None
        if self.planned:
            if self._reconcile_planned_target(out, total_sec, target_sec):
                return None
        res = jt._format(out, total_min)
        if self.planned:
            self._attach_walk_truth(res, out)
        if geom_provider is not None:
            res["geom"] = jt._geometry(ci, res["legs"], geom_provider, s_star=s)
            if self.planned:
                self._attach_walk_truth(res, out, res["geom"])
            for l in res["legs"]:
                l.pop("segs", None)
        return res

    @staticmethod
    def _raw_chain_valid_after_start(legs_raw, start_at_stop):
        """True iff the traced raw chain can actually be boarded after reaching the access stop.

        ``arrivalW`` is an inverted, row-wise cummax-MONOTONIZED profile readout, so its anchor can
        disagree with the raw single-deadline tree: the traced chain's first board may depart before
        the requested at-stop time, which would let ``_clock``'s wait clamp turn an unboardable ride
        into a zero-wait one. Applied (B3) to EVERY planned trace path: the primary
        (planned selection publication), the per-stop alt anchor (``_validated_stop_anchor``, shared by
        ``itinerary_via_stop("planned")`` / ``committed_legs_via_stops`` /
        ``_planned_alt_lines_window``), and branch enumeration (``_planned_itinerary_from_anchor``),
        each with its own deterministic fallback (see those docstrings). Validity is monotone in
        ``start_at_stop``: lowering the start can only stay valid."""
        t = int(start_at_stop)
        for leg in legs_raw:
            k = leg[0]
            if k == "access":
                continue
            if k in ("walk", "walk_t", "egress"):
                t += int(leg[1])
                continue
            if k == "ride":
                dep = int(leg[2])
                if dep < t:
                    return False
                t = int(leg[3])
        return True

    @staticmethod
    def _geom_route_label(geom):
        names = []
        for g in geom or ():
            if g.get("mode") != "transit":
                continue
            n = g.get("name") or g.get("line")
            if n:
                names.append(str(n))
        return " > ".join(names) if names else "walk only"

    @staticmethod
    def _geom_route_sig(geom):
        return tuple((g.get("mode") or "", str(g.get("name") or g.get("line") or ""),
                      int(g.get("min") or 0), int(g.get("wait") or 0))
                     for g in (geom or ()))

    def _planned_pattern_identity(self, pi):
        """Opaque service-pattern identity that never includes a public display name.

        The RAPTOR bake does not retain GTFS ``direction_id``. Its ordered stop topology is a
        stronger discriminator here: reverse directions and branch variants differ, while FIFO-
        split copies of the same pattern remain equal. Feed + route id keeps equal public labels
        and cross-feed route-id collisions separate; ``pat_line`` is the sparse-data fallback.
        """
        pi = int(pi)
        cache = getattr(self, "_planned_pattern_identity_cache", None)
        if cache is None:
            cache = self._planned_pattern_identity_cache = {}
        cached = cache.get(pi)
        if cached is not None:
            return cached
        li = int(self.pat_line[pi])
        row = self.line_table[li]
        feed = str(row[0] if len(row) > 0 and row[0] is not None else "")
        route_id = row[1] if len(row) > 1 else None
        rid = str(route_id).strip() if route_id is not None else ""
        service = ("route", feed, rid) if rid else ("line", feed, li)

        offsets = self.d.get("pat_stop_off")
        stops = self.d.get("pat_stops")
        if offsets is None or stops is None or pi >= len(offsets):
            identity = service + (("pattern", pi),)
            cache[pi] = identity
            return identity
        base = int(offsets[pi])
        nstops = self.d.get("pat_nstops")
        if nstops is not None and pi < len(nstops):
            end = base + int(nstops[pi])
        elif pi + 1 < len(offsets):
            end = int(offsets[pi + 1])
        else:
            end = len(stops)
        topology = tuple(int(s) for s in stops[base:end])
        identity = service + (("stops", topology),)
        cache[pi] = identity
        return identity

    def _planned_structural_pattern_rank(self):
        """Lexicographic ranks of the exact topology identities consumed by compact tracing.

        The primitive uses this only to choose a representative pattern when ride durations tie;
        route equivalence still uses the full ``_planned_pattern_identity`` tuple occurrence by
        occurrence.  Equal identities receive equal ranks, so FIFO-split copies do not acquire an
        incidental pattern-id preference.
        """
        cached = getattr(self, "_planned_structural_rank_cache", None)
        if cached is not None:
            return cached
        identities = [self._planned_pattern_identity(pi) for pi in range(len(self.pat_line))]
        order = sorted(range(len(identities)), key=lambda pi: identities[pi])
        rank = np.empty(len(identities), np.int64)
        last = None
        current = -1
        for pi in order:
            identity = identities[pi]
            if last is None or identity != last:
                current += 1
                last = identity
            rank[pi] = current
        self._planned_structural_rank_cache = rank
        return rank

    def _planned_route_identity(self, legs_raw):
        """Structural planned-candidate identity, independent of display labels and trip times.

        Ride occurrences are not collapsed, so a same-service reboard stays distinct from a
        one-seat ride. Pattern topology distinguishes directions/branches; ridden positions retain
        genuine loop segments. Excluding times dedupes repeated probes of the same route shape.
        """
        rides = []
        for leg in legs_raw or ():
            if leg[0] != "ride":
                continue
            pi = int(leg[1])
            rides.append((self._planned_pattern_identity(pi), int(leg[4]), int(leg[5])))
        return tuple(rides)

    def _planned_candidate_identity(self, cand):
        key = cand.get("route_key")
        if key is None:
            key = self._planned_route_identity(cand.get("raw") or ())
            cand["route_key"] = key
        return key

    def _planned_itinerary_from_anchor(self, ci, s, awk, B, T, geom_provider=None,
                                       planned_total=None, planned_target_sec=None,
                                       include_raw=False):
        """Trace one planned branch candidate for a fixed access stop and board-time anchor.

        Unlike ``itinerary_via_stop(..., percentile="planned")``, this does not collapse the access
        stop to its single best anchor. It traces the exact chain available at ``B``/``T`` and then
        uses the same planned display rule as the primary: home departs at ``B - access_walk`` (the
        anchored minute's own home, B1) and the genuine pre-board wait is attached to the access
        leg as a separate schedule allowance, so it is not described as physical walking or shown
        as a transit wait chip.
        """
        jt = self._tree_at(int(T))
        latest_home = int(B) - int(awk)
        tr = jt._trace_from(int(s), int(awk), latest_home)
        if tr is None:
            return None
        legs_raw, latest_home = tr
        if not self._raw_chain_valid_after_start(legs_raw, int(B)):
            return None                                  # B3: probe anchors must be boardable
        out, total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None,
                                   fold_tiny=False)
        self._fold_first_visible_wait(out)
        target_sec = int(total_sec)
        total_min = int(np.ceil(target_sec / 60.0))
        if total_min >= self.max_min:
            return None
        if total_min != int(np.ceil(target_sec / 60.0)):
            return None
        if self._reconcile_planned_target(out, total_sec, target_sec):
            return None
        res = jt._format(out, total_min)
        self._attach_walk_truth(res, out)
        if geom_provider is not None:
            res["geom"] = jt._geometry(ci, res["legs"], geom_provider, s_star=int(s))
            self._attach_walk_truth(res, out, res["geom"])
            for l in res["legs"]:
                l.pop("segs", None)
        if include_raw:
            return res, legs_raw, latest_home, jt
        return res

    def _format_planned_raw(self, ci, s, legs_raw, latest_home, jt, geom_provider=None,
                            planned_total=None, planned_target_sec=None):
        # ``latest_home`` is the candidate's anchored home (B - access_walk, B1) as stored by the
        # branch enumeration; clock from it directly so the pre-board wait stays genuine and the
        # fold below exposes it as schedule allowance (no home shift to the vehicle departure).
        out, total_sec = jt._clock(legs_raw, latest_home, segs=geom_provider is not None,
                                   fold_tiny=False)
        self._fold_first_visible_wait(out)
        target_sec = int(total_sec)
        total_min = int(np.ceil(target_sec / 60.0))
        if total_min >= self.max_min:
            return None
        if total_min != int(np.ceil(target_sec / 60.0)):
            return None
        if self._reconcile_planned_target(out, total_sec, target_sec):
            return None
        res = jt._format(out, total_min)
        self._attach_walk_truth(res, out)
        if geom_provider is not None:
            res["geom"] = jt._geometry(ci, res["legs"], geom_provider, s_star=int(s))
            self._attach_walk_truth(res, out, res["geom"])
            for l in res["legs"]:
                l.pop("segs", None)
        return res

    def _planned_one_tail_branches(self, ci, candidates, max_total, geom_provider=None):
        """Force one extra transit tail after one-seat candidates with long egress walks."""
        from .raptor_planned import discover_one_tail_variants
        # A busy corridor can expose thousands of scheduled tail probes but only a few hundred
        # distinct structural ride shapes.  Formatting every probe used to dominate a pinned
        # request even though ``_planned_branch_closure`` immediately deduplicated them by
        # ``route_key``.  Keep the exact same rank here, but defer clock/rounding/geometry work
        # until a shape has won.  Ties on the cheap rank are materialized before comparing the
        # final displayed signature, preserving the old deterministic winner exactly.
        best_by_route = {}

        def materialize(cand):
            if cand.get("it") is not None:
                return True
            it = self._format_planned_raw(
                ci, cand["stop"], cand["raw"], cand["home"], cand["jt"],
                geom_provider=geom_provider, planned_total=cand["total"],
                planned_target_sec=cand.get("target_sec", cand.get("metric_sec")))
            if it is None:
                return False
            cand["it"] = it
            cand["sig"] = self._geom_route_sig(it.get("geom") or it.get("legs") or [])
            return True

        def cheap_rank(cand):
            return self._planned_candidate_cheap_rank(cand)

        scan_seeds = []
        seen_scan_seeds = set()
        for cand in candidates:
            geom = cand["it"].get("geom") or cand["it"].get("legs") or []
            transit = [g for g in geom if g.get("mode") == "transit"]
            if len(transit) != 1:
                continue
            final_walk = next((g for g in reversed(geom) if g.get("mode") == "walk"), None)
            if not final_walk or int(final_walk.get("min") or 0) < 5:
                continue
            raw = cand.get("raw") or []
            first_ride_idx = next((i for i, leg in enumerate(raw) if leg[0] == "ride"), None)
            if first_ride_idx is None:
                continue
            first = raw[first_ride_idx]
            first_pi = int(first[1]); first_dep = int(first[2])
            first_bpos = int(first[4])
            # Every one-seat sibling on the same boarded trip scans the same downstream alights,
            # transfers, and next trips.  Its current walking alight is deliberately replaced by
            # ``xapos`` below, so repeating that scan cannot discover a new tail.  Retain the
            # access prefix/home/JourneyTree in the key because those do affect the formatted
            # result and its rank.
            scan_key = (
                tuple(raw[:first_ride_idx]), first_pi, first_dep, first_bpos,
                int(cand["stop"]), int(cand["home"]), id(cand["jt"]),
            )
            if scan_key in seen_scan_seeds:
                continue
            seen_scan_seeds.add(scan_key)
            scan_seeds.append((cand, raw, first_ride_idx, first_pi, first_dep, first_bpos))

        for cand, raw, first_ride_idx, first_pi, first_dep, first_bpos in scan_seeds:
            # Try transfer tails from downstream stops along the first ride. The walk-finish alight
            # and the best transfer alight are not necessarily the same stop; a useful tail may
            # require staying on the first vehicle past the best walking alight. Schedule scanning
            # is a flat-array compiled kernel; its pure-Python sibling is the permanent oracle.
            jt = cand["jt"]
            variants = discover_one_tail_variants(
                self.d, jt.egress_sec, first_pi, first_dep, first_bpos,
                self.board_slack, EGRESS_INF)
            for row in variants:
                (xapos, alight_stop, arr0, stop, walk_sec, pi, bpos,
                 dep2, arr2, apos, st2, eg) = (int(v) for v in row)
                prefix = list(raw[:first_ride_idx]) + [
                    ("ride", first_pi, first_dep, arr0, first_bpos, xapos, alight_stop)
                ]
                tail = []
                if walk_sec > 0:
                    tail.append(("walk_t", walk_sec, alight_stop, stop))
                tail.append(("ride", pi, dep2, arr2, bpos, apos, st2))
                tail.append(("egress", eg, st2))
                raw_out = prefix + tail
                total = int(np.ceil(
                    self._planned_raw_total_sec(raw_out, cand["home"]) / 60.0))
                if total > max_total or total >= self.max_min:
                    continue
                route_key = self._planned_route_identity(raw_out)
                metric_sec = self._planned_raw_total_sec(raw_out, cand["home"])
                proposal = {
                    "line": self._planned_route_label(raw_out),
                    "total": total,
                    "metric_sec": metric_sec, "target_sec": metric_sec,
                    "board_anchor": int(cand["home"]) + self._raw_access_sec(raw_out),
                    "it": None,
                    "stop": cand["stop"],
                    "sig": None,
                    "raw": raw_out,
                    "route_key": route_key,
                    "home": cand["home"],
                    "jt": jt,
                }
                cur = best_by_route.get(route_key)
                if cur is None or cheap_rank(proposal) < cheap_rank(cur):
                    best_by_route[route_key] = proposal
                elif cheap_rank(proposal) == cheap_rank(cur):
                    if (materialize(proposal) and materialize(cur)
                            and self._planned_candidate_better(proposal, cur)):
                        best_by_route[route_key] = proposal

        out = []
        for cand in best_by_route.values():
            if materialize(cand):
                out.append(cand)
        return out

    def _planned_walk_finish_branches(self, ci, candidates, max_total, geom_provider=None):
        """Force the one-seat walk-finish sibling of multi-leg planned branches.

        Branch enumeration used to be one-way: if a one-seat ride ended in a long walk, we tried to
        add a transit tail. The reverse case is just as important for a commute-family view: if the
        exact schedule's best branch is already ``22 > 19`` (or ``K > 9``), the user still needs to
        see whether staying on the first vehicle and walking from a downstream stop is a meaningful
        sibling. This keeps end-bus choices like ``22`` vs ``22 > 19`` stable across walk speeds.
        """
        d = self.d
        pat_nstops = d["pat_nstops"]; pat_ntrips = d["pat_ntrips"]
        pat_stop_off = d["pat_stop_off"]; pat_mat_off = d["pat_mat_off"]
        pat_stops = d["pat_stops"]; pat_dep = d["pat_dep"]; pat_arr = d["pat_arr"]
        # Just as in ``_planned_one_tail_branches``, a frequent branch corridor can enumerate
        # the same structural one-seat finish from many scheduled probes.  The closure below
        # deduplicates by ``route_key`` anyway, so do the timestamp-only work first and leave
        # clock/rounding/geometry allocation for the structural winner.  Do not use public
        # labels in this gate: equal display text can represent distinct feed/direction shapes.
        best_by_route = {}

        def materialize(cand):
            if cand.get("it") is not None:
                return True
            it = self._format_planned_raw(
                ci, cand["stop"], cand["raw"], cand["home"], cand["jt"],
                geom_provider=geom_provider, planned_total=cand["total"],
                planned_target_sec=cand.get("target_sec", cand.get("metric_sec")))
            if it is None:
                return False
            geom = it.get("geom") or it.get("legs") or []
            label = self._geom_route_label(geom)
            if label == "walk only":
                return False
            cand["it"] = it
            cand["line"] = label
            # ``geom_provider=None`` is the server's branch-enumeration call shape.  Fall back
            # to display legs so a missing geometry payload cannot silently sort as ``()``.
            cand["sig"] = self._geom_route_sig(geom)
            return True

        def cheap_rank(cand):
            return self._planned_candidate_cheap_rank(cand)

        for cand in candidates:
            geom = cand["it"].get("geom") or cand["it"].get("legs") or []
            transit = [g for g in geom if g.get("mode") == "transit"]
            if len(transit) < 2:
                continue
            raw = cand.get("raw") or []
            first_ride_idx = next((i for i, leg in enumerate(raw) if leg[0] == "ride"), None)
            if first_ride_idx is None:
                continue
            first = raw[first_ride_idx]
            first_pi = int(first[1]); first_dep = int(first[2])
            first_bpos = int(first[4])
            ns0 = int(pat_nstops[first_pi])
            mb0 = int(pat_mat_off[first_pi]); sb0 = int(pat_stop_off[first_pi])
            deps0 = pat_dep[mb0 + first_bpos: mb0 + first_bpos + int(pat_ntrips[first_pi]) * ns0: ns0]
            trip0 = int(np.searchsorted(deps0, first_dep, side="left"))
            if trip0 >= len(deps0) or int(deps0[trip0]) != first_dep:
                continue
            jt = cand["jt"]
            for xapos in range(first_bpos + 1, ns0):
                st2 = int(pat_stops[sb0 + xapos])
                eg = int(jt.egress_sec[st2])
                if eg >= EGRESS_INF:
                    continue
                arr0 = int(pat_arr[mb0 + trip0 * ns0 + xapos])
                prefix = list(raw[:first_ride_idx]) + [
                    ("ride", first_pi, first_dep, arr0, first_bpos, xapos, st2),
                    ("egress", int(eg), st2),
                ]
                total = int(np.ceil(
                    self._planned_raw_total_sec(prefix, cand["home"]) / 60.0))
                if total > max_total or total >= self.max_min:
                    continue
                route_key = self._planned_route_identity(prefix)
                metric_sec = self._planned_raw_total_sec(prefix, cand["home"])
                proposal = {
                    "line": None,
                    "total": total,
                    "metric_sec": metric_sec, "target_sec": metric_sec,
                    "board_anchor": int(cand["home"]) + self._raw_access_sec(prefix),
                    "it": None,
                    "stop": cand["stop"],
                    "sig": None,
                    "raw": prefix,
                    "route_key": route_key,
                    "home": cand["home"],
                    "jt": jt,
                }
                cur = best_by_route.get(route_key)
                if cur is None or cheap_rank(proposal) < cheap_rank(cur):
                    best_by_route[route_key] = proposal
                elif cheap_rank(proposal) == cheap_rank(cur):
                    if (materialize(proposal) and materialize(cur)
                            and self._planned_candidate_better(proposal, cur)):
                        best_by_route[route_key] = proposal

        out = []
        for cand in best_by_route.values():
            if materialize(cand):
                out.append(cand)
        return out

    @staticmethod
    def _raw_final_walk_sec(raw):
        if not raw:
            return 1 << 60
        last = raw[-1]
        if last[0] in ("egress", "walk"):
            return int(last[1])
        return 0

    @staticmethod
    def _raw_access_sec(raw):
        """Physical walk seconds before the first transit board in a raw planned chain."""
        total = 0
        for leg in raw or ():
            if leg[0] == "ride":
                break
            if leg[0] in ("access", "walk", "walk_t", "egress"):
                total += max(0, int(leg[1]))
        return total

    @staticmethod
    def _raw_transfer_walk_sec(raw):
        return sum(max(0, int(leg[1])) for leg in raw or () if leg[0] == "walk_t")

    @staticmethod
    def _raw_transfer_count(raw):
        return max(0, sum(1 for leg in raw or () if leg[0] == "ride") - 1)

    @staticmethod
    def _raw_transit_sec(raw):
        total = 0
        for leg in raw or []:
            if leg[0] == "ride":
                total += max(0, int(leg[3]) - int(leg[2]))
        return total

    @staticmethod
    def _planned_raw_total_sec(raw, latest_home):
        """Exact ``_clock`` elapsed seconds without allocating formatted display legs."""
        t = int(latest_home)
        for leg in raw or ():
            if leg[0] in ("access", "walk", "walk_t", "egress"):
                t += int(leg[1])
            elif leg[0] == "ride":
                # Waiting is implicit in the jump from the current clock to the scheduled
                # departure; the arrival timestamp already includes both wait and ride.
                t = int(leg[3])
        return max(0, t - int(latest_home))

    @staticmethod
    def _alight_tail_better(finish, apos, eg, best):
        if best is None:
            return True
        best_finish, best_apos, _best_stop, best_eg = best
        return (int(finish), int(eg), -int(apos)) < (
            int(best_finish), int(best_eg), -int(best_apos))

    @classmethod
    def _planned_candidate_quality(cls, cand):
        """Common quality tuple for planned structural-candidate collapse.

        Every branch producer supplies raw legs and a home anchor, so ranking must not discard a
        longer physical-access variant merely because it tied after minute rounding.  The exact
        metric is first; displayed minutes retain the public metric tie-break; a later board then
        wins equal journeys. Remaining terms prefer less physical access, fewer/shorter transfers,
        and less final walking before the durable structural identity settles true ties.
        """
        raw = cand.get("raw") or ()
        home = int(cand.get("home", 0))
        metric_sec = int(cand.get("metric_sec", cls._planned_raw_total_sec(raw, home)))
        board_anchor = int(cand.get("board_anchor", home + cls._raw_access_sec(raw)))
        return (
            metric_sec,
            int(cand.get("total", int(np.ceil(metric_sec / 60.0)))),
            -board_anchor,
            cls._raw_access_sec(raw),
            cls._raw_transfer_count(raw),
            cls._raw_transfer_walk_sec(raw),
            cls._raw_final_walk_sec(raw),
            cand.get("route_key") or (),
        )

    @classmethod
    def _planned_candidate_cheap_rank(cls, cand):
        """Geometry-free prefix shared by every planned branch producer."""
        return cls._planned_candidate_quality(cand)

    @classmethod
    def _planned_candidate_rank(cls, cand):
        return (
            *cls._planned_candidate_quality(cand),
            cand.get("sig", ()),
        )

    @classmethod
    def _planned_candidate_better(cls, cand, cur):
        return cur is None or cls._planned_candidate_rank(cand) < cls._planned_candidate_rank(cur)

    def _planned_branch_closure(self, ci, candidates, max_total, geom_provider=None):
        """Close planned branch seeds over the two supported finish shapes.

        The helpers intentionally form a finite, canonical pipeline rather than an open-ended
        fixed-point loop:

        1. derive one-seat walk finishes from every multi-leg seed and deduplicate them by route;
        2. derive one additional transit tail from the resulting unique one-seat candidates.

        This ordering is structural, not service-specific.  In particular, a seed whose best
        scheduled chain already has two rides can first reveal its one-seat sibling and then use
        that sibling to discover a different useful tail.  Generating tails before walk siblings
        made that result depend on which shape happened to be present in the initial trace.

        Stopping after the one-tail phase is deliberate: the walk sibling of any generated tail
        was already considered in phase 1, so iterating tail -> walk -> tail would only rediscover
        the same structural ride shapes while multiplying schedule probes.
        """
        closed = {}

        def merge(items):
            for cand in items:
                if not cand.get("line"):
                    continue
                route_key = self._planned_candidate_identity(cand)
                if not route_key:
                    continue
                if self._planned_candidate_better(cand, closed.get(route_key)):
                    closed[route_key] = cand

        merge(candidates)
        merge(self._planned_walk_finish_branches(
            ci, list(closed.values()), max_total, geom_provider=geom_provider))
        merge(self._planned_one_tail_branches(
            ci, list(closed.values()), max_total, geom_provider=geom_provider))
        return closed

    def planned_branch_itineraries(self, ci, base_min, window_min, geom_provider=None,
                                   access_stops=None):
        """Exact current-speed branch alternatives for one pinned planned depart-after cell.

        The full-grid alt window intentionally stays cheap: one best route per access stop. A pinned
        card needs a richer commute-diversity view, so for this single cell we enumerate every
        morning-window anchor at every reachable access stop, trace the exact scheduled chain, then
        keep the best instance of each distinct route sequence within ``base_min + window_min``.
        This surfaces sibling tails such as ``22`` vs ``22 > 19`` when walking faster makes the walk
        tail the best chain from the stop but the transit tail remains a meaningful option.
        """
        if not self.planned:
            return []
        try:
            base = int(base_min)
        except Exception:
            return []
        if base < 0:
            return []
        max_total = base + int(round(window_min))
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        aw = np.asarray(self.access_w, np.int64)
        a0, a1 = int(off[ci]), int(off[ci + 1])
        if a1 <= a0:
            return []
        allowed_stops = ({int(stop) for stop in access_stops}
                         if access_stops is not None else None)
        kk = np.searchsorted(self.dep_grid, self.cell_deps, side="left")
        best = {}

        def materialize(cand):
            if cand.get("it") is not None:
                return True
            it = self._format_planned_raw(
                ci, cand["stop"], cand["raw"], cand["home"], cand["jt"],
                geom_provider=geom_provider, planned_total=cand["total"],
                planned_target_sec=cand.get("target_sec", cand.get("metric_sec")))
            if it is None:
                return False
            geom = it.get("geom") or it.get("legs") or []
            label = self._geom_route_label(geom)
            if label == "walk only":
                return False
            cand["it"] = it
            cand["line"] = label
            cand["sig"] = self._geom_route_sig(geom)
            return True

        def cheap_rank(cand):
            return self._planned_candidate_cheap_rank(cand)

        # Gather first, then trace once per (access stop, representative deadline).  The trace
        # walker's node chain is chosen only by that pair: its ``aw`` argument merely becomes the
        # leading ``("access", aw)`` tuple and ``latest_home`` is returned untouched.  Board B
        # still stays per-probe below — B3 validity can reject a later B while retaining an
        # earlier B on the exact same tail.
        probes = []
        for k in range(a0, a1):
            s = int(to[k])
            if allowed_stops is not None and s not in allowed_stops:
                continue
            awk = int(aw[k])
            row = self.arrivalW[s]
            for di, ridx in enumerate(kk):
                if ridx >= len(row):
                    continue
                T = int(row[ridx])
                if T >= _INF:
                    continue
                B = int(self.cell_deps[di])
                probes.append((s, B, T, awk))

        from .raptor_planned import evaluate_planned_branch_probes

        def trace_at_stop(s, T):
            jt = self._tree_at(int(T))
            # ``_trace_from`` reads neither ``aw`` nor ``latest_home`` while walking the node
            # chain; zero supplies a canonical access-free tail.  Keep its exact tree handle in
            # the payload, rather than reacquiring it after the bounded tree cache may evict T.
            traced = jt._trace_from(int(s), 0, 0)
            if traced is None:
                return None
            legs_raw, _ignored_home = traced
            return tuple(legs_raw[1:]), jt

        def valid_after_start(trace, B):
            raw_tail, _jt = trace
            # ``_raw_chain_valid_after_start`` ignores the leading access tuple, so validating
            # the canonical tail at B is byte-for-byte the old ``[access(awk), *tail]`` check.
            return self._raw_chain_valid_after_start(raw_tail, int(B))

        for s, B, T, awk, trace in evaluate_planned_branch_probes(
                probes, trace_at_stop, valid_after_start):
            raw_tail, jt = trace
            legs_raw = [("access", int(awk)), *raw_tail]
            latest_home = int(B) - int(awk)
            metric_sec = self._planned_raw_total_sec(legs_raw, latest_home)
            planned_total = int(np.ceil(metric_sec / 60.0))
            if planned_total > max_total or planned_total >= self.max_min:
                continue
            route_key = self._planned_route_identity(legs_raw)
            cand = {"line": None, "total": planned_total,
                    "metric_sec": metric_sec,
                    "target_sec": metric_sec,
                    "board_anchor": int(B), "it": None,
                    "stop": s, "sig": None,
                    "raw": legs_raw, "route_key": route_key,
                    "home": latest_home, "jt": jt}
            cur = best.get(route_key)
            if cur is None or cheap_rank(cand) < cheap_rank(cur):
                best[route_key] = cand
            elif cheap_rank(cand) == cheap_rank(cur):
                if (materialize(cand) and materialize(cur)
                        and self._planned_candidate_better(cand, cur)):
                    best[route_key] = cand
        best = {key: cand for key, cand in best.items() if materialize(cand)}
        best = self._planned_branch_closure(ci, best.values(), max_total,
                                            geom_provider=geom_provider)
        return sorted(best.values(), key=lambda x: (x["total"], x["route_key"]))

    def planned_access_stop_closure(self, ci, seed_stops, station_radius_m=300.0):
        """Expand proven chip/primary stops to their structural boarding corridor.

        The citywide dominance window already contributes the best access stop for every proven
        route choice. A nearby platform can carry a sibling branch of that same boarding station,
        so close the seeds over a generic 300m station/corridor radius. Do not expand a service to
        every other walk-accessible stop along its route: those are different boarding corridors,
        multiply schedule probes, and contradict the family model's station/direction identity.
        """
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        a0, a1 = int(off[ci]), int(off[ci + 1])
        accessible = {int(stop) for stop in to[a0:a1]}
        seeds = {int(stop) for stop in (seed_stops or ()) if int(stop) in accessible}
        if not seeds:
            return set()

        lat = self.d.get("stop_lat"); lon = self.d.get("stop_lon")
        nearby = set(seeds)
        if lat is not None and lon is not None:
            for stop in accessible:
                la, lo = float(lat[stop]), float(lon[stop])
                if np.isnan(la) or np.isnan(lo):
                    continue
                for seed in seeds:
                    sla, slo = float(lat[seed]), float(lon[seed])
                    if np.isnan(sla) or np.isnan(slo):
                        continue
                    north = (la - sla) * 111_320.0
                    east = ((lo - slo) * 111_320.0
                            * math.cos(math.radians((la + sla) / 2.0)))
                    if math.hypot(east, north) <= float(station_radius_m):
                        nearby.add(stop)
                        break

        return nearby

    def planned_branch_access_stops(self, ci, base_min, window_min):
        """Coverage-safe access-stop filter for pinned structural branch enumeration.

        A stop can contain an in-window branch only if its *best* validated planned anchor plus the
        cell's access walk is also within the window.  This necessary predicate is independent of
        public labels, chip caps, station radii, and whichever structural route happened to win the
        citywide overlay.  It therefore removes provably irrelevant stops without hiding a second
        corridor/direction that shares a label or falls after the UI chip cap.
        """
        try:
            max_total = int(base_min) + int(round(window_min))
        except (TypeError, ValueError):
            return None
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        aw = np.asarray(self.access_w, np.int64)
        a0, a1 = int(off[ci]), int(off[ci + 1])
        if a1 <= a0:
            return set()

        _stop_tail, best_B, best_T = self._planned_stop_anchors()
        allowed = set()
        for k in range(a0, a1):
            s = int(to[k]); awk = int(aw[k])
            # Access walking is the only safe pre-validation lower bound.  ``T-B`` is a deadline
            # upper bound and may contain nearly a minute of grid slack.
            lower = int((awk + 59) // 60)
            if lower > max_total or lower >= self.max_min:
                continue
            template = self._validated_stop_template(s)
            if template is None:
                continue
            total = int((self._planned_template_tail_sec(template) + awk + 59) // 60)
            if total <= max_total and total < self.max_min:
                allowed.add(s)
        return allowed

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
        if self.planned:
            return self._planned_alt_lines_window(perfect, window_min)
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

    def _planned_route_label(self, legs_raw):
        seq = []
        for leg in legs_raw:
            if leg[0] != "ride":
                continue
            pi = int(leg[1])
            name = self.line_table[int(self.pat_line[pi])][2]
            seq.append(name)
        return " > ".join(seq) if seq else "walk only"

    def _planned_alt_lines_window(self, perfect, window_min):
        """Planned-mode alternatives scored by the same first-board model as the headline.

        Unlike the legacy depart-after wrapper, this scans access stops even when the primary route
        is pure walk. That keeps useful transit choices visible when walking is fastest but a bus or
        metro option is still close enough to be worth comparing.
        """
        _s_star, _aw_sel, _Dstar, _Tstar, _is_walk, painted = self._select_arrays()
        off = np.asarray(self.access_off, np.int64)
        to = np.asarray(self.access_to, np.int64)
        aw = np.asarray(self.access_w, np.int64)
        win = int(round(window_min))
        out = [None] * self.n_cells
        stop_tail, best_B, best_T = self._planned_stop_anchors()

        # Every access pair is already contiguous by cell. Build its cell index once, then do the
        # prefilter citywide in numpy instead of 2,999 tiny slices + ~168k Python set insertions.
        counts = np.diff(off)
        access_cell = np.repeat(np.arange(self.n_cells, dtype=np.int64), counts)
        bases = np.asarray(painted, np.int64).copy()
        p = np.asarray(perfect, np.int64)
        upto = min(self.n_cells, len(p))
        if upto:
            valid_perfect = p[:upto] >= 0
            # Boolean indexing creates a temporary, so assigning through
            # ``bases[:upto][valid_perfect]`` silently left every row on its
            # painted fallback.  ``perfect`` is normally the planned painted
            # value in the served endpoint, but callers may legitimately pass
            # a distinct non-negative baseline.  Write through explicit row
            # indices so those rows drive the window, while negative sentinels
            # intentionally retain their painted value.
            valid_rows = np.flatnonzero(valid_perfect)
            bases[valid_rows] = p[valid_rows]
        # Only access time is a safe lower bound before raw template validation; the profile's
        # deadline tail can include quantization slack and must not exclude a truthful route.
        pair_totals = (aw + 59) // 60
        pair_ok = ((bases[access_cell] >= 0)
                   & (pair_totals < self.max_min)
                   & (pair_totals <= bases[access_cell] + win))
        candidates = np.unique(to[pair_ok])

        n_stops = self.arrivalW.shape[0]
        valid_tail = np.full(n_stops, _INF, np.int64)
        labels = {}
        route_keys = {}
        # Keep lightweight test/subclass overrides on the established scalar hook.  Production
        # takes the grouped path: a deadline tree is resolved once for all candidate stops whose
        # immutable profile anchor shares T, while the template/B3 validation itself stays exactly
        # one trace per stop.
        scalar_hook = getattr(self._validated_stop_template, "__func__", None)
        if scalar_hook is DepartAfterJourneyTree._validated_stop_template:
            templates = self._validated_stop_templates_grouped(candidates, best_B, best_T)
        else:
            templates = {int(s): self._validated_stop_template(int(s)) for s in candidates}
        for s in candidates:
            s = int(s)
            template = templates[s]
            if template is None:
                continue
            _B, _T, label, route_key = self._planned_template_summary(template)
            valid_tail[s] = self._planned_template_tail_sec(template)
            labels[s] = label
            route_keys[s] = route_key

        # Assign opaque route identities lexicographic integer ids. Lexicographic ids let numpy's
        # final sort reproduce ``sorted(best_by_route, key=(minute, route_key))`` exactly.
        ordered_keys = sorted(set(route_keys.values()))
        rid_for_key = {key: i for i, key in enumerate(ordered_keys)}
        rid_at_stop = np.full(n_stops, -1, np.int64)
        label_by_rid = [None] * len(ordered_keys)
        for s, route_key in route_keys.items():
            rid = rid_for_key[route_key]
            rid_at_stop[s] = rid
            label_by_rid[rid] = labels[s]

        pair_totals = (valid_tail[to] + aw + 59) // 60
        pair_rids = rid_at_stop[to]
        keep = np.flatnonzero(
            (bases[access_cell] >= 0)
            & (pair_rids >= 0)
            & (pair_totals < self.max_min)
            & (pair_totals <= bases[access_cell] + win))
        if keep.size:
            cells = access_cell[keep]
            rids = pair_rids[keep]
            totals = pair_totals[keep]
            stops = to[keep]
            # Per (cell, opaque route), the old rule kept minimum (minute, stop).
            order = np.lexsort((stops, totals, rids, cells))
            cells = cells[order]; rids = rids[order]
            totals = totals[order]; stops = stops[order]
            first = np.ones(len(order), bool)
            first[1:] = (cells[1:] != cells[:-1]) | (rids[1:] != rids[:-1])
            cells = cells[first]; rids = rids[first]
            totals = totals[first]; stops = stops[first]

            # The coarse overlay's public contract is {display_label: [minute, stop]}. Preserve
            # every structural route through ranking, then coalesce equal public labels at this
            # final boundary in the exact old (minute, route_key) iteration order.
            order = np.lexsort((rids, totals, cells))
            displays = {}
            for pos in order:
                ci = int(cells[pos]); rid = int(rids[pos]); pm = int(totals[pos])
                s = int(stops[pos]); label = label_by_rid[rid]
                if label == "walk only":
                    continue
                display = displays.setdefault(ci, {})
                route_key = ordered_keys[rid]
                cur = display.get(label)
                # Rows already arrive in exact old (minute, route_key) order. The first public
                # label wins; a later equal-minute structural route must not replace it merely
                # because its chosen stop id is smaller (stop id was never part of this tie).
                if cur is None:
                    display[label] = [pm, s, route_key]
            for ci, display in displays.items():
                out[ci] = {label: [v[0], v[1]] for label, v in display.items()}
        return out
