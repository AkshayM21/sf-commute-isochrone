"""RaptorEngine — the server-facing grid travel-time engine.

Loads the RAPTOR structures (raptor_build) plus the graph-native baked cell-to-stop access
table.  Its reverse transfer CSR is consumed by the backward RAPTOR kernels while the paired
forward CSR/path arrays remain available to geometry consumers.  A workplace's egress/pure-walk
can still be supplied by any caller-side walk router.

Three semantics from the SAME reverse range-RAPTOR inputs:
  * planned scheduled depart-after — the served, first-boarding-anchored product metric.
  * legacy depart-after p5/p50 over [DEP, DEP+WINDOW] — retained validation output.
  * legacy arrive-by over an arrival window ending at the target (default [TARGET-WINDOW, TARGET]);
    single-deadline arrive-by is the WINDOW=0 case.

The hot reverse sweep runs in the numba kernel automatically when available; assembly is numpy.
"""
import os
import time
from pathlib import Path
import numpy as np

from . import config, graph_transfers, raptor_build, raptor as R, raptor_journey


def _perf_add(perf, name, started):
    """Record one request-local benchmark phase without affecting normal calls."""
    if perf is not None and started is not None:
        perf[name] = round((time.perf_counter() - started) * 1000.0, 3)

ACCESS_CAP_MIN = int(os.environ.get("RAPTOR_ACCESS_CAP", "25"))  # 25 cleared the worst access-starved periphery (max err 15->7, mism 20->5) vs 20
DEADLINE_STEP = int(os.environ.get("RAPTOR_DEADLINE_STEP", "180"))
MC_DEADLINE_STEP = int(os.environ.get("RAPTOR_MC_DEADLINE_STEP", "60"))  # MC tail readout (finer; see _make_grids)
PLANNED_DEADLINE_STEP = int(os.environ.get("RAPTOR_PLANNED_DEADLINE_STEP", "60"))
DEP_STEP = 60
BOARD_SLACK = int(os.environ.get("RAPTOR_BOARD_SLACK", "60"))
MAX_ROUNDS = 8                                    # rides = transfers + 1
ARRIVE_BY_HM = (9, 0)                             # product target: arrive by 09:00

# Runtime transfer timing/path arrays emitted by the canonical graph-native access bake.  The
# metadata arrays are retained on the engine separately; RAPTOR itself consumes only tr_off,
# tr_to, and the effective tr_time view.
_ACCESS_TRANSFER_KEYS = (
    # Reverse target -> source runtime view for RAPTOR.
    "tr_off", "tr_to", "tr_walk_time", "tr_min_time", "tr_time", "tr_path_fallback",
    # Forward source -> target geometry view.
    "tr_forward_off", "tr_forward_to", "tr_forward_walk_time", "tr_forward_min_time",
    "tr_forward_time", "tr_forward_path_off", "tr_forward_path_points",
    "tr_forward_path_fallback", "tr_forward_pathway_off", "tr_forward_pathway_id",
    "tr_forward_pathway_time", "tr_forward_pathway_mode", "tr_forward_pathway_length_m",
    "tr_forward_pathway_reversed", "transfer_scoped_source", "transfer_scoped_target",
    "transfer_scoped_from_route", "transfer_scoped_to_route", "transfer_scoped_from_trip",
    "transfer_scoped_to_trip", "transfer_scoped_min_time", "transfer_scoped_prohibited",
    "transfer_scoped_type",
    "transfer_scoped_physical_time", "transfer_scoped_path_fallback",
    "transfer_scoped_path_off", "transfer_scoped_path_points",
    "transfer_pathway_source", "transfer_pathway_target", "transfer_pathway_id",
    "transfer_pathway_time", "transfer_pathway_mode", "transfer_pathway_length_m",
    "transfer_pathway_reversed",
)

# --- service-noise Monte-Carlo (realistic + fragility + alt-lines) --------------------
MC_DRAWS = int(os.environ.get("RAPTOR_MC_DRAWS", "24"))       # draws for realistic/fragility
# Alt-lines are a DOMINANCE WINDOW over the UNPERTURBED tree's per-access-stop journeys (NOT a
# draw-vote): a transit line whose best per-access-stop door-to-door time is within ALT_WINDOW_MIN
# of the cell's best is an alternative — deterministic, K-free, and walk-speed-STABLE (a line within
# the window at slow walk stays within it at fast; its gap to best changes only by the walk-speed
# delta on the access leg). This replaced the old K-draw ">=1-draw vote" lottery that dropped
# near-best short-walk buses when walking sped up — and a perturbation-draw candidate pool only ADDS
# churn back (measured), so alts come purely from the deterministic tree (see
# .plans/alt_walkspeed_diag.md). MC_ALT_DRAWS now just GATES alts on/off (>0 = on) — it no longer
# drives any perturbed traces; the realistic/fragility MC keeps its own RAPTOR_MC_DRAWS.
MC_ALT_DRAWS = int(os.environ.get("RAPTOR_MC_ALT_DRAWS", "12"))  # >0 enables the alt-lines window
ALT_WINDOW_MIN = float(os.environ.get("RAPTOR_ALT_WINDOW_MIN", "5"))  # alt = within this of the cell best
PLANNED_ALT_WINDOW_MIN = float(os.environ.get("RAPTOR_PLANNED_ALT_WINDOW_MIN", "10"))
MC_SHAPE = float(os.environ.get("RAPTOR_MC_SHAPE", "2.0"))   # Gamma shape (spread); mean=shape*scale
WALK_RELUCTANCE = config.WALK_RELUCTANCE          # mild walk prior (decision-only; see config.py)
WALK_PRIOR_EPS = config.WALK_PRIOR_EPS_SEC        # hard cap (sec) on the prior's true-time change
# mean INITIAL delay (sec, env-overridable via RAPTOR_MC_MU_*) + fractional DRIFT slope (fixed,
# see _SLOPE). Buses are the noisiest, rail the steadiest; BART/Caltrain resolved by feed STEM in
# _mc_mode_params (NOT by position — gtfs_paths() drops missing files and appends extras, so
# positional indices can silently shift).
_MU = dict(bus=float(os.environ.get("RAPTOR_MC_MU_BUS", "70")),
           metro=float(os.environ.get("RAPTOR_MC_MU_METRO", "45")),
           cable=float(os.environ.get("RAPTOR_MC_MU_CABLE", "40")),
           bart=float(os.environ.get("RAPTOR_MC_MU_BART", "25")),
           caltrain=float(os.environ.get("RAPTOR_MC_MU_CALTRAIN", "40")))
# fractional DRIFT (delay grows with time on the vehicle); the most uncertain knob, kept modest
# so a long peripheral bus ride does not over-inflate the median in the validation corpus.
_SLOPE = dict(bus=0.035, metro=0.025, cable=0.03, bart=0.012, caltrain=0.02)


def _committed_deadline_prefix(deadlines, legs, max_min):
    """Losslessly trim a committed-MC deadline grid to the rows' useful horizon.

    A transit row's raw result is capped at ``max_min``.  Therefore every deadline strictly
    after ``commit_home + max_min * 60`` can only produce the same cap, regardless of the
    perturbed schedule.  Keep the inclusive prefix through the latest such bound across transit
    rows.  Rows without transit are deadline-independent in the committed kernel, so a single
    deadline is sufficient when the batch contains no transit at all.

    The helper is deliberately conservative: malformed/mismatched leg arrays, an invalid transit
    home, a negative cap, or a non-increasing grid return the full grid.  Production callers pass
    the engine's non-empty ascending ``Tgrid_mc``; keeping the fallback here makes this optimization
    incapable of changing results if a future caller violates those assumptions.
    """
    grid = np.asarray(deadlines, dtype=np.int64)
    if grid.ndim != 1 or grid.size == 0:
        return grid
    if grid.size > 1 and not bool(np.all(np.diff(grid) > 0)):
        return grid
    try:
        kind = np.asarray(legs["commit_kind"])
        home = np.asarray(legs["commit_home"], dtype=np.int64)
        cap_sec = int(max_min) * 60
    except (KeyError, TypeError, ValueError, OverflowError):
        return grid
    if kind.ndim != 1 or home.ndim != 1 or kind.shape != home.shape or cap_sec < 0:
        return grid
    transit = kind == 2
    if not bool(np.any(transit)):
        return grid[:1]
    transit_home = home[transit]
    # A kind-2 row must carry a real time-of-day home departure.  NEG belongs only to unreachable
    # kind-0 rows; treating it as a bound would incorrectly collapse the grid.
    if transit_home.size == 0 or bool(np.any(transit_home < 0)):
        return grid
    bound = int(transit_home.max()) + cap_sec
    # side="right" is load-bearing: a deadline exactly at home+cap can still be the first feasible
    # readout and must remain.  If even the first grid point lies beyond the bound, one column still
    # suffices: every transit result is necessarily capped, and the kernel requires a non-empty grid.
    end = max(1, int(np.searchsorted(grid, bound, side="right")))
    return grid[:end]


def _mc_summary_from_draws(commute_draws, perfect, max_min):
    """Return the public MC statistics with one ordering pass per cell.

    ``numpy.percentile`` was previously called three times over the same tiny draw axis: once on
    raw draws for the diagnostic p50, then twice more after applying the per-cell best-case
    floor.  Flooring every draw by one scalar is monotone, so the raw sorted order stays sorted
    after that floor.  Sort once, read the exact ``method='linear'`` order statistics before and
    after flooring, and keep NumPy reductions for standard deviation and stuck probability.

    The explicit NaN propagation matches ``np.percentile`` rather than letting ``np.sort`` place
    a NaN at the end and accidentally hide it from a median whose ranks precede that position.
    Inputs are only produced by the float64 committed kernel in production; accepting every
    numeric dtype here keeps this seam independently testable and preserves NumPy's float64
    percentile promotion for smaller input dtypes.
    """
    draws = np.asarray(commute_draws)
    # ``np.percentile`` promotes integer and low-precision float inputs to float64.  Retain a
    # wider input (notably longdouble) if present, while production's float64 kernel stays
    # allocation-free before the single sort.
    stat_dtype = np.result_type(draws.dtype, np.float64)
    ordered = np.sort(draws.astype(stat_dtype, copy=False), axis=1)
    n_draws = ordered.shape[1]

    # NumPy's default percentile method is linear: h=(n-1)q, then a + (b-a) * fractional(h).
    # The draw count is normally 24, but test and diagnostic callers use smaller values too.
    has_nan = np.isnan(ordered).any(axis=1)

    def linear_percentile(q):
        h = (n_draws - 1) * (float(q) / 100.0)
        lo = int(np.floor(h))
        hi = int(np.ceil(h))
        out = ordered[:, lo].copy()
        if hi != lo:
            out += (ordered[:, hi] - out) * (h - lo)
        if bool(np.any(has_nan)):
            out[has_nan] = np.nan
        return out

    raw_p50 = linear_percentile(50.0)
    # Keep the historical floor semantics exactly: non-positive/sentinel perfect values floor
    # to zero, and the floor applies before every served distribution statistic.
    floor = np.where(np.asarray(perfect) >= 0, perfect, 0).astype(np.float64)[:, None]
    floored = np.maximum(draws, floor)
    # Applying the scalar row floor preserves ordering, so no second sort is needed.
    np.maximum(ordered, floor, out=ordered)
    p50 = linear_percentile(50.0)
    p90 = linear_percentile(90.0)

    return (np.ceil(raw_p50).astype(np.int32),
            np.ceil(p50).astype(np.int32),
            np.maximum(0, np.round(p90 - p50)).astype(np.int32),
            np.round(p90).astype(np.int32),
            np.round(np.std(floored, axis=1)).astype(np.int32),
            np.mean(floored >= float(max_min) - 1e-9, axis=1))


class MonteCarloScenario:
    """Private, lossless reverse-profile snapshot used only to accelerate a later route pin.

    The public Monte-Carlo result remains ordinary JSON-shaped arrays/dicts.  A server may retain
    one instance out-of-band and hand it back to ``route_typicals``; every routing/model input that
    affects the profile is checked before reuse, otherwise the normal full kernel runs.
    """
    VERSION = 2
    __slots__ = ("version", "tail_lag", "deadlines", "delta0_all", "slope_all", "seed",
                 "n_draws", "max_rounds", "board_slack", "max_min", "walk_scalar",
                 "data", "egress_g", "egress_w")

    def __init__(self, *, tail_lag, deadlines, delta0_all, slope_all, seed, max_rounds,
                 board_slack, max_min, walk_scalar, data, egress_g, egress_w):
        self.version = self.VERSION
        self.tail_lag = np.ascontiguousarray(tail_lag, np.uint16)
        self.deadlines = np.ascontiguousarray(deadlines, np.int64)
        self.delta0_all = np.ascontiguousarray(delta0_all, np.float64)
        self.slope_all = np.ascontiguousarray(slope_all, np.float64)
        self.seed = seed
        self.n_draws = int(self.delta0_all.shape[0])
        self.max_rounds = int(max_rounds)
        self.board_slack = int(board_slack)
        self.max_min = int(max_min)
        self.walk_scalar = float(walk_scalar)
        # Keep the exact graph object rather than only its numeric id.  CPython may recycle an
        # id after an in-process rebuild; identity makes a retained profile ineligible for an
        # equivalent-looking but different graph even in that pathological lifecycle.
        self.data = data
        self.egress_g = np.ascontiguousarray(egress_g, np.int32)
        self.egress_w = np.ascontiguousarray(egress_w, np.int64)
        for a in (self.tail_lag, self.deadlines, self.delta0_all, self.slope_all,
                  self.egress_g, self.egress_w):
            a.flags.writeable = False

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.tail_lag, self.deadlines, self.delta0_all,
                                      self.slope_all, self.egress_g, self.egress_w))

    def compatible(self, *, data, deadlines, egress_g, egress_w, seed, n_draws, max_rounds,
                   board_slack, max_min, walk_scalar):
        deadlines = np.asarray(deadlines, np.int64)
        return bool(
            self.version == self.VERSION
            and seed is not None and self.seed == seed
            and self.n_draws == int(n_draws)
            and self.max_rounds == int(max_rounds)
            and self.board_slack == int(board_slack)
            and self.max_min == int(max_min)
            and self.walk_scalar == float(walk_scalar)
            and self.data is data
            and np.array_equal(self.egress_g, np.asarray(egress_g, np.int32))
            and np.array_equal(self.egress_w, np.asarray(egress_w, np.int64))
            and deadlines.size <= self.deadlines.size
            and np.array_equal(deadlines, self.deadlines[:deadlines.size])
            and self.tail_lag.shape == (self.n_draws, int(data["n_stops"]),
                                        self.deadlines.size)
            and self.delta0_all.shape == self.slope_all.shape)


class RaptorEngine:
    def __init__(self, gtfs_paths=None, service_date=None, access_path=None,
                 access_cap_min=ACCESS_CAP_MIN, verbose=True):
        gtfs_paths = gtfs_paths or config.gtfs_paths()
        from . import feeds
        self.service_date = service_date or feeds.pick_service_date(gtfs_paths)
        self.data = raptor_build.load_or_build(gtfs_paths, self.service_date, verbose=verbose)
        self._walk_scaled_data = {}
        self._walk_grid_cache = {}
        self.access_cap_min = access_cap_min
        self._load_access(access_path, verbose)
        self._make_grids()
        self._warm()

    # -- access table (baked, workplace-independent) --------------------------------------
    def _load_access(self, access_path, verbose):
        expect_grid_m = None                      # set only when WE pick the table (default path);
        if access_path is None:                   # explicit callers choose their own grid
            # Exact canonical name, not a glob: the bake is selected by explicit grid and
            # service date, while its direct source mtimes are validated below.
            access_path = (raptor_build.CACHE_DIR /
                           f"access_walk_{config.GRID_M}m_{self.service_date:%Y%m%d}.npz")
            if not access_path.exists():
                raise FileNotFoundError(
                    f"no baked access table {access_path.name} in {raptor_build.CACHE_DIR}; "
                    f"run scripts/bake_walk_access.py")
            expect_grid_m = config.GRID_M
        with np.load(access_path, allow_pickle=False) as z:
            self._validate_access_artifact(z, access_path, expect_grid_m)
            self.cell_ids = list(z["cell_ids"].astype(str))
            self.cell_index = {c: i for i, c in enumerate(self.cell_ids)}
            # The graph bake is the runtime source of truth for directed transfers.  Keep
            # timing/path metadata on the engine for geometry consumers, while RAPTOR sees the
            # effective reference-time CSR in the same keys it has always consumed.
            self.data = dict(self.data)
            for key in _ACCESS_TRANSFER_KEYS:
                self.data[key] = np.array(z[key], copy=True)
            self.transfer_path_off = np.array(z["tr_forward_path_off"], copy=True)
            self.transfer_path_points = np.array(z["tr_forward_path_points"], copy=True)
            self.transfer_path_fallback = np.array(z["tr_forward_path_fallback"], copy=True)
            self.transfer_pathway_off = np.array(z["tr_forward_pathway_off"], copy=True)
            self.transfer_scoped_physical_time = np.array(
                z["transfer_scoped_physical_time"], copy=True)
            self.transfer_scoped_path_off = np.array(
                z["transfer_scoped_path_off"], copy=True)
            self.transfer_scoped_path_points = np.array(
                z["transfer_scoped_path_points"], copy=True)
            self.transfer_scoped_path_fallback = np.array(
                z["transfer_scoped_path_fallback"], copy=True)
            self.transfer_scoped_source = np.array(z["transfer_scoped_source"], copy=True)
            self.transfer_scoped_target = np.array(z["transfer_scoped_target"], copy=True)
            self.transfer_pathway_source = np.array(z["transfer_pathway_source"], copy=True)
            self.transfer_pathway_target = np.array(z["transfer_pathway_target"], copy=True)
            # filter to the access cap and re-pack CSR (seconds)
            off = np.asarray(z["access_off"]); to = np.asarray(z["access_to"]); w = np.asarray(z["access_w"])
            # Keep only arrays needed after the archive is closed.
            n = len(self.cell_ids)
        cap = self.access_cap_min * 60
        new_off = np.zeros(n + 1, dtype=np.int64)
        keep_to, keep_w = [], []
        for ci in range(n):
            a0, a1 = int(off[ci]), int(off[ci + 1])
            sub_to = to[a0:a1]; sub_w = w[a0:a1]
            m = sub_w <= cap
            keep_to.append(sub_to[m]); keep_w.append(sub_w[m])
            new_off[ci + 1] = new_off[ci] + int(m.sum())
        self.access_off = new_off
        self.access_to = (np.concatenate(keep_to).astype(np.int32) if keep_to
                          else np.zeros(0, np.int32))
        self.access_w = (np.concatenate(keep_w).astype(np.int64) if keep_w
                         else np.zeros(0, np.int64))
        if verbose:
            print(f"[engine] access table {access_path.name}: {n} cells, "
                  f"{len(self.access_to)} pairs <= {self.access_cap_min}min walk")

    def _validate_access_artifact(self, z, access_path, expect_grid_m=None):
        """Validate the graph-native access archive and its direct source metadata."""
        required = {"cell_ids", "access_off", "access_to", "access_w", "grid_m", "n_stops",
                    "service_date", "walk_ref_kmh", "slope_aware", "raptor_source_names",
                    "raptor_source_sizes", "raptor_source_mtimes_ns", "walk_graph_size",
                    "walk_graph_mtime_ns", "footpath_m", "raptor_build_version",
                    "grid_source_names", "grid_source_sizes", "grid_source_mtimes_ns",
                    *_ACCESS_TRANSFER_KEYS,
                    "transfer_scoped_source", "transfer_scoped_target", "transfer_scoped_from_route",
                    "transfer_scoped_to_route", "transfer_scoped_from_trip", "transfer_scoped_to_trip",
                    "transfer_scoped_min_time", "transfer_scoped_prohibited",
                    "transfer_pathway_source", "transfer_pathway_target", "transfer_pathway_id",
                    "transfer_pathway_time", "transfer_pathway_mode", "transfer_pathway_length_m",
                    "transfer_pathway_reversed"}
        if not required.issubset(z.files):
            raise ValueError(f"access table {access_path.name} is missing required metadata")
        for scalar_key in (
                "grid_m", "n_stops", "service_date", "walk_ref_kmh", "slope_aware",
                "walk_graph_size", "walk_graph_mtime_ns", "footpath_m",
                "raptor_build_version"):
            if np.asarray(z[scalar_key]).ndim != 0:
                raise ValueError(f"access table {access_path.name} has invalid scalar metadata")
        cell_ids = np.asarray(z["cell_ids"])
        off = np.asarray(z["access_off"]); to = np.asarray(z["access_to"]); w = np.asarray(z["access_w"])
        if cell_ids.ndim != 1 or cell_ids.dtype.kind not in "OUS":
            raise ValueError(f"access table {access_path.name} has invalid cell ids")
        n_cells = len(cell_ids)
        if len(set(cell_ids.astype(str))) != n_cells:
            raise ValueError(f"access table {access_path.name} has duplicate cells")
        if off.ndim != 1 or off.dtype.kind not in "iu" or len(off) != n_cells + 1:
            raise ValueError(f"access table {access_path.name} has invalid CSR offsets")
        if off[0] != 0 or np.any(np.diff(off) < 0) or int(off[-1]) != len(to) or len(to) != len(w):
            raise ValueError(f"access table {access_path.name} has inconsistent CSR arrays")
        if to.ndim != 1 or to.dtype.kind not in "iu" or w.ndim != 1 or w.dtype.kind not in "iu":
            raise ValueError(f"access table {access_path.name} has invalid CSR dtypes")
        n_stops = int(np.asarray(z["n_stops"]))
        if n_stops != int(self.data["n_stops"]) or np.any(to < 0) or np.any(to >= n_stops) or np.any(w < 0):
            raise ValueError("access table / raptor structures stop-count mismatch (stale cache)")
        grid_m_value = np.asarray(z["grid_m"])
        if grid_m_value.ndim != 0 or int(grid_m_value) <= 0:
            raise ValueError(f"access table {access_path.name} has invalid grid size metadata")
        grid_m = int(grid_m_value)
        if expect_grid_m is not None and grid_m != expect_grid_m:
            raise ValueError(f"access table {access_path.name} is a {grid_m}m bake but the configured grid is {expect_grid_m}m")
        if str(np.asarray(z["service_date"]).item()) != self.service_date.strftime("%Y%m%d"):
            raise ValueError(f"access table {access_path.name} has a different service date")
        if abs(float(np.asarray(z["walk_ref_kmh"])) - float(config.WALK_KMH)) > 1e-6:
            raise ValueError(f"access table {access_path.name} uses a different walk reference speed")
        if int(np.asarray(z["slope_aware"])) != 1:
            raise ValueError(f"access table {access_path.name} is not the hill-aware bake")
        if (abs(float(np.asarray(z["footpath_m"])) - float(self.data["footpath_m"])) > 1e-9
                or int(np.asarray(z["raptor_build_version"]))
                != int(self.data["build_version"])):
            raise ValueError(f"access table {access_path.name} has incompatible transfer bake parameters")
        if (int(np.asarray(z["walk_graph_size"])) < -1
                or int(np.asarray(z["walk_graph_mtime_ns"])) < -1):
            raise ValueError(f"access table {access_path.name} has invalid walking graph metadata")
        grid_names = np.asarray(z["grid_source_names"]).astype(str)
        grid_sizes = np.asarray(z["grid_source_sizes"])
        grid_mtimes = np.asarray(z["grid_source_mtimes_ns"])
        if (grid_names.ndim != 1 or grid_sizes.ndim != 1 or grid_mtimes.ndim != 1
                or not (len(grid_names) == len(grid_sizes) == len(grid_mtimes))
                or len(grid_names) == 0 or any(not str(name).strip() for name in grid_names)
                or grid_sizes.dtype.kind not in "iu" or grid_mtimes.dtype.kind not in "iu"
                or np.any(grid_sizes < -1) or np.any(grid_mtimes < -1)):
            raise ValueError(f"access table {access_path.name} has invalid grid source metadata")
        grid_source = config.neigh_path()
        try:
            grid_st = grid_source.stat()
            actual_grid = (grid_source.name, int(grid_st.st_size), int(grid_st.st_mtime_ns))
        except OSError:
            actual_grid = (Path(grid_source).name, -1, -1)
        stored_grid = tuple((str(name), int(size), int(mtime))
                            for name, size, mtime in zip(grid_names, grid_sizes, grid_mtimes))
        if stored_grid != (actual_grid,):
            raise ValueError(f"access table {access_path.name} is stale relative to the neighborhood/grid source")

        def _check_view(prefix, *, with_paths=False):
            off = np.asarray(z[f"{prefix}off"]); to = np.asarray(z[f"{prefix}to"])
            walk = np.asarray(z[f"{prefix}walk_time"]); minimum = np.asarray(z[f"{prefix}min_time"])
            effective = np.asarray(z[f"{prefix}time"]); fallback = np.asarray(z[f"{prefix}path_fallback"])
            if (off.ndim != 1 or off.dtype.kind not in "iu" or len(off) != n_stops + 1
                    or off[0] != 0 or np.any(np.diff(off) < 0) or int(off[-1]) != len(to)
                    or to.ndim != 1 or to.dtype.kind not in "iu"
                    or np.any(to < 0) or np.any(to >= n_stops)):
                return False
            edge_count = len(to)
            if any(a.ndim != 1 or len(a) != edge_count for a in (walk, minimum, effective, fallback)):
                return False
            if (walk.dtype.kind not in "fiu" or minimum.dtype.kind not in "fiu"
                    or effective.dtype.kind not in "iu" or fallback.dtype.kind not in "biu"
                    or not np.isfinite(walk).all() or not np.isfinite(minimum).all()
                    or np.any(walk < 0) or np.any(minimum < 0) or np.any(effective < 0)
                    or not np.array_equal(effective, np.floor(np.maximum(walk, minimum) + 0.5).astype(effective.dtype))
                    or np.any((fallback != 0) & (fallback != 1))):
                return False
            for row in range(n_stops):
                a, b = int(off[row]), int(off[row + 1])
                if b - a > 1 and np.any(np.diff(to[a:b]) <= 0):
                    return False
            if with_paths:
                path_off = np.asarray(z["tr_forward_path_off"]); points = np.asarray(z["tr_forward_path_points"])
                if (path_off.ndim != 1 or path_off.dtype.kind not in "iu"
                        or len(path_off) != edge_count + 1 or path_off[0] != 0
                        or np.any(np.diff(path_off) < 2) or int(path_off[-1]) != len(points)
                        or points.ndim != 2 or points.shape[1:] != (2,)
                        or points.dtype.kind not in "f" or not np.isfinite(points).all()
                        or np.any(points[:, 0] < -90.0) or np.any(points[:, 0] > 90.0)
                        or np.any(points[:, 1] < -180.0) or np.any(points[:, 1] > 180.0)):
                    return False
                # Every stored path is a reusable route result, not merely a polyline blob:
                # its endpoints must remain attached to the forward CSR edge's canonical stops.
                try:
                    stop_lat = np.asarray(self.data["stop_lat"], dtype=np.float64)
                    stop_lon = np.asarray(self.data["stop_lon"], dtype=np.float64)
                    sources = np.repeat(np.arange(n_stops, dtype=np.int64), np.diff(off))
                    if (stop_lat.ndim != 1 or stop_lon.ndim != 1
                            or len(stop_lat) != n_stops or len(stop_lon) != n_stops
                            or len(sources) != edge_count
                            or (edge_count and (
                                not np.isfinite(stop_lat[sources]).all()
                                or not np.isfinite(stop_lon[sources]).all()
                                or not np.isfinite(stop_lat[to]).all()
                                or not np.isfinite(stop_lon[to]).all()))):
                        return False
                    if (not np.allclose(
                            points[path_off[:-1]],
                            np.column_stack((stop_lat[sources], stop_lon[sources])),
                            atol=1e-5, rtol=0.0)
                            or not np.allclose(
                                points[path_off[1:] - 1],
                                np.column_stack((stop_lat[to], stop_lon[to])),
                                atol=1e-5, rtol=0.0)):
                        return False
                except (KeyError, IndexError, TypeError, ValueError):
                    return False
                edge_path_off = np.asarray(z["tr_forward_pathway_off"])
                if (edge_path_off.ndim != 1 or edge_path_off.dtype.kind not in "iu"
                        or len(edge_path_off) != edge_count + 1 or edge_path_off[0] != 0
                        or np.any(np.diff(edge_path_off) < 0)):
                    return False
                edge_meta_len = int(edge_path_off[-1])
                for key in ("tr_forward_pathway_id", "tr_forward_pathway_time",
                            "tr_forward_pathway_mode", "tr_forward_pathway_length_m",
                            "tr_forward_pathway_reversed"):
                    if np.asarray(z[key]).ndim != 1 or len(np.asarray(z[key])) != edge_meta_len:
                        return False
                if (np.asarray(z["tr_forward_pathway_id"]).dtype.kind not in "OUS"
                        or np.asarray(z["tr_forward_pathway_mode"]).dtype.kind not in "OUS"):
                    return False
                pt = np.asarray(z["tr_forward_pathway_time"]); pr = np.asarray(z["tr_forward_pathway_reversed"])
                pl = np.asarray(z["tr_forward_pathway_length_m"])
                if (pt.dtype.kind not in "iu" or pr.dtype.kind not in "biu"
                        or pl.dtype.kind not in "f" or np.any(pt < -1)
                        or np.any((pr != 0) & (pr != 1))
                        or np.any(~np.isnan(pl) & (pl < 0))):
                    return False
            return True

        if not _check_view("tr_") or not _check_view("tr_forward_", with_paths=True):
            raise ValueError(f"access table {access_path.name} has invalid transfer CSR/timing/path arrays")
        if not graph_transfers.validate_transfer_views(
                n_stops,
                forward_off=z["tr_forward_off"], forward_to=z["tr_forward_to"],
                forward_walk=z["tr_forward_walk_time"], forward_min=z["tr_forward_min_time"],
                forward_time=z["tr_forward_time"], forward_fallback=z["tr_forward_path_fallback"],
                reverse_off=z["tr_off"], reverse_to=z["tr_to"],
                reverse_walk=z["tr_walk_time"], reverse_min=z["tr_min_time"],
                reverse_time=z["tr_time"], reverse_fallback=z["tr_path_fallback"],
                forward_pathway_off=z["tr_forward_pathway_off"],
                forward_pathway_time=z["tr_forward_pathway_time"]):
            raise ValueError(f"access table {access_path.name} has mismatched forward/reverse transfer views")
        pathway_names = [key for key in z.files if key.startswith("transfer_pathway_")]
        if (not pathway_names
                or len({len(np.asarray(z[key])) for key in pathway_names}) != 1):
            raise ValueError(f"access table {access_path.name} has invalid preserved pathway metadata")
        scoped_n = len(np.asarray(z["transfer_scoped_source"]))
        pathway_n = len(np.asarray(z["transfer_pathway_source"]))
        for source_key, target_key, count in (
                ("transfer_scoped_source", "transfer_scoped_target", scoped_n),
                ("transfer_pathway_source", "transfer_pathway_target", pathway_n)):
            source = np.asarray(z[source_key]); target = np.asarray(z[target_key])
            if (source.dtype.kind not in "iu" or target.dtype.kind not in "iu"
                    or np.any(source < 0) or np.any(source >= n_stops)
                or np.any(target < 0) or np.any(target >= n_stops)
                or len(source) != count or len(target) != count):
                raise ValueError(f"access table {access_path.name} has invalid preserved rule indexes")
        scoped_scalar_keys = (
            "transfer_scoped_target", "transfer_scoped_from_route",
            "transfer_scoped_to_route", "transfer_scoped_from_trip",
            "transfer_scoped_to_trip", "transfer_scoped_min_time",
            "transfer_scoped_prohibited", "transfer_scoped_type",
            "transfer_scoped_physical_time",
            "transfer_scoped_path_fallback")
        if any(len(np.asarray(z[key])) != scoped_n for key in scoped_scalar_keys):
            raise ValueError(f"access table {access_path.name} has invalid scoped metadata lengths")
        scoped_path_off = np.asarray(z["transfer_scoped_path_off"])
        scoped_path_points = np.asarray(z["transfer_scoped_path_points"])
        if (scoped_path_off.ndim != 1 or scoped_path_off.dtype.kind not in "iu"
                or len(scoped_path_off) != scoped_n + 1 or scoped_path_off[0] != 0
                or np.any(np.diff(scoped_path_off) < 0)
                or int(scoped_path_off[-1]) != len(scoped_path_points)
                or scoped_path_points.ndim != 2 or scoped_path_points.shape[1:] != (2,)
                or scoped_path_points.dtype.kind not in "f"
                or not np.isfinite(scoped_path_points).all()
                or np.any(scoped_path_points[:, 0] < -90.0)
                or np.any(scoped_path_points[:, 0] > 90.0)
                or np.any(scoped_path_points[:, 1] < -180.0)
                or np.any(scoped_path_points[:, 1] > 180.0)):
            raise ValueError(f"access table {access_path.name} has invalid scoped paths")
        scoped_physical = np.asarray(z["transfer_scoped_physical_time"])
        scoped_fallback = np.asarray(z["transfer_scoped_path_fallback"])
        if (scoped_physical.ndim != 1 or scoped_physical.dtype.kind not in "f"
                or not np.all((scoped_physical == -1)
                               | (np.isfinite(scoped_physical) & (scoped_physical >= 0)))
                or scoped_fallback.ndim != 1 or scoped_fallback.dtype.kind not in "biu"
                or np.any((scoped_fallback != 0) & (scoped_fallback != 1))):
            raise ValueError(f"access table {access_path.name} has invalid scoped path values")
        scoped_source = np.asarray(z["transfer_scoped_source"])
        scoped_target = np.asarray(z["transfer_scoped_target"])
        stop_lat = np.asarray(self.data.get("stop_lat", ()), dtype=np.float64)
        stop_lon = np.asarray(self.data.get("stop_lon", ()), dtype=np.float64)
        if stop_lat.ndim != 1 or stop_lon.ndim != 1 or len(stop_lat) != n_stops or len(stop_lon) != n_stops:
            raise ValueError(f"access table {access_path.name} has no canonical stop coordinates")
        for index, (start, end, physical) in enumerate(
                zip(scoped_path_off[:-1], scoped_path_off[1:], scoped_physical)):
            if int(end) > int(start) and (int(end) - int(start) < 2 or float(physical) < 0.0):
                raise ValueError(f"access table {access_path.name} has disconnected scoped paths")
            if int(end) == int(start) and float(physical) >= 0.0:
                raise ValueError(f"access table {access_path.name} has a scoped time without a path")
            if int(end) > int(start):
                source = int(scoped_source[index]); target = int(scoped_target[index])
                if (not np.isfinite(stop_lat[source]) or not np.isfinite(stop_lon[source])
                        or not np.isfinite(stop_lat[target]) or not np.isfinite(stop_lon[target])
                        or not np.allclose(
                            scoped_path_points[int(start)],
                            (stop_lat[source], stop_lon[source]), atol=1e-5, rtol=0.0)
                        or not np.allclose(
                            scoped_path_points[int(end) - 1],
                            (stop_lat[target], stop_lon[target]), atol=1e-5, rtol=0.0)):
                    raise ValueError(f"access table {access_path.name} has disconnected scoped paths")
        scoped_min = np.asarray(z["transfer_scoped_min_time"])
        scoped_prohibited = np.asarray(z["transfer_scoped_prohibited"])
        if (scoped_min.dtype.kind not in "iu" or scoped_prohibited.dtype.kind not in "biu"
                or np.any(scoped_min < -1)
                or np.any((scoped_prohibited != 0) & (scoped_prohibited != 1))):
            raise ValueError(f"access table {access_path.name} has invalid scoped rule values")
        if any(np.asarray(z[key]).dtype.kind not in "OUS" for key in (
                "transfer_scoped_from_route", "transfer_scoped_to_route",
                "transfer_scoped_from_trip", "transfer_scoped_to_trip",
                "transfer_scoped_type")):
            raise ValueError(f"access table {access_path.name} has invalid scoped rule strings")
        pathway_time = np.asarray(z["transfer_pathway_time"])
        pathway_rev = np.asarray(z["transfer_pathway_reversed"])
        pathway_len = np.asarray(z["transfer_pathway_length_m"])
        if (pathway_time.dtype.kind not in "iu" or pathway_rev.dtype.kind not in "biu"
                or pathway_len.dtype.kind not in "f" or np.any(pathway_time < -1)
                or np.any((pathway_rev != 0) & (pathway_rev != 1))
                or np.any(~np.isnan(pathway_len) & (pathway_len < 0))):
            raise ValueError(f"access table {access_path.name} has invalid preserved pathway values")
        if any(np.asarray(z[key]).dtype.kind not in "OUS" for key in (
                "transfer_pathway_id", "transfer_pathway_mode")):
            raise ValueError(f"access table {access_path.name} has invalid pathway strings")
        names = np.asarray(z["raptor_source_names"]).astype(str)
        sizes = np.asarray(z["raptor_source_sizes"])
        mtimes = np.asarray(z["raptor_source_mtimes_ns"])
        expected_sources = tuple(self.data.get("source_mtimes", ()))
        if (names.ndim != 1 or sizes.ndim != 1 or mtimes.ndim != 1
                or not (len(names) == len(sizes) == len(mtimes))
                or any(not str(name).strip() for name in names)
                or sizes.dtype.kind not in "iu" or mtimes.dtype.kind not in "iu"
                or np.any(sizes < -1) or np.any(mtimes < -1)):
            raise ValueError(f"access table {access_path.name} has invalid RAPTOR source metadata")
        actual_sources = tuple((str(name), None if int(size) < 0 else int(size),
                                None if int(mtime) < 0 else int(mtime))
                               for name, size, mtime in zip(names, sizes, mtimes))
        if expected_sources and actual_sources != expected_sources:
            raise ValueError(f"access table {access_path.name} is stale relative to the RAPTOR feeds")
        graph_path = config.DATA / "walk_graph.npz"
        try:
            st = graph_path.stat()
            graph_source = (int(st.st_size), int(st.st_mtime_ns))
        except OSError:
            graph_source = (None, None)
        stored_graph = (int(np.asarray(z["walk_graph_size"])), int(np.asarray(z["walk_graph_mtime_ns"])))
        if graph_source != (None, None) and stored_graph != graph_source:
            raise ValueError(f"access table {access_path.name} is stale relative to the walking graph")

    def _make_grids(self):
        dep = config.DEP_HM[0] * 3600 + config.DEP_HM[1] * 60
        win = int(config.window().total_seconds())
        maxsec = config.MAX_MIN * 60
        self.dep_sec, self.win_sec, self.max_min = dep, win, config.MAX_MIN
        # depart-after: cell departures + the stop-departure grid (extended by the access cap)
        self.cell_deps = np.arange(dep, dep + win + 1, DEP_STEP)
        self.dep_grid = np.arange(dep, dep + win + self.access_cap_min * 60 + 1, DEP_STEP)
        # deadlines swept for the reverse profile (cover all departures + the routing cap)
        self.Tgrid = np.arange(dep, self.dep_grid[-1] + maxsec + 1, DEADLINE_STEP)
        # arrive-by: arrival-window deadlines (fine grid ending at the target)
        self.target_sec = ARRIVE_BY_HM[0] * 3600 + ARRIVE_BY_HM[1] * 60
        # Planned/free-departure display grid: the pinned card compares concrete scheduled branches,
        # so keep planned map/hover scoring at minute resolution without coupling it to MC tuning.
        self.Tgrid_planned = np.arange(dep, self.Tgrid[-1] + 1, PLANNED_DEADLINE_STEP)
        # MC tail-readout grid: the committed kernel returns the first GRID deadline reachable
        # from the actual (late) arrival, so the step size rounds every draw up by U(0, step)s
        # before the percentile — at 180s that's a systematic ~+1.5 min baked into `realistic`.
        # A 60s step keeps the quantization within the served 1-min resolution (dep + target are
        # whole minutes, so 09:00:00 lands exactly on it). MC is lazy + cached off the hover
        # path, so the ~3x deadline count is affordable; the map's Tgrid stays at DEADLINE_STEP.
        self.Tgrid_mc = np.arange(dep, self.Tgrid[-1] + 1, MC_DEADLINE_STEP)

    def _warm(self):
        """Compile the numba kernel once (cheap dummy) so the first real request is fast."""
        eg = self.access_to[:1] if len(self.access_to) else np.zeros(1, np.int32)
        try:
            R.reverse_profile(self.data, eg, np.array([0], np.int64),
                              self.Tgrid[:2], board_slack=BOARD_SLACK, max_rounds=1)
        except Exception:
            pass

    # -- walk-speed scalar ----------------------------------------------------------------
    def _scale_walk(self, egress_w, purewalk, walk_scalar, access_w=None):
        """Scale all WALK reference-seconds (access/egress/pure-walk, baked at 4.8 km/h) to the
        user's pace: walk_scalar = 4.8/v (slow 1.412, med 1.143, fast 0.923). ``access_w`` None
        scales the engine's baked grid access table; callers with their own access CSR
        (``commute_for_access``) pass theirs. A public API default of ``walk_scalar=1.0`` means
        the 4.8 km/h bake/reference pace, *not* the served Medium preset (which passes its
        calibrated scalar explicitly). Returns (egress_w, purewalk, access_w) as int64 at the
        user's pace."""
        ew = np.asarray(egress_w, np.int64); pw = np.asarray(purewalk, np.int64)
        aw = self.access_w if access_w is None else np.asarray(access_w, np.int64)
        if walk_scalar != 1.0:
            ew = np.rint(ew.astype(np.float64) * walk_scalar).astype(np.int64)
            pw = pw.copy(); m = pw >= 0
            pw[m] = np.rint(pw[m].astype(np.float64) * walk_scalar).astype(np.int64)
            aw = np.rint(aw.astype(np.float64) * walk_scalar).astype(np.int64)
        return ew, pw, aw

    def _data_for_walk_scalar(self, walk_scalar):
        """RAPTOR data with transfer footpath seconds scaled to the user's walking pace."""
        scalar = float(walk_scalar)
        if abs(scalar - 1.0) < 1e-9:
            return self.data
        key = round(scalar, 6)
        cached = self._walk_scaled_data.get(key)
        if cached is not None:
            return cached
        data = dict(self.data)
        # Graph-baked timing keeps physical walk, fixed GTFS minimum, and effective time
        # separate.  Only ordinary physical walking scales; minimum dwell and authoritative
        # pathway traversal remain reference-authoritative.  The legacy fallback preserves
        # synthetic callers and old in-memory fixtures that have only tr_time.
        if {"tr_walk_time", "tr_min_time", "tr_path_fallback"}.issubset(self.data):
            def _scale_view(prefix):
                walk = np.asarray(self.data[f"{prefix}walk_time"], dtype=np.float64)
                minimum = np.asarray(self.data[f"{prefix}min_time"], dtype=np.float64)
                pathway = np.asarray(self.data[f"{prefix}path_fallback"], dtype=bool)
                # Artifact timing is already integer-rounded at scalar 1.  For non-unit pace,
                # apply the same half-up rounding once to physical walking only.
                scaled_walk = np.where(pathway, walk, np.floor(walk * scalar + 0.5))
                scaled_effective = np.floor(np.maximum(scaled_walk, minimum) + 0.5).astype(np.int64)
                return scaled_walk, scaled_effective
            scaled_walk, scaled = _scale_view("tr_")
            data["tr_walk_time"] = scaled_walk
            data["tr_time"] = scaled
            if {"tr_forward_walk_time", "tr_forward_min_time", "tr_forward_path_fallback"}.issubset(self.data):
                forward_walk, forward_time = _scale_view("tr_forward_")
                data["tr_forward_walk_time"] = forward_walk
                data["tr_forward_time"] = forward_time
        else:
            base = np.asarray(self.data["tr_time"])
            scaled64 = np.rint(base.astype(np.float64) * scalar).astype(np.int64)
            scaled64 = np.where((base > 0) & (scaled64 < 1), 1, scaled64)
            data["tr_time"] = scaled64.astype(base.dtype, copy=False)
        self._walk_scaled_data[key] = data
        return data

    def _departafter_grids(self, walk_scalar, *, planned=False):
        """Return profile grids wide enough for the selected access-walk pace.

        The access table is capped in *reference* seconds.  A slower selected pace can therefore
        make its outermost allowed access leg longer than the reference ``dep_grid`` tail.  Extend
        both grids in that case so ``D + access_w`` remains representable; scalar 1.0 returns the
        original objects byte-for-byte for the reference/oracle path.
        """
        scalar = float(walk_scalar)
        if scalar <= 1.0:
            return self.dep_grid, self.Tgrid_planned if planned else self.Tgrid
        key = (round(scalar, 6), bool(planned))
        cached = getattr(self, "_walk_grid_cache", {}).get(key)
        if cached is not None:
            return cached
        access_tail = int(np.ceil(self.access_cap_min * 60 * scalar / DEP_STEP)) * DEP_STEP
        dep_grid = np.arange(self.dep_sec, self.dep_sec + self.win_sec + access_tail + 1,
                             DEP_STEP, dtype=np.int64)
        deadline_step = PLANNED_DEADLINE_STEP if planned else DEADLINE_STEP
        deadline_end = int(dep_grid[-1]) + self.max_min * 60
        deadlines = np.arange(self.dep_sec, deadline_end + deadline_step, deadline_step,
                              dtype=np.int64)
        cache = getattr(self, "_walk_grid_cache", None)
        if cache is None:
            cache = self._walk_grid_cache = {}
        cache[key] = (dep_grid, deadlines)
        return cache[key]

    # -- compute -------------------------------------------------------------------------
    def _reverse(self, egress_g, egress_w, deadlines, max_rounds=MAX_ROUNDS, data=None):
        data = self.data if data is None else data
        return R.reverse_profile(data, egress_g, egress_w, deadlines,
                                 board_slack=BOARD_SLACK, max_rounds=max_rounds)

    def commute_for_access(self, access_off, access_to, access_w, egress_g, egress_w, purewalk,
                           semantic="departafter", percentiles=(5, 50), max_rounds=MAX_ROUNDS,
                           walk_scalar=1.0, target_sec=None, window_sec=None,
                           walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS):
        """Door-to-door commute minutes for a CALLER-SUPPLIED origin set (all three semantics).

        The public API for off-grid origins (e.g. scripts/commute_origins.py): the caller builds
        its own CSR access table (origin -> stop walk REFERENCE seconds at config.WALK_KMH,
        capped like the baked table at ``self.access_cap_min`` so the engine's departure grids
        cover every boarding) + per-origin ``purewalk`` (origin->W reference seconds, -1 if
        unwalkable). Only ``semantic='planned'`` runs the SERVED PRODUCT MODEL on this branch;
        the other two are the legacy/validation models (see below) — callers that must match the
        live map must ask for 'planned' explicitly.

          semantic   'planned'     — THE SERVED PRODUCT MODEL (this branch): the first-boarding-
                     anchored scheduled depart-after value (``raptor.select_planned_departafter``
                     over the SAME inputs the served map uses — walk-scalar-scaled transfer
                     footpaths via ``_data_for_walk_scalar``, the minute-resolution
                     ``Tgrid_planned`` sweep, ``stop_arrival_profile`` on the 60s ``dep_grid``,
                     mirroring ``journey_tree_departafter(planned=True)``). ONE scheduled value
                     per origin, broadcast into every requested percentile column exactly like the
                     served ``[scheduled, scheduled]`` cells; ``percentiles`` selects only the
                     column count and ``walk_reluctance``/``walk_prior_eps`` are inert (the
                     planned selection takes no walk prior).
                     'departafter' — p-percentiles over the [DEP, DEP+WINDOW] departure window:
                     the LEGACY comparison model retained for offline validation, NOT
                     what the map serves on this branch;
                     'arriveby'    — perfect-timing arrive-by window ending at ``target_sec``
                     (default 09:00; ``window_sec`` None -> config.window(), 0 -> single deadline).
          walk_scalar  4.8/pace scalar applied to ALL walk legs, including transfer footpaths.
                     ``1.0`` deliberately denotes the 4.8 km/h bake/reference pace, not the
                     served Medium preset; product callers pass Medium's calibrated scalar.

        Returns int32[n_origins, n_pct] minutes, -1 = unreachable."""
        access_off = np.asarray(access_off, np.int64)
        access_to = np.asarray(access_to, np.int32)
        egress_g = np.asarray(egress_g, np.int32)
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar, access_w=access_w)
        if semantic == "planned":
            # The served product model. Build the same truthful traced tree the map uses instead
            # of returning the profile kernel's deadline-rounded provisional values. Planned
            # publication validates boardability and ranks concrete chains by their exact raw
            # clock seconds; using that single source here keeps public off-grid callers in
            # lockstep with map/hover even when a profile deadline contains sub-minute slack.
            data = self._data_for_walk_scalar(walk_scalar)
            dep_grid, deadlines = self._departafter_grids(walk_scalar, planned=True)
            latest = self._reverse(egress_g, ew, deadlines, max_rounds, data=data)
            arrivalW = R.stop_arrival_profile(latest, deadlines, dep_grid)
            tree = raptor_journey.DepartAfterJourneyTree(
                data, access_off, access_to, aw, pw, arrivalW,
                dep_grid, self.cell_deps, self.max_min, egress_g, ew,
                percentile=50.0, walk_reluctance=walk_reluctance,
                walk_prior_eps=walk_prior_eps, max_rounds=max_rounds,
                board_slack=BOARD_SLACK, planned=True)
            painted = tree.commute()
            # one scheduled value; broadcast across the requested percentile columns exactly like
            # the served map's [scheduled, scheduled] cells (return contract: int32[n, n_pct]).
            npct = max(1, np.atleast_1d(np.asarray(percentiles)).shape[0])
            return np.repeat(np.asarray(painted, np.int32)[:, None], npct, axis=1)
        if semantic == "departafter":
            data = self._data_for_walk_scalar(walk_scalar)
            dep_grid, deadlines = self._departafter_grids(walk_scalar)
            latest = self._reverse(egress_g, ew, deadlines, max_rounds, data=data)
            arrivalW = R.stop_arrival_profile(latest, deadlines, dep_grid)
            return R.assemble_departafter(access_off, access_to, aw, pw, arrivalW,
                                          dep_grid, self.cell_deps, self.max_min,
                                          percentiles=percentiles, beta=walk_reluctance,
                                          eps=walk_prior_eps)
        if semantic != "arriveby":
            raise ValueError(
                f"semantic must be 'planned', 'departafter' or 'arriveby', got {semantic!r}")
        target = self.target_sec if target_sec is None else int(target_sec)
        win = int(config.window().total_seconds()) if window_sec is None else int(window_sec)
        deadlines = (np.array([target], np.int64) if win <= 0
                     else np.arange(target - win, target + 1, DEP_STEP, dtype=np.int64))
        data = self._data_for_walk_scalar(walk_scalar)
        latest = self._reverse(egress_g, ew, deadlines, max_rounds, data=data)
        return _assemble_arriveby_window(access_off, access_to, aw, pw, latest, deadlines,
                                         self.max_min, np.asarray(percentiles, np.float64),
                                         beta=walk_reluctance, eps=walk_prior_eps)

    def departafter(self, egress_g, egress_w, purewalk, percentiles=(5, 50),
                    max_rounds=MAX_ROUNDS, walk_scalar=1.0, walk_reluctance=WALK_RELUCTANCE,
                    walk_prior_eps=WALK_PRIOR_EPS, planned=False):
        """{cell_id: [p5, p50]} minutes over the depart-after window.

        Default (``planned=False``) = the LEGACY percentile model: p5/p50 over the
        [DEP, DEP+WINDOW] departure window. This path remains stable for offline
        validation. At its reference ``walk_scalar=1.0`` it stays byte-identical for
        validation/existing callers; other scalars now correctly scale every walk leg. It is no
        longer what the map serves on this branch.

        ``planned=True`` = THE SERVED PRODUCT MODEL on this branch: the first-boarding-anchored
        scheduled value (``semantic='planned'`` in ``commute_for_access`` — the SAME grids/inputs
        ``journey_tree_departafter(planned=True)``/the served map use), broadcast into every
        percentile slot exactly like the served ``[scheduled, scheduled]`` cells;
        ``walk_reluctance``/``walk_prior_eps`` are inert under it.

        ``purewalk`` is cell->W walk seconds aligned to self.cell_ids (-1 if > cap).
        ``max_rounds`` caps public-transport rides (rides = transfers + 1). ``walk_scalar`` sets
        the walk pace; ``walk_reluctance``/``walk_prior_eps`` the mild walk prior (decision-only,
        legacy path)."""
        out = self.commute_for_access(self.access_off, self.access_to, self.access_w,
                                      egress_g, egress_w, purewalk,
                                      semantic="planned" if planned else "departafter",
                                      percentiles=percentiles, max_rounds=max_rounds,
                                      walk_scalar=walk_scalar, walk_reluctance=walk_reluctance,
                                      walk_prior_eps=walk_prior_eps)
        return {c: [int(out[i, k]) if out[i, k] >= 0 else None
                    for k in range(out.shape[1])] for i, c in enumerate(self.cell_ids)}

    def arriveby(self, egress_g, egress_w, purewalk, target_sec=None, window_sec=None,
                 percentiles=(5, 50), max_rounds=MAX_ROUNDS, walk_scalar=1.0,
                 walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS):
        """{cell_id: [p5, p50]} minutes, arrive-by an arrival window ending at ``target_sec``
        (default 09:00). ``window_sec`` None -> use config.window(); 0 -> single deadline.
        ``max_rounds`` caps public-transport rides (rides = transfers + 1). ``walk_reluctance``/
        ``walk_prior_eps`` are the mild walk prior (decision-only)."""
        out = self.commute_for_access(self.access_off, self.access_to, self.access_w,
                                      egress_g, egress_w, purewalk, semantic="arriveby",
                                      percentiles=percentiles, max_rounds=max_rounds,
                                      walk_scalar=walk_scalar, target_sec=target_sec,
                                      window_sec=window_sec, walk_reluctance=walk_reluctance,
                                      walk_prior_eps=walk_prior_eps)
        return {c: [int(out[i, k]) if out[i, k] >= 0 else None
                    for k in range(out.shape[1])] for i, c in enumerate(self.cell_ids)}


    # -- Phase 2: traced arrive-by tree -> journey breakdown + color-by-line ---------------
    def journey_tree(self, egress_g, egress_w, purewalk, target_sec=None, max_rounds=MAX_ROUNDS,
                     walk_scalar=1.0, walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS):
        """A JourneyTree for the single arrive-by deadline: serves the per-cell breakdown
        (hover), color-by-line, AND the arrive-by map value (actual commute = arrival - latest
        home departure), all from ONE traced reverse tree so hover == map by construction.
        ``walk_reluctance`` (decision-only) steers the access-stop choice toward less walking on
        ties (drives hover + map + color-by-line + the committed-MC first legs)."""
        target = self.target_sec if target_sec is None else int(target_sec)
        egress_g = np.asarray(egress_g, np.int32)
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar)
        data = self._data_for_walk_scalar(walk_scalar)
        par = R.reverse_raptor_traced_fast(
            data, egress_g, target - ew, ew,
            max_rounds=max_rounds, board_slack=BOARD_SLACK)
        return raptor_journey.JourneyTree(data, par, self.access_off, self.access_to,
                                          aw, pw, target, self.max_min,
                                          walk_reluctance=walk_reluctance,
                                          walk_prior_eps=walk_prior_eps,
                                          egress_g=egress_g, egress_w=ew)

    # -- depart-after traced tree -> hover==map breakdown + color-by-line ------------------
    def journey_tree_departafter(self, egress_g, egress_w, purewalk, percentile=50.0,
                                 max_rounds=MAX_ROUNDS, walk_scalar=1.0,
                                 walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS,
                                 planned=False):
        """A ``DepartAfterJourneyTree`` for the depart-after window. With ``planned=True`` it serves
        the first-boarding-anchored scheduled value used by the product; otherwise it preserves the
        legacy percentile mode used by validation tests.

        Mirrors ``journey_tree`` (the arrive-by sibling) but is driven by the depart-after window
        instead of a single 09:00 deadline: it reuses THIS engine's depart-after value computation
        (the SAME ``reverse_profile`` + ``stop_arrival_profile`` + grids ``departafter`` uses), so
        the tree's painted grid equals ``self.departafter(...)`` for that percentile EXACTLY. The
        tracer then builds one ``reverse_raptor_traced`` tree per representative arrival deadline T*
        (only ~15-17 distinct per workplace) and reads each cell's journey from its painted access
        stop s*. ``walk_scalar`` is applied end-to-end, including transfer footpaths, so access,
        transfer, egress, and pure walking follow the same pace. JVM-free."""
        egress_g = np.asarray(egress_g, np.int32)
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar)
        data = self._data_for_walk_scalar(walk_scalar)
        # Planned/free-departure displays compare concrete scheduled branches in the pinned card.
        # Use the minute grid here so the map-selected primary and branch enumeration score against
        # the same resolution as the displayed leg times; the legacy percentile mode keeps the
        # coarser validated grid.
        dep_grid, deadlines = self._departafter_grids(walk_scalar, planned=planned)
        latest = self._reverse(egress_g, ew, deadlines, max_rounds, data=data)
        arrivalW = R.stop_arrival_profile(latest, deadlines, dep_grid)
        return raptor_journey.DepartAfterJourneyTree(
            data, self.access_off, self.access_to, aw, pw, arrivalW,
            dep_grid, self.cell_deps, self.max_min, egress_g, ew,
            percentile=float(percentile), walk_reluctance=walk_reluctance,
            walk_prior_eps=walk_prior_eps, max_rounds=max_rounds, board_slack=BOARD_SLACK,
            planned=planned)

    # -- Phase A: service-noise Monte-Carlo (realistic + fragility + alt-lines) ------------
    def _mc_mode_params(self):
        """Per-pattern (mean initial delay sec, fractional drift slope) from mode + operator.
        BART/Caltrain are resolved by feed STEM (``data["feeds"]``, e.g. 'bart_gtfs'), never by
        position: config.gtfs_paths() drops missing files and appends ``extra`` feeds, so a fixed
        index could silently hand BART's tight rail profile to a bus feed. A feed whose stem
        matches neither keeps its mode-bucket default (bus/metro/cable)."""
        pf = np.asarray(self.data["pat_feed"]); pm = np.asarray(self.data["pat_mode"])
        mu = np.full(len(pm), _MU["bus"], np.float64); sl = np.full(len(pm), _SLOPE["bus"], np.float64)
        mu[pm == 0] = _MU["metro"]; sl[pm == 0] = _SLOPE["metro"]      # Muni Metro
        mu[pm == 2] = _MU["cable"]; sl[pm == 2] = _SLOPE["cable"]      # cable/streetcar
        stems = [str(s).lower() for s in self.data["feeds"]]
        for op in ("bart", "caltrain"):                                # operator noise by feed stem
            fis = [fi for fi, s in enumerate(stems) if op in s]
            if fis:
                m = np.isin(pf, fis)
                mu[m] = _MU[op]; sl[m] = _SLOPE[op]
        return mu, sl

    def _mc_draw_arrays(self, nR, seed):
        """Per-trip service-delay draw arrays (delta0, slope) for the committed MC, shared by
        ``montecarlo`` and ``route_typicals`` so a per-route typical is drawn from the SAME
        distribution as the served realistic map — with the same seed the primary's per-route
        typical is then byte-identical to ``realistic``. One source so the two can never drift.
        Returns (delta0_all, slope_all), each float64[nR, n_trips]."""
        mu_pat, slope_pat = self._mc_mode_params()
        ntrips = np.asarray(self.data["pat_ntrips"], dtype=np.int64)
        mu_trip = np.repeat(mu_pat, ntrips)
        slope_trip = np.repeat(slope_pat, ntrips)
        rng = np.random.default_rng(seed)
        k = MC_SHAPE
        T = mu_trip.shape[0]
        delta0_all = rng.gamma(k, mu_trip / k, size=(nR, T))
        slope_all = rng.gamma(k, np.maximum(slope_trip, 1e-9) / k, size=(nR, T))
        return delta0_all, slope_all

    def montecarlo(self, egress_g, egress_w, purewalk, perfect=None, n_draws=None,
                   seed=None, alt_draws=None, walk_scalar=1.0, max_rounds=MAX_ROUNDS, tree=None,
                   walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS,
                   capture_scenario=False, perf=None):
        """Committed-plan service-noise MC for a workplace. Returns dict of cell-aligned arrays:
          realistic int32  p50 door-to-door commute over draws FLOORED at ``perfect`` (so
                           realistic/frag/std/stuck all describe one consistent distribution
                           and realistic >= perfect holds by construction)
          realistic_raw int32  pre-floor p50 — the non-vacuous perfect<=committed regression
                           signal for tests/validators (never served)
          frag      int32  p90-p50 "bad-day delta" minutes (the headline fragility number;
                           arrive-by serves it directly because its headline IS committed p50)
          committed_p90 int32  committed bad-day ABSOLUTE minute (round(p90) of the floored draws).
                           Depart-after derives its frag = committed_p90 - served_p50 off this (its
                           headline is the bare served p50, not committed p50); arrive-by ignores it.
          std       int32  commute std minutes (secondary)
          stuck     float  fraction of draws where the cell hits the cap (last-train/peak risk)
          alt       list[dict|None]  {line: min_minutes} alternative lines by a DOMINANCE WINDOW
                                     over the UNPERTURBED tree's per-access-stop journeys — every
                                     distinct transit line whose best door-to-door time is within
                                     ALT_WINDOW_MIN of the cell's best (deterministic + walk-speed-
                                     stable; sorted closest-first). The server drops the PRIMARY line
                                     + caps at 4. See ``_alt_window`` + ``JourneyTree.alt_lines_window``.
          alt_bundle dict|None  the geometry handle for the drawn alternatives (None when
                                 alt_draws<=0): ``{"alt_stop": list[{line: access_stop}|None],
                                 "draws": []}``. The alt routes trace from the SAME (unperturbed)
                                 tree via ``JourneyTree.itinerary_via_stop(ci, alt_stop[ci][line])``
                                 — no perturbed schedules, no re-trace; ``draws`` is empty (the tree,
                                 cached by the caller, IS the source).
        You commit the first leg from the published plan and re-optimize the tail from the actual
        late arrival -> the honest "I missed my transfer and ate a headway" cost. The committed plan
        + ``perfect`` map come from the SAME unperturbed arrive-by tree; the caller can pass an
        already-built one (the server reuses its cached tree -> no re-trace). Lazy + cached per
        workplace by the caller; never on the hover path. ``seed`` makes it reproducible."""
        t_phase = time.perf_counter() if perf is not None else None
        nR = MC_DRAWS if n_draws is None else int(n_draws)
        egress_g = np.asarray(egress_g, np.int32)
        ew, _pw, _aw = self._scale_walk(egress_w, purewalk, walk_scalar)   # ew feeds the committed kernel
        delta0_all, slope_all = self._mc_draw_arrays(nR, seed)
        if tree is None:
            tree = self.journey_tree(egress_g, egress_w, purewalk, max_rounds=max_rounds,
                                     walk_scalar=walk_scalar, walk_reluctance=walk_reluctance,
                                     walk_prior_eps=walk_prior_eps)
        legs = tree.committed_first_legs()
        if perfect is None:
            perfect, _ = tree.commute_and_dominant()
        perfect = np.asarray(perfect, np.int32)          # always materialized (tree fallback above)
        _perf_add(perf, "variance.committed_ms", t_phase)
        t_phase = time.perf_counter() if perf is not None else None
        # Truncate the MC tail-readout grid: any deadline beyond max(commit_home) + cap yields
        # tt > cap -> capf for EVERY cell regardless of the draw, so dropping those deadlines is
        # provably output-identical (including `stuck`) and trims the kernel's dominant cost (the
        # per-draw sweep; how many deadlines survive is workplace-dependent — later committed
        # departures keep more of the grid). Computed HERE, after committed_first_legs(), so the
        # bound sees the walk-scalar-scaled legs; _make_grids' Tgrid_mc is untouched (tests pass
        # engine.Tgrid_mc straight to the kernel). Only transit cells (commit_kind == 2) carry a
        # real commit_home (kind 0 holds NEG); deterministic-only batches need one non-empty column.
        Tmc = _committed_deadline_prefix(self.Tgrid_mc, legs, self.max_min)
        # the kernel's cummax + _first_ge readout require an ascending grid (np.arange +
        # prefix mask preserve this; assert in case a future caller breaks the invariant)
        assert Tmc.size and np.all(np.diff(Tmc) > 0), "MC deadline grid must be ascending"
        data = getattr(tree, "d", self.data)
        captured = R.montecarlo_commute_committed(
            data, egress_g, ew, Tmc, legs, perfect, self.max_min, delta0_all, slope_all,
            board_slack=BOARD_SLACK, max_rounds=max_rounds, capture_tail=capture_scenario)
        _perf_add(perf, "variance.kernel_ms", t_phase)
        t_phase = time.perf_counter() if perf is not None else None
        scenario = None
        if capture_scenario:
            cm, tail_lag, tail_valid = captured
            if bool(np.all(np.asarray(tail_valid) == 1)):
                scenario = MonteCarloScenario(
                    tail_lag=tail_lag, deadlines=Tmc, delta0_all=delta0_all,
                    slope_all=slope_all, seed=seed, max_rounds=max_rounds,
                    board_slack=BOARD_SLACK, max_min=self.max_min,
                    walk_scalar=walk_scalar, data=data, egress_g=egress_g, egress_w=ew)
        else:
            cm = captured
        # The helper makes ONE sorted pass over each row, then uses monotonicity of the perfect
        # floor to read both the raw diagnostic p50 and floored public stats exactly.  The floored
        # quartet remains self-consistent: "bad day = realistic + frag" reads the floored p90.
        # Draws can legitimately dip below perfect: the tail re-optimizes earliest-arrival over
        # the deadline grid where the traced tree optimized latest-departure — small + two-sided,
        # see the zero-perturbation test.
        (realistic_raw, realistic, frag, committed_p90, std,
         stuck) = _mc_summary_from_draws(cm, perfect, self.max_min)
        _perf_add(perf, "variance.stats_ms", t_phase)
        t_phase = time.perf_counter() if perf is not None else None
        # committed_p90 (rounded) — the committed bad-day ABSOLUTE minute. Arrive-by reconciles its
        # chip as headline(=realistic=committed_p50)+frag; depart-after's headline is the BARE served
        # p50 (the painted floor, != committed_p50 wherever committed_p50 drifted above the floor), so
        # it can't use frag=p90-p50 — it derives frag = committed_p90 - served_p50 off this absolute
        # so served_p50 + frag == committed_p90 exactly. Exposed for that depart-after caller;
        # arrive-by ignores it and keeps `frag` byte-identical.
        alt, alt_bundle = self._alt_window(tree, perfect,
                                           MC_ALT_DRAWS if alt_draws is None else int(alt_draws))
        _perf_add(perf, "variance.planned_alts_ms", t_phase)
        if perf is not None:
            perf["variance.draws"] = nR
            perf["variance.deadlines"] = int(Tmc.size)
        result = dict(realistic=realistic, realistic_raw=realistic_raw, frag=frag,
                      committed_p90=committed_p90, std=std,
                      stuck=stuck, alt=alt, alt_bundle=alt_bundle)
        return (result, scenario) if capture_scenario else result

    def _alt_window(self, tree, perfect, alt_draws):
        """Alternatives by a DOMINANCE WINDOW over the UNPERTURBED arrive-by tree (NOT a draw-vote).
        Returns (alt, alt_bundle):
          alt        list[dict|None]  per cell ``{line: min_minutes}`` — every distinct transit line
                     whose best per-access-stop door-to-door time is within ``ALT_WINDOW_MIN`` of the
                     cell's best, sorted closest-first (the PRIMARY line is dropped by the server).
          alt_bundle dict  the geometry handle for /itinerary's drawn alternatives — it carries the
                     per-cell ``{line: access_stop}`` map (``alt_stop``) so the route is traced from
                     the SAME tree via ``JourneyTree.itinerary_via_stop`` (no perturbed re-trace,
                     no separate bundle of schedules). ``draws`` is empty (the tree IS the source).

        ``alt_draws <= 0`` disables alts (returns (None, None)), preserving the old knob's "off"
        semantics for callers/tests. (The old K perturbation draws no longer drive alts — the window
        is K-free and deterministic; the realistic/fragility MC keeps its own draws.)"""
        if alt_draws <= 0:
            return None, None
        # Planned depart-after intentionally uses a wider "also useful" window. The headline can
        # be a timed walk or walk-heavy route, but riders still expect nearby transit options to
        # stay visible for orientation and choice.
        window = PLANNED_ALT_WINDOW_MIN if getattr(tree, "planned", False) else ALT_WINDOW_MIN
        win = tree.alt_lines_window(perfect, window)   # per cell {line/signature: (min, stop)}
        alt = [None] * len(win)
        alt_stop = [None] * len(win)
        for ci, d in enumerate(win):
            if not d:
                continue
            alt[ci] = {ln: int(ms[0]) for ln, ms in d.items()}
            alt_stop[ci] = {ln: int(ms[1]) for ln, ms in d.items()}
        bundle = {"alt_stop": alt_stop, "draws": []}
        return alt, bundle

    def route_typicals(self, tree, ci, stops, egress_g, egress_w, perfect_route_mins=None,
                       n_draws=None, seed=None, walk_scalar=1.0, max_rounds=MAX_ROUNDS,
                       return_committed_p90=False, scenario=None):
        """Per-ROUTE committed-plan TYPICAL (p50) + FRAGILITY (p90-p50) for ONE pinned cell ``ci``.

        ``stops`` is the list of access stops (gids) of the routes to score — the PRIMARY's selected
        stop first, then each alternative's access stop (from ``alt_lines_window`` / the MC alt
        bundle). Each route's committed first leg is extracted from the journey traced FROM its stop
        (``JourneyTree.committed_legs_via_stops``), so every route is scored with the SAME committed
        Monte-Carlo as the served ``realistic`` map (catch the next trip on the committed line, ride
        to the committed alight, re-optimize the tail from the actual late arrival).

        Efficiency: all routes for the cell are ONE combined committed-leg batch -> a SINGLE
        ``montecarlo_commute_committed`` call, so the R per-draw reverse profiles (the dominant cost,
        cell-independent) are computed once and shared across every route. Lazy + cached by the caller
        per pinned cell; NEVER on the hover path. Serialized under ``raptor._MC_KERNEL_LOCK`` (inside
        ``montecarlo_commute_committed``).

        ``perfect_route_mins`` (optional, aligned to ``stops``) is each route's best-case door-to-door
        minutes; the returned typical is FLOORED at it so ``perfect <= committed`` holds PER ROUTE.

        Returns list[(real_min, frag_min) | None] aligned to ``stops`` — None for a route whose stop
        is unreachable / off-cell (so the caller falls back to that route's best-case). With
        ``return_committed_p90=True`` each tuple is (real_min, frag_min, committed_p90_min) so the
        depart-after caller can derive frag = committed_p90 - that route's served p50 (its displayed
        typical is the bare served p50, not committed p50); arrive-by leaves the default 2-tuple
        untouched (byte-identical)."""
        stops = [int(s) for s in stops]
        if not stops:
            return []
        nR = MC_DRAWS if n_draws is None else int(n_draws)
        egress_g = np.asarray(egress_g, np.int32)
        ew, _pw, _aw = self._scale_walk(egress_w, np.zeros(1, np.int64), walk_scalar)
        legs = tree.committed_legs_via_stops(ci, stops)             # one row per route
        # per-route best-case floor (so committed >= perfect PER route); -1 where unknown/unreachable
        n = len(stops)
        if perfect_route_mins is None:
            perfect = np.full(n, -1, np.int64)
        else:
            perfect = np.array([int(p) if p is not None else -1 for p in perfect_route_mins],
                               np.int64)
        data = getattr(tree, "d", self.data)
        Tmc = _committed_deadline_prefix(self.Tgrid_mc, legs, self.max_min)
        reuse = (isinstance(scenario, MonteCarloScenario)
                 and scenario.compatible(
                     data=data, deadlines=Tmc, egress_g=egress_g, egress_w=ew, seed=seed,
                     n_draws=nR, max_rounds=max_rounds, board_slack=BOARD_SLACK,
                     max_min=self.max_min, walk_scalar=walk_scalar))
        if reuse:
            # The retained profile may cover a longer horizon than this small route batch. The
            # replay kernel uses ``len(Tmc)`` as its read bound, so pass the already-contiguous full
            # tensor and let it ignore the suffix. Materializing ``[:, :, :len(Tmc)]`` made an
            # 8-12MiB non-contiguous prefix copy on every first pin for no semantic benefit.
            cm = R.montecarlo_commute_committed_from_tail(
                data, Tmc, scenario.tail_lag, legs, perfect, self.max_min,
                scenario.delta0_all, scenario.slope_all, board_slack=BOARD_SLACK)
        else:
            # Same per-pattern delay model + RNG as montecarlo() (shared helper) so a compatible
            # seed remains byte-identical even when reuse is unavailable or deliberately rejected.
            delta0_all, slope_all = self._mc_draw_arrays(nR, seed)
            cm = R.montecarlo_commute_committed(
                data, egress_g, ew, Tmc, legs, perfect, self.max_min, delta0_all, slope_all,
                board_slack=BOARD_SLACK, max_rounds=max_rounds)
        # floor the draws at each route's own best-case BEFORE the percentile, mirroring montecarlo()
        floor = np.where(perfect >= 0, perfect, 0).astype(np.float64)
        cm = np.maximum(cm, floor[:, None])
        p50 = np.percentile(cm, 50, axis=1)
        p90 = np.percentile(cm, 90, axis=1)
        real = np.ceil(p50).astype(np.int32)
        frag = np.maximum(0, np.round(p90 - p50)).astype(np.int32)
        committed_p90 = np.round(p90).astype(np.int32)   # absolute bad-day minute (depart-after frag)
        kind = np.asarray(legs["commit_kind"])
        out = []
        for row in range(n):
            if kind[row] == 0:                # stop unreachable / off-cell -> no typical
                out.append(None)
            elif return_committed_p90:
                out.append((int(real[row]), int(frag[row]), int(committed_p90[row])))
            else:
                out.append((int(real[row]), int(frag[row])))
        return out


def _assemble_arriveby_window(access_off, access_to, access_w, purewalk, latest, deadlines,
                              max_min, percentiles, beta=1.0, eps=60.0):
    """Per cell, per arrival deadline T: travel(T) = T - latest_home_departure(cell, T);
    then percentile over the arrival window. ``beta``/``eps`` are the walk-reluctance multiplier +
    the true-time cap (decision-only): among access stops whose TRUE home is within ``eps`` of the
    latest one, the access stop maximizing the PENALIZED home ``home - (beta-1)*access_w`` is chosen,
    but the reported travel uses the TRUE chosen home, so the map minutes stay exact clock time.
    Returns int32[n_cells, n_pct] (-1 unreachable)."""
    n_cells = len(access_off) - 1
    nd = len(deadlines)
    deadlines = np.asarray(deadlines, np.int64)
    beta = float(beta); eps = float(eps)
    if R._select_kernel() == "numba":
        from . import raptor_numba
        return raptor_numba.assemble_arriveby(
            np.asarray(access_off, np.int64), np.asarray(access_to, np.int64),
            np.asarray(access_w, np.int64), np.asarray(purewalk, np.int64),
            latest, deadlines, np.int64(max_min), np.asarray(percentiles, np.float64),
            np.float64(beta), np.float64(eps))
    out = np.full((n_cells, len(percentiles)), -1, dtype=np.int32)
    bw = beta - 1.0
    epsi = int(round(eps))
    BIG = np.iinfo(np.int64).max
    for ci in range(n_cells):
        a0, a1 = int(access_off[ci]), int(access_off[ci + 1])
        gids = access_to[a0:a1]
        awalk = access_w[a0:a1].astype(np.int64)
        tt = np.full(nd, BIG, dtype=np.int64)              # TRUE travel of the chosen stop
        pen_tt = np.full(nd, np.inf, dtype=np.float64)     # penalized travel that drove the choice
        if len(gids):
            sub = latest[gids]                                # (nstops_cell, nd)
            home = sub - awalk[:, None]                       # latest home departure per stop,T
            reachable_each = home > R.NEG // 2
            opt = np.where(reachable_each, home, R.NEG).max(axis=0)    # time-optimal true home per T
            in_eps = reachable_each & (home >= opt[None, :] - epsi)    # eps window: ~same time only
            pen_home = home.astype(np.float64) - bw * awalk[:, None]   # penalized home (decision)
            pen_home = np.where(in_eps, pen_home, -np.inf)
            sel = np.argmax(pen_home, axis=0)                 # penalized argmax within the eps window
            best_home = home[sel, np.arange(nd)]
            reachable = opt > R.NEG // 2
            tt = np.where(reachable, deadlines - best_home, BIG)
            pen_tt = np.where(reachable, deadlines - pen_home[sel, np.arange(nd)], np.inf)
        pw = purewalk[ci]
        if pw >= 0:
            # pure walk competes on penalized travel (pw*beta) vs the chosen transit's penalized
            # travel; reports the TRUE pw seconds where it wins and the deadline allows it.
            walk_ok = (deadlines - pw) >= 0
            win = walk_ok & (pw * beta < pen_tt)
            tt = np.where(win, pw, tt)
        ttm = tt.astype(np.float64) / 60.0
        ttm = np.where(ttm < 0, 0.0, ttm)
        ttm = np.ceil(np.where(ttm > max_min, max_min, ttm))
        vals = np.percentile(ttm, percentiles, method="lower")
        out[ci] = [(-1 if v >= max_min else int(v)) for v in np.atleast_1d(vals)]
    return out
