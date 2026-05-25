"""RaptorEngine — the JVM-free, server-facing grid travel-time engine.

Loads the RAPTOR structures (raptor_build) + the baked cell->stop walk-access table
(raptor_oracle), and turns a workplace's egress/pure-walk (computed by the caller, e.g. one
R5 walk matrix, or any walk router) into per-cell door-to-door times. No r5py here.

Two semantics from the SAME reverse range-RAPTOR profile:
  * depart-after p5/p50 over the [DEP, DEP+WINDOW] departure window — bit-for-bit comparable to
    R5's departure-window percentile model, so this is what we validate against the R5 oracle.
  * arrive-by over an arrival window ending at the target (default [TARGET-WINDOW, TARGET]) —
    the product semantic ("where can you live and reach work by 9:00"), read off the same
    profile. Single-deadline arrive-by is the WINDOW=0 case.

The hot reverse sweep runs in the numba kernel automatically when available; assembly is numpy.
"""
import os
import numpy as np

from . import config, raptor_build, raptor as R, raptor_journey

ACCESS_CAP_MIN = int(os.environ.get("RAPTOR_ACCESS_CAP", "25"))  # 25 cleared the worst access-starved periphery (max err 15->7, mism 20->5) vs 20
DEADLINE_STEP = int(os.environ.get("RAPTOR_DEADLINE_STEP", "180"))
DEP_STEP = 60
BOARD_SLACK = int(os.environ.get("RAPTOR_BOARD_SLACK", "60"))
MAX_ROUNDS = 8                                    # rides = transfers + 1 (R5 default cap)
ARRIVE_BY_HM = (9, 0)                             # product target: arrive by 09:00

# --- service-noise Monte-Carlo (realistic + fragility + alt-lines) --------------------
MC_DRAWS = int(os.environ.get("RAPTOR_MC_DRAWS", "24"))       # draws for realistic/fragility
MC_ALT_DRAWS = int(os.environ.get("RAPTOR_MC_ALT_DRAWS", "4"))  # traced draws for alt-lines (pricier)
# committed-plan vs clairvoyant realistic. Committed (default) FIXES the first leg from the
# published plan then re-optimizes the tail from the actual late arrival -> a higher, more honest
# realistic + fragility. Clairvoyant (RAPTOR_MC_COMMITTED=0) re-optimizes the whole journey with
# foreknowledge -> an optimistic lower bound (route resilience). See RAPTOR.md.
MC_COMMITTED = os.environ.get("RAPTOR_MC_COMMITTED", "1").lower() in ("1", "true", "yes", "on")
MC_SHAPE = float(os.environ.get("RAPTOR_MC_SHAPE", "2.0"))   # Gamma shape (spread); mean=shape*scale
# mean INITIAL delay (sec) + fractional DRIFT slope, by mode/operator (env-overridable). Buses are
# the noisiest, rail the steadiest; Caltrain/BART keyed on feed (BART=1, Caltrain=2).
_MU = dict(bus=float(os.environ.get("RAPTOR_MC_MU_BUS", "70")),
           metro=float(os.environ.get("RAPTOR_MC_MU_METRO", "45")),
           cable=float(os.environ.get("RAPTOR_MC_MU_CABLE", "40")),
           bart=float(os.environ.get("RAPTOR_MC_MU_BART", "25")),
           caltrain=float(os.environ.get("RAPTOR_MC_MU_CALTRAIN", "40")))
# fractional DRIFT (delay grows with time on the vehicle); the most uncertain knob, kept modest
# so a long peripheral bus ride doesn't over-inflate the median (validated vs R5 p50 bias).
_SLOPE = dict(bus=0.035, metro=0.025, cable=0.03, bart=0.012, caltrain=0.02)


class RaptorEngine:
    def __init__(self, gtfs_paths=None, service_date=None, access_path=None,
                 access_cap_min=ACCESS_CAP_MIN, verbose=True):
        gtfs_paths = gtfs_paths or config.gtfs_paths()
        from . import feeds
        self.service_date = service_date or feeds.pick_service_date(gtfs_paths)
        self.data = raptor_build.load_or_build(gtfs_paths, self.service_date, verbose=verbose)
        self.access_cap_min = access_cap_min
        self._load_access(access_path, verbose)
        self._make_grids()
        self._warm()

    # -- access table (baked, workplace-independent) --------------------------------------
    def _load_access(self, access_path, verbose):
        if access_path is None:
            fp = raptor_build._fingerprint(
                config.gtfs_paths(), self.service_date.strftime("%Y%m%d"),
                raptor_build.band_seconds(), raptor_build.FOOTPATH_M)
            cands = sorted(raptor_build.CACHE_DIR.glob(f"access_*m_{fp}.npz"))
            if not cands:
                raise FileNotFoundError(
                    f"no baked access table for fingerprint {fp} in {raptor_build.CACHE_DIR}; "
                    f"run scripts/raptor_oracle.py")
            access_path = cands[0]
        z = np.load(access_path, allow_pickle=True)
        if int(z["n_stops"]) != self.data["n_stops"]:
            raise ValueError("access table / raptor structures stop-count mismatch (stale cache)")
        self.cell_ids = list(z["cell_ids"].astype(str))
        self.cell_index = {c: i for i, c in enumerate(self.cell_ids)}
        # filter to the access cap and re-pack CSR (seconds)
        off = z["access_off"]; to = z["access_to"]; w = z["access_w"]
        cap = self.access_cap_min * 60
        n = len(self.cell_ids)
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

    def _warm(self):
        """Compile the numba kernel once (cheap dummy) so the first real request is fast."""
        eg = self.access_to[:1] if len(self.access_to) else np.zeros(1, np.int32)
        try:
            R.reverse_profile(self.data, eg, np.array([0], np.int64),
                              self.Tgrid[:2], board_slack=BOARD_SLACK, max_rounds=1)
        except Exception:
            pass

    # -- walk-speed scalar ----------------------------------------------------------------
    def _scale_walk(self, egress_w, purewalk, walk_scalar):
        """Scale all WALK reference-seconds (access/egress/pure-walk, baked at 4.8 km/h) to the
        user's pace: walk_scalar = 4.8/v (slow 1.20, med 1.00, fast 0.857). Returns
        (egress_w, purewalk, access_w) as int64 at the user's pace."""
        ew = np.asarray(egress_w, np.int64); pw = np.asarray(purewalk, np.int64)
        aw = self.access_w
        if walk_scalar != 1.0:
            ew = np.rint(ew.astype(np.float64) * walk_scalar).astype(np.int64)
            pw = pw.copy(); m = pw >= 0
            pw[m] = np.rint(pw[m].astype(np.float64) * walk_scalar).astype(np.int64)
            aw = np.rint(self.access_w.astype(np.float64) * walk_scalar).astype(np.int64)
        return ew, pw, aw

    # -- compute -------------------------------------------------------------------------
    def _reverse(self, egress_g, egress_w, deadlines, max_rounds=MAX_ROUNDS):
        return R.reverse_profile(self.data, egress_g, egress_w, deadlines,
                                 board_slack=BOARD_SLACK, max_rounds=max_rounds)

    def departafter(self, egress_g, egress_w, purewalk, percentiles=(5, 50),
                    max_rounds=MAX_ROUNDS, walk_scalar=1.0):
        """{cell_id: [p5, p50]} minutes, depart-after window (R5-comparable). ``purewalk`` is
        cell->W walk seconds aligned to self.cell_ids (-1 if > cap). ``max_rounds`` caps
        public-transport rides (rides = transfers + 1). ``walk_scalar`` sets the walk pace."""
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar)
        latest = self._reverse(egress_g, ew, self.Tgrid, max_rounds)
        arrivalW = R.stop_arrival_profile(latest, self.Tgrid, self.dep_grid)
        out = R.assemble_departafter(self.access_off, self.access_to, aw, pw, arrivalW,
                                     self.dep_grid, self.cell_deps, self.max_min,
                                     percentiles=percentiles)
        return {c: [int(out[i, k]) if out[i, k] >= 0 else None
                    for k in range(out.shape[1])] for i, c in enumerate(self.cell_ids)}

    def arriveby(self, egress_g, egress_w, purewalk, target_sec=None, window_sec=None,
                 percentiles=(5, 50), max_rounds=MAX_ROUNDS, walk_scalar=1.0):
        """{cell_id: [p5, p50]} minutes, arrive-by an arrival window ending at ``target_sec``
        (default 09:00). ``window_sec`` None -> use config.window(); 0 -> single deadline.
        ``max_rounds`` caps public-transport rides (rides = transfers + 1)."""
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar)
        target = self.target_sec if target_sec is None else int(target_sec)
        win = int(config.window().total_seconds()) if window_sec is None else int(window_sec)
        deadlines = (np.array([target], np.int64) if win <= 0
                     else np.arange(target - win, target + 1, DEP_STEP, dtype=np.int64))
        latest = self._reverse(egress_g, ew, deadlines, max_rounds)
        out = _assemble_arriveby_window(self.access_off, self.access_to, aw, pw, latest, deadlines,
                                        self.max_min, np.asarray(percentiles, np.float64))
        return {c: [int(out[i, k]) if out[i, k] >= 0 else None
                    for k in range(out.shape[1])] for i, c in enumerate(self.cell_ids)}


    # -- Phase 2: traced arrive-by tree -> journey breakdown + color-by-line ---------------
    def journey_tree(self, egress_g, egress_w, purewalk, target_sec=None, max_rounds=MAX_ROUNDS,
                     walk_scalar=1.0):
        """A JourneyTree for the single arrive-by deadline: serves the per-cell breakdown
        (hover), color-by-line, AND the arrive-by map value (actual commute = arrival - latest
        home departure), all from ONE traced reverse tree so hover == map by construction."""
        target = self.target_sec if target_sec is None else int(target_sec)
        egress_g = np.asarray(egress_g, np.int32)
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar)
        par = R.reverse_raptor_traced(self.data, egress_g, target - ew, ew,
                                      max_rounds=max_rounds, board_slack=BOARD_SLACK)
        return raptor_journey.JourneyTree(self.data, par, self.access_off, self.access_to,
                                          aw, pw, target, self.max_min)

    # -- Phase A: service-noise Monte-Carlo (realistic + fragility + alt-lines) ------------
    def _mc_mode_params(self):
        """Per-pattern (mean initial delay sec, fractional drift slope) from mode + operator."""
        pf = np.asarray(self.data["pat_feed"]); pm = np.asarray(self.data["pat_mode"])
        mu = np.full(len(pm), _MU["bus"], np.float64); sl = np.full(len(pm), _SLOPE["bus"], np.float64)
        mu[pm == 0] = _MU["metro"]; sl[pm == 0] = _SLOPE["metro"]      # Muni Metro
        mu[pm == 2] = _MU["cable"]; sl[pm == 2] = _SLOPE["cable"]      # cable/streetcar
        mu[pf == 1] = _MU["bart"]; sl[pf == 1] = _SLOPE["bart"]        # BART (feed 1)
        mu[pf == 2] = _MU["caltrain"]; sl[pf == 2] = _SLOPE["caltrain"]  # Caltrain (feed 2)
        return mu, sl

    def montecarlo(self, egress_g, egress_w, purewalk, perfect=None, n_draws=None,
                   seed=None, alt_draws=None, walk_scalar=1.0, committed=None,
                   max_rounds=MAX_ROUNDS, tree=None):
        """Service-noise MC for a workplace. Returns dict of cell-aligned arrays:
          realistic int32  p50 door-to-door commute over draws (clamped >= ``perfect``)
          frag      int32  p90-p50 "bad-day delta" minutes (the headline fragility number)
          std       int32  commute std minutes (secondary)
          stuck     float  fraction of draws where the cell hits the cap (last-train/peak risk)
          alt       list[dict|None]  {line: votes} alternative lines that serve the cell under
                                     delays (re-routes), from ``alt_draws`` traced perturbed trees
        ``committed`` (default ``MC_COMMITTED``=True): fix the first leg from the published plan and
        re-optimize the tail from the actual late arrival -> the honest "I missed my transfer and ate
        a headway" cost. ``committed=False`` is the clairvoyant lower bound (re-route the whole
        journey with foreknowledge). Lazy + cached per workplace by the caller; never on the hover
        path. ``seed`` makes it reproducible per workplace."""
        committed = MC_COMMITTED if committed is None else bool(committed)
        nR = MC_DRAWS if n_draws is None else int(n_draws)
        egress_g = np.asarray(egress_g, np.int32)
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar)
        mu_pat, slope_pat = self._mc_mode_params()
        ntrips = np.asarray(self.data["pat_ntrips"])
        mu_trip = np.repeat(mu_pat, ntrips); slope_trip = np.repeat(slope_pat, ntrips)
        rng = np.random.default_rng(seed)
        k = MC_SHAPE
        T = mu_trip.shape[0]
        delta0_all = rng.gamma(k, mu_trip / k, size=(nR, T))
        slope_all = rng.gamma(k, np.maximum(slope_trip, 1e-9) / k, size=(nR, T))
        if committed:
            # the committed plan + perfect map come from the SAME unperturbed arrive-by tree; the
            # caller can pass an already-built one (server reuses its cached tree -> no re-trace)
            if tree is None:
                tree = self.journey_tree(egress_g, egress_w, purewalk, max_rounds=max_rounds,
                                         walk_scalar=walk_scalar)
            legs = tree.committed_first_legs()
            if perfect is None:
                perfect, _ = tree.commute_and_dominant()
            cm = R.montecarlo_commute_committed(self.data, egress_g, ew, self.Tgrid, legs,
                                                np.asarray(perfect, np.int32), self.max_min,
                                                delta0_all, slope_all, board_slack=BOARD_SLACK,
                                                max_rounds=max_rounds)
        else:
            d_med = self.dep_sec + self.win_sec // 2
            cm = R.montecarlo_commute(self.data, egress_g, ew, self.Tgrid,
                                      self.access_off, self.access_to, aw, pw,
                                      np.int64(d_med), self.max_min, delta0_all, slope_all,
                                      board_slack=BOARD_SLACK, max_rounds=max_rounds)
        p50 = np.percentile(cm, 50, axis=1)
        p90 = np.percentile(cm, 90, axis=1)
        realistic = np.ceil(p50).astype(np.int32)
        frag = np.maximum(0, np.round(p90 - p50)).astype(np.int32)
        std = np.round(np.std(cm, axis=1)).astype(np.int32)
        stuck = np.mean(cm >= self.max_min - 1e-9, axis=1)
        if perfect is not None:
            perfect = np.asarray(perfect)
            m = perfect >= 0
            realistic[m] = np.maximum(realistic[m], perfect[m].astype(np.int32))
        alt = self._mc_alt_lines(egress_g, ew, pw, aw, delta0_all, slope_all,
                                 MC_ALT_DRAWS if alt_draws is None else int(alt_draws), max_rounds)
        return dict(realistic=realistic, frag=frag, std=std, stuck=stuck, alt=alt)

    def _mc_alt_lines(self, egress_g, egress_w, purewalk, access_w, delta0_all, slope_all, K,
                      max_rounds=MAX_ROUNDS):
        """{line: votes} per cell: dominant line across ``K`` perturbed arrive-by traced trees
        (so a delayed express lets the next local show up as an alternative). Reuses the validated
        JourneyTree; pricier than the numpy MC, so K is small + env-tunable."""
        if K <= 0:
            return None
        off = R.pat_trip_off(self.data)
        target = self.target_sec
        n = len(self.cell_ids)
        votes = [None] * n
        K = min(K, delta0_all.shape[0])
        for kk in range(K):
            pdata, _dep, _arr = R.perturbed_data(self.data, delta0_all[kk], slope_all[kk], off)
            par = R.reverse_raptor_traced(pdata, egress_g, target - egress_w, egress_w,
                                          max_rounds=max_rounds, board_slack=BOARD_SLACK)
            tree = raptor_journey.JourneyTree(pdata, par, self.access_off, self.access_to,
                                              access_w, purewalk, target, self.max_min)
            _, dom = tree.commute_and_dominant()
            for ci, line in enumerate(dom):
                if line and line != "walk only":
                    d = votes[ci]
                    if d is None:
                        d = votes[ci] = {}
                    d[line] = d.get(line, 0) + 1
        return [None if not d else dict(sorted(d.items(), key=lambda kv: -kv[1])) for d in votes]


def _assemble_arriveby_window(access_off, access_to, access_w, purewalk, latest, deadlines,
                              max_min, percentiles):
    """Per cell, per arrival deadline T: travel(T) = T - latest_home_departure(cell, T);
    then percentile over the arrival window. Returns int32[n_cells, n_pct] (-1 unreachable)."""
    n_cells = len(access_off) - 1
    nd = len(deadlines)
    deadlines = np.asarray(deadlines, np.int64)
    if R._select_kernel() == "numba":
        from . import raptor_numba
        return raptor_numba.assemble_arriveby(
            np.asarray(access_off, np.int64), np.asarray(access_to, np.int64),
            np.asarray(access_w, np.int64), np.asarray(purewalk, np.int64),
            latest, deadlines, np.int64(max_min), np.asarray(percentiles, np.float64))
    out = np.full((n_cells, len(percentiles)), -1, dtype=np.int32)
    for ci in range(n_cells):
        a0, a1 = int(access_off[ci]), int(access_off[ci + 1])
        gids = access_to[a0:a1]
        awalk = access_w[a0:a1].astype(np.int64)
        tt = np.full(nd, np.iinfo(np.int64).max, dtype=np.int64)
        if len(gids):
            sub = latest[gids]                                # (nstops_cell, nd)
            home = sub - awalk[:, None]                       # latest home departure per stop,T
            best_home = home.max(axis=0)                      # over access stops
            reachable = best_home > R.NEG // 2
            tt = np.where(reachable, deadlines - best_home, np.iinfo(np.int64).max)
        pw = purewalk[ci]
        if pw >= 0:
            tt = np.minimum(tt, np.where((deadlines - pw) >= 0, pw, np.iinfo(np.int64).max))
        ttm = tt.astype(np.float64) / 60.0
        ttm = np.where(ttm < 0, 0.0, ttm)
        ttm = np.ceil(np.where(ttm > max_min, max_min, ttm))
        vals = np.percentile(ttm, percentiles, method="lower")
        out[ci] = [(-1 if v >= max_min else int(v)) for v in np.atleast_1d(vals)]
    return out
