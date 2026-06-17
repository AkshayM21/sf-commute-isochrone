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
MC_DEADLINE_STEP = int(os.environ.get("RAPTOR_MC_DEADLINE_STEP", "60"))  # MC tail readout (finer; see _make_grids)
DEP_STEP = 60
BOARD_SLACK = int(os.environ.get("RAPTOR_BOARD_SLACK", "60"))
MAX_ROUNDS = 8                                    # rides = transfers + 1 (R5 default cap)
ARRIVE_BY_HM = (9, 0)                             # product target: arrive by 09:00

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
        expect_grid_m = None                      # set only when WE pick the table (default path);
        if access_path is None:                   # explicit callers choose their own grid
            fp = raptor_build._fingerprint(
                config.gtfs_paths(), self.service_date.strftime("%Y%m%d"),
                raptor_build.band_seconds(), raptor_build.FOOTPATH_M)
            # exact name, NOT a glob: a glob would also match access_walk*/access_walkflat*
            # (and any leftover bake at another resolution would shadow the intended grid)
            access_path = raptor_build.CACHE_DIR / f"access_{config.GRID_M}m_{fp}.npz"
            if not access_path.exists():
                raise FileNotFoundError(
                    f"no baked access table {access_path.name} in {raptor_build.CACHE_DIR}; "
                    f"run scripts/raptor_oracle.py")
            expect_grid_m = config.GRID_M
        z = np.load(access_path, allow_pickle=True)
        if int(z["n_stops"]) != self.data["n_stops"]:
            raise ValueError("access table / raptor structures stop-count mismatch (stale cache)")
        if (expect_grid_m is not None and "grid_m" in z.files
                and int(z["grid_m"]) != expect_grid_m):
            raise ValueError(f"access table {access_path.name} is a {int(z['grid_m'])}m bake "
                             f"but the configured grid is {expect_grid_m}m")
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
        user's pace: walk_scalar = 4.8/v (slow 1.20, med 1.00, fast 0.857). ``access_w`` None
        scales the engine's baked grid access table; callers with their own access CSR
        (``commute_for_access``) pass theirs. Returns (egress_w, purewalk, access_w) as int64
        at the user's pace."""
        ew = np.asarray(egress_w, np.int64); pw = np.asarray(purewalk, np.int64)
        aw = self.access_w if access_w is None else np.asarray(access_w, np.int64)
        if walk_scalar != 1.0:
            ew = np.rint(ew.astype(np.float64) * walk_scalar).astype(np.int64)
            pw = pw.copy(); m = pw >= 0
            pw[m] = np.rint(pw[m].astype(np.float64) * walk_scalar).astype(np.int64)
            aw = np.rint(aw.astype(np.float64) * walk_scalar).astype(np.int64)
        return ew, pw, aw

    # -- compute -------------------------------------------------------------------------
    def _reverse(self, egress_g, egress_w, deadlines, max_rounds=MAX_ROUNDS):
        return R.reverse_profile(self.data, egress_g, egress_w, deadlines,
                                 board_slack=BOARD_SLACK, max_rounds=max_rounds)

    def commute_for_access(self, access_off, access_to, access_w, egress_g, egress_w, purewalk,
                           semantic="departafter", percentiles=(5, 50), max_rounds=MAX_ROUNDS,
                           walk_scalar=1.0, target_sec=None, window_sec=None,
                           walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS):
        """Door-to-door commute minutes for a CALLER-SUPPLIED origin set (both semantics).

        The public API for off-grid origins (e.g. scripts/commute_origins.py): the caller builds
        its own CSR access table (origin -> stop walk REFERENCE seconds at config.WALK_KMH,
        capped like the baked table at ``self.access_cap_min`` so the engine's departure grids
        cover every boarding) + per-origin ``purewalk`` (origin->W reference seconds, -1 if
        unwalkable), and the engine runs the SAME grids/steps/assembly the served map uses —
        so external callers can never drift from the product model.

          semantic   'departafter' — p-percentiles over the [DEP, DEP+WINDOW] departure window
                     (the R5-validated realistic model);
                     'arriveby'    — perfect-timing arrive-by window ending at ``target_sec``
                     (default 09:00; ``window_sec`` None -> config.window(), 0 -> single deadline).
          walk_scalar  4.8/pace scalar applied to ALL walk legs (access/egress/pure-walk).

        Returns int32[n_origins, n_pct] minutes, -1 = unreachable."""
        access_off = np.asarray(access_off, np.int64)
        access_to = np.asarray(access_to, np.int32)
        egress_g = np.asarray(egress_g, np.int32)
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar, access_w=access_w)
        if semantic == "departafter":
            latest = self._reverse(egress_g, ew, self.Tgrid, max_rounds)
            arrivalW = R.stop_arrival_profile(latest, self.Tgrid, self.dep_grid)
            return R.assemble_departafter(access_off, access_to, aw, pw, arrivalW,
                                          self.dep_grid, self.cell_deps, self.max_min,
                                          percentiles=percentiles, beta=walk_reluctance,
                                          eps=walk_prior_eps)
        if semantic != "arriveby":
            raise ValueError(f"semantic must be 'departafter' or 'arriveby', got {semantic!r}")
        target = self.target_sec if target_sec is None else int(target_sec)
        win = int(config.window().total_seconds()) if window_sec is None else int(window_sec)
        deadlines = (np.array([target], np.int64) if win <= 0
                     else np.arange(target - win, target + 1, DEP_STEP, dtype=np.int64))
        latest = self._reverse(egress_g, ew, deadlines, max_rounds)
        return _assemble_arriveby_window(access_off, access_to, aw, pw, latest, deadlines,
                                         self.max_min, np.asarray(percentiles, np.float64),
                                         beta=walk_reluctance, eps=walk_prior_eps)

    def departafter(self, egress_g, egress_w, purewalk, percentiles=(5, 50),
                    max_rounds=MAX_ROUNDS, walk_scalar=1.0, walk_reluctance=WALK_RELUCTANCE,
                    walk_prior_eps=WALK_PRIOR_EPS):
        """{cell_id: [p5, p50]} minutes, depart-after window (R5-comparable). ``purewalk`` is
        cell->W walk seconds aligned to self.cell_ids (-1 if > cap). ``max_rounds`` caps
        public-transport rides (rides = transfers + 1). ``walk_scalar`` sets the walk pace;
        ``walk_reluctance``/``walk_prior_eps`` the mild walk prior (decision-only)."""
        out = self.commute_for_access(self.access_off, self.access_to, self.access_w,
                                      egress_g, egress_w, purewalk, semantic="departafter",
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
        par = R.reverse_raptor_traced(self.data, egress_g, target - ew, ew,
                                      max_rounds=max_rounds, board_slack=BOARD_SLACK)
        return raptor_journey.JourneyTree(self.data, par, self.access_off, self.access_to,
                                          aw, pw, target, self.max_min,
                                          walk_reluctance=walk_reluctance,
                                          walk_prior_eps=walk_prior_eps,
                                          egress_g=egress_g, egress_w=ew)

    # -- depart-after traced tree -> hover==map breakdown + color-by-line ------------------
    def journey_tree_departafter(self, egress_g, egress_w, purewalk, percentile=50.0,
                                 max_rounds=MAX_ROUNDS, walk_scalar=1.0,
                                 walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS):
        """A ``DepartAfterJourneyTree`` for the depart-after window percentile (default p50): serves
        the per-cell breakdown (hover), color-by-line, AND the depart-after map value, all anchored
        on the SAME ``arrivalW`` the served depart-after map paints with — so hover == map by
        construction (Stage 1 of the depart-after map migration).

        Mirrors ``journey_tree`` (the arrive-by sibling) but is driven by the depart-after window
        instead of a single 09:00 deadline: it reuses THIS engine's depart-after value computation
        (the SAME ``reverse_profile`` + ``stop_arrival_profile`` + grids ``departafter`` uses), so
        the tree's painted grid equals ``self.departafter(...)`` for that percentile EXACTLY. The
        tracer then builds one ``reverse_raptor_traced`` tree per representative arrival deadline T*
        (only ~15-17 distinct per workplace) and reads each cell's journey from its painted access
        stop s*. ``walk_scalar`` is applied end-to-end (access/egress/pure-walk seconds are scaled),
        so the depart-after breakdown honors the walk-speed toggle — unlike the legacy R5-backed
        departafter, which did not. JVM-free.

        NOTE: not yet wired into the server (Stage 2). ``RAPTOR_SEMANTIC`` default stays
        ``arriveby`` and the arrive-by path is byte-unchanged."""
        egress_g = np.asarray(egress_g, np.int32)
        ew, pw, aw = self._scale_walk(egress_w, purewalk, walk_scalar)
        latest = self._reverse(egress_g, ew, self.Tgrid, max_rounds)
        arrivalW = R.stop_arrival_profile(latest, self.Tgrid, self.dep_grid)
        return raptor_journey.DepartAfterJourneyTree(
            self.data, self.access_off, self.access_to, aw, pw, arrivalW,
            self.dep_grid, self.cell_deps, self.max_min, egress_g, ew,
            percentile=float(percentile), walk_reluctance=walk_reluctance,
            walk_prior_eps=walk_prior_eps, max_rounds=max_rounds, board_slack=BOARD_SLACK)

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
        ntrips = np.asarray(self.data["pat_ntrips"])
        mu_trip = np.repeat(mu_pat, ntrips); slope_trip = np.repeat(slope_pat, ntrips)
        rng = np.random.default_rng(seed)
        k = MC_SHAPE
        T = mu_trip.shape[0]
        delta0_all = rng.gamma(k, mu_trip / k, size=(nR, T))
        slope_all = rng.gamma(k, np.maximum(slope_trip, 1e-9) / k, size=(nR, T))
        return delta0_all, slope_all

    def montecarlo(self, egress_g, egress_w, purewalk, perfect=None, n_draws=None,
                   seed=None, alt_draws=None, walk_scalar=1.0, max_rounds=MAX_ROUNDS, tree=None,
                   walk_reluctance=WALK_RELUCTANCE, walk_prior_eps=WALK_PRIOR_EPS):
        """Committed-plan service-noise MC for a workplace. Returns dict of cell-aligned arrays:
          realistic int32  p50 door-to-door commute over draws FLOORED at ``perfect`` (so
                           realistic/frag/std/stuck all describe one consistent distribution
                           and realistic >= perfect holds by construction)
          realistic_raw int32  pre-floor p50 — the non-vacuous perfect<=committed regression
                           signal for tests/validators (never served)
          frag      int32  p90-p50 "bad-day delta" minutes (the headline fragility number)
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
        # Truncate the MC tail-readout grid: any deadline beyond max(commit_home) + cap yields
        # tt > cap -> capf for EVERY cell regardless of the draw, so dropping those deadlines is
        # provably output-identical (including `stuck`) and trims the kernel's dominant cost (the
        # per-draw sweep; how many deadlines survive is workplace-dependent — later committed
        # departures keep more of the grid). Computed HERE, after committed_first_legs(), so the
        # bound sees the walk-scalar-scaled legs; _make_grids' Tgrid_mc is untouched (tests pass
        # engine.Tgrid_mc straight to the kernel). Only transit cells (commit_kind == 2) carry a
        # real commit_home (kind 0 holds NEG); no transit cells -> no bound -> keep the full grid.
        Tmc = self.Tgrid_mc
        kind_c = np.asarray(legs["commit_kind"])
        if (kind_c == 2).any():
            bound = int(np.asarray(legs["commit_home"])[kind_c == 2].max()) + self.max_min * 60
            trunc = Tmc[Tmc <= bound]                    # INCLUSIVE: keep deadlines <= bound
            if trunc.size:
                Tmc = trunc
        # the kernel's cummax + _first_ge readout require an ascending grid (np.arange +
        # prefix mask preserve this; assert in case a future caller breaks the invariant)
        assert Tmc.size and np.all(np.diff(Tmc) > 0), "MC deadline grid must be ascending"
        cm = R.montecarlo_commute_committed(self.data, egress_g, ew, Tmc, legs,
                                            perfect, self.max_min,
                                            delta0_all, slope_all, board_slack=BOARD_SLACK,
                                            max_rounds=max_rounds)
        # pre-floor p50: the regression signal tests/validators assert on (the floored stats
        # below satisfy perfect <= realistic by construction, so they carry no signal).
        realistic_raw = np.ceil(np.percentile(cm, 50, axis=1)).astype(np.int32)
        # Floor the DRAWS at perfect once, BEFORE any statistic, so the served quartet
        # (realistic, frag, std, stuck) describes one self-consistent distribution and a frontend
        # "bad day = realistic + frag" read matches p90. (The old post-hoc clamp lifted only
        # realistic, so frag was measured from the lower unclamped p50 and overstated the bad day
        # wherever the clamp fired.) Draws can legitimately dip below perfect: the tail
        # re-optimizes earliest-arrival over the deadline grid where the traced tree optimized
        # latest-departure — small + two-sided, see the zero-perturbation test.
        cm = np.maximum(cm, np.where(perfect >= 0, perfect, 0).astype(np.float64)[:, None])
        p50 = np.percentile(cm, 50, axis=1)
        p90 = np.percentile(cm, 90, axis=1)
        realistic = np.ceil(p50).astype(np.int32)
        frag = np.maximum(0, np.round(p90 - p50)).astype(np.int32)
        std = np.round(np.std(cm, axis=1)).astype(np.int32)
        stuck = np.mean(cm >= self.max_min - 1e-9, axis=1)
        alt, alt_bundle = self._alt_window(tree, perfect,
                                           MC_ALT_DRAWS if alt_draws is None else int(alt_draws))
        return dict(realistic=realistic, realistic_raw=realistic_raw, frag=frag, std=std,
                    stuck=stuck, alt=alt, alt_bundle=alt_bundle)

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
        win = tree.alt_lines_window(perfect, ALT_WINDOW_MIN)   # per cell {line: (min, stop)}
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
                       n_draws=None, seed=None, walk_scalar=1.0, max_rounds=MAX_ROUNDS):
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
        is unreachable / off-cell (so the caller falls back to that route's best-case)."""
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
        # same per-pattern delay model + RNG as montecarlo() (shared helper) so the alt numbers are
        # drawn from the SAME distribution as the served realistic map, not a fresh ad-hoc noise
        delta0_all, slope_all = self._mc_draw_arrays(nR, seed)
        cm = R.montecarlo_commute_committed(self.data, egress_g, ew, self.Tgrid_mc, legs,
                                            perfect, self.max_min,
                                            delta0_all, slope_all, board_slack=BOARD_SLACK,
                                            max_rounds=max_rounds)
        # floor the draws at each route's own best-case BEFORE the percentile, mirroring montecarlo()
        floor = np.where(perfect >= 0, perfect, 0).astype(np.float64)
        cm = np.maximum(cm, floor[:, None])
        p50 = np.percentile(cm, 50, axis=1)
        p90 = np.percentile(cm, 90, axis=1)
        real = np.ceil(p50).astype(np.int32)
        frag = np.maximum(0, np.round(p90 - p50)).astype(np.int32)
        kind = np.asarray(legs["commit_kind"])
        out = []
        for row in range(n):
            if kind[row] == 0:                # stop unreachable / off-cell -> no typical
                out.append(None)
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
