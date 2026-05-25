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

from . import config, raptor_build, raptor as R

ACCESS_CAP_MIN = int(os.environ.get("RAPTOR_ACCESS_CAP", "25"))  # 25 cleared the worst access-starved periphery (max err 15->7, mism 20->5) vs 20
DEADLINE_STEP = int(os.environ.get("RAPTOR_DEADLINE_STEP", "180"))
DEP_STEP = 60
BOARD_SLACK = int(os.environ.get("RAPTOR_BOARD_SLACK", "60"))
MAX_ROUNDS = 8                                    # rides = transfers + 1 (R5 default cap)
ARRIVE_BY_HM = (9, 0)                             # product target: arrive by 09:00


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

    # -- compute -------------------------------------------------------------------------
    def _reverse(self, egress_g, egress_w, deadlines):
        return R.reverse_profile(self.data, egress_g, egress_w, deadlines,
                                 board_slack=BOARD_SLACK, max_rounds=MAX_ROUNDS)

    def departafter(self, egress_g, egress_w, purewalk, percentiles=(5, 50)):
        """{cell_id: [p5, p50]} minutes, depart-after window (R5-comparable). ``purewalk`` is
        cell->W walk seconds aligned to self.cell_ids (-1 if > cap)."""
        latest = self._reverse(egress_g, egress_w, self.Tgrid)
        arrivalW = R.stop_arrival_profile(latest, self.Tgrid, self.dep_grid)
        out = R.assemble_departafter(self.access_off, self.access_to, self.access_w,
                                     np.asarray(purewalk, np.int64), arrivalW,
                                     self.dep_grid, self.cell_deps, self.max_min,
                                     percentiles=percentiles)
        return {c: [int(out[i, k]) if out[i, k] >= 0 else None
                    for k in range(out.shape[1])] for i, c in enumerate(self.cell_ids)}

    def arriveby(self, egress_g, egress_w, purewalk, target_sec=None, window_sec=None,
                 percentiles=(5, 50)):
        """{cell_id: [p5, p50]} minutes, arrive-by an arrival window ending at ``target_sec``
        (default 09:00). ``window_sec`` None -> use config.window(); 0 -> single deadline."""
        target = self.target_sec if target_sec is None else int(target_sec)
        win = int(config.window().total_seconds()) if window_sec is None else int(window_sec)
        deadlines = (np.array([target], np.int64) if win <= 0
                     else np.arange(target - win, target + 1, DEP_STEP, dtype=np.int64))
        latest = self._reverse(egress_g, egress_w, deadlines)
        out = _assemble_arriveby_window(self.access_off, self.access_to, self.access_w,
                                        np.asarray(purewalk, np.int64), latest, deadlines,
                                        self.max_min, np.asarray(percentiles, np.float64))
        return {c: [int(out[i, k]) if out[i, k] >= 0 else None
                    for k in range(out.shape[1])] for i, c in enumerate(self.cell_ids)}


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
