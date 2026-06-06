#!/usr/bin/env python
"""Door-to-door morning commute (walk + Muni/BART/Caltrain, arrive ~9am weekday) from a set of
ARBITRARY origin lat/lons to a chosen workplace, reusing the JVM-free RAPTOR engine + hill-aware
walk graph.

This is the same math the live server runs for grid cells, but for off-grid origins: we root the
reverse range-RAPTOR at the workplace once (giving the latest feasible departure from every transit
stop), build a tiny CSR access table (origin -> nearby stops walk seconds) for our origins via the
walk graph, then assemble the depart-after p50 (the R5-validated "realistic median, typical wait
included") commute per origin. Output: minutes int per origin id.
"""
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/

import numpy as np
from core import config, raptor as R
from core.raptor_engine import RaptorEngine, BOARD_SLACK, DEADLINE_STEP
from core.walk import WalkGraph


def build_origin_access(wg, engine, origins_ll, cap_min):
    """CSR access table origin->stops (reference walk seconds, capped), + origin->work pure-walk.
    origins_ll: list of (lon, lat). Returns (access_off, access_to, access_w, purewalk_placeholder).
    purewalk is computed separately (origin->W)."""
    stop_lon = engine.data["stop_lon"]
    stop_lat = engine.data["stop_lat"]
    n_stops = engine.data["n_stops"]
    # pre-snap all stops once (forward graph: origin -> stop, normal walking direction)
    stop_nodes, stop_conn = wg.snap(np.column_stack((stop_lon, stop_lat)))
    cap = cap_min * 60
    offs = [0]; tos = []; ws = []
    for (lon, lat) in origins_ll:
        d = wg.one_to_many((lon, lat), stop_nodes, stop_conn, cap, reverse=False)  # origin->stops
        fin = np.isfinite(d)
        gids = np.nonzero(fin)[0].astype(np.int32)
        secs = np.rint(d[fin]).astype(np.int64)
        tos.append(gids); ws.append(secs)
        offs.append(offs[-1] + len(gids))
    access_off = np.asarray(offs, np.int64)
    access_to = (np.concatenate(tos).astype(np.int32) if tos else np.zeros(0, np.int32))
    access_w = (np.concatenate(ws).astype(np.int64) if ws else np.zeros(0, np.int64))
    return access_off, access_to, access_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest-lat", type=float, required=True)
    ap.add_argument("--dest-lon", type=float, required=True)
    ap.add_argument("--origins", required=True, help="JSON file: list of {id,lat,lon}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--semantic", default="departafter", choices=["departafter", "arriveby"])
    args = ap.parse_args()

    origins = json.load(open(args.origins))
    ids = [o["id"] for o in origins]
    origins_ll = [(float(o["lon"]), float(o["lat"])) for o in origins]

    print(f"[commute] loading RAPTOR engine + walk graph ...", flush=True)
    engine = RaptorEngine(verbose=True)
    wg = WalkGraph.load()

    Wlon, Wlat = args.dest_lon, args.dest_lat
    cap_min = engine.access_cap_min

    # --- workplace egress (stops -> W) + reverse RAPTOR profile rooted at W --------------------
    stop_lon = engine.data["stop_lon"]; stop_lat = engine.data["stop_lat"]
    stop_nodes, stop_conn = wg.snap(np.column_stack((stop_lon, stop_lat)))
    # egress = stop -> W ; root at W on the TRANSPOSED graph (reverse=True) like the server does
    eg = wg.one_to_many((Wlon, Wlat), stop_nodes, stop_conn, cap_min * 60, reverse=True)
    fin = np.isfinite(eg)
    egress_g = np.nonzero(fin)[0].astype(np.int32)
    egress_w = np.rint(eg[fin]).astype(np.int64)
    print(f"[commute] workplace reaches {len(egress_g)} stops within {cap_min}min walk", flush=True)

    # --- origin access table (origin -> stops) + origin -> W pure walk -------------------------
    access_off, access_to, access_w = build_origin_access(wg, engine, origins_ll, cap_min)
    # pure walk origin -> W: one W-rooted Dijkstra on the transposed graph to all origins
    orig_nodes, orig_conn = wg.snap(np.array(origins_ll, dtype=np.float64))
    pw = wg.one_to_many((Wlon, Wlat), orig_nodes, orig_conn, config.MAX_MIN * 60, reverse=True)  # origin->W
    purewalk = np.where(np.isfinite(pw), np.rint(pw), -1).astype(np.int64)

    if args.semantic == "departafter":
        # reverse profile over the deadline grid, then depart-after p50 assembly (R5-validated median)
        latest = R.reverse_profile(engine.data, egress_g, egress_w, engine.Tgrid,
                                   board_slack=BOARD_SLACK)
        arrivalW = R.stop_arrival_profile(latest, engine.Tgrid, engine.dep_grid)
        out = R.assemble_departafter(access_off, access_to, access_w, purewalk, arrivalW,
                                     engine.dep_grid, engine.cell_deps, engine.max_min,
                                     percentiles=(5, 50))
        vals = [(int(out[i, 1]) if out[i, 1] >= 0 else None) for i in range(len(ids))]
        method = ("real RAPTOR routing (JVM-free reverse range-RAPTOR + hill-aware walk graph); "
                  "depart-after p50 (realistic median, typical wait included; R5-validated MAE 0.75) "
                  "over the 08:35-09:05 departure window; walk+Muni+BART+Caltrain to "
                  "650 Townsend St; walk speed 4.8 km/h; access cap 25 min.")
    else:
        from core.raptor_engine import _assemble_arriveby_window
        target = engine.target_sec
        win = int(config.window().total_seconds())
        deadlines = np.arange(target - win, target + 1, 60, dtype=np.int64)
        latest = R.reverse_profile(engine.data, egress_g, egress_w, deadlines, board_slack=BOARD_SLACK)
        out = _assemble_arriveby_window(access_off, access_to, access_w, purewalk, latest, deadlines,
                                        engine.max_min, np.asarray((5, 50), np.float64))
        vals = [(int(out[i, 1]) if out[i, 1] >= 0 else None) for i in range(len(ids))]
        method = ("real RAPTOR routing (arrive-by-09:00 perfect-timing, JVM-free); optimistic.")

    result = {ids[i]: vals[i] for i in range(len(ids))}
    result["_method"] = method
    json.dump(result, open(args.out, "w"), indent=1)
    reachable = [v for v in vals if v is not None]
    print(f"[commute] wrote {args.out}: {len(reachable)}/{len(ids)} reachable, "
          f"range {min(reachable)}-{max(reachable)} min, "
          f"null ids: {[ids[i] for i in range(len(ids)) if vals[i] is None]}", flush=True)


if __name__ == "__main__":
    main()
