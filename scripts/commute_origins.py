#!/usr/bin/env python
"""Door-to-door morning commute (walk + Muni/BART/Caltrain, arrive ~9am weekday) from a set of
ARBITRARY origin lat/lons to a chosen workplace, reusing the JVM-free RAPTOR engine + hill-aware
walk graph.

This is the same math the live server runs for grid cells, but for off-grid origins: we build a
tiny CSR access table (origin -> nearby stops walk seconds) for our origins via the walk graph,
compute the workplace egress (stops -> W) the way the server does, and hand both to the engine's
public ``commute_for_access`` — which roots the reverse range-RAPTOR at the workplace and
assembles the served planned scheduled commute by default. The legacy depart-after percentile and
arrive-by perfect-timing semantics remain available explicitly for validation/comparison. Output:
minutes int per origin id.
"""
import sys, json, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/

import numpy as np
from core import config
from core.raptor_engine import RaptorEngine, ARRIVE_BY_HM
from core.walk import WalkGraph

# Walk-speed toggle: one source of truth with the live server. The graph remains baked at the
# reference pace; requests rescale it by reference / selected product pace.
WALK_SPEEDS = config.WALK_SPEEDS


def build_origin_access(wg, valid_gids, stop_nodes, stop_conn, origins_ll, cap_min):
    """CSR access table origin->stops (reference walk seconds, capped at the engine's access
    cap, like the baked grid table). ``valid_gids``/``stop_nodes``/``stop_conn`` are the
    NaN-filtered, pre-snapped stops from main() (snapped ONCE, shared with the egress pass).
    origins_ll: list of (lon, lat). Returns (access_off, access_to, access_w)."""
    cap = cap_min * 60
    offs = [0]; tos = []; ws = []
    for (lon, lat) in origins_ll:
        d = wg.one_to_many((lon, lat), stop_nodes, stop_conn, cap, reverse=False)  # origin->stops
        fin = np.isfinite(d)
        gids = valid_gids[np.nonzero(fin)[0]].astype(np.int32)   # back to GLOBAL stop gids
        secs = np.rint(d[fin]).astype(np.int64)
        tos.append(gids); ws.append(secs)
        offs.append(offs[-1] + len(gids))
    access_off = np.asarray(offs, np.int64)
    access_to = (np.concatenate(tos).astype(np.int32) if tos else np.zeros(0, np.int32))
    access_w = (np.concatenate(ws).astype(np.int64) if ws else np.zeros(0, np.int64))
    return access_off, access_to, access_w


def _hm(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}"


def _method(engine, semantic, speed_kmh):
    """Human-readable provenance string, built from the LIVE model constants (config + engine)
    so it can never drift from what was actually computed."""
    common = (f"walk+Muni+BART+Caltrain to the workplace passed via --dest-lat/--dest-lon; "
              f"walk speed {speed_kmh:g} km/h; reference access cap {engine.access_cap_min} min "
              f"(rescaled at the selected pace).")
    if semantic == "planned":
        return (f"served planned scheduled commute (JVM-free reverse range-RAPTOR + hill-aware "
                f"walk graph); one first-boarding-anchored scheduled value matching the live map "
                f"(not a percentile); " + common)
    if semantic == "departafter":
        dep0 = config.DEP_HM[0] * 3600 + config.DEP_HM[1] * 60
        win = int(config.window().total_seconds())
        return (f"legacy RAPTOR validation routing (JVM-free reverse range-RAPTOR + hill-aware "
                f"walk graph); depart-after p50 (realistic median, typical wait included) over "
                f"the {_hm(dep0)}-{_hm(dep0 + win)} departure window; not the live-map metric; "
                + common)
    target = ARRIVE_BY_HM[0] * 3600 + ARRIVE_BY_HM[1] * 60
    return (f"real RAPTOR routing (arrive-by-{_hm(target)} perfect-timing, JVM-free); "
            f"optimistic; " + common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest-lat", type=float, required=True)
    ap.add_argument("--dest-lon", type=float, required=True)
    ap.add_argument("--origins", required=True, help="JSON file: list of {id,lat,lon}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--semantic", default="planned", choices=["planned", "departafter", "arriveby"],
                    help="routing metric: planned matches the live map; the others are legacy comparison modes")
    ap.add_argument("--speed", default="med", choices=sorted(WALK_SPEEDS),
                    help="walk pace (mirrors the server toggle): slow 3.4 / med 4.2 / fast 5.2 km/h")
    args = ap.parse_args()

    origins = json.load(open(args.origins))
    ids = [o["id"] for o in origins]
    origins_ll = [(float(o["lon"]), float(o["lat"])) for o in origins]
    speed_kmh = WALK_SPEEDS[args.speed]
    walk_scalar = config.WALK_KMH / speed_kmh

    print(f"[commute] loading RAPTOR engine + walk graph ...", flush=True)
    engine = RaptorEngine(verbose=True)
    wg = WalkGraph.load()

    Wlon, Wlat = args.dest_lon, args.dest_lat
    cap_min = engine.access_cap_min

    # --- snap all stops ONCE (shared by the egress + access passes). raptor_build leaves NaN
    # coords for dangling stop_ids (referenced by stop_times but absent from stops.txt) and
    # cKDTree.query raises on NaN — filter to stops with coordinates and map results back to
    # global gids via `valid` (same pattern as server.py / bake_walk_access.py).
    stop_lon = engine.data["stop_lon"]; stop_lat = engine.data["stop_lat"]
    valid = np.where(~(np.isnan(stop_lat) | np.isnan(stop_lon)))[0]
    stop_nodes, stop_conn = wg.snap(np.column_stack((stop_lon[valid], stop_lat[valid])))

    # --- workplace egress (stops -> W): root at W on the TRANSPOSED graph like the server does
    eg = wg.one_to_many((Wlon, Wlat), stop_nodes, stop_conn, cap_min * 60, reverse=True)
    fin = np.isfinite(eg)
    egress_g = valid[np.nonzero(fin)[0]].astype(np.int32)
    egress_w = np.rint(eg[fin]).astype(np.int64)
    print(f"[commute] workplace reaches {len(egress_g)} stops within {cap_min}min walk", flush=True)

    # --- origin access table (origin -> stops) + origin -> W pure walk -------------------------
    access_off, access_to, access_w = build_origin_access(
        wg, valid, stop_nodes, stop_conn, origins_ll, cap_min)
    # pure walk origin -> W: one W-rooted Dijkstra on the transposed graph to all origins
    orig_nodes, orig_conn = wg.snap(np.array(origins_ll, dtype=np.float64))
    pw = wg.one_to_many((Wlon, Wlat), orig_nodes, orig_conn, config.MAX_MIN * 60, reverse=True)  # origin->W
    purewalk = np.where(np.isfinite(pw), np.rint(pw), -1).astype(np.int64)

    # --- the engine's public off-grid API: same grids/steps/assembly as the served map ---------
    out = engine.commute_for_access(access_off, access_to, access_w, egress_g, egress_w,
                                    purewalk, semantic=args.semantic, percentiles=(5, 50),
                                    walk_scalar=walk_scalar)
    vals = [(int(out[i, 1]) if out[i, 1] >= 0 else None) for i in range(len(ids))]

    result = {ids[i]: vals[i] for i in range(len(ids))}
    result["_method"] = _method(engine, args.semantic, speed_kmh)
    json.dump(result, open(args.out, "w"), indent=1)
    reachable = [v for v in vals if v is not None]
    if reachable:
        print(f"[commute] wrote {args.out}: {len(reachable)}/{len(ids)} reachable, "
              f"range {min(reachable)}-{max(reachable)} min, "
              f"null ids: {[ids[i] for i in range(len(ids)) if vals[i] is None]}", flush=True)
    else:
        # min()/max() would raise on an empty list AFTER the output file was written,
        # making a valid all-null run look like a crash. Summarize honestly instead.
        print(f"[commute] wrote {args.out}: 0/{len(ids)} reachable "
              f"(no origin reaches the workplace within the caps; all values null)", flush=True)


if __name__ == "__main__":
    main()
