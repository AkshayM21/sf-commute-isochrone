#!/usr/bin/env python
"""
Reverse transit isochrone for commuting TO a configurable SF workplace.

For a fine grid of origin points across SF, compute door-to-door travel time
(walk + wait + transit incl. transfers + walk) to the destination, leaving in the
canonical ~8:35am weekday window (~30-min span). We report:
  - best-case time  (5th percentile over the departure window; you time it well)
  - realistic time  (50th percentile / median; typical wait included)
Then aggregate to the 117 SF "Find Neighborhoods" and rank by realistic time.

Routes on the canonical commute model (Muni + BART + Caltrain), shared via the
`core` package so this offline reference can never drift from the live server.

Outputs (in out/):
  grid_traveltimes.gpkg   grid cells with best/realistic minutes
  isochrone_map.html      interactive Folium map (banded)
  neighborhoods_ranked.csv  ranked neighborhood table
"""
import argparse, sys
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import config, feeds, grid as gridmod, network

config.OUT.mkdir(exist_ok=True)

from destination import DEST_LAT, DEST_LON, DEST_LABEL  # configurable; see .env

# discrete bands by design; live ramp lives in scripts/assets/viz.js
ISOCHRONE_BANDS = [(0, 15, "#08589e", "≤15 min"),
                   (15, 20, "#2b8cbe", "15–20"),
                   (20, 25, "#7bccc4", "20–25"),
                   (25, 30, "#bae4bc", "25–30"),
                   (30, 40, "#fdae6b", "30–40"),
                   (40, 60, "#e34a33", "40–60")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtfs", nargs="+", default=None,
                    help="extra GTFS zip paths added to the default Muni+BART+Caltrain feeds")
    ap.add_argument("--limit", type=int, default=None,
                    help="randomly subsample N grid origins (fast validation)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files (e.g. 'full', 'munionly')")
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""

    gtfs = config.gtfs_paths(extra=args.gtfs)
    print("GTFS feeds:", [p.name for p in gtfs])

    service_date = feeds.pick_service_date(gtfs)
    departure = config.departure(service_date)
    print(f"Service date (Wed): {service_date}  departure {departure}")

    neigh = gridmod.load_neighborhoods()
    grid = gridmod.build_grid(neigh)
    if args.limit:
        grid = grid.sample(min(args.limit, len(grid)), random_state=1).reset_index(drop=True)
    print(f"Grid origins: {len(grid)} cells @ {config.GRID_M}m")

    dest = gpd.GeoDataFrame(
        {"id": ["dest"]},
        geometry=[Point(DEST_LON, DEST_LAT)], crs=config.WGS)

    print("Building R5 transport network ...")
    net = network.build_network(gtfs)

    print("Computing travel-time matrix (this is the slow part) ...")
    ttm = network.travel_time_matrix(
        net, origins=grid, destinations=dest, dep=departure, snap_to_network=True)
    ttm = pd.DataFrame(ttm)
    print("matrix columns:", ttm.columns.tolist())

    tt_cols = sorted([c for c in ttm.columns if c.startswith("travel_time")])
    # map lowest percentile -> best, p50 -> realistic
    best_col = tt_cols[0]
    real_col = [c for c in tt_cols if "50" in c]
    real_col = real_col[0] if real_col else tt_cols[-1]
    ttm = ttm.rename(columns={best_col: "best_min", real_col: "real_min"})
    ttm["id"] = ttm["from_id"].astype(str)

    grid = grid.merge(ttm[["id", "best_min", "real_min"]], on="id", how="left")
    grid.to_file(config.OUT / f"grid_traveltimes{suf}.gpkg", driver="GPKG")

    reach = grid.dropna(subset=["real_min"])
    print(f"Reachable within {config.MAX_MIN}min: {len(reach)}/{len(grid)} cells")

    # ---- neighborhood aggregation
    j = gpd.sjoin(grid, neigh[["name", "geometry"]], how="inner", predicate="within")
    agg = (j.groupby("name")
             .agg(real_median=("real_min", "median"),
                  best_min=("best_min", "min"),
                  real_min=("real_min", "min"),
                  cells=("real_min", "size"),
                  reachable_30=("real_min", lambda s: (s <= 30).mean()))
             .reset_index())
    agg = agg.sort_values("real_median").round(1)
    agg.to_csv(config.OUT / f"neighborhoods_ranked{suf}.csv", index=False)

    print("\n=== TOP NEIGHBORHOODS by realistic median door-to-door (min) ===")
    show = agg[agg["real_median"] <= 35].copy()
    for _, r in show.iterrows():
        print(f"{r['real_median']:5.1f} med | best {r['best_min']:4.1f} | "
              f"{int(r['reachable_30']*100):3d}% of area <=30 | {r['name']}")

    make_map(grid, neigh, agg, service_date, suf)
    print(f"\nWrote out/isochrone_map{suf}.html, out/neighborhoods_ranked{suf}.csv, out/grid_traveltimes{suf}.gpkg")


def make_map(grid, neigh, agg, service_date, suf=""):
    import folium
    reach = grid.dropna(subset=["real_min"]).reset_index(drop=True)
    # build square cells
    cells = reach.copy()
    cells["geometry"] = gridmod.square_cells(reach.geometry).values

    bands = ISOCHRONE_BANDS
    def color(v):
        for lo,hi,c,_ in bands:
            if lo < v <= hi or (lo==0 and v<=hi): return c
        return "#999999"
    cells["fill"] = cells["real_min"].apply(color)

    m = folium.Map(location=[DEST_LAT, DEST_LON], zoom_start=13,
                   tiles="CartoDB positron")
    folium.GeoJson(
        cells.to_json(),
        style_function=lambda f: {"fillColor": f["properties"]["fill"],
                                  "color": f["properties"]["fill"],
                                  "weight": 0, "fillOpacity": 0.62},
        tooltip=folium.GeoJsonTooltip(
            fields=["real_min", "best_min"],
            aliases=["realistic min", "best-case min"]),
    ).add_to(m)
    # neighborhood outlines (subtle)
    folium.GeoJson(neigh.to_json(),
                   style_function=lambda f: {"color":"#444","weight":0.6,
                                             "fill":False,"opacity":0.4}).add_to(m)
    # label reachable neighborhoods at centroid
    for _, r in agg[agg["real_median"] <= 32].iterrows():
        sub = neigh[neigh["name"] == r["name"]]
        if not len(sub): continue
        c = sub.to_crs(config.UTM).geometry.centroid.to_crs(config.WGS).iloc[0]
        folium.map.Marker(
            [c.y, c.x],
            icon=folium.DivIcon(html=(
                f'<div style="font:600 10px sans-serif;color:#111;'
                f'text-shadow:0 0 3px #fff,0 0 3px #fff;white-space:nowrap">'
                f'{r["name"]} · {r["real_median"]:.0f}m</div>'))).add_to(m)
    # destination
    folium.CircleMarker([DEST_LAT, DEST_LON], radius=7, color="#000",
                        fill=True, fill_color="#ff0", fill_opacity=1,
                        popup=DEST_LABEL).add_to(m)
    # legend
    items = "".join(
        f'<div><span style="background:{c};width:14px;height:14px;'
        f'display:inline-block;margin-right:6px;"></span>{lbl}</div>'
        for _,_,c,lbl in bands)
    legend = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;'
              f'background:#fff;padding:10px 12px;border:1px solid #999;'
              f'border-radius:6px;font:12px sans-serif">'
              f'<b>Door-to-door to {DEST_LABEL}</b><br>'
              f'<span style="color:#666">realistic (median) · {service_date} · ~8:35am</span>'
              f'<br>{items}</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(config.OUT / f"isochrone_map{suf}.html"))


if __name__ == "__main__":
    main()
