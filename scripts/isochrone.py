#!/usr/bin/env python
"""
Reverse transit isochrone for commuting TO a configurable SF workplace.

For a fine grid of origin points across SF, compute door-to-door travel time
(walk + wait + transit incl. transfers + walk) to the destination, leaving in the
~8:00-8:45am weekday window. We report:
  - best-case time  (5th percentile over the departure window; you time it well)
  - realistic time  (50th percentile / median; typical wait included)
Then aggregate to the 117 SF "Find Neighborhoods" and rank by realistic time.

Outputs (in out/):
  grid_traveltimes.gpkg   grid cells with best/realistic minutes
  isochrone_map.html      interactive Folium map (banded)
  neighborhoods_ranked.csv  ranked neighborhood table
"""
import argparse, datetime as dt, sys, json, zipfile, io, csv
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

import r5py
from r5py import TransportNetwork, TravelTimeMatrix, TransportMode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

from destination import DEST_LAT, DEST_LON, DEST_LABEL  # configurable; see .env

GRID_M = 200          # grid spacing / cell size (meters)
DEP_START = (8, 0)    # departure window start (h, m), local time
DEP_WINDOW_MIN = 45   # window length minutes
MAX_MIN = 60          # cap travel time computed
PERCENTILES = [5, 50] # 5th = best-case, 50th = realistic
UTM = "EPSG:32610"    # UTM 10N for metric grid


def _feed_has_trips(path, date):
    """True if feed `path` actually runs trips on YYYYMMDD `date`."""
    wd = dt.datetime.strptime(date, "%Y%m%d").strftime("%A").lower()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        sids = set()
        if "calendar.txt" in names:
            cal = pd.read_csv(io.BytesIO(z.read("calendar.txt")), dtype=str)
            m = cal[(cal[wd] == "1") & (cal["start_date"] <= date) & (cal["end_date"] >= date)]
            sids = set(m["service_id"])
        if "calendar_dates.txt" in names:
            cd = pd.read_csv(io.BytesIO(z.read("calendar_dates.txt")), dtype=str)
            sids |= set(cd[(cd["date"] == date) & (cd["exception_type"] == "1")]["service_id"])
            sids -= set(cd[(cd["date"] == date) & (cd["exception_type"] == "2")]["service_id"])
        if not sids:
            return False
        trips = pd.read_csv(io.BytesIO(z.read("trips.txt")), dtype=str)
        return bool(trips["service_id"].isin(sids).any())


def pick_service_date(gtfs_paths):
    """First Wednesday where EVERY feed actually runs trips (verified, in-range)."""
    starts, ends = [], []
    for p in gtfs_paths:
        with zipfile.ZipFile(p) as z:
            cal = pd.read_csv(io.BytesIO(z.read("calendar.txt")), dtype=str)
            w = cal[cal["wednesday"] == "1"]
            if len(w):
                starts.append(int(w["start_date"].min()))
                ends.append(int(w["end_date"].max()))
    lo = dt.datetime.strptime(str(max(starts)), "%Y%m%d").date()
    hi = dt.datetime.strptime(str(min(ends)), "%Y%m%d").date()
    d = lo + dt.timedelta(days=7)              # a week in, past signup-boundary swaps
    while d.weekday() != 2:                    # 2 == Wednesday
        d += dt.timedelta(days=1)
    while d <= hi:
        ds = d.strftime("%Y%m%d")
        if all(_feed_has_trips(p, ds) for p in gtfs_paths):
            return d
        d += dt.timedelta(days=7)
    raise SystemExit("No common service Wednesday found across feeds")


def build_grid(neigh_gdf):
    poly = neigh_gdf.to_crs(UTM).union_all()
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + GRID_M / 2, maxx, GRID_M)
    ys = np.arange(miny + GRID_M / 2, maxy, GRID_M)
    pts = [Point(x, y) for x in xs for y in ys]
    g = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    g = g[g.within(poly)].reset_index(drop=True)
    g["id"] = g.index.astype(str)
    return g.to_crs("EPSG:4326")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtfs", nargs="+", default=None,
                    help="GTFS zip paths (default: muni_current.zip + bart_gtfs.zip, else fallback)")
    ap.add_argument("--limit", type=int, default=None,
                    help="randomly subsample N grid origins (fast validation)")
    ap.add_argument("--tag", default="",
                    help="suffix for output files (e.g. 'full', 'munionly')")
    args = ap.parse_args()
    suf = f"_{args.tag}" if args.tag else ""

    if args.gtfs:
        gtfs = [DATA / g if not Path(g).is_absolute() else Path(g) for g in args.gtfs]
    else:
        muni = DATA / "muni_current.zip"
        if not muni.exists():
            muni = DATA / "muni_gtfs_2022_fallback.zip"
            print(f"!! WARNING: using STALE Muni feed {muni.name} (no current feed found)")
        gtfs = [muni, DATA / "bart_gtfs.zip"]
    gtfs = [p for p in gtfs if p.exists()]
    print("GTFS feeds:", [p.name for p in gtfs])
    osm = DATA / "osm_sf.pbf"

    service_date = pick_service_date(gtfs)
    departure = dt.datetime(service_date.year, service_date.month, service_date.day,
                            DEP_START[0], DEP_START[1])
    print(f"Service date (Wed): {service_date}  departure window start {departure}")

    neigh = gpd.read_file(DATA / "sf_neighborhoods.geojson").to_crs("EPSG:4326")
    grid = build_grid(neigh)
    if args.limit:
        grid = grid.sample(min(args.limit, len(grid)), random_state=1).reset_index(drop=True)
    print(f"Grid origins: {len(grid)} cells @ {GRID_M}m")

    dest = gpd.GeoDataFrame(
        {"id": ["dest"]},
        geometry=[Point(DEST_LON, DEST_LAT)], crs="EPSG:4326")

    print("Building R5 transport network ...")
    net = TransportNetwork(str(osm), [str(p) for p in gtfs])

    print("Computing travel-time matrix (this is the slow part) ...")
    ttm = TravelTimeMatrix(
        net,
        origins=grid,
        destinations=dest,
        snap_to_network=True,
        departure=departure,
        departure_time_window=dt.timedelta(minutes=DEP_WINDOW_MIN),
        max_time=dt.timedelta(minutes=MAX_MIN),
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        percentiles=PERCENTILES,
        speed_walking=4.8,
    )
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
    grid.to_file(OUT / f"grid_traveltimes{suf}.gpkg", driver="GPKG")

    reach = grid.dropna(subset=["real_min"])
    print(f"Reachable within {MAX_MIN}min: {len(reach)}/{len(grid)} cells")

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
    agg.to_csv(OUT / f"neighborhoods_ranked{suf}.csv", index=False)

    print("\n=== TOP NEIGHBORHOODS by realistic median door-to-door (min) ===")
    show = agg[agg["real_median"] <= 35].copy()
    for _, r in show.iterrows():
        print(f"{r['real_median']:5.1f} med | best {r['best_min']:4.1f} | "
              f"{int(r['reachable_30']*100):3d}% of area <=30 | {r['name']}")

    make_map(grid, neigh, agg, service_date, suf)
    print(f"\nWrote out/isochrone_map{suf}.html, out/neighborhoods_ranked{suf}.csv, out/grid_traveltimes{suf}.gpkg")


def make_map(grid, neigh, agg, service_date, suf=""):
    import folium
    reach = grid.dropna(subset=["real_min"]).to_crs(UTM)
    # build square cells
    half = GRID_M / 2
    cells = reach.copy()
    cells["geometry"] = cells.geometry.apply(
        lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half))
    cells = cells.to_crs("EPSG:4326")

    bands = [(0,15,"#08589e","≤15 min"),
             (15,20,"#2b8cbe","15–20"),
             (20,25,"#7bccc4","20–25"),
             (25,30,"#bae4bc","25–30"),
             (30,40,"#fdae6b","30–40"),
             (40,60,"#e34a33","40–60")]
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
        c = sub.to_crs(UTM).geometry.centroid.to_crs("EPSG:4326").iloc[0]
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
              f'<span style="color:#666">realistic (median) · {service_date} · ~8am</span>'
              f'<br>{items}</div>')
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(OUT / f"isochrone_map{suf}.html"))


if __name__ == "__main__":
    main()
