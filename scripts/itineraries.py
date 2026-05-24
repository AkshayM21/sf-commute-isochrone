#!/usr/bin/env python
"""
Fastest door-to-door itineraries from representative SF origins to the workplace,
using r5py DetailedItineraries (shows the actual legs: walk / which line / transfer).
"""
import argparse
import pandas as pd, geopandas as gpd
from shapely.geometry import Point
from r5py import DetailedItineraries, TransportMode

from core import config, feeds, network
from destination import DEST_LAT, DEST_LON, DEST_LABEL  # configurable; see .env

# representative points across the city (lat, lon)
ORIGINS = {
    "NoPa (Divisadero/Hayes)":      (37.7745, -122.4378),
    "Lower Haight":                 (37.7720, -122.4310),
    "Hayes Valley":                 (37.7765, -122.4241),
    "Alamo Square":                 (37.7765, -122.4347),
    "Castro (Castro/Market)":       (37.7625, -122.4350),
    "Duboce Triangle":              (37.7690, -122.4290),
    "Cole Valley":                  (37.7659, -122.4500),
    "Noe Valley (24th/Church)":     (37.7510, -122.4290),
    "Mission (16th BART)":          (37.7650, -122.4197),
    "Mission (24th BART)":          (37.7522, -122.4180),
    "Bernal Heights":               (37.7390, -122.4160),
    "Glen Park (BART)":             (37.7330, -122.4337),
    "Potrero Hill":                 (37.7620, -122.3970),
    "Dogpatch (20th/3rd)":          (37.7595, -122.3880),
    "North Beach (Wash. Sq)":       (37.8006, -122.4103),
    "Jackson Square":               (37.7965, -122.4036),
    "Nob Hill":                     (37.7919, -122.4097),
    "Russian Hill":                 (37.7990, -122.4220),
    "Marina":                       (37.8004, -122.4360),
    "Pacific Heights":              (37.7925, -122.4350),
    "Inner Richmond (Clement)":     (37.7827, -122.4640),
    "Inner Sunset (Irving/9th)":    (37.7640, -122.4665),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtfs", nargs="+", default=None)
    args = ap.parse_args()
    gtfs = config.gtfs_paths(extra=args.gtfs)
    print("GTFS:", [p.name for p in gtfs])

    departure = config.departure(feeds.pick_service_date(gtfs))
    print("departure:", departure)

    origins = gpd.GeoDataFrame(
        {"id": list(ORIGINS.keys())},
        geometry=[Point(lon, lat) for lat, lon in ORIGINS.values()],
        crs=config.WGS)
    dest = gpd.GeoDataFrame({"id": [DEST_LABEL]},
                            geometry=[Point(DEST_LON, DEST_LAT)], crs=config.WGS)

    net = network.build_network(gtfs)
    di = DetailedItineraries(
        net, origins=origins, destinations=dest,
        departure=departure,
        transport_modes=[TransportMode.TRANSIT, TransportMode.WALK],
        snap_to_network=True, force_all_to_all=True,
    )
    df = pd.DataFrame(di)
    df.to_csv(config.OUT / "itineraries_raw.csv", index=False)
    summarize(df, feeds.load_routes(gtfs))


def summarize(df, routes):
    df = df.copy()
    df["tt"] = pd.to_timedelta(df["travel_time"]).dt.total_seconds() / 60
    df["wt"] = pd.to_timedelta(df["wait_time"]).dt.total_seconds() / 60

    def modestr(m):
        return str(m).split(".")[-1]

    print(f"\n=== FASTEST DOOR-TO-DOOR ITINERARIES TO {DEST_LABEL.upper()} ===")
    print("   (total includes walk + wait + ride; xfer = transfers)\n")
    rows = []
    for oid, g in df.groupby("from_id"):
        totals = g.groupby("option").apply(
            lambda x: x["tt"].sum() + x["wt"].sum(), include_groups=False)
        best = totals.idxmin()
        legs = g[g["option"] == best].sort_values("segment")
        parts, nrides = [], 0
        for _, leg in legs.iterrows():
            mode = modestr(leg["transport_mode"])
            if mode == "WALK":
                if leg["tt"] >= 1:
                    parts.append(f"walk {leg['tt']:.0f}m")
            else:
                nrides += 1
                rn = feeds.route_name(leg["route_id"], leg.get("feed"), routes) or mode
                parts.append(f"{rn} {leg['tt']:.0f}m")
        rows.append((totals[best], oid, max(0, nrides - 1), " → ".join(parts)))
    for total, oid, nxfer, chain in sorted(rows):
        print(f"{total:5.0f}m  {oid:28s} [{nxfer}x]  {chain}")


if __name__ == "__main__":
    main()
