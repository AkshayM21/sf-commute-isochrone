#!/usr/bin/env python
"""Build data/sf_neighborhoods.geojson used for per-cell neighborhood labels.

Source: DataSF **Realtor Neighborhoods** (92 areas) — granular AND with accurate, colloquial
boundaries (unlike the old "Find Neighborhoods" set, whose North Beach polygon reached south
to ~Washington St and mislabeled FiDi/Chinatown/Jackson Square). Realtor has no standalone
Chinatown or Japantown, so we graft those two polygons from the old **Find Neighborhoods**
set — whose Chinatown/Japantown polygons are TIGHT and accurate (the Analysis set's are too
coarse and would swallow the eastern FiDi). We carve the grafted areas out of the overlapping
Realtor polygons (so coverage stays non-overlapping) and add them back as their own areas.
Output field: `name`.

Run by scripts/setup.sh; re-run anytime to refresh. Public domain (PDDL); EPSG:4326.
"""
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import config

REALTOR = "https://data.sfgov.org/resource/2kjj-ysvr.geojson?$limit=2000"   # field: nbrhood
FIND117 = "https://data.sfgov.org/resource/gfpk-269f.geojson?$limit=300"    # field: name (tight Chinatown/Japantown)
GRAFT = ["Chinatown", "Japantown"]


def main():
    realtor = (gpd.read_file(REALTOR).to_crs(config.WGS)
               .rename(columns={"nbrhood": "name"})[["name", "geometry"]])
    find117 = gpd.read_file(FIND117).to_crs(config.WGS)   # field already 'name'
    graft = find117[find117["name"].isin(GRAFT)][["name", "geometry"]].reset_index(drop=True)
    if len(graft) != len(GRAFT):
        raise SystemExit(f"expected {GRAFT} in Find Neighborhoods set, found {graft['name'].tolist()}")

    # Carve the grafted areas out of the Realtor polygons that overlap them, so the final
    # coverage is non-overlapping and a point in Chinatown/Japantown labels as such.
    carve = graft.geometry.union_all()
    realtor["geometry"] = realtor.geometry.difference(carve)
    realtor = realtor[~(realtor.geometry.is_empty | realtor.geometry.isna())]

    merged = gpd.GeoDataFrame(pd.concat([realtor, graft], ignore_index=True),
                              geometry="geometry", crs=config.WGS)
    out = config.neigh_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()                      # GeoJSON driver won't overwrite in place
    merged.to_file(out, driver="GeoJSON")
    print(f"wrote {out} — {len(merged)} neighborhoods "
          f"({len(realtor)} realtor + {len(graft)} grafted: {graft['name'].tolist()})")


if __name__ == "__main__":
    main()
