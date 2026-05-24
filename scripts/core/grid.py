"""SF grid construction: neighborhoods, the regular origin grid, and square cells.

No r5py here.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
from . import config


def load_neighborhoods():
    """SF 'Find Neighborhoods' polygons (DataSF), in WGS84."""
    return gpd.read_file(config.neigh_path()).to_crs(config.WGS)


def build_grid(neigh, grid_m=config.GRID_M):
    """Regular point grid (cell centers) covering `neigh`, in WGS84, with a string `id`.

    Built in UTM for a true metric spacing, filtered to the neighborhood union, then
    reset so ids are 0..N-1 in (x-major, y-minor) order."""
    poly = neigh.to_crs(config.UTM).union_all()
    minx, miny, maxx, maxy = poly.bounds
    pts = [Point(x, y) for x in np.arange(minx + grid_m / 2, maxx, grid_m)
           for y in np.arange(miny + grid_m / 2, maxy, grid_m)]
    g = gpd.GeoDataFrame(geometry=pts, crs=config.UTM)
    g = g[g.within(poly)].reset_index(drop=True)
    g["id"] = g.index.astype(str)
    return g.to_crs(config.WGS)


def square_cells(points, grid_m=config.GRID_M):
    """GeoSeries of square polygons (grid_m on a side) centered on each WGS84 point,
    index-aligned to `points`. Squares are built in UTM then returned in WGS84."""
    gm = points.to_crs(config.UTM)
    half = grid_m / 2
    return gpd.GeoSeries(
        [box(p.x - half, p.y - half, p.x + half, p.y + half) for p in gm.geometry],
        crs=config.UTM).to_crs(config.WGS)


def attach_neighborhoods(cells, neigh, how="left", predicate="intersects"):
    """Spatial-join a neighborhood `name` onto each cell (one row per cell)."""
    j = gpd.sjoin(cells, neigh[["name", "geometry"]], how=how, predicate=predicate)
    return j.drop_duplicates("id").drop(columns="index_right")
