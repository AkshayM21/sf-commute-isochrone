"""Build a compact, hill-aware pedestrian graph from OSM + a DEM (offline, JVM-free).

Replaces R5's only remaining RUNTIME job (the per-workplace walk matrix) with a custom
pedestrian router. SF is steep and R5 ignores grade, so a slope-aware walk router is MORE
accurate, not just a JVM-free swap. We bake, once:

  data/walk_graph.npz  — a directed CSR pedestrian graph whose edge weights are REFERENCE
  WALK SECONDS at config.WALK_KMH (4.8 km/h), adjusted for grade via Tobler's hiking function
  (uphill slower, gentle downhill fastest; stairs penalized). Directed half-edges so uphill !=
  downhill. Plus node coords + elevation. The runtime (`core/walk.py`) loads this, rebuilds a
  cKDTree for snapping, and runs scipy one-to-many Dijkstra; a per-request walk-speed scalar
  (slow/med/fast) multiplies the reference seconds (no rebuild).

Pipeline (see esy-osm-pbf two-pass requirement):
  1. PASS 1: keep pedestrian `highway` ways (drop motorway/trunk + links, foot=no, private),
     collect their ordered node refs.
  2. PASS 2: resolve ONLY the referenced node ids -> (lon, lat).
  3. Sample DEM elevation per node (rasterio).
  4. Build directed edges (both directions per OSM way segment); weight = horizontal_len *
     tobler(slope) / speed; stairs get an extra penalty.
  5. Keep the largest connected component (so snapping can't land on an isolated path).
  6. Save CSR npz.

Usage: scripts/fetch_dem.sh && .venv/bin/python scripts/build_walk_graph.py
"""
import os, sys, time, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

from core import config

PBF = config.DATA / "osm_sf.pbf"
DEM = config.DATA / "dem_sf.tif"
OUT = config.DATA / "walk_graph.npz"

# Walkable highway classes. Exclude motorway/trunk (+ their _link) and rail/proposed/construction.
PED_OK = {"footway", "path", "pedestrian", "steps", "living_street", "residential",
          "service", "unclassified", "tertiary", "tertiary_link", "secondary",
          "secondary_link", "primary", "primary_link", "track", "road", "cycleway",
          "crossing", "corridor", "platform"}
SPEED_MPS = config.WALK_KMH * 1000.0 / 3600.0      # reference 4.8 km/h -> 1.333 m/s
STAIRS_MULT = float(os.environ.get("WALK_STAIRS_MULT", "2.5"))   # stairs are ~this much slower
MAX_GRADE = 0.5                                     # clamp |slope| (cliffs/DEM noise)


def log(*a):
    print(*a, flush=True)


def tobler_factor(slope):
    """Time multiplier vs flat from Tobler's hiking function W=6*exp(-3.5*|S+0.05|).
    factor = flat_speed/W = exp(3.5*(|S+0.05|-0.05)); flat (S=0) -> 1.0, gentle downhill
    (S=-0.05) -> ~0.84 (fastest), uphill > 1."""
    s = np.clip(slope, -MAX_GRADE, MAX_GRADE)
    return np.exp(3.5 * (np.abs(s + 0.05) - 0.05))


def parse_osm():
    """Two-pass parse -> (ways: list[(node_id list, is_steps)], coords: dict[node_id->(lon,lat)])."""
    import esy.osm.pbf
    from esy.osm.pbf import Node, Way
    t = time.time()
    ways, needed = [], set()
    for e in esy.osm.pbf.File(str(PBF)):
        if isinstance(e, Way):
            hw = e.tags.get("highway")
            if hw not in PED_OK:
                continue
            if e.tags.get("foot") in ("no", "private"):
                continue
            if e.tags.get("access") in ("no", "private") and e.tags.get("foot") not in ("yes", "designated", "permissive"):
                continue
            refs = list(e.refs)
            if len(refs) < 2:
                continue
            ways.append((refs, hw == "steps"))
            needed.update(refs)
    log(f"[osm] pass1: {len(ways)} pedestrian ways, {len(needed)} referenced nodes "
        f"({time.time()-t:.1f}s)")
    t = time.time()
    coords = {}
    for e in esy.osm.pbf.File(str(PBF)):
        if isinstance(e, Node) and e.id in needed:
            coords[e.id] = e.lonlat                 # (lon, lat)
    log(f"[osm] pass2: resolved {len(coords)}/{len(needed)} node coords ({time.time()-t:.1f}s)")
    return ways, coords


def build():
    if not PBF.exists():
        sys.exit(f"missing {PBF}")
    if not DEM.exists():
        sys.exit(f"missing {DEM} — run scripts/fetch_dem.sh first")
    ways, coords = parse_osm()

    # dedup referenced nodes that actually resolved -> contiguous index
    nid_list = [nid for nid in coords]              # insertion order
    nid_to_ix = {nid: i for i, nid in enumerate(nid_list)}
    N = len(nid_list)
    lon = np.array([coords[nid][0] for nid in nid_list], dtype=np.float64)
    lat = np.array([coords[nid][1] for nid in nid_list], dtype=np.float64)

    # local equirectangular metres (matches raptor_build._footpaths; <0.1% at city scale)
    lat0 = float(np.mean(lat))
    mlat, mlon = 111320.0, 111320.0 * math.cos(math.radians(lat0))
    xs, ys = lon * mlon, lat * mlat

    # DEM elevation per node
    import rasterio
    t = time.time()
    with rasterio.open(DEM) as ds:
        elev = np.array([v[0] for v in ds.sample(zip(lon, lat))], dtype=np.float64)
    elev = np.nan_to_num(elev, nan=float(np.nanmedian(elev)))
    log(f"[dem] sampled {N} nodes ({time.time()-t:.1f}s); elev {elev.min():.0f}..{elev.max():.0f} m")

    # directed edges from way segments (both directions)
    t = time.time()
    src, dst, w, wf = [], [], [], []     # wf = FLAT reference sec (distance/speed; matches R5's
    for refs, is_steps in ways:          #      grade-agnostic model, for validating graph mechanics)
        ix = [nid_to_ix.get(r) for r in refs]
        for a, b in zip(ix, ix[1:]):
            if a is None or b is None or a == b:
                continue
            dx = xs[a] - xs[b]; dy = ys[a] - ys[b]
            hlen = math.hypot(dx, dy)
            if hlen <= 0:
                continue
            dz = elev[b] - elev[a]
            slope = dz / hlen
            base = hlen / SPEED_MPS
            mult = STAIRS_MULT if is_steps else 1.0
            src.append(a); dst.append(b); w.append(base * float(tobler_factor(slope)) * mult); wf.append(base)
            src.append(b); dst.append(a); w.append(base * float(tobler_factor(-slope)) * mult); wf.append(base)
    src = np.asarray(src, np.int32); dst = np.asarray(dst, np.int32)
    w = np.asarray(w, np.float64); wf = np.asarray(wf, np.float64)
    log(f"[edges] {len(src)} directed half-edges ({time.time()-t:.1f}s)")

    # giant connected component (undirected sense) so snapping can't land on an island
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    A = csr_matrix((np.ones(len(src)), (src, dst)), shape=(N, N))
    ncomp, labels = connected_components(A, directed=False)
    if ncomp > 1:
        sizes = np.bincount(labels)
        giant = int(np.argmax(sizes))
        keep = labels == giant
        log(f"[cc] {ncomp} components; keeping giant ({keep.sum()}/{N} nodes, "
            f"{100*keep.sum()/N:.1f}%)")
        remap = np.full(N, -1, np.int64)
        remap[keep] = np.arange(keep.sum())
        em = keep[src] & keep[dst]
        src, dst, w, wf = remap[src[em]], remap[dst[em]], w[em], wf[em]
        lon, lat, elev = lon[keep], lat[keep], elev[keep]
        N = int(keep.sum())

    # build directed CSR (sorted by source)
    order = np.lexsort((dst, src))
    src, dst, w, wf = src[order], dst[order], w[order], wf[order]
    indptr = np.zeros(N + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    np.cumsum(indptr, out=indptr)

    np.savez(OUT,
             node_lon=lon.astype(np.float32), node_lat=lat.astype(np.float32),
             node_elev=elev.astype(np.float32),
             indptr=indptr, indices=dst.astype(np.int32), w_ref=w.astype(np.float32),
             w_flat=wf.astype(np.float32),       # grade-agnostic (R5-comparable) reference seconds
             walk_ref_kmh=np.float32(config.WALK_KMH), slope_aware=np.int8(1),
             stairs_mult=np.float32(STAIRS_MULT))
    log(f"[save] {OUT.name}: {N} nodes, {len(dst)} edges, "
        f"{OUT.stat().st_size/1e6:.1f} MB; ref={config.WALK_KMH}km/h slope_aware=1")


if __name__ == "__main__":
    build()
