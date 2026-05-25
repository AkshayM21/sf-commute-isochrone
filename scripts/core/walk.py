"""Runtime hill-aware pedestrian router (JVM-free) over the baked walk graph.

Loads ``data/walk_graph.npz`` (built by ``scripts/build_walk_graph.py``), rebuilds a cKDTree for
snapping + a scipy CSR for routing, and answers one-to-many WALK queries with C-speed Dijkstra.
This replaces R5's last runtime job (the per-workplace walk matrix) and the offline access bake.

Weights are REFERENCE seconds at ``config.WALK_KMH`` (4.8): ``w_ref`` is grade-aware (Tobler +
stairs), ``w_flat`` is the grade-agnostic distance/speed (R5-comparable, for validation). A
per-request walk-speed scalar is applied by the CALLER (engine), not here — this module is
scalar-agnostic and always returns reference seconds, so the same graph serves every speed.

Do NOT import r5py here — the whole point is to drop the JVM.
"""
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import config

_DEFAULT = config.DATA / "walk_graph.npz"


class WalkGraph:
    def __init__(self, path=None):
        z = np.load(path or _DEFAULT)
        self.lon = z["node_lon"].astype(np.float64)
        self.lat = z["node_lat"].astype(np.float64)
        self.elev = z["node_elev"]
        self.indptr = z["indptr"].astype(np.int32)
        self.indices = z["indices"].astype(np.int32)
        self.w_ref = z["w_ref"].astype(np.float64)
        self.w_flat = z["w_flat"].astype(np.float64) if "w_flat" in z else self.w_ref
        self.ref_kmh = float(z["walk_ref_kmh"]) if "walk_ref_kmh" in z else config.WALK_KMH
        self.speed_mps = self.ref_kmh * 1000.0 / 3600.0
        self._n = len(self.lon)
        # CSR graphs are built LAZILY per (flat, reverse) combo — at server runtime (arrive-by) only
        # the transposed grade-aware graph is ever used, so we materialize just that one (~1/4 the
        # RAM of building all four). Transposed: a Dijkstra from X on the transpose gives, per node,
        # the cost NODE->X in the original graph. Walking is DIRECTIONAL (uphill != downhill), so the
        # egress (alight stop -> work) + pure-walk (home -> work) legs, rooted at W, route on the
        # transpose to get the true stop->W / cell->W uphill cost.
        self._graphs = {}                            # (flat, reverse) -> csr_matrix
        # local equirectangular metres (matches build_walk_graph + raptor_build._footpaths)
        self.lat0 = float(self.lat.mean())
        self.mlat = 111320.0
        self.mlon = 111320.0 * math.cos(math.radians(self.lat0))
        self._xy = np.column_stack((self.lon * self.mlon, self.lat * self.mlat))
        self._tree = cKDTree(self._xy)

    @classmethod
    def load(cls, path=None):
        return cls(path)

    def _graph(self, flat, reverse=False):
        key = (flat, reverse)
        g = self._graphs.get(key)
        if g is None:
            w = self.w_flat if flat else self.w_ref
            base = csr_matrix((w, self.indices, self.indptr), shape=(self._n, self._n))
            g = base.T.tocsr() if reverse else base
            self._graphs[key] = g
        return g

    K_SNAP = 4              # connect each endpoint to its K nearest nodes (nearest-EDGE approx:
    #                        a mid-block point reaches the graph via whichever node is best by
    #                        graph distance, not just the geometrically nearest one)

    def snap(self, lonlats, k=K_SNAP):
        """lon/lat array [m,2] -> (node_idx int32[m,k], connector REFERENCE seconds float[m,k]),
        the k nearest graph nodes + the straight-line offset to each at the reference speed."""
        ll = np.asarray(lonlats, dtype=np.float64).reshape(-1, 2)
        q = np.column_stack((ll[:, 0] * self.mlon, ll[:, 1] * self.mlat))
        d, ix = self._tree.query(q, k=k)
        d = np.atleast_2d(d); ix = np.atleast_2d(ix)
        return ix.astype(np.int32), (d / self.speed_mps)

    def _base_from(self, src_lonlat, flat, cap_ref_sec, reverse=False):
        """min over the source's k snap-nodes of (connector + Dijkstra dist) -> dist[n_nodes].
        reverse=True roots on the transpose, so base[node] is the cost NODE->src (uphill-aware)."""
        snodes, sconn = self.snap([src_lonlat])           # [1,k]
        D = dijkstra(self._graph(flat, reverse), directed=True, indices=snodes[0], limit=cap_ref_sec)
        return (D + sconn[0][:, None]).min(axis=0)        # [n_nodes]

    def one_to_many(self, src_lonlat, target_nodes, target_conn, cap_ref_sec, flat=False,
                    reverse=False):
        """Walk REFERENCE seconds from one point to many pre-snapped targets (np.inf if
        unreachable or > cap). ``target_nodes``/``target_conn`` are [n_tgt, k] from ``snap``.
        reverse=True gives TARGET->src cost (use for egress stop->W and pure-walk cell->W rooted
        at the workplace), respecting uphill/downhill asymmetry."""
        base = self._base_from(src_lonlat, flat, cap_ref_sec, reverse)
        ref = (base[np.asarray(target_nodes)] + np.asarray(target_conn)).min(axis=1)
        ref[~np.isfinite(ref)] = np.inf
        ref[ref > cap_ref_sec] = np.inf
        return ref

    def many_to_targets(self, src_nodes, src_conn, target_nodes, target_conn, cap_ref_sec,
                        flat=False, batch=64):
        """[n_src, n_tgt] REFERENCE seconds (np.inf beyond cap) for many pre-snapped sources to
        many pre-snapped targets, both [*, k]. Batched so the transient [batch*k, n_nodes]
        Dijkstra block stays small. Used by the offline cell->stop access bake."""
        src_nodes = np.asarray(src_nodes); src_conn = np.asarray(src_conn)
        target_nodes = np.asarray(target_nodes); target_conn = np.asarray(target_conn)
        ns, nt = len(src_nodes), len(target_nodes)
        k = src_nodes.shape[1]
        out = np.full((ns, nt), np.inf, dtype=np.float64)
        g = self._graph(flat)
        for s in range(0, ns, batch):
            sn = src_nodes[s:s + batch]; sc = src_conn[s:s + batch]; b = sn.shape[0]
            D = dijkstra(g, directed=True, indices=sn.ravel(), limit=cap_ref_sec)  # [b*k, n_nodes]
            base = (D.reshape(b, k, -1) + sc[:, :, None]).min(axis=1)              # [b, n_nodes]
            sub = base[:, target_nodes] + target_conn[None, :, :]                  # [b, nt, k]
            out[s:s + b] = sub.min(axis=2)
        out[out > cap_ref_sec] = np.inf
        return out
