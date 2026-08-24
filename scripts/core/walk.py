"""Runtime hill-aware pedestrian router over the baked walk graph.

Loads ``data/walk_graph.npz`` (built by ``scripts/build_walk_graph.py``), rebuilds a cKDTree for
snapping + a scipy CSR for routing, and answers one-to-many WALK queries with C-speed Dijkstra.
Weights are REFERENCE seconds at ``config.WALK_KMH`` (4.8): ``w_ref`` is grade-aware (Tobler +
stairs), ``w_flat`` is the grade-agnostic distance/speed reference used for validation. A
per-request walk-speed scalar is applied by the CALLER (engine), not here — this module is
scalar-agnostic and always returns reference seconds, so the same graph serves every speed.
"""
import math
from typing import NamedTuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import config

_DEFAULT = config.DATA / "walk_graph.npz"


def _wgs84_lonlat(value):
    """Normalize one ``(lon, lat)`` pair, or return ``None`` when it is not valid WGS84."""
    try:
        lon, lat = value
        lon, lat = float(lon), float(lat)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return lon, lat


class WalkRouteResult(NamedTuple):
    """The duration and predecessor-chain path for one graph route choice.

    ``seconds`` includes both straight-line snap connectors and the directed graph cost, in
    the graph's reference seconds.  ``points`` includes those exact endpoint connectors around
    the predecessor-chain graph path, in walking order.  A named tuple keeps this result cheap
    to pass around while still allowing callers to destructure it as ``seconds, points``.
    """

    seconds: float
    points: list


class WalkGraph:
    # The graph is a bounded SF pedestrian network.  A connector this long is already a
    # substantial off-network walk and, more importantly, is where the audit found false
    # positives from snapping points outside the baked graph (East Bay/water examples).  Keep
    # this policy in the graph owner so every caller can validate against the same boundary.
    MAX_CONNECTOR_M = 300.0

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

    def nearest_distance_m(self, lon, lat):
        """Return the nearest graph-node distance in metres, or ``inf`` for invalid input.

        This is deliberately an opt-in validation query: :meth:`snap` keeps its historical
        permissive behavior for existing callers, while API boundaries can reject locations
        that would otherwise be silently snapped to a distant edge of the SF-only graph.
        Coordinates are WGS84 ``(lon, lat)`` scalars.  Non-finite, out-of-range, or otherwise
        malformed values are not sent to cKDTree.
        """
        try:
            lon = float(lon)
            lat = float(lat)
        except (TypeError, ValueError, OverflowError):
            return math.inf
        if not (math.isfinite(lon) and math.isfinite(lat)):
            return math.inf
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            return math.inf
        q = np.array([[lon * self.mlon, lat * self.mlat]], dtype=np.float64)
        distance, _ = self._tree.query(q, k=1)
        return float(distance[0])

    def supports_point(self, lon, lat, max_connector_m=None):
        """Whether a WGS84 point is close enough to this graph to be routable.

        ``max_connector_m`` is an explicit override for tests or a caller with a narrower
        product policy.  The default is the authoritative :attr:`MAX_CONNECTOR_M` policy.
        Invalid coordinates, invalid thresholds, and negative thresholds return ``False``.
        """
        if max_connector_m is None:
            max_connector_m = self.MAX_CONNECTOR_M
        try:
            max_connector_m = float(max_connector_m)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(max_connector_m) or max_connector_m < 0.0:
            return False
        return self.nearest_distance_m(lon, lat) <= max_connector_m

    def snap(self, lonlats, k=K_SNAP):
        """lon/lat array [m,2] -> (node_idx int32[m,k], connector REFERENCE seconds float[m,k]),
        the k nearest graph nodes + the straight-line offset to each at the reference speed."""
        ll = np.asarray(lonlats, dtype=np.float64).reshape(-1, 2)
        q = np.column_stack((ll[:, 0] * self.mlon, ll[:, 1] * self.mlat))
        d, ix = self._tree.query(q, k=k)
        # cKDTree squeezes the k axis when k=1 -> (m,); reshape (not atleast_2d, which would
        # transpose that case to (1, m)) keeps the documented [m, k] contract for every k.
        d = d.reshape(len(ll), k); ix = ix.reshape(len(ll), k)
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

    def path_tree(self, src_lonlat, cap_ref_sec, flat=False, reverse=False):
        """A PathTree rooted at ``src_lonlat`` for extracting the actual street path of walk
        legs (one Dijkstra with predecessors from the point's k snap nodes; each target's
        path is then a cheap predecessor-chain walk). ``reverse=True`` roots on the
        TRANSPOSED graph (use for a tree rooted at the workplace: paths are node->W in the
        original graph, matching one_to_many(reverse=True)'s costs)."""
        return PathTree(self, src_lonlat, cap_ref_sec, flat=flat, reverse=reverse)

    def many_to_targets(self, src_nodes, src_conn, target_nodes, target_conn, cap_ref_sec,
                        flat=False, batch=64):
        """[n_src, n_tgt] REFERENCE seconds (np.inf beyond cap) for many pre-snapped sources to
        many pre-snapped targets, both [*, k]. Batched so the transient [batch*k, n_nodes]
        Dijkstra block stays bounded (~0.5 GB at batch=64 on the SF graph vs ~20 GB
        unbatched). Used by the offline cell->stop access bake."""
        src_nodes = np.asarray(src_nodes); src_conn = np.asarray(src_conn)
        target_nodes = np.asarray(target_nodes); target_conn = np.asarray(target_conn)
        ns, nt = len(src_nodes), len(target_nodes)
        k = src_nodes.shape[1]
        out = np.full((ns, nt), np.inf, dtype=np.float64)
        g = self._graph(flat)
        for s in range(0, ns, batch):
            sn = src_nodes[s:s + batch]; sc = src_conn[s:s + batch]; b = sn.shape[0]
            D = dijkstra(g, directed=True, indices=sn.ravel(), limit=cap_ref_sec)  # [b*k, n_nodes]
            Dv = D.reshape(b, k, -1)                       # view; D is dead after the in-place add
            Dv += sc[:, :, None]                           # avoids a second full-size temporary
            base = Dv.min(axis=1)                                                  # [b, n_nodes]
            sub = base[:, target_nodes] + target_conn[None, :, :]                  # [b, nt, k]
            out[s:s + b] = sub.min(axis=2)
        out[out > cap_ref_sec] = np.inf
        return out


class PathTree:
    """Predecessor tree from one Dijkstra root point, for walk-leg PATH EXTRACTION (numpy
    only, JVM-free). Mirrors ``one_to_many``'s cost model exactly: the root snaps to its k
    nearest nodes (straight connector seconds), one multi-source Dijkstra runs with
    ``return_predecessors=True``, and a target's path takes the (root-snap, target-snap)
    pair that minimizes connector + graph + connector — the SAME argmin one_to_many's
    ``.min`` takes, so the drawn path is the path the served TIME used.

    Orientation: ``path_points`` always returns points in WALKING order.
      * forward tree (reverse=False, root = the walk's origin): predecessor chains run
        target->root, so they are reversed -> origin->target.
      * reverse tree (reverse=True, root = the walk's DESTINATION, e.g. the workplace, on
        the transposed graph): a transpose-path root->target is the original-graph path
        target->root reversed, so the raw chain (target ... root) IS the walking order.
    """

    def __init__(self, wg, src_lonlat, cap_ref_sec, flat=False, reverse=False):
        self.wg = wg
        self.reverse = bool(reverse)
        self.cap_ref_sec = float(cap_ref_sec)
        # Retain the exact point because the Dijkstra duration includes its graph connector.
        # Validation is intentionally local to the new route-result contract: legacy path-only
        # callers keep their historical snapping behavior.
        self._src_lonlat = _wgs84_lonlat(src_lonlat)
        snodes, sconn = wg.snap([src_lonlat])             # [1,k] each
        self._snodes = snodes[0]
        dist, pred = dijkstra(wg._graph(flat, reverse), directed=True,
                              indices=self._snodes, limit=cap_ref_sec,
                              return_predecessors=True)
        self._dist = dist + sconn[0][:, None]             # [k, n_nodes] incl. root connector
        self._pred = pred                                 # -9999 sentinel = no predecessor

    def distances_to(self, target_nodes, target_conn, cap_ref_sec=None):
        """Vectorized REFERENCE seconds to pre-snapped targets.

        This is the PathTree equivalent of :meth:`WalkGraph.one_to_many` for the tree's
        already-routed root.  ``target_nodes`` and ``target_conn`` have the same ``[n, k]``
        contract as ``WalkGraph.snap``.  A caller may apply any cap no larger than the cap used
        to build the tree; values outside it are returned as ``np.inf`` exactly as
        ``one_to_many`` does.

        Keeping the per-root-snap distance rows (rather than collapsing them in the tree) is
        required by ``path_points``.  The projection below takes their minimum only at the
        requested target nodes, avoiding another graph-sized distance array.
        """
        cap = self.cap_ref_sec if cap_ref_sec is None else float(cap_ref_sec)
        if cap > self.cap_ref_sec:
            raise ValueError(
                f"requested cap {cap:g}s exceeds PathTree build cap {self.cap_ref_sec:g}s")
        nodes = np.asarray(target_nodes)
        conn = np.asarray(target_conn)
        if nodes.ndim != 2 or conn.shape != nodes.shape:
            raise ValueError("target_nodes and target_conn must have matching [n, k] shapes")
        # This is the same reduction order as WalkGraph.one_to_many: first choose the best root
        # snap for each graph node, then choose the best target snap after adding its connector.
        base_at_targets = self._dist[:, nodes].min(axis=0)
        ref = (base_at_targets + conn).min(axis=1)
        ref[~np.isfinite(ref)] = np.inf
        ref[ref > cap] = np.inf
        return ref

    def _target_choice(self, tgt_lonlat, enforce_cap=False):
        """Return the shared root/target snap argmin for a target point.

        The graph distance rows retain one value per root snap node so geometry can follow the
        same choice as timing.  Keep the argmin in one helper: callers must not independently
        ask for a distance and a path and accidentally select different connector pairs.
        ``path_points`` historically did not apply the final connector cap, so that behavior is
        opt-in here for the new result API, which follows ``one_to_many``'s cap semantics.
        """
        tn, tc = self.wg.snap([tgt_lonlat])
        tn, tc = tn[0], tc[0]
        total = self._dist[:, tn] + tc[None, :]
        valid = np.isfinite(total)
        if enforce_cap:
            valid &= total <= self.cap_ref_sec
        if not valid.any():
            return None
        choices = np.where(valid, total, np.inf)
        i, j = np.unravel_index(int(np.argmin(choices)), choices.shape)
        return float(total[i, j]), tn, i, j

    def _path_for_choice(self, target_nodes, root_ix, target_ix):
        """Materialize a path for a previously selected root/target snap pair."""
        pred_row = self._pred[root_ix]
        chain = [int(target_nodes[target_ix])]
        cur = chain[0]
        for _ in range(len(self.wg.lat)):                 # bounded (acyclic by construction)
            p = int(pred_row[cur])
            if p < 0:                                     # reached the root snap node
                break
            chain.append(p)
            cur = p
        if not self.reverse:
            chain.reverse()                               # forward tree: root -> target
        lat, lon = self.wg.lat, self.wg.lon
        return [[round(float(lat[c]), 6), round(float(lon[c]), 6)] for c in chain]

    def route_result(self, tgt_lonlat):
        """Return the selected graph duration and exact path for ``tgt_lonlat``.

        The duration and path are produced from one root-snap/target-snap argmin, so they
        cannot disagree about which of the K connector candidates was selected.  The returned
        points include the exact requested endpoints around that graph-node chain, deduplicated
        when an endpoint already equals its snap node.  ``None`` is returned for invalid WGS84
        endpoints or when the target is unreachable within the tree's build cap, matching
        :meth:`one_to_many`'s unreachable result.  Duration is reference seconds and follows
        the tree's directed graph orientation (including ``reverse=True`` semantics).
        """
        target_lonlat = _wgs84_lonlat(tgt_lonlat)
        if self._src_lonlat is None or target_lonlat is None:
            return None
        choice = self._target_choice(target_lonlat, enforce_cap=True)
        if choice is None:
            return None
        seconds, target_nodes, root_ix, target_ix = choice
        graph_points = self._path_for_choice(target_nodes, root_ix, target_ix)
        # A reverse tree is rooted at the walk's destination, so walking order is target->root.
        start, end = ((target_lonlat, self._src_lonlat) if self.reverse
                      else (self._src_lonlat, target_lonlat))
        endpoint_points = (
            [round(float(start[1]), 6), round(float(start[0]), 6)],
            [round(float(end[1]), 6), round(float(end[0]), 6)],
        )
        points = []
        for point in (endpoint_points[0], *graph_points, endpoint_points[1]):
            if not points or points[-1] != point:
                points.append(point)
        return WalkRouteResult(seconds, points)

    def path_points(self, tgt_lonlat):
        """[[lat, lon], ...] node path to/from ``tgt_lonlat`` (lon, lat — same convention
        as ``snap``) in walking order, or None when unreachable within the cap."""
        choice = self._target_choice(tgt_lonlat)
        if choice is None:
            return None
        _seconds, target_nodes, root_ix, target_ix = choice
        return self._path_for_choice(target_nodes, root_ix, target_ix)
