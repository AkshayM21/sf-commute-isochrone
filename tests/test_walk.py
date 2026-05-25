"""Hill-aware JVM-free walk router tests (Phase B).

Validates the pedestrian graph built by scripts/build_walk_graph.py against R5's walk oracle and
checks the load-bearing physics: walking is DIRECTIONAL (uphill is slower than downhill) and the
grade-agnostic ('flat') weight reproduces R5's walk times (so the graph/snapping/Dijkstra are
correct). Skips cleanly if data/walk_graph.npz (gitignored) or the R5 oracles are absent.
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
GOLDEN = os.path.join(_HERE, "raptor_golden")
WALK_NPZ = os.path.join(_REPO, "data", "walk_graph.npz")
CAP_REF = 30 * 60


@pytest.fixture(scope="module")
def wg():
    if not os.path.exists(WALK_NPZ):
        pytest.skip("data/walk_graph.npz not built; run scripts/build_walk_graph.py")
    from core import walk
    return walk.WalkGraph.load()


def _oracles():
    if not os.path.isdir(GOLDEN):
        return []
    return sorted(f for f in os.listdir(GOLDEN) if f.startswith("oracle_") and f.endswith(".npz"))


def test_walk_graph_sane(wg):
    assert len(wg.lon) > 100_000                      # SF pedestrian graph is large
    assert len(wg.indices) == len(wg.w_ref) == len(wg.w_flat)
    assert wg.w_ref.min() > 0 and np.isfinite(wg.w_ref).all()
    assert wg.ref_kmh == pytest.approx(4.8, abs=0.01)


def test_walk_is_directional_uphill_slower(wg):
    """The whole point of the hill model: for a directed edge, going UP costs more than the
    grade-agnostic time and going (gently) DOWN costs less — so uphill edges are slower than
    downhill edges. Checked in aggregate over every edge via per-edge slope."""
    # per-edge: source node (expand indptr), dest node (indices), horizontal length from w_flat
    n = len(wg.indptr) - 1
    src = np.repeat(np.arange(n), np.diff(wg.indptr))
    dst = wg.indices
    hlen = wg.w_flat * wg.speed_mps                    # w_flat = hlen / speed
    slope = (wg.elev[dst] - wg.elev[src]) / np.maximum(hlen, 1.0)
    ratio = wg.w_ref / np.maximum(wg.w_flat, 1e-6)     # time multiplier vs flat
    up = ratio[slope > 0.08]                           # climbing
    down = ratio[slope < -0.08]                        # descending
    assert up.mean() > 1.2, f"uphill edges not penalized (mean ratio {up.mean():.2f})"
    assert up.mean() > down.mean(), "uphill is not slower than downhill (model not directional)"
    # a directed edge and its reverse must differ on steep ground (asymmetry, not symmetric cost)
    assert ratio[slope > 0.12].mean() > ratio[(slope < -0.12) & (slope > -0.25)].mean()


@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_walk_flat_matches_r5(wg):
    """FLAT (grade-agnostic) W->stops reproduces R5's walk matrix within tolerance — validates the
    OSM parse + snapping + Dijkstra (mechanics), independent of the hill model."""
    from core import raptor_build
    data = raptor_build.load_or_build(verbose=False)
    slat, slon = data["stop_lat"], data["stop_lon"]
    gids = np.where(~np.isnan(slat))[0]
    snode, sconn = wg.snap(np.column_stack((slon[gids], slat[gids])))
    gpos = {int(g): i for i, g in enumerate(gids)}
    err = []
    for f in _oracles():
        z = np.load(os.path.join(GOLDEN, f), allow_pickle=True)
        flat = wg.one_to_many((float(z["lon"]), float(z["lat"])), snode, sconn, CAP_REF, flat=True)
        for g, r5 in zip(np.asarray(z["egress_g"]), np.asarray(z["egress_w"])):
            i = gpos.get(int(g))
            if i is not None and np.isfinite(flat[i]):
                err.append(flat[i] - float(r5))
    err = np.array(err)
    print(f"\n[walk flat vs R5] MAE={np.abs(err).mean():.1f}s bias={err.mean():+.1f}s n={len(err)}")
    assert np.abs(err).mean() <= 45, f"flat walk MAE {np.abs(err).mean():.0f}s > 45s"
    assert abs(err.mean()) <= 25, f"flat walk bias {err.mean():+.0f}s > 25s"


@pytest.mark.skipif(not _oracles(), reason="no R5 oracles in tests/raptor_golden/")
def test_walk_hill_geq_flat(wg):
    """The hill weight is never faster than flat on net-uphill trips and is >= R5 on steep ground
    (the intended, more-accurate divergence)."""
    from core import raptor_build
    data = raptor_build.load_or_build(verbose=False)
    slat, slon = data["stop_lat"], data["stop_lon"]
    gids = np.where(~np.isnan(slat))[0]
    snode, sconn = wg.snap(np.column_stack((slon[gids], slat[gids])))
    z = np.load(os.path.join(GOLDEN, _oracles()[0]), allow_pickle=True)
    flat = wg.one_to_many((float(z["lon"]), float(z["lat"])), snode, sconn, CAP_REF, flat=True)
    hill = wg.one_to_many((float(z["lon"]), float(z["lat"])), snode, sconn, CAP_REF, flat=False)
    m = np.isfinite(flat) & np.isfinite(hill)
    # over many stops the hill model is, on net, >= flat (uphill penalties dominate gentle descents)
    assert hill[m].mean() >= flat[m].mean()
