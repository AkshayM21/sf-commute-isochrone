"""Contract tests for the graph-native access archive."""
import importlib.util
import datetime as dt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("bake_walk_access", ROOT / "scripts" / "bake_walk_access.py")
bake_walk_access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bake_walk_access)


def _write(path, *, bad=None):
    values = dict(
        cell_ids=np.asarray(["0", "1"], dtype="U"),
        access_off=np.asarray([0, 1, 2], dtype=np.int64),
        access_to=np.asarray([0, 1], dtype=np.int32),
        access_w=np.asarray([30, 45], dtype=np.int32),
        grid_m=np.int32(200), n_stops=np.int32(2), service_date=np.asarray("20260819"),
        walk_ref_kmh=np.float32(4.8), slope_aware=np.int8(1),
        raptor_source_names=np.asarray(["muni.zip"], dtype="U"),
        raptor_source_sizes=np.asarray([10], dtype=np.int64),
        raptor_source_mtimes_ns=np.asarray([20], dtype=np.int64),
        walk_graph_size=np.int64(30), walk_graph_mtime_ns=np.int64(40),
        footpath_m=np.float64(250.0), raptor_build_version=np.int32(6),
        grid_source_names=np.asarray(["sf_neighborhoods.geojson"], dtype="U"),
        grid_source_sizes=np.asarray([-1], dtype=np.int64),
        grid_source_mtimes_ns=np.asarray([-1], dtype=np.int64),
        tr_off=np.asarray([0, 1, 2], dtype=np.int64),
        tr_to=np.asarray([1, 0], dtype=np.int32),
        # Reverse rows are target -> source: forward 1->0 precedes forward 0->1.
        tr_walk_time=np.asarray([13.0, 12.0], dtype=np.float64),
        tr_min_time=np.asarray([0.0, 0.0], dtype=np.float64),
        tr_time=np.asarray([13, 12], dtype=np.int64),
        tr_path_fallback=np.asarray([0, 0], dtype=np.int8),
        tr_forward_off=np.asarray([0, 1, 2], dtype=np.int64),
        tr_forward_to=np.asarray([1, 0], dtype=np.int32),
        tr_forward_walk_time=np.asarray([12.0, 13.0], dtype=np.float64),
        tr_forward_min_time=np.asarray([0.0, 0.0], dtype=np.float64),
        tr_forward_time=np.asarray([12, 13], dtype=np.int64),
        tr_forward_path_off=np.asarray([0, 2, 4], dtype=np.int64),
        tr_forward_path_points=np.asarray([[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3]], dtype=np.float64),
        tr_forward_path_fallback=np.asarray([0, 0], dtype=np.int8),
        tr_forward_pathway_off=np.asarray([0, 0, 0], dtype=np.int64),
        tr_forward_pathway_id=np.asarray([], dtype="U"),
        tr_forward_pathway_time=np.asarray([], dtype=np.int32),
        tr_forward_pathway_mode=np.asarray([], dtype="U"),
        tr_forward_pathway_length_m=np.asarray([], dtype=np.float64),
        tr_forward_pathway_reversed=np.asarray([], dtype=np.int8),
        transfer_scoped_source=np.asarray([], dtype=np.int32),
        transfer_scoped_target=np.asarray([], dtype=np.int32),
        transfer_scoped_from_route=np.asarray([], dtype="U"),
        transfer_scoped_to_route=np.asarray([], dtype="U"),
        transfer_scoped_from_trip=np.asarray([], dtype="U"),
        transfer_scoped_to_trip=np.asarray([], dtype="U"),
        transfer_scoped_min_time=np.asarray([], dtype=np.int32),
        transfer_scoped_prohibited=np.asarray([], dtype=np.int8),
        transfer_scoped_type=np.asarray([], dtype="U"),
        transfer_scoped_physical_time=np.asarray([], dtype=np.float64),
        transfer_scoped_path_fallback=np.asarray([], dtype=np.int8),
        transfer_scoped_path_off=np.asarray([0], dtype=np.int64),
        transfer_scoped_path_points=np.asarray([], dtype=np.float64).reshape(0, 2),
        transfer_pathway_source=np.asarray([], dtype=np.int32),
        transfer_pathway_target=np.asarray([], dtype=np.int32),
        transfer_pathway_id=np.asarray([], dtype="U"),
        transfer_pathway_time=np.asarray([], dtype=np.int32),
        transfer_pathway_mode=np.asarray([], dtype="U"),
        transfer_pathway_length_m=np.asarray([], dtype=np.float64),
        transfer_pathway_reversed=np.asarray([], dtype=np.int8),
    )
    if bad:
        values.update(bad)
    np.savez(path, **values)


def test_graph_native_archive_contract_accepts_canonical_payload(tmp_path):
    path = tmp_path / "access_walk_200m_20260819.npz"
    _write(path)
    assert bake_walk_access.validate_artifact(path, n_stops=2,
                                              service_date="20260819", grid_m=200)
    assert bake_walk_access.validate_artifact(path, n_stops=2,
                                              service_date=dt.date(2026, 8, 19), grid_m=200)


def test_archive_contract_rejects_duplicate_cells_and_bad_csr(tmp_path):
    duplicate = tmp_path / "duplicate.npz"
    _write(duplicate, bad={"cell_ids": np.asarray(["0", "0"], dtype="U")})
    assert not bake_walk_access.validate_artifact(duplicate)

    broken = tmp_path / "broken.npz"
    _write(broken, bad={"access_off": np.asarray([0, 2, 1], dtype=np.int64)})
    assert not bake_walk_access.validate_artifact(broken)

    bad_scalar = tmp_path / "bad-scalar.npz"
    _write(bad_scalar, bad={"n_stops": np.asarray([2, 2], dtype=np.int32)})
    assert not bake_walk_access.validate_artifact(bad_scalar)

    bad_transfer = tmp_path / "bad-transfer.npz"
    _write(bad_transfer, bad={"tr_time": np.asarray([12, 5], dtype=np.int64)})
    assert not bake_walk_access.validate_artifact(bad_transfer)

    mismatched_reverse = tmp_path / "mismatched-reverse.npz"
    _write(mismatched_reverse, bad={"tr_walk_time": np.asarray([13.0, 99.0], dtype=np.float64)})
    assert not bake_walk_access.validate_artifact(mismatched_reverse)

    bad_path = tmp_path / "bad-path.npz"
    _write(bad_path, bad={"tr_forward_path_points": np.asarray([[100.0, 2.0], [1.2, 2.2],
                                                          [1.3, 2.3], [1.4, 2.4]])})
    assert not bake_walk_access.validate_artifact(bad_path)

    empty_path = tmp_path / "empty-path.npz"
    _write(empty_path, bad={"tr_forward_path_off": np.asarray([0, 0, 4], dtype=np.int64)})
    assert not bake_walk_access.validate_artifact(empty_path)

    scoped_time_without_path = tmp_path / "scoped-time-without-path.npz"
    _write(scoped_time_without_path, bad={
        "transfer_scoped_source": np.asarray([0], dtype=np.int32),
        "transfer_scoped_target": np.asarray([1], dtype=np.int32),
        "transfer_scoped_physical_time": np.asarray([10.0], dtype=np.float64),
        "transfer_scoped_path_fallback": np.asarray([0], dtype=np.int8),
        "transfer_scoped_path_off": np.asarray([0, 0], dtype=np.int64),
    })
    assert not bake_walk_access.validate_artifact(scoped_time_without_path)


def test_archive_contract_accepts_and_checks_one_way_forward_reverse_pair(tmp_path):
    path = tmp_path / "one-way.npz"
    _write(path, bad={
        "tr_forward_off": np.asarray([0, 1, 1], dtype=np.int64),
        "tr_forward_to": np.asarray([1], dtype=np.int32),
        "tr_forward_walk_time": np.asarray([12.0], dtype=np.float64),
        "tr_forward_min_time": np.asarray([0.0], dtype=np.float64),
        "tr_forward_time": np.asarray([12], dtype=np.int64),
        "tr_forward_path_off": np.asarray([0, 2], dtype=np.int64),
        "tr_forward_path_points": np.asarray([[1.0, 2.0], [1.1, 2.1]], dtype=np.float64),
        "tr_forward_path_fallback": np.asarray([0], dtype=np.int8),
        "tr_forward_pathway_off": np.asarray([0, 0], dtype=np.int64),
        "tr_off": np.asarray([0, 0, 1], dtype=np.int64),
        "tr_to": np.asarray([0], dtype=np.int32),
        "tr_walk_time": np.asarray([12.0], dtype=np.float64),
        "tr_min_time": np.asarray([0.0], dtype=np.float64),
        "tr_time": np.asarray([12], dtype=np.int64),
        "tr_path_fallback": np.asarray([0], dtype=np.int8),
        "tr_forward_pathway_id": np.asarray([], dtype="U"),
        "tr_forward_pathway_time": np.asarray([], dtype=np.int32),
        "tr_forward_pathway_mode": np.asarray([], dtype="U"),
        "tr_forward_pathway_length_m": np.asarray([], dtype=np.float64),
        "tr_forward_pathway_reversed": np.asarray([], dtype=np.int8),
    })
    assert bake_walk_access.validate_artifact(path)
    broken = tmp_path / "one-way-broken.npz"
    _write(broken, bad={
        "tr_off": np.asarray([0, 0, 1], dtype=np.int64),
        "tr_to": np.asarray([1], dtype=np.int32),
        "tr_walk_time": np.asarray([12.0], dtype=np.float64),
        "tr_min_time": np.asarray([0.0], dtype=np.float64),
        "tr_time": np.asarray([12], dtype=np.int64),
        "tr_path_fallback": np.asarray([0], dtype=np.int8),
        "tr_forward_off": np.asarray([0, 1, 1], dtype=np.int64),
        "tr_forward_to": np.asarray([1], dtype=np.int32),
        "tr_forward_walk_time": np.asarray([12.0], dtype=np.float64),
        "tr_forward_min_time": np.asarray([0.0], dtype=np.float64),
        "tr_forward_time": np.asarray([12], dtype=np.int64),
        "tr_forward_path_off": np.asarray([0, 2], dtype=np.int64),
        "tr_forward_path_points": np.asarray([[1.0, 2.0], [1.1, 2.1]], dtype=np.float64),
        "tr_forward_path_fallback": np.asarray([0], dtype=np.int8),
        "tr_forward_pathway_off": np.asarray([0, 0], dtype=np.int64),
        "tr_forward_pathway_id": np.asarray([], dtype="U"),
        "tr_forward_pathway_time": np.asarray([], dtype=np.int32),
        "tr_forward_pathway_mode": np.asarray([], dtype="U"),
        "tr_forward_pathway_length_m": np.asarray([], dtype=np.float64),
        "tr_forward_pathway_reversed": np.asarray([], dtype=np.int8),
    })
    # The reverse row now claims 1 -> 1 rather than the forward edge's 0 -> 1.
    assert not bake_walk_access.validate_artifact(broken)


def test_archive_contract_rejects_pathway_metadata_timing_mismatch(tmp_path):
    path = tmp_path / "timed-pathway.npz"
    common = {
        "tr_forward_walk_time": np.asarray([40.0, 13.0], dtype=np.float64),
        "tr_forward_time": np.asarray([40, 13], dtype=np.int64),
        "tr_forward_path_fallback": np.asarray([1, 0], dtype=np.int8),
        "tr_walk_time": np.asarray([13.0, 40.0], dtype=np.float64),
        "tr_time": np.asarray([13, 40], dtype=np.int64),
        "tr_path_fallback": np.asarray([0, 1], dtype=np.int8),
        "tr_forward_pathway_off": np.asarray([0, 1, 1], dtype=np.int64),
        "tr_forward_pathway_id": np.asarray(["p"], dtype="U"),
        "tr_forward_pathway_time": np.asarray([40], dtype=np.int32),
        "tr_forward_pathway_mode": np.asarray(["1"], dtype="U"),
        "tr_forward_pathway_length_m": np.asarray([10.0], dtype=np.float64),
        "tr_forward_pathway_reversed": np.asarray([0], dtype=np.int8),
    }
    _write(path, bad=common)
    assert bake_walk_access.validate_artifact(path)
    broken = tmp_path / "timed-pathway-broken.npz"
    broken_values = dict(common)
    broken_values["tr_forward_pathway_time"] = np.asarray([41], dtype=np.int32)
    _write(broken, bad=broken_values)
    assert not bake_walk_access.validate_artifact(broken)


def test_canonical_cache_name_contains_parameters_not_content_digest(tmp_path, monkeypatch):
    from core import raptor_build

    monkeypatch.setattr(raptor_build, "CACHE_DIR", tmp_path)
    path = raptor_build._cache_path([tmp_path / "muni.zip"], "20260819", (18000, 39600), 250.0)
    assert path.name == "raptor_20260819_18000-39600_footpath250m.pkl"
    assert "sha" not in path.name and "finger" not in path.name


def test_engine_rejects_stale_neighborhood_grid_source(tmp_path, monkeypatch):
    from core import raptor_engine

    grid = tmp_path / "sf_neighborhoods.geojson"
    grid.write_text("current grid")
    graph = tmp_path / "walk_graph.npz"
    graph.write_bytes(b"graph")
    monkeypatch.setattr(raptor_engine.config, "DATA", tmp_path)
    artifact = tmp_path / "access.npz"
    _write(artifact, bad={
        "grid_source_names": np.asarray([grid.name], dtype="U"),
        "grid_source_sizes": np.asarray([grid.stat().st_size + 1], dtype=np.int64),
        "grid_source_mtimes_ns": np.asarray([grid.stat().st_mtime_ns], dtype=np.int64),
        "walk_graph_size": np.int64(graph.stat().st_size),
        "walk_graph_mtime_ns": np.int64(graph.stat().st_mtime_ns),
    })
    engine = object.__new__(raptor_engine.RaptorEngine)
    engine.service_date = dt.date(2026, 8, 19)
    engine.data = {
        "n_stops": 2, "footpath_m": 250.0, "build_version": 6,
        "source_mtimes": (("muni.zip", 10, 20),),
        "stop_lat": np.asarray([1.0, 1.1]), "stop_lon": np.asarray([2.0, 2.1]),
    }
    with np.load(artifact, allow_pickle=False) as payload:
        try:
            engine._validate_access_artifact(payload, artifact)
        except ValueError as exc:
            assert "neighborhood/grid source" in str(exc)
        else:  # pragma: no cover - the assertion above is the contract
            raise AssertionError("stale grid source was accepted")
