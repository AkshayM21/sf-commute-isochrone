"""Bake the graph-native cell-to-stop access table used by the RAPTOR engine.

The archive is derived directly from the canonical neighborhood grid and the directed walking
graph. It has a readable identity (walk mode, grid size, service date) and stores direct source
size/mtime metadata so the runtime can reject stale artifacts directly.

Usage: ``.venv/bin/python scripts/bake_walk_access.py``
       ``WALK_FLAT=1 .venv/bin/python scripts/bake_walk_access.py`` (mechanics-only flat bake)
"""
import os
import sys
import threading
import time
import datetime as dt
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

from core import config, feeds, graph_transfers, raptor_build, transfer_rules, walk

FLAT = os.environ.get("WALK_FLAT", "").lower() in ("1", "true", "yes")
GRID_M = int(os.environ.get("GRID_M", str(config.GRID_M)))
CACHE = config.DATA / "raptor_cache"
WALK_CAP_MIN = 30
_PUBLICATION_LOCK = threading.Lock()


def log(*a):
    print(*a, flush=True)


@contextmanager
def _publication_guard():
    """Serialize publication within a process and, where available, across processes.

    The lock is held only for validation/source recheck/rename.  Work remains parallelizable,
    while a directory-backed ``flock`` prevents two independent bakes from interleaving their
    final checks.  No lock artifact or compatibility metadata is published.
    """
    with _PUBLICATION_LOCK:
        fd = None
        flock = None
        try:
            import fcntl
            flock = fcntl.flock
            fd = os.open(CACHE, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            flock(fd, fcntl.LOCK_EX)
        except (ImportError, OSError):
            if fd is not None:
                os.close(fd)
            fd = None
            flock = None
        try:
            yield
        finally:
            if fd is not None:
                try:
                    flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


def _source_arrays(source_mtimes):
    names, sizes, mtimes = [], [], []
    for name, size, mtime in source_mtimes:
        names.append(str(name))
        sizes.append(-1 if size is None else int(size))
        mtimes.append(-1 if mtime is None else config.normalize_mtime_ns(mtime))
    return (np.asarray(names, dtype="U"), np.asarray(sizes, dtype=np.int64),
            np.asarray(mtimes, dtype=np.int64))


def _rule_arrays(transfer_bake):
    """Encode graph-transfer dataclass metadata into pickle-free NPZ arrays."""
    scoped = tuple(transfer_bake.scoped_rules)
    pathway = tuple(transfer_bake.pathway_records)
    scoped_rule = tuple(record.rule for record in scoped)
    scoped_arrays = {
        "transfer_scoped_source": np.asarray([r.source for r in scoped], dtype=np.int32),
        "transfer_scoped_target": np.asarray([r.target for r in scoped], dtype=np.int32),
        "transfer_scoped_from_route": np.asarray([r.from_route_id or "" for r in scoped_rule], dtype="U"),
        "transfer_scoped_to_route": np.asarray([r.to_route_id or "" for r in scoped_rule], dtype="U"),
        "transfer_scoped_from_trip": np.asarray([r.from_trip_id or "" for r in scoped_rule], dtype="U"),
        "transfer_scoped_to_trip": np.asarray([r.to_trip_id or "" for r in scoped_rule], dtype="U"),
        "transfer_scoped_min_time": np.asarray([
            -1 if r.min_transfer_seconds is None else int(r.min_transfer_seconds)
            for r in scoped_rule], dtype=np.int32),
        "transfer_scoped_prohibited": np.asarray([int(r.prohibited) for r in scoped_rule], dtype=np.int8),
        "transfer_scoped_type": np.asarray([r.transfer_type or "" for r in scoped_rule], dtype="U"),
        "transfer_scoped_physical_time": np.asarray([
            -1.0 if record.physical_seconds is None else float(record.physical_seconds)
            for record in scoped], dtype=np.float64),
        "transfer_scoped_path_fallback": np.asarray(
            [int(record.pathway_fallback) for record in scoped], dtype=np.int8),
    }
    scoped_path_off = np.zeros(len(scoped) + 1, dtype=np.int64)
    scoped_path_off[1:] = np.cumsum(
        [len(record.path_points) for record in scoped], dtype=np.int64)
    scoped_path_points = np.asarray(
        [point for record in scoped for point in record.path_points], dtype=np.float64
    ).reshape(-1, 2)
    scoped_arrays.update({
        "transfer_scoped_path_off": scoped_path_off,
        "transfer_scoped_path_points": scoped_path_points,
    })
    pathway_arrays = {
        "transfer_pathway_source": np.asarray([r.source for r in pathway], dtype=np.int32),
        "transfer_pathway_target": np.asarray([r.target for r in pathway], dtype=np.int32),
        "transfer_pathway_id": np.asarray([r.edge.pathway_id for r in pathway], dtype="U"),
        "transfer_pathway_time": np.asarray([
            -1 if r.edge.traversal_seconds is None else int(r.edge.traversal_seconds)
            for r in pathway], dtype=np.int32),
        "transfer_pathway_mode": np.asarray([r.edge.pathway_mode or "" for r in pathway], dtype="U"),
        "transfer_pathway_length_m": np.asarray([
            np.nan if r.edge.length_meters is None else float(r.edge.length_meters)
            for r in pathway], dtype=np.float64),
        "transfer_pathway_reversed": np.asarray([
            int(r.edge.reversed_from_bidirectional) for r in pathway], dtype=np.int8),
    }
    # Keep all pathway records attached to the emitted CSR edge as well.  This captures
    # duplicate pathway rows deterministically (the shortest timed record drives timing,
    # while the full tuple remains available to future geometry consumers).
    edge_groups = tuple(transfer_bake.tr_pathway_metadata)
    edge_flat = tuple(edge for group in edge_groups for edge in group)
    edge_off = np.zeros(len(edge_groups) + 1, dtype=np.int64)
    edge_off[1:] = np.cumsum([len(group) for group in edge_groups], dtype=np.int64)
    edge_arrays = {
        "tr_forward_pathway_off": edge_off,
        "tr_forward_pathway_id": np.asarray([e.pathway_id for e in edge_flat], dtype="U"),
        "tr_forward_pathway_time": np.asarray([
            -1 if e.traversal_seconds is None else int(e.traversal_seconds)
            for e in edge_flat], dtype=np.int32),
        "tr_forward_pathway_mode": np.asarray([e.pathway_mode or "" for e in edge_flat], dtype="U"),
        "tr_forward_pathway_length_m": np.asarray([
            np.nan if e.length_meters is None else float(e.length_meters)
            for e in edge_flat], dtype=np.float64),
        "tr_forward_pathway_reversed": np.asarray([
            int(e.reversed_from_bidirectional) for e in edge_flat], dtype=np.int8),
    }
    return {**scoped_arrays, **pathway_arrays, **edge_arrays}


_TRANSFER_ARRAY_KEYS = (
    # Reverse target -> source CSR consumed by RAPTOR.
    "tr_off", "tr_to", "tr_walk_time", "tr_min_time", "tr_time", "tr_path_fallback",
    # Forward source -> target CSR and reusable geometry consumed by the server.
    "tr_forward_off", "tr_forward_to", "tr_forward_walk_time", "tr_forward_min_time",
    "tr_forward_time", "tr_forward_path_off", "tr_forward_path_points",
    "tr_forward_path_fallback", "tr_forward_pathway_off",
    "tr_forward_pathway_id", "tr_forward_pathway_time", "tr_forward_pathway_mode",
    "tr_forward_pathway_length_m", "tr_forward_pathway_reversed",
    # Preserved optional rule metadata.
    "transfer_scoped_source", "transfer_scoped_target",
    "transfer_scoped_from_route", "transfer_scoped_to_route", "transfer_scoped_from_trip",
    "transfer_scoped_to_trip", "transfer_scoped_min_time", "transfer_scoped_prohibited",
    "transfer_scoped_type",
    "transfer_scoped_physical_time", "transfer_scoped_path_fallback",
    "transfer_scoped_path_off", "transfer_scoped_path_points",
    "transfer_pathway_source", "transfer_pathway_target", "transfer_pathway_id",
    "transfer_pathway_time", "transfer_pathway_mode", "transfer_pathway_length_m",
    "transfer_pathway_reversed",
)


def validate_artifact(path, *, n_stops=None, service_date=None, grid_m=None,
                      walk_graph_path=None, grid_source_path=None,
                      stop_lat=None, stop_lon=None):
    """Reopen and structurally validate one access archive before/after publication."""
    with np.load(path, allow_pickle=False) as z:
        required = {"cell_ids", "access_off", "access_to", "access_w", "grid_m", "n_stops",
                    "service_date", "walk_ref_kmh", "slope_aware", "raptor_source_names",
                    "raptor_source_sizes", "raptor_source_mtimes_ns", "walk_graph_size",
                    "walk_graph_mtime_ns", "footpath_m", "raptor_build_version",
                    "grid_source_names", "grid_source_sizes", "grid_source_mtimes_ns",
                    *_TRANSFER_ARRAY_KEYS}
        if not required.issubset(z.files):
            return False
        for scalar_key in (
                "grid_m", "n_stops", "service_date", "walk_ref_kmh", "slope_aware",
                "walk_graph_size", "walk_graph_mtime_ns", "footpath_m",
                "raptor_build_version"):
            if np.asarray(z[scalar_key]).ndim != 0:
                return False
        cells = np.asarray(z["cell_ids"]); off = np.asarray(z["access_off"])
        to = np.asarray(z["access_to"]); weights = np.asarray(z["access_w"])
        if cells.ndim != 1 or cells.dtype.kind not in "OUS" or len(set(cells.astype(str))) != len(cells):
            return False
        if off.ndim != 1 or off.dtype.kind not in "iu" or len(off) != len(cells) + 1:
            return False
        if off[0] != 0 or np.any(np.diff(off) < 0) or int(off[-1]) != len(to) or len(to) != len(weights):
            return False
        if to.ndim != 1 or to.dtype.kind not in "iu" or weights.ndim != 1 or weights.dtype.kind not in "iu":
            return False
        stored_stops = int(np.asarray(z["n_stops"]))
        if stored_stops < 0 or np.any(to < 0) or np.any(to >= stored_stops) or np.any(weights < 0):
            return False
        if n_stops is not None and stored_stops != int(n_stops):
            return False
        grid_value = np.asarray(z["grid_m"])
        if grid_value.ndim != 0 or int(grid_value) <= 0:
            return False
        if grid_m is not None and int(grid_value) != int(grid_m):
            return False
        if service_date is not None:
            expected_service_date = (
                service_date.strftime("%Y%m%d")
                if isinstance(service_date, (dt.date, dt.datetime))
                else str(service_date)
            )
            if str(np.asarray(z["service_date"]).item()) != expected_service_date:
                return False
        if abs(float(np.asarray(z["walk_ref_kmh"])) - float(config.WALK_KMH)) > 1e-6:
            return False
        if int(np.asarray(z["slope_aware"])) not in (0, 1):
            return False
        if (not np.isfinite(float(np.asarray(z["footpath_m"])))
                or float(np.asarray(z["footpath_m"])) < 0.0
                or int(np.asarray(z["raptor_build_version"])) < 1
                or int(np.asarray(z["walk_graph_size"])) < -1
                or int(np.asarray(z["walk_graph_mtime_ns"])) < -1):
            return False
        grid_names = np.asarray(z["grid_source_names"])
        grid_sizes = np.asarray(z["grid_source_sizes"])
        grid_mtimes = np.asarray(z["grid_source_mtimes_ns"])
        if (grid_names.ndim != 1 or grid_sizes.ndim != 1 or grid_mtimes.ndim != 1
                or not (len(grid_names) == len(grid_sizes) == len(grid_mtimes))
                or len(grid_names) == 0 or any(not str(name).strip() for name in grid_names)
                or grid_names.dtype.kind not in "OUS" or grid_sizes.dtype.kind not in "iu"
                or grid_mtimes.dtype.kind not in "iu"
                or np.any(grid_sizes < -1) or np.any(grid_mtimes < -1)):
            return False
        if grid_source_path is not None:
            try:
                grid_source = Path(grid_source_path)
                stat = grid_source.stat()
                expected_grid = (
                    grid_source.name,
                    int(stat.st_size),
                    config.portable_mtime_ns(stat),
                )
            except OSError:
                expected_grid = (Path(grid_source_path).name, -1, -1)
            stored_grid = tuple((str(name), int(size), config.normalize_mtime_ns(mtime))
                                for name, size, mtime in zip(grid_names, grid_sizes, grid_mtimes))
            if stored_grid != (expected_grid,):
                return False
        def _check_view(prefix, *, with_paths=False):
            off = np.asarray(z[f"{prefix}off"]); to = np.asarray(z[f"{prefix}to"])
            walk = np.asarray(z[f"{prefix}walk_time"]); minimum = np.asarray(z[f"{prefix}min_time"])
            effective = np.asarray(z[f"{prefix}time"]); fallback = np.asarray(z[f"{prefix}path_fallback"])
            if (off.ndim != 1 or off.dtype.kind not in "iu" or len(off) != stored_stops + 1
                    or off[0] != 0 or np.any(np.diff(off) < 0) or int(off[-1]) != len(to)
                    or to.ndim != 1 or to.dtype.kind not in "iu"
                    or np.any(to < 0) or np.any(to >= stored_stops)):
                return False
            edge_count = len(to)
            if any(a.ndim != 1 or len(a) != edge_count for a in (walk, minimum, effective, fallback)):
                return False
            if (walk.dtype.kind not in "fiu" or minimum.dtype.kind not in "fiu"
                    or effective.dtype.kind not in "iu" or fallback.dtype.kind not in "biu"
                    or not np.isfinite(walk).all() or not np.isfinite(minimum).all()
                    or np.any(walk < 0) or np.any(minimum < 0) or np.any(effective < 0)
                    or not np.array_equal(effective, np.floor(np.maximum(walk, minimum) + 0.5).astype(effective.dtype))
                    or np.any((fallback != 0) & (fallback != 1))):
                return False
            for row in range(stored_stops):
                a, b = int(off[row]), int(off[row + 1])
                if b - a > 1 and np.any(np.diff(to[a:b]) <= 0):
                    return False
            if with_paths:
                path_off = np.asarray(z["tr_forward_path_off"])
                points = np.asarray(z["tr_forward_path_points"])
                if (path_off.ndim != 1 or path_off.dtype.kind not in "iu"
                        or len(path_off) != edge_count + 1 or path_off[0] != 0
                        or np.any(np.diff(path_off) < 2) or int(path_off[-1]) != len(points)
                        or points.ndim != 2 or points.shape[1:] != (2,)
                        or points.dtype.kind not in "f" or not np.isfinite(points).all()
                        or np.any(points[:, 0] < -90.0) or np.any(points[:, 0] > 90.0)
                        or np.any(points[:, 1] < -180.0) or np.any(points[:, 1] > 180.0)):
                    return False
                if stop_lat is not None or stop_lon is not None:
                    if stop_lat is None or stop_lon is None:
                        return False
                    stop_lat_array = np.asarray(stop_lat, dtype=np.float64).reshape(-1)
                    stop_lon_array = np.asarray(stop_lon, dtype=np.float64).reshape(-1)
                    sources = np.repeat(np.arange(stored_stops, dtype=np.int64), np.diff(off))
                    if (len(stop_lat_array) != stored_stops or len(stop_lon_array) != stored_stops
                            or len(sources) != edge_count):
                        return False
                    if edge_count:
                        if (not np.isfinite(stop_lat_array[sources]).all()
                                or not np.isfinite(stop_lon_array[sources]).all()
                                or not np.isfinite(stop_lat_array[to]).all()
                                or not np.isfinite(stop_lon_array[to]).all()
                                or not np.allclose(
                                    points[path_off[:-1]],
                                    np.column_stack((stop_lat_array[sources], stop_lon_array[sources])),
                                    atol=1e-5, rtol=0.0)
                                or not np.allclose(
                                    points[path_off[1:] - 1],
                                    np.column_stack((stop_lat_array[to], stop_lon_array[to])),
                                    atol=1e-5, rtol=0.0)):
                            return False
                edge_path_off = np.asarray(z["tr_forward_pathway_off"])
                if (edge_path_off.ndim != 1 or edge_path_off.dtype.kind not in "iu"
                        or len(edge_path_off) != edge_count + 1 or edge_path_off[0] != 0
                        or np.any(np.diff(edge_path_off) < 0)):
                    return False
                edge_meta_len = int(edge_path_off[-1])
                for key in ("tr_forward_pathway_id", "tr_forward_pathway_time",
                            "tr_forward_pathway_mode", "tr_forward_pathway_length_m",
                            "tr_forward_pathway_reversed"):
                    if np.asarray(z[key]).ndim != 1 or len(np.asarray(z[key])) != edge_meta_len:
                        return False
                if (np.asarray(z["tr_forward_pathway_id"]).dtype.kind not in "OUS"
                        or np.asarray(z["tr_forward_pathway_mode"]).dtype.kind not in "OUS"):
                    return False
                pt = np.asarray(z["tr_forward_pathway_time"])
                pr = np.asarray(z["tr_forward_pathway_reversed"])
                pl = np.asarray(z["tr_forward_pathway_length_m"])
                if (pt.dtype.kind not in "iu" or pr.dtype.kind not in "biu"
                        or pl.dtype.kind not in "f" or np.any(pt < -1)
                        or np.any((pr != 0) & (pr != 1))
                        or np.any(~np.isnan(pl) & (pl < 0))):
                    return False
            return True

        if not _check_view("tr_") or not _check_view("tr_forward_", with_paths=True):
            return False
        if not graph_transfers.validate_transfer_views(
                stored_stops,
                forward_off=z["tr_forward_off"], forward_to=z["tr_forward_to"],
                forward_walk=z["tr_forward_walk_time"], forward_min=z["tr_forward_min_time"],
                forward_time=z["tr_forward_time"], forward_fallback=z["tr_forward_path_fallback"],
                reverse_off=z["tr_off"], reverse_to=z["tr_to"],
                reverse_walk=z["tr_walk_time"], reverse_min=z["tr_min_time"],
                reverse_time=z["tr_time"], reverse_fallback=z["tr_path_fallback"],
                forward_pathway_off=z["tr_forward_pathway_off"],
                forward_pathway_time=z["tr_forward_pathway_time"]):
            return False
        pathway_names = [key for key in z.files if key.startswith("transfer_pathway_")]
        if (not pathway_names
                or len({len(np.asarray(z[key])) for key in pathway_names}) != 1):
            return False
        scoped_n = len(np.asarray(z["transfer_scoped_source"]))
        pathway_n = len(np.asarray(z["transfer_pathway_source"]))
        for source_key, target_key, count in (
                ("transfer_scoped_source", "transfer_scoped_target", scoped_n),
                ("transfer_pathway_source", "transfer_pathway_target", pathway_n)):
            source = np.asarray(z[source_key]); target = np.asarray(z[target_key])
            if (source.dtype.kind not in "iu" or target.dtype.kind not in "iu"
                    or np.any(source < 0) or np.any(source >= stored_stops)
                or np.any(target < 0) or np.any(target >= stored_stops)
                or len(source) != count or len(target) != count):
                return False
        scoped_scalar_keys = (
            "transfer_scoped_target", "transfer_scoped_from_route",
            "transfer_scoped_to_route", "transfer_scoped_from_trip",
            "transfer_scoped_to_trip", "transfer_scoped_min_time",
            "transfer_scoped_prohibited", "transfer_scoped_type",
            "transfer_scoped_physical_time",
            "transfer_scoped_path_fallback")
        if any(len(np.asarray(z[key])) != scoped_n for key in scoped_scalar_keys):
            return False
        scoped_path_off = np.asarray(z["transfer_scoped_path_off"])
        scoped_path_points = np.asarray(z["transfer_scoped_path_points"])
        if (scoped_path_off.ndim != 1 or scoped_path_off.dtype.kind not in "iu"
                or len(scoped_path_off) != scoped_n + 1 or scoped_path_off[0] != 0
                or np.any(np.diff(scoped_path_off) < 0)
                or int(scoped_path_off[-1]) != len(scoped_path_points)
                or scoped_path_points.ndim != 2 or scoped_path_points.shape[1:] != (2,)
                or scoped_path_points.dtype.kind not in "f"
                or not np.isfinite(scoped_path_points).all()
                or np.any(scoped_path_points[:, 0] < -90.0)
                or np.any(scoped_path_points[:, 0] > 90.0)
                or np.any(scoped_path_points[:, 1] < -180.0)
                or np.any(scoped_path_points[:, 1] > 180.0)):
            return False
        scoped_physical = np.asarray(z["transfer_scoped_physical_time"])
        scoped_fallback = np.asarray(z["transfer_scoped_path_fallback"])
        if (scoped_physical.ndim != 1 or scoped_physical.dtype.kind not in "f"
                or not np.all((scoped_physical == -1)
                               | (np.isfinite(scoped_physical) & (scoped_physical >= 0)))
                or scoped_fallback.ndim != 1 or scoped_fallback.dtype.kind not in "biu"
                or np.any((scoped_fallback != 0) & (scoped_fallback != 1))):
            return False
        for index, (start, end, physical) in enumerate(zip(
                scoped_path_off[:-1], scoped_path_off[1:], scoped_physical)):
            if int(end) > int(start) and (int(end) - int(start) < 2 or float(physical) < 0.0):
                return False
            if int(end) == int(start) and float(physical) >= 0.0:
                return False
            if stop_lat is not None and int(end) > int(start):
                source_index = int(np.asarray(z["transfer_scoped_source"])[index])
                target_index = int(np.asarray(z["transfer_scoped_target"])[index])
                source_ll = (float(np.asarray(stop_lat)[source_index]),
                             float(np.asarray(stop_lon)[source_index]))
                target_ll = (float(np.asarray(stop_lat)[target_index]),
                             float(np.asarray(stop_lon)[target_index]))
                if (not np.allclose(scoped_path_points[int(start)], source_ll, atol=1e-5, rtol=0.0)
                        or not np.allclose(scoped_path_points[int(end) - 1], target_ll,
                                           atol=1e-5, rtol=0.0)):
                    return False
        scoped_min = np.asarray(z["transfer_scoped_min_time"])
        scoped_prohibited = np.asarray(z["transfer_scoped_prohibited"])
        if (scoped_min.dtype.kind not in "iu" or scoped_prohibited.dtype.kind not in "biu"
                or np.any(scoped_min < -1)
                or np.any((scoped_prohibited != 0) & (scoped_prohibited != 1))):
            return False
        if any(np.asarray(z[key]).dtype.kind not in "OUS" for key in (
                "transfer_scoped_from_route", "transfer_scoped_to_route",
                "transfer_scoped_from_trip", "transfer_scoped_to_trip",
                "transfer_scoped_type")):
            return False
        pathway_time = np.asarray(z["transfer_pathway_time"])
        pathway_rev = np.asarray(z["transfer_pathway_reversed"])
        pathway_len = np.asarray(z["transfer_pathway_length_m"])
        if (pathway_time.dtype.kind not in "iu" or pathway_rev.dtype.kind not in "biu"
                or pathway_len.dtype.kind not in "f" or np.any(pathway_time < -1)
                or np.any((pathway_rev != 0) & (pathway_rev != 1))
                or np.any(~np.isnan(pathway_len) & (pathway_len < 0))):
            return False
        if any(np.asarray(z[key]).dtype.kind not in "OUS" for key in (
                "transfer_pathway_id", "transfer_pathway_mode")):
            return False
        names = np.asarray(z["raptor_source_names"]); sizes = np.asarray(z["raptor_source_sizes"])
        mtimes = np.asarray(z["raptor_source_mtimes_ns"])
        if (names.ndim != 1 or sizes.ndim != 1 or mtimes.ndim != 1
                or not (len(names) == len(sizes) == len(mtimes))
                or any(not str(name).strip() for name in names)
                or names.dtype.kind not in "OUS" or sizes.dtype.kind not in "iu"
                or mtimes.dtype.kind not in "iu" or np.any(sizes < -1)
                or np.any(mtimes < -1)):
            return False
        if walk_graph_path is not None:
            try:
                st = Path(walk_graph_path).stat()
                if (
                    int(np.asarray(z["walk_graph_size"])),
                    config.normalize_mtime_ns(np.asarray(z["walk_graph_mtime_ns"])),
                ) != (st.st_size, config.portable_mtime_ns(st)):
                    return False
            except OSError:
                pass
        return True


def bake():
    from core import grid as gridmod
    gtfs = config.gtfs_paths()
    requested_date = os.environ.get("SFCI_SERVICE_DATE", "").strip()
    if requested_date:
        try:
            svc = dt.datetime.strptime(requested_date, "%Y%m%d").date()
        except ValueError as exc:
            raise ValueError("SFCI_SERVICE_DATE must be YYYYMMDD") from exc
        if not feeds._feed_has_trips(gtfs[0], svc.strftime("%Y%m%d")):
            raise ValueError(f"SFCI_SERVICE_DATE has no service in the current feeds: {requested_date}")
    else:
        svc = feeds.pick_service_date(gtfs)
    data = raptor_build.load_or_build(gtfs, svc, verbose=False)
    n_stops = int(data["n_stops"])
    slat, slon = np.asarray(data["stop_lat"]), np.asarray(data["stop_lon"])

    # The grid itself is the source of truth: ids and order are deterministic x-major/y-minor
    # output from build_grid, with no dependency on a legacy access archive.
    grid_source = Path(config.neigh_path())
    try:
        grid_st = grid_source.stat()
        grid_names = np.asarray([grid_source.name], dtype="U")
        grid_sizes = np.asarray([int(grid_st.st_size)], dtype=np.int64)
        grid_mtimes = np.asarray([config.portable_mtime_ns(grid_st)], dtype=np.int64)
    except OSError:
        grid_names = np.asarray([grid_source.name], dtype="U")
        grid_sizes = np.asarray([-1], dtype=np.int64)
        grid_mtimes = np.asarray([-1], dtype=np.int64)
    g = gridmod.build_grid(gridmod.load_neighborhoods(), GRID_M)
    cell_ids = [str(v) for v in g["id"]]
    cell_ll = np.asarray([(float(geom.x), float(geom.y)) for geom in g.geometry], dtype=np.float64)

    graph_path = config.DATA / "walk_graph.npz"
    try:
        graph_st = graph_path.stat()
        graph_size = int(graph_st.st_size)
        graph_mtime = config.portable_mtime_ns(graph_st)
    except OSError:
        graph_size = graph_mtime = -1
    wg = walk.WalkGraph.load()
    valid = np.where(
        np.isfinite(slat) & np.isfinite(slon)
        & (slat >= -90.0) & (slat <= 90.0)
        & (slon >= -180.0) & (slon <= 180.0))[0]
    stop_nodes, stop_conn = wg.snap(np.column_stack((slon[valid], slat[valid])))
    cell_nodes, cell_conn = wg.snap(cell_ll)

    t = time.time()
    cap_ref = WALK_CAP_MIN * 60
    matrix = wg.many_to_targets(cell_nodes, cell_conn, stop_nodes, stop_conn, cap_ref, flat=FLAT)
    log(f"[bake] {len(cell_ids)} cells x {len(valid)} stops in {time.time()-t:.1f}s "
        f"({'flat' if FLAT else 'hill'} weights)")

    off = np.zeros(len(cell_ids) + 1, dtype=np.int64)
    to_l, weights_l = [], []
    for ci, row in enumerate(matrix):
        for j in np.where(np.isfinite(row))[0]:
            to_l.append(int(valid[j])); weights_l.append(int(round(float(row[j]))))
        off[ci + 1] = len(to_l)
    access_to = np.asarray(to_l, dtype=np.int32)
    access_w = np.asarray(weights_l, dtype=np.int32)
    log(f"[bake] {len(access_to)} pairs (avg {len(access_to)/max(1, len(cell_ids)):.1f}/cell)")

    source_names, source_sizes, source_mtimes = _source_arrays(data.get("source_mtimes", ()))
    stop_keys = tuple(transfer_rules.StopKey(str(feed), str(sid))
                      for feed, sid in zip(data["stop_feed"], data["stop_id"]))
    rule_sets = transfer_rules.parse_transfer_rules_many(gtfs)
    transfer_bake = graph_transfers.bake_graph_transfers(
        stop_keys, slon, slat, wg, rule_sets, radius_m=raptor_build.FOOTPATH_M,
        cap_ref_sec=cap_ref)
    transfer_arrays = {
        # Forward source -> target geometry view.
        "tr_forward_off": np.asarray(transfer_bake.tr_forward_off, dtype=np.int64),
        "tr_forward_to": np.asarray(transfer_bake.tr_forward_to, dtype=np.int32),
        "tr_forward_walk_time": np.asarray(transfer_bake.tr_forward_walk_time, dtype=np.float64),
        "tr_forward_min_time": np.asarray(transfer_bake.tr_forward_min_time, dtype=np.float64),
        "tr_forward_time": np.asarray(transfer_bake.tr_forward_time, dtype=np.int64),
        "tr_forward_path_off": np.asarray(transfer_bake.tr_forward_path_off, dtype=np.int64),
        "tr_forward_path_points": np.asarray(transfer_bake.tr_forward_path_points, dtype=np.float64).reshape(-1, 2),
        "tr_forward_path_fallback": np.asarray(transfer_bake.tr_forward_path_fallback, dtype=np.int8),
        # Reverse target -> source runtime view for RAPTOR.
        "tr_off": np.asarray(transfer_bake.tr_reverse_off, dtype=np.int64),
        "tr_to": np.asarray(transfer_bake.tr_reverse_to, dtype=np.int32),
        "tr_walk_time": np.asarray(transfer_bake.tr_reverse_walk_time, dtype=np.float64),
        "tr_min_time": np.asarray(transfer_bake.tr_reverse_min_time, dtype=np.float64),
        "tr_time": np.asarray(transfer_bake.tr_reverse_time, dtype=np.int64),
        "tr_path_fallback": np.asarray(transfer_bake.tr_reverse_path_fallback, dtype=np.int8),
    }
    transfer_arrays.update(_rule_arrays(transfer_bake))
    date_str = svc.strftime("%Y%m%d")
    name = f"access_walk{'flat' if FLAT else ''}_{GRID_M}m_{date_str}.npz"
    out = CACHE / name
    CACHE.mkdir(parents=True, exist_ok=True)
    # A per-process/thread temporary avoids two concurrent bakes clobbering one another before
    # they reach the publication lock.  The canonical filename remains parameter-readable.
    tmp = out.with_name(
        f".{out.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
    np.savez(tmp, cell_ids=np.asarray(cell_ids, dtype="U"), access_off=off,
             access_to=access_to, access_w=access_w, grid_m=np.int32(GRID_M),
             n_stops=np.int32(n_stops), service_date=np.asarray(date_str),
             footpath_m=np.float64(raptor_build.FOOTPATH_M),
             raptor_build_version=np.int32(data["build_version"]),
             grid_source_names=grid_names, grid_source_sizes=grid_sizes,
             grid_source_mtimes_ns=grid_mtimes,
             slope_aware=np.int8(0 if FLAT else 1), walk_ref_kmh=np.float32(config.WALK_KMH),
             raptor_source_names=source_names, raptor_source_sizes=source_sizes,
             raptor_source_mtimes_ns=source_mtimes, walk_graph_size=np.int64(graph_size),
             walk_graph_mtime_ns=np.int64(graph_mtime), **transfer_arrays)
    try:
        with _publication_guard():
            # The archive is a complete, closed NPZ before it becomes canonical.  Validate the
            # exact temporary path while holding the publication lock, then fsync it so the
            # single rename cannot expose a partially written file to another process.
            if not validate_artifact(tmp, n_stops=n_stops, service_date=date_str, grid_m=GRID_M,
                                     walk_graph_path=graph_path, grid_source_path=grid_source,
                                     stop_lat=slat, stop_lon=slon):
                raise ValueError("access archive failed structural validation")
            with open(tmp, "rb") as payload:
                os.fsync(payload.fileno())
            # Feed/graph changes during the expensive bake must not publish an artifact whose
            # source metadata describes an earlier graph.  The old canonical artifact remains
            # untouched on this failure; a later invocation retries from fresh inputs.
            if raptor_build._source_mtimes(gtfs) != \
                    config.normalize_source_mtimes(data.get("source_mtimes", ())):
                raise RuntimeError("GTFS source changed during access bake")
            try:
                current_graph = graph_path.stat()
                current_graph_meta = (
                    int(current_graph.st_size),
                    config.portable_mtime_ns(current_graph),
                )
            except OSError:
                current_graph_meta = (-1, -1)
            if current_graph_meta != (graph_size, graph_mtime):
                raise RuntimeError("walking graph changed during access bake")
            try:
                current_grid = grid_source.stat()
                current_grid_meta = (
                    int(current_grid.st_size),
                    config.portable_mtime_ns(current_grid),
                )
            except OSError:
                current_grid_meta = (-1, -1)
            if current_grid_meta != (int(grid_sizes[0]), int(grid_mtimes[0])):
                raise RuntimeError("neighborhood/grid source changed during access bake")
            os.replace(tmp, out)
            # Persist the directory entry as well as the closed NPZ so a crash cannot leave
            # readers with a missing/partially published canonical artifact.
            raptor_build._fsync_directory(CACHE)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    log(f"[bake] saved {out.name} ({out.stat().st_size/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    bake()
