"""Cheap, direct checks used to decide whether the routing runtime is ready.

This module deliberately has no Flask, cache, or deployment concerns.  It
validates the things the process must be able to read and use: the three GTFS archives,
the RAPTOR arrays, the walking graph/access bake, and the static browser bundle.  Every
failure is represented by a small, stable reason code; exception text and filesystem
paths never cross this boundary.

Feed/archive validation is intentionally a boot-time or data-refresh operation, not a
per-request health-endpoint operation.  Server integration must compute it once for the
current boot/data generation and expose the cached result until the next controlled
refresh; this module performs the direct checks and returns their stable result.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np

from . import config


LA = ZoneInfo("America/Los_Angeles")
DEFAULT_REQUIRED_FEEDS = ("muni", "bart", "caltrain")

# These strings are API-safe and intentionally independent of implementation details.
OK = "ok"
REASON_CODES = frozenset({
    OK,
    "missing_feed",
    "invalid_feed",
    "no_service",
    "calendar_expired",
    "service_date_stale",
    "missing_cache",
    "invalid_cache",
    "missing_walk_graph",
    "missing_access_bake",
    "missing_static_bundle",
    "runtime_uninitialized",
    "runtime_load_failed",
    "wrong_engine",
})


@dataclass(frozen=True)
class CheckResult:
    """A safe result for one readiness check.

    ``detail`` is a small role label (for example ``"bart"``), never a path or an
    exception.  Callers may expose ``reason_code`` directly in a health response.
    """

    ready: bool
    reason_code: str = OK
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": self.ready, "reason_code": self.reason_code}
        if self.detail is not None:
            result["detail"] = self.detail
        return result


@dataclass(frozen=True)
class ReadinessResult:
    """Aggregate result, including the modeled service date used by the checks."""

    ready: bool
    reason_code: str = OK
    service_date: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": self.ready,
            "reason_code": self.reason_code,
        }
        if self.service_date is not None:
            result["service_date"] = self.service_date
        if self.detail is not None:
            result["detail"] = self.detail
        return result


def modeled_wednesday(now: dt.datetime | None = None,
                      dep_hm: tuple[int, int] = config.DEP_HM) -> dt.date:
    """Return the next relevant modeled Wednesday in Los Angeles time.

    On Wednesday before the model departure, that same Wednesday is relevant.  Once
    the departure has passed, use the following Wednesday.  This prevents a healthy
    morning service from becoming falsely stale merely because the clock crossed the
    model departure time.
    """
    if now is None:
        now = dt.datetime.now(LA)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=LA)
    else:
        now = now.astimezone(LA)
    days = (2 - now.weekday()) % 7
    candidate = now.date() + dt.timedelta(days=days)
    departure = dt.datetime(candidate.year, candidate.month, candidate.day,
                            int(dep_hm[0]), int(dep_hm[1]), tzinfo=LA)
    if days == 0 and now >= departure:
        candidate += dt.timedelta(days=7)
    return candidate


def _result(code: str, detail: str | None = None) -> CheckResult:
    if code not in REASON_CODES:
        raise ValueError(f"unknown readiness reason code: {code}")
    return CheckResult(code == OK, code, detail)


def _date(value: dt.date | str) -> dt.date | None:
    """Parse API/artifact dates without allowing malformed data to escape."""
    try:
        if isinstance(value, dt.datetime):
            return value.date()
        if isinstance(value, dt.date):
            return value
        text = str(value).strip()
        if len(text) == 8 and text.isdigit():
            return dt.datetime.strptime(text, "%Y%m%d").date()
        return dt.date.fromisoformat(text)
    except (TypeError, ValueError, OverflowError):
        return None


def _gtfs_date(value: str) -> dt.date | None:
    try:
        text = str(value).strip()
        if len(text) != 8 or not text.isdigit():
            return None
        return dt.datetime.strptime(text, "%Y%m%d").date()
    except (TypeError, ValueError, OverflowError):
        return None


def _stream_table(z: zipfile.ZipFile, name: str, headers: tuple[str, ...], consume) -> CheckResult:
    """Validate a GTFS table while keeping only caller-selected identifiers.

    In particular, ``stop_times.txt`` can contain millions of rows.  It is opened as a
    text stream and handed to ``consume`` one row at a time; no table bytes or row list
    is retained.  The same bounded approach is used for the smaller tables so future
    feed growth cannot quietly turn readiness into a large allocation.
    """
    try:
        with z.open(name, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or any(h not in fields for h in headers):
                return _result("invalid_feed")
            count = 0
            for row in reader:
                count += 1
                if None in row or not consume(row):
                    return _result("invalid_feed")
            if count == 0:
                return _result("invalid_feed")
            return _result(OK)
    except (KeyError, UnicodeDecodeError, csv.Error, OSError, ValueError, TypeError):
        return _result("invalid_feed")


def _nonempty(row: Mapping[str, str], *fields: str) -> bool:
    return all(isinstance(row.get(field), str) and row[field].strip() for field in fields)


def _gtfs_time(value: str) -> int | None:
    try:
        parts = value.strip().split(":")
        if len(parts) != 3:
            return None
        hour, minute, second = (int(x) for x in parts)
        if hour < 0 or minute not in range(60) or second not in range(60):
            return None
        return hour * 3600 + minute * 60 + second
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _active_service_ids(calendar_rows: list[dict[str, str]],
                        exception_rows: list[dict[str, str]],
                        date: dt.date) -> set[str]:
    ds = date.strftime("%Y%m%d")
    weekday = date.strftime("%A").lower()
    active: set[str] = set()
    for row in calendar_rows:
        start = _gtfs_date(row.get("start_date", ""))
        end = _gtfs_date(row.get("end_date", ""))
        if start and end and start <= date <= end and row.get(weekday) == "1":
            active.add(row.get("service_id", ""))
    for row in exception_rows:
        if row.get("date") == ds:
            sid = row.get("service_id", "")
            if row.get("exception_type") == "1":
                active.add(sid)
            elif row.get("exception_type") == "2":
                active.discard(sid)
    return {sid for sid in active if sid}


def validate_gtfs_feed(path: str | Path, service_date: dt.date | str,
                       role: str | None = None) -> CheckResult:
    """Validate a GTFS ZIP and verify trips on the exact modeled service date."""
    label = role if role in DEFAULT_REQUIRED_FEEDS else None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return _result("missing_feed", label)
    try:
        with zipfile.ZipFile(p) as z:
            if z.testzip() is not None:
                return _result("invalid_feed", label)
            names = set(z.namelist())
            required_names = {"routes.txt", "stops.txt", "trips.txt", "stop_times.txt"}
            if not required_names.issubset(names):
                return _result("invalid_feed", label)
            if "calendar.txt" not in names and "calendar_dates.txt" not in names:
                return _result("invalid_feed", label)

            route_ids: set[str] = set()
            stop_ids: set[str] = set()
            trip_ids: set[str] = set()
            trips_by_service: dict[str, set[str]] = {}

            def route_row(row):
                rid = row.get("route_id", "").strip()
                if not _nonempty(row, "route_id") or rid in route_ids:
                    return False
                route_ids.add(rid)
                return True

            def stop_row(row):
                sid = row.get("stop_id", "").strip()
                if not _nonempty(row, "stop_id") or sid in stop_ids:
                    return False
                try:
                    lat = float(row["stop_lat"]); lon = float(row["stop_lon"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    return False
                if not (math.isfinite(lat) and math.isfinite(lon)
                        and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    return False
                stop_ids.add(sid)
                return True

            def trip_row(row):
                rid = row.get("route_id", "").strip()
                sid = row.get("service_id", "").strip()
                tid = row.get("trip_id", "").strip()
                if (not _nonempty(row, "route_id", "service_id", "trip_id")
                        or rid not in route_ids or tid in trip_ids):
                    return False
                trip_ids.add(tid)
                trips_by_service.setdefault(sid, set()).add(tid)
                return True

            checks = (
                _stream_table(z, "routes.txt", ("route_id",), route_row),
                _stream_table(z, "stops.txt", ("stop_id", "stop_lat", "stop_lon"), stop_row),
                _stream_table(z, "trips.txt", ("route_id", "service_id", "trip_id"), trip_row),
            )
            if any(not check.ready for check in checks):
                return _result("invalid_feed", label)

            # One compact record per declared trip; the table itself is never retained.
            # Keeping state keyed by trip makes interleaved trip rows safe while still
            # bounding memory by trips.txt rather than by stop_times.txt row count.
            stop_time_state: dict[str, tuple[int, int, int]] = {}

            def stop_time_row(row):
                tid = row.get("trip_id", "").strip()
                sid = row.get("stop_id", "").strip()
                seq = row.get("stop_sequence", "").strip()
                arrival = _gtfs_time(row.get("arrival_time", ""))
                departure = _gtfs_time(row.get("departure_time", ""))
                if (not _nonempty(row, "trip_id", "stop_id", "stop_sequence",
                                  "arrival_time", "departure_time")
                        or tid not in trip_ids or sid not in stop_ids or arrival is None
                        or departure is None):
                    return False
                try:
                    sequence = int(seq)
                    if sequence < 0:
                        return False
                except (TypeError, ValueError, OverflowError):
                    return False
                previous = stop_time_state.get(tid)
                if arrival > departure:
                    return False
                if previous is not None:
                    count, last_sequence, last_departure = previous
                    if sequence <= last_sequence or arrival < last_departure:
                        return False
                else:
                    count = 0
                stop_time_state[tid] = (count + 1, sequence, departure)
                return True

            # This is intentionally streaming: do not replace with ZipFile.read() or
            # list(DictReader(...)), even for today's relatively small feeds.
            check = _stream_table(
                z, "stop_times.txt", ("trip_id", "stop_id", "stop_sequence",
                                      "arrival_time", "departure_time"), stop_time_row)
            if (not check.ready or set(stop_time_state) != trip_ids
                    or any(count < 2 for count, _, _ in stop_time_state.values())):
                return _result("invalid_feed", label)
            usable_stop_time_trips = set(stop_time_state)

            calendars: list[dict[str, str]] = []
            exceptions: list[dict[str, str]] = []
            if "calendar.txt" in names:
                def calendar_row(row):
                    if not _nonempty(row, "service_id", "start_date", "end_date"):
                        return False
                    start = _gtfs_date(row["start_date"]); end = _gtfs_date(row["end_date"])
                    if start is None or end is None or start > end:
                        return False
                    if any(row.get(day) not in ("0", "1") for day in
                           ("monday", "tuesday", "wednesday", "thursday", "friday",
                            "saturday", "sunday")):
                        return False
                    calendars.append(dict(row))
                    return True

                check = _stream_table(
                    z, "calendar.txt", ("service_id", "start_date", "end_date",
                                         "monday", "tuesday", "wednesday", "thursday",
                                         "friday", "saturday", "sunday"), calendar_row)
                if not check.ready:
                    return _result("invalid_feed", label)
            if "calendar_dates.txt" in names:
                exception_types: dict[tuple[str, str], str] = {}

                def exception_row(row):
                    if (not _nonempty(row, "service_id", "date", "exception_type")
                            or _gtfs_date(row["date"]) is None
                            or row["exception_type"] not in ("1", "2")):
                        return False
                    key = (row["service_id"], row["date"])
                    previous = exception_types.get(key)
                    if previous is not None:
                        return previous == row["exception_type"]
                    exception_types[key] = row["exception_type"]
                    exceptions.append(dict(row))
                    return True

                check = _stream_table(
                    z, "calendar_dates.txt", ("service_id", "date", "exception_type"), exception_row)
                if not check.ready:
                    return _result("invalid_feed", label)

            target = _date(service_date)
            if target is None:
                return _result("invalid_feed", label)
            # Expiration is a distinct operational failure from a feed that is merely
            # empty for a weekday.  Both calendar forms are accepted by GTFS.
            end_dates = [d for row in calendars
                         if (d := _gtfs_date(row.get("end_date", ""))) is not None]
            end_dates += [d for row in exceptions
                          if (d := _gtfs_date(row.get("date", ""))) is not None]
            if not end_dates or max(end_dates) < target:
                return _result("calendar_expired", label)
            target_sids = _active_service_ids(calendars, exceptions, target)
            target_trips = set().union(*(trips_by_service.get(sid, set()) for sid in target_sids))
            if not target_trips & usable_stop_time_trips:
                return _result("no_service", label)
            return _result(OK)
    except (OSError, PermissionError, zipfile.BadZipFile, EOFError):
        return _result("invalid_feed", label)
    except Exception:
        # A readiness endpoint must never serialize implementation-specific parser
        # details.  Treat unexpected archive/parser failures as invalid feed data.
        return _result("invalid_feed", label)


def validate_required_feeds(feeds: Mapping[str, str | Path],
                            service_date: dt.date | str) -> CheckResult:
    """Validate exactly Muni, BART, and Caltrain under stable role labels."""
    for role in DEFAULT_REQUIRED_FEEDS:
        path = feeds.get(role)
        if path is None:
            return _result("missing_feed", role)
        check = validate_gtfs_feed(path, service_date, role)
        if not check.ready:
            return check
    return _result(OK)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _array(value: Any, ndim: int = 1) -> np.ndarray | None:
    try:
        a = np.asarray(value)
        return a if a.ndim == ndim else None
    except (TypeError, ValueError):
        return None


def _integer_array(value: Any) -> np.ndarray | None:
    array = _array(value)
    return array if array is not None and array.dtype.kind in "iu" else None


def _scalar_int(value: Any) -> int | None:
    try:
        number = float(value)
        if not math.isfinite(number) or number != int(number):
            return None
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


def _valid_geojson_coordinates(value: Any) -> bool:
    """Check nested GeoJSON coordinate arrays without copying the geometry."""
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            lon, lat = float(value[0]), float(value[1])
            return (math.isfinite(lon) and math.isfinite(lat)
                    and -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0)
        return bool(value) and all(_valid_geojson_coordinates(item) for item in value)
    return False


def _validate_raptor_mapping(data: Mapping[str, Any],
                             service_date: dt.date | str | None = None) -> CheckResult:
    """Delegate cache structure validation to the canonical RAPTOR builder."""
    try:
        from . import raptor_build
        if not raptor_build._validate_cache_data(data):
            return _result("invalid_cache")
        cached_date = _date(str(data["date"]))
        if cached_date is None:
            return _result("invalid_cache")
        if service_date is not None:
            expected_date = _date(service_date)
            if expected_date is None:
                return _result("runtime_load_failed")
            if cached_date != expected_date:
                return _result("service_date_stale")
        return _result(OK)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, IndexError,
            OverflowError):
        return _result("runtime_load_failed")


def load_raptor_state(data: Mapping[str, Any] | str | Path,
                      service_date: dt.date | str | None = None) -> tuple[CheckResult, Mapping[str, Any] | None]:
    """Load a cache at most once and return it for downstream stop-count checks."""
    if isinstance(data, (str, Path)):
        try:
            with Path(data).open("rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            return _result("missing_cache"), None
        except Exception:
            return _result("runtime_load_failed"), None
    d = _as_mapping(data)
    if d is None:
        return _result("invalid_cache"), None
    check = _validate_raptor_mapping(d, service_date)
    return check, d if check.ready else None


def validate_raptor_state(data: Mapping[str, Any] | str | Path,
                          service_date: dt.date | str | None = None) -> CheckResult:
    """Validate a RAPTOR cache, delegating structure checks to ``raptor_build``."""
    return load_raptor_state(data, service_date)[0]


def _load_npz(value: str | Path | Mapping[str, Any]) -> tuple[Any, bool]:
    if isinstance(value, (str, Path)):
        try:
            return np.load(value, allow_pickle=True), True
        except FileNotFoundError:
            return None, False
        except Exception:
            return ..., False
    return value, True


def _npz_get(z: Any, key: str) -> Any:
    if isinstance(z, Mapping):
        return z[key]
    return z[key]


def _npz_keys(z: Any) -> set[str]:
    if isinstance(z, Mapping):
        return set(z)
    return set(z.files)


def validate_walk_graph(value: str | Path | Mapping[str, Any]) -> CheckResult:
    """Validate the directed walk CSR graph required by runtime snapping/routing."""
    z, loaded = _load_npz(value)
    if not loaded:
        return _result("missing_walk_graph" if z is None else "runtime_load_failed")
    try:
        if not {"node_lon", "node_lat", "node_elev", "indptr", "indices", "w_ref"}.issubset(_npz_keys(z)):
            return _result("runtime_load_failed")
        lon = _array(_npz_get(z, "node_lon"))
        lat = _array(_npz_get(z, "node_lat"))
        elev = _array(_npz_get(z, "node_elev"))
        indptr = _integer_array(_npz_get(z, "indptr"))
        indices = _integer_array(_npz_get(z, "indices"))
        weights = _array(_npz_get(z, "w_ref"))
        n = len(lon) if lon is not None else 0
        if any(a is None for a in (lon, lat, elev, indptr, indices, weights)) or n <= 0:
            return _result("runtime_load_failed")
        if len(lat) != n or len(elev) != n or len(indptr) != n + 1 or indptr[0] != 0 \
                or np.any(np.diff(indptr) < 0) or int(indptr[-1]) != len(indices) \
                or len(weights) != len(indices):
            return _result("runtime_load_failed")
        if not np.isfinite(lon).all() or not np.isfinite(lat).all() \
                or not np.isfinite(elev).all() or not np.isfinite(weights).all() \
                or np.any(weights < 0) or np.any(lon < -180.0) or np.any(lon > 180.0) \
                or np.any(lat < -90.0) or np.any(lat > 90.0):
            return _result("runtime_load_failed")
        if len(indices) and (np.min(indices) < 0 or np.max(indices) >= n):
            return _result("runtime_load_failed")
        return _result(OK)
    except Exception:
        return _result("runtime_load_failed")
    finally:
        if hasattr(z, "close"):
            z.close()


def validate_access_bake(value: str | Path | Mapping[str, Any],
                         expected_n_stops: int | None = None,
                         expected_grid_m: int | None = None,
                         service_date: dt.date | str | None = None,
                         walk_graph_path: str | Path | None = None,
                         grid_source_path: str | Path | None = None,
                         stop_lat: Any = None,
                         stop_lon: Any = None) -> CheckResult:
    """Validate the access bake using the same complete contract as runtime loading.

    Mapping inputs remain a small compatibility seam for unit callers that only need the
    generic cell-access checks.  Published NPZ artifacts go through the canonical validator
    in ``bake_walk_access`` so readiness cannot silently accept an artifact the engine would
    reject; the import is lazy to keep this module cheap at process start.
    """
    if isinstance(value, (str, Path)):
        if not Path(value).exists():
            return _result("missing_access_bake")
        try:
            import bake_walk_access
            valid = bake_walk_access.validate_artifact(
                value, n_stops=expected_n_stops, service_date=service_date,
                grid_m=expected_grid_m, walk_graph_path=walk_graph_path,
                grid_source_path=grid_source_path, stop_lat=stop_lat, stop_lon=stop_lon)
        except Exception:
            return _result("runtime_load_failed")
        return _result(OK if valid else "runtime_load_failed")
    z, loaded = _load_npz(value)
    if not loaded:
        return _result("missing_access_bake" if z is None else "runtime_load_failed")
    try:
        required = {"cell_ids", "access_off", "access_to", "access_w", "grid_m", "n_stops",
                    "service_date"}
        if not required.issubset(_npz_keys(z)):
            return _result("runtime_load_failed")
        cells = _array(_npz_get(z, "cell_ids"))
        off = _integer_array(_npz_get(z, "access_off"))
        to = _integer_array(_npz_get(z, "access_to"))
        weights = _array(_npz_get(z, "access_w"))
        n_cells = len(cells) if cells is not None else 0
        n_stops = _scalar_int(_npz_get(z, "n_stops"))
        grid_m = _scalar_int(_npz_get(z, "grid_m"))
        if any(a is None for a in (cells, off, to, weights)) or n_cells <= 0 or n_stops <= 0 \
                or grid_m <= 0:
            return _result("runtime_load_failed")
        if expected_n_stops is not None and n_stops != int(expected_n_stops):
            return _result("runtime_load_failed")
        if expected_grid_m is not None and grid_m != int(expected_grid_m):
            return _result("runtime_load_failed")
        if len(off) != n_cells + 1 or off[0] != 0 or np.any(np.diff(off) < 0) \
                or int(off[-1]) != len(to) or len(to) != len(weights):
            return _result("runtime_load_failed")
        if len(to) and (np.min(to) < 0 or np.max(to) >= n_stops):
            return _result("runtime_load_failed")
        if not np.isfinite(weights).all() or np.any(weights < 0):
            return _result("runtime_load_failed")
        baked_date = _date(str(_npz_get(z, "service_date")))
        if baked_date is None:
            return _result("runtime_load_failed")
        if service_date is not None:
            expected_date = _date(service_date)
            if expected_date is None:
                return _result("runtime_load_failed")
            if baked_date != expected_date:
                return _result("service_date_stale")
        return _result(OK)
    except Exception:
        return _result("runtime_load_failed")
    finally:
        if hasattr(z, "close"):
            z.close()


def validate_static_bundle(value: str | Path | Mapping[str, Any],
                           service_date: dt.date | str | None = None,
                           expected_grid_m: int | None = None,
                           expected_grid_source: str | Path | Mapping[str, Any] | None = None,
                           expected_gtfs_sources: tuple[Any, ...] | list[Any] | None = None) \
        -> CheckResult:
    """Validate the JSON data needed to render the initial map.

    The static page bundle is derived from the neighborhood source and the grid
    resolution as well as from GTFS.  Keep those direct, readable freshness values
    in the bundle so an old map cannot be paired with a newly shaped grid.  A caller
    that has the live source path can pass it as ``expected_grid_source`` to compare
    the current size and nanosecond mtime without hashing the source.
    """
    try:
        if isinstance(value, (str, Path)):
            p = Path(value)
            if not p.exists():
                return _result("missing_static_bundle")
            try:
                value = json.loads(p.read_text())
            except Exception:
                return _result("runtime_load_failed")
        if not isinstance(value, Mapping):
            return _result("runtime_load_failed")
        if not {"origin_ll", "cells", "lines", "grid_m", "grid_source_name",
                "grid_source_size", "grid_source_mtime_ns"}.issubset(value):
            return _result("runtime_load_failed")
        grid_m = _scalar_int(value["grid_m"])
        source_name = value["grid_source_name"]
        source_size = _metadata_int(value["grid_source_size"])
        source_mtime_ns = _metadata_int(value["grid_source_mtime_ns"])
        if (grid_m is None or grid_m <= 0 or not isinstance(source_name, str)
                or not source_name.strip() or source_size is None or source_size < 0
                or source_mtime_ns is None or source_mtime_ns < 0):
            return _result("runtime_load_failed")
        if expected_grid_m is not None and grid_m != int(expected_grid_m):
            return _result("runtime_load_failed")
        if expected_grid_source is not None:
            expected = _grid_source_metadata(expected_grid_source)
            if expected is None:
                return _result("runtime_load_failed")
            if (source_name, source_size, config.normalize_mtime_ns(source_mtime_ns)) != expected:
                return _result("runtime_load_failed")
        if expected_gtfs_sources is not None:
            actual_sources = value.get("source_mtimes")
            if not _same_source_metadata(actual_sources, expected_gtfs_sources):
                return _result("runtime_load_failed")
        svc = value.get("svc_date", value.get("service_date"))
        if svc is None:
            return _result("runtime_load_failed")
        actual = _date(str(svc))
        if actual is None:
            return _result("runtime_load_failed")
        if service_date is not None:
            expected_date = _date(service_date)
            if expected_date is None:
                return _result("runtime_load_failed")
            if actual != expected_date:
                return _result("service_date_stale")
        origins = value["origin_ll"]
        if not isinstance(origins, Mapping) or not origins:
            return _result("runtime_load_failed")
        for coords in origins.values():
            if not isinstance(coords, (list, tuple)) or len(coords) != 2 \
                    or not all(math.isfinite(float(x)) for x in coords) \
                    or not (-90.0 <= float(coords[0]) <= 90.0
                            and -180.0 <= float(coords[1]) <= 180.0):
                return _result("runtime_load_failed")
        for key in ("cells", "lines"):
            obj = value[key]
            if not isinstance(obj, Mapping) or obj.get("type") != "FeatureCollection" \
                    or not isinstance(obj.get("features"), list):
                return _result("runtime_load_failed")
            for feature in obj["features"]:
                if not isinstance(feature, Mapping):
                    return _result("runtime_load_failed")
                geometry = feature.get("geometry")
                if geometry is not None:
                    if not isinstance(geometry, Mapping) \
                            or not _valid_geojson_coordinates(geometry.get("coordinates")):
                        return _result("runtime_load_failed")
        return _result(OK)
    except Exception:
        return _result("runtime_load_failed")


def _same_source_metadata(actual: Any, expected: Any) -> bool:
    """Compare direct feed name/size/mtime tuples without reading feed contents."""
    try:
        actual_t = tuple(tuple(item) for item in actual)
        expected_t = tuple(tuple(item) for item in expected)
    except (TypeError, ValueError):
        return False
    if len(actual_t) != len(expected_t):
        return False
    for got, want in zip(actual_t, expected_t):
        if len(got) != 3 or len(want) != 3 or str(got[0]) != str(want[0]):
            return False
        try:
            if int(got[1]) != int(want[1]) or \
                    config.normalize_mtime_ns(got[2]) != config.normalize_mtime_ns(want[2]):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def _direct_source_metadata(feeds: Mapping[str, str | Path]) -> tuple[tuple[str, int, int], ...] | None:
    """Return canonical Muni/BART/Caltrain direct metadata, or None for mapping-only tests."""
    values = []
    for role in DEFAULT_REQUIRED_FEEDS:
        value = feeds.get(role)
        if not isinstance(value, (str, Path)):
            return None
        path = Path(value)
        try:
            st = path.stat()
        except OSError:
            return None
        values.append((path.name, int(st.st_size), config.portable_mtime_ns(st)))
    return tuple(values)


def _grid_source_metadata(value: str | Path | Mapping[str, Any] | tuple[Any, ...] | list[Any]) \
        -> tuple[str, int, int] | None:
    """Resolve a direct neighborhood-source identity without reading its contents."""
    try:
        if isinstance(value, Mapping):
            # Accept the readable flat form used by ``server_static.json`` and a
            # labeled mapping for integration callers.  Paths are deliberately not
            # retained in the bundle; only the basename and direct stat values matter.
            name = value.get("grid_source_name", value.get("name"))
            size = value.get("grid_source_size", value.get("size"))
            mtime = value.get("grid_source_mtime_ns", value.get("mtime_ns"))
            if name is None or size is None or mtime is None:
                return None
            size_i = _metadata_int(size)
            mtime_i = _metadata_int(mtime)
            if not isinstance(name, str) or not name.strip() or size_i is None or size_i < 0 \
                    or mtime_i is None or mtime_i < 0:
                return None
            return name, size_i, config.normalize_mtime_ns(mtime_i)
        if isinstance(value, (tuple, list)) and len(value) == 3:
            return _grid_source_metadata({"name": value[0], "size": value[1],
                                          "mtime_ns": value[2]})
        path = Path(value)
        st = path.stat()
        return path.name, int(st.st_size), config.portable_mtime_ns(st)
    except (OSError, TypeError, ValueError, OverflowError):
        return None


def _metadata_int(value: Any) -> int | None:
    """Parse a non-lossy integer metadata value (nanosecond mtimes exceed float precision)."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            text = value.strip()
            if text and (text.isdigit() or (text[0] in "+-" and text[1:].isdigit())):
                return int(text)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def validate_runtime_state(*, engine_kind: str | None,
                           semantic: str | None,
                           graph_backed: bool | None,
                           engine: Any = None,
                           walk_graph: Any = None,
                           service_date: dt.date | str | None = None,
                           now: dt.datetime | None = None,
                           initialized: bool | None = None) -> CheckResult:
    """Validate the live engine contract independently of Flask/server globals.

    ``engine`` and ``walk_graph`` are intentionally opaque objects: callers pass the
    already-loaded runtime instances, while this pure seam only verifies that both
    are present.  The boot coordinator must explicitly attest ``graph_backed=True``
    and ``initialized=True``; truth is never inferred from a merely allocated object.
    """
    if (engine_kind != "raptor" or semantic not in ("arriveby", "departafter")
            or graph_backed is not True):
        return _result("wrong_engine")
    if initialized is not True or engine is None or walk_graph is None:
        return _result("runtime_uninitialized")
    loaded_date = _date(service_date) if service_date is not None else None
    if loaded_date is None or loaded_date != modeled_wednesday(now):
        return _result("service_date_stale")
    return _result(OK)


def check_readiness(feeds: Mapping[str, str | Path], raptor: Mapping[str, Any] | str | Path,
                    walk_graph: str | Path | Mapping[str, Any],
                    access_bake: str | Path | Mapping[str, Any],
                    static_bundle: str | Path | Mapping[str, Any],
                    now: dt.datetime | None = None,
                    grid_m: int | None = None,
                    grid_source: str | Path | Mapping[str, Any] | tuple[Any, ...] | list[Any] | None = None,
                    expected_grid_source: str | Path | Mapping[str, Any] | tuple[Any, ...] | list[Any] | None = None) \
        -> ReadinessResult:
    """Run all direct checks in deterministic order and return the first safe failure."""
    target = modeled_wednesday(now)
    svc = target.strftime("%Y%m%d")
    check = validate_required_feeds(feeds, target)
    if not check.ready:
        return ReadinessResult(False, check.reason_code, svc, check.detail)
    check, raptor_data = load_raptor_state(raptor, target)
    if not check.ready:
        return ReadinessResult(False, check.reason_code, svc, check.detail)
    check = validate_walk_graph(walk_graph)
    if not check.ready:
        return ReadinessResult(False, check.reason_code, svc, check.detail)
    n_stops = int(raptor_data["n_stops"]) if raptor_data is not None else None
    if grid_source is None:
        grid_source = expected_grid_source
    graph_source = walk_graph if isinstance(walk_graph, (str, Path)) else None
    check = validate_access_bake(
        access_bake, n_stops, grid_m, target,
        walk_graph_path=graph_source,
        grid_source_path=grid_source if isinstance(grid_source, (str, Path)) else None,
        stop_lat=raptor_data.get("stop_lat") if raptor_data is not None else None,
        stop_lon=raptor_data.get("stop_lon") if raptor_data is not None else None)
    if not check.ready:
        return ReadinessResult(False, check.reason_code, svc, check.detail)
    check = validate_static_bundle(static_bundle, target, expected_grid_m=grid_m,
                                   expected_grid_source=grid_source,
                                   expected_gtfs_sources=_direct_source_metadata(feeds))
    if not check.ready:
        return ReadinessResult(False, check.reason_code, svc, check.detail)
    return ReadinessResult(True, OK, svc)
