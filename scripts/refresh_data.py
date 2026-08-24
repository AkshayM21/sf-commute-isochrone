#!/usr/bin/env python3
"""Fetch and build a complete inactive data generation.

This is intentionally a local, stage-first workflow.  It never writes the default
``data/`` tree or any of its descendants.  It produces only an inactive, isolated
stage; production promotion is handled by the deployment data-refresh helper after
every artifact has passed structural/readiness checks.

Examples::

    .venv/bin/python scripts/refresh_data.py run \
        --stage /tmp/sfci-stage-20260823

    .venv/bin/python scripts/refresh_data.py fetch --stage /tmp/sfci-stage
    .venv/bin/python scripts/refresh_data.py build --stage /tmp/sfci-stage

Production promotion is owned by ``deploy/data-refresh.sh`` after this command
prints a complete, validated stage path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA_FILES = ("sf_neighborhoods.geojson", "walk_graph.npz", "osm_sf.pbf", "dem_sf.tif")
FEED_NAMES = {"muni": "muni_current.zip", "bart": "bart_gtfs.zip", "caltrain": "caltrain.zip"}
FEED_URLS = {
    "muni": "https://api.511.org/transit/datafeeds?operator_id=SF&format=gtfs",
    "caltrain": "https://api.511.org/transit/datafeeds?operator_id=CT&format=gtfs",
    "bart": "https://www.bart.gov/dev/schedules/google_transit.zip",
}
DOWNLOAD_USER_AGENT = "sf-commute-isochrone-data-refresh/1.0"
REQUIRED_GTFS = ("routes.txt", "trips.txt", "stops.txt", "stop_times.txt")


class _FeedValidationError(ValueError):
    """Internal validation failure whose message contains no request information."""


def _required_names(names: Iterable[str]) -> set[str]:
    return {Path(name).name for name in names}


def valid_gtfs_zip(path: str | Path, *, required: Iterable[str] = REQUIRED_GTFS) -> bool:
    """Return true only for a readable archive with all core GTFS tables.

    ``calendar.txt`` and ``calendar_dates.txt`` are alternatives because a valid feed
    may use either calendar representation.  Nested archive paths are accepted.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                return False
            names = _required_names(archive.namelist())
            if not set(required).issubset(names):
                return False
            return "calendar.txt" in names or "calendar_dates.txt" in names
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False


def _download(url: str, destination: Path, *, opener: Callable = urllib.request.urlopen) -> Path:
    """Download to a same-directory temporary file, validate, then atomically replace."""
    if destination.is_symlink():
        raise ValueError(f"download destination must not be a symlink: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": DOWNLOAD_USER_AGENT,
            "Accept": "application/zip, application/octet-stream;q=0.9, */*;q=0.1",
        },
    )
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part",
                                          dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as out, opener(request, timeout=240) as response:
            shutil.copyfileobj(response, out)
            out.flush()
            os.fsync(out.fileno())
        if not valid_gtfs_zip(temporary):
            raise _FeedValidationError(
                f"downloaded feed is not a complete GTFS archive: {destination.name}")
        os.replace(temporary, destination)
        return destination
    except _FeedValidationError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception:
        temporary.unlink(missing_ok=True)
        # urllib exceptions can echo the request URL, which would disclose API511_TOKEN.
        raise RuntimeError(f"feed download failed: {destination.name}") from None


def fetch_feeds(stage: str | Path, token: str | None = None,
                *, opener: Callable = urllib.request.urlopen) -> Mapping[str, Path]:
    """Fetch Muni, BART, and Caltrain into an inactive stage.

    Muni is intentionally required as the current feed; there is no fallback in this
    refresh path.  The token is passed in the URL only for 511 requests and is never
    printed or included in an exception of our making.
    """
    stage = _inactive_directory(stage, "stage", create=True)
    stage.mkdir(parents=True, exist_ok=True)
    _assert_isolated_tree(stage)
    token = token or os.environ.get("API511_TOKEN")
    if not token:
        raise ValueError("API511_TOKEN is required to refresh current Muni and Caltrain feeds")
    if any(ch in token for ch in '"\\\r\n'):
        raise ValueError("API511_TOKEN contains invalid characters")
    urls = dict(FEED_URLS)
    urls["muni"] += f"&api_key={token}"
    urls["caltrain"] += f"&api_key={token}"
    results = {}
    for role in ("muni", "bart", "caltrain"):
        destination = stage / FEED_NAMES[role]
        results[role] = _download(urls[role], destination, opener=opener)
    if not valid_gtfs_zip(results["muni"]):
        raise ValueError("current Muni feed failed validation")
    _assert_isolated_tree(stage)
    return results


def _link_or_copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("stable data inputs and stage targets must not be symlinks")
    if not source.exists():
        raise FileNotFoundError(f"missing stable input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _inactive_directory(value: str | Path, label: str, *, create: bool = False) -> Path:
    """Resolve a real directory outside the configured active data tree."""
    raw = Path(value).expanduser()
    # Check both the caller's lexical path and the resolved path.  The lexical check
    # matters for deployment pointers such as ``data-current/child``: resolving a
    # symlink first could otherwise make a protected deployment tree look like an
    # ordinary external stage.  Keep this list intentionally narrow so /tmp stages
    # (including a user's own ``data-releases`` scratch directory) remain allowed.
    lexical = Path(os.path.abspath(os.fspath(raw)))
    protected_roots = tuple(
        base / name
        for base in (ROOT, Path("/opt/sfci"))
        for name in ("data-releases", "data-current", "data-incoming")
    )
    if any(path == protected or protected in path.parents
           for path in (lexical, raw.resolve(strict=False))
           for protected in protected_roots):
        raise ValueError(f"{label} must not use a protected deployment data path")
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = raw.resolve(strict=False)
    active_roots = {(ROOT / "data").resolve()}
    configured = os.environ.get("SFCI_DATA_DIR")
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = ROOT / configured_path
        active_roots.add(configured_path.resolve(strict=False))
    if any(resolved == active or active in resolved.parents for active in active_roots):
        raise ValueError(f"{label} must be an inactive data directory")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"{label} is not a directory")
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _assert_isolated_tree(stage: Path) -> None:
    """Reject links and multi-linked payloads before any build or final validation."""
    if stage.is_symlink():
        raise ValueError("stage must not be a symlink")
    try:
        paths = [stage, *stage.rglob("*")]
    except OSError:
        raise ValueError("stage cannot be inspected safely") from None
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"stage contains a symlink: {path.name}")
        try:
            if path.is_file() and path.stat().st_nlink != 1:
                raise ValueError(f"stage contains a hard-linked payload: {path.name}")
        except OSError:
            raise ValueError("stage cannot be inspected safely") from None


def prepare_stage(stage: str | Path, source_data: str | Path | None = None) -> Path:
    """Create an empty stage and carry stable graph/grid inputs into it."""
    stage = _inactive_directory(stage, "stage", create=False)
    if stage.exists():
        if not stage.is_dir():
            raise ValueError(f"stage is not a directory: {stage}")
        if any(stage.iterdir()):
            raise ValueError(f"stage must be empty to avoid mixing generations: {stage}")
    else:
        stage.mkdir(parents=True)
    source_raw = Path(source_data).expanduser() if source_data else (ROOT / "data")
    if source_raw.is_symlink():
        raise ValueError("source data root must not be a symlink")
    source = source_raw.resolve()
    # A refresh needs the existing walking graph and grid source.  OSM/DEM are carried
    # when available so a later graph rebuild can run from the same generation, but are
    # not made prerequisites when the already-baked walk graph is sufficient.
    for name in DATA_FILES:
        candidate = source / name
        if candidate.exists():
            _link_or_copy(candidate, stage / name)
    _assert_isolated_tree(stage)
    return stage


def _service_date(value: str | None = None) -> str:
    if value:
        parsed = dt.datetime.strptime(value, "%Y%m%d").date()
        return parsed.strftime("%Y%m%d")
    # Importing readiness is intentionally delayed until the refresh subprocess has
    # selected its data root.  It supplies the canonical next-Wednesday policy.
    sys.path.insert(0, str(SCRIPTS))
    from core.readiness import modeled_wednesday
    return modeled_wednesday().strftime("%Y%m%d")


def _run_build_command(stage: Path, command: list[str], service_date: str) -> None:
    env = os.environ.copy()
    env["SFCI_DATA_DIR"] = str(stage)
    env["SFCI_SERVICE_DATE"] = service_date
    scripts_text = str(SCRIPTS)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = scripts_text + ((os.pathsep + existing_pythonpath)
                                         if existing_pythonpath else "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def build_stage(stage: str | Path, *, service_date: str | None = None,
                python: str | None = None) -> Path:
    """Build RAPTOR/access/transfer artifacts, static bundle, and run readiness."""
    stage = _inactive_directory(stage, "stage", create=False)
    if not stage.is_dir():
        raise FileNotFoundError(stage)
    _assert_isolated_tree(stage)
    svc = _service_date(service_date)
    py = python or sys.executable
    _run_build_command(stage, [py, str(SCRIPTS / "bake_walk_access.py")], svc)
    static_code = (
        "import datetime as dt; "
        "from core import config, feeds, raptor_build, static_bundle; "
        "gtfs=config.gtfs_paths(); "
        "svc=dt.datetime.strptime(__import__('os').environ['SFCI_SERVICE_DATE'], '%Y%m%d').date(); "
        "static_bundle.build_static_bundle(config.DATA/'server_static.json', gtfs, "
        "grid_m=int(__import__('os').environ.get('GRID_M', str(config.GRID_M))), "
        "service_date=svc, source_mtimes=raptor_build._source_mtimes(gtfs))"
    )
    _run_build_command(stage, [py, "-c", static_code], svc)
    _assert_isolated_tree(stage)
    check_code = (
        "import datetime as dt, json, os; "
        "from pathlib import Path; from core import config, readiness; "
        "svc=dt.datetime.strptime(os.environ['SFCI_SERVICE_DATE'], '%Y%m%d').date(); "
        "feeds={'muni':config.DATA/config.MUNI_CURRENT, 'bart':config.DATA/config.BART, "
        "'caltrain':config.DATA/config.CALTRAIN}; "
        "cache=next(config.DATA.joinpath('raptor_cache').glob('*.pkl')); "
        "access=next(config.DATA.joinpath('raptor_cache').glob('access_walk*.npz')); "
        "r=readiness.check_readiness(feeds, cache, config.DATA/'walk_graph.npz', access, "
        "config.DATA/'server_static.json', now=dt.datetime.combine(svc, dt.time(0), tzinfo=readiness.LA), "
        "grid_m=int(os.environ.get('GRID_M', str(config.GRID_M))), grid_source=config.neigh_path()); "
        "print(json.dumps(r.as_dict(), sort_keys=True)); "
        "raise SystemExit(0 if r.ready else 1)"
    )
    _run_build_command(stage, [py, "-c", check_code], svc)
    _assert_isolated_tree(stage)
    return stage


def _clear_failed_stage(stage: Path, *, remove_root: bool) -> None:
    """Remove only an inactive failed stage so a retry cannot mix generations."""
    try:
        safe = _inactive_directory(stage, "stage", create=False)
    except (OSError, ValueError):
        return
    if remove_root:
        shutil.rmtree(safe, ignore_errors=True)
        return
    for child in list(safe.iterdir()):
        if child.is_symlink() or child.is_file():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)


def _load_repo_env() -> None:
    """Load the repo's ignored .env without overriding explicit environment values."""
    sys.path.insert(0, str(SCRIPTS))
    from core import config
    config.load_dotenv()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "build"):
        p = sub.add_parser(name)
        p.add_argument("--stage", required=True, type=Path)
        if name == "build":
            p.add_argument("--service-date", help="YYYYMMDD; defaults to modeled next Wednesday")
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, type=Path)
    p.add_argument("--service-date")
    p.add_argument("--source-data", type=Path)
    args = parser.parse_args(argv)
    if args.command in ("fetch", "run"):
        _load_repo_env()
    if args.command == "fetch":
        fetch_feeds(args.stage)
    elif args.command == "build":
        build_stage(args.stage, service_date=args.service_date)
    else:
        stage_input = Path(args.stage)
        existed = stage_input.exists()
        stage = prepare_stage(stage_input, args.source_data)
        try:
            fetch_feeds(stage)
            build_stage(stage, service_date=args.service_date)
            print(stage)
        except Exception:
            _clear_failed_stage(stage, remove_root=not existed)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
