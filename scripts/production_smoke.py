#!/usr/bin/env python3
"""Small, dependency-free smoke test for the public commute service.

This probe intentionally checks the browser's map module as well as the API.  A map provider can
return an HTTP 200 image containing an "API key required" watermark, so health checks and API
checks alone are not sufficient.  The probe fetches the module that selects the tile provider and
asserts the explicit, keyless OpenStreetMap contract; it does not fetch any third-party tiles.

Usage::

    python scripts/production_smoke.py
    python scripts/production_smoke.py --base-url https://sfcommutemap.com

The script is deliberately stdlib-only so it can run from a clean checkout and from a scheduled
GitHub Actions job.  Failure messages contain endpoint paths/statuses only; response bodies are
never printed because upstream responses can contain private or operational data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://sfcommutemap.com"
DEFAULT_TIMEOUT_SECONDS = 20.0
MAX_JSON_BYTES = 12 * 1024 * 1024
MAX_MODULE_BYTES = 512 * 1024
DESTINATION = (37.7955, -122.3937)  # public Ferry Building coordinate, not a user address
APPROVED_TILE_ENDPOINT = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"


class SmokeFailure(RuntimeError):
    """A safe-to-display smoke failure (never carries a response body)."""


Opener = Callable[..., Any]


def _base_url(value: str) -> str:
    """Normalize a base URL while rejecting credentials and non-HTTP schemes."""
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SmokeFailure("base URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise SmokeFailure("base URL may not contain credentials")
    # A base URL's path is useful for a local reverse proxy, but fragments/queries would make
    # endpoint construction ambiguous and could accidentally leak values into logs.
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _endpoint(base: str, path: str, query: Mapping[str, Any] | None = None) -> str:
    url = f"{base}{path}"
    return f"{url}?{urlencode(query)}" if query else url


def _path_for_message(url: str) -> str:
    """Return only a URL path for errors; never echo query strings or credentials."""
    parsed = urlsplit(url)
    return parsed.path or "/"


def _read_limited(response: Any, limit: int) -> bytes:
    body = response.read(limit + 1)
    if len(body) > limit:
        raise SmokeFailure("response exceeded smoke-test size limit")
    return body


def _fetch(url: str, *, timeout: float, opener: Opener, limit: int) -> bytes:
    request = Request(url, headers={"Accept": "application/json, text/javascript, */*",
                                    "User-Agent": "sfcommutemap-production-smoke/1"})
    path = _path_for_message(url)
    try:
        with opener(request, timeout=timeout) as response:
            status = int(getattr(response, "status", response.getcode()))
            if status != 200:
                raise SmokeFailure(f"GET {path} returned HTTP {status}")
            return _read_limited(response, limit)
    except SmokeFailure:
        raise
    except HTTPError as exc:
        # Do not stringify HTTPError: it can include an upstream response body or URL.
        raise SmokeFailure(f"GET {path} returned HTTP {exc.code}") from None
    except (URLError, TimeoutError, OSError):
        raise SmokeFailure(f"GET {path} was unavailable or timed out") from None


def _json_get(base: str, path: str, *, query: Mapping[str, Any] | None,
              timeout: float, opener: Opener) -> Mapping[str, Any]:
    raw = _fetch(_endpoint(base, path, query), timeout=timeout, opener=opener, limit=MAX_JSON_BYTES)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SmokeFailure(f"GET {path} returned invalid JSON") from None
    if not isinstance(value, dict):
        raise SmokeFailure(f"GET {path} returned an unexpected JSON shape")
    return value


def _check_health(base: str, *, timeout: float, opener: Opener) -> None:
    live = _json_get(base, "/livez", query=None, timeout=timeout, opener=opener)
    if live.get("ok") is not True:
        raise SmokeFailure("GET /livez reported not ready")
    ready = _json_get(base, "/readyz", query=None, timeout=timeout, opener=opener)
    if ready.get("ok") is not True:
        raise SmokeFailure("GET /readyz reported not ready")


def _check_map_provider(base: str, *, timeout: float, opener: Opener) -> None:
    raw = _fetch(_endpoint(base, "/static/map-renderer.mjs"), timeout=timeout,
                 opener=opener, limit=MAX_MODULE_BYTES)
    try:
        module = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise SmokeFailure("GET /static/map-renderer.mjs was not UTF-8") from None
    lowered = module.lower()
    if "api key required" in lowered or "apikey" in lowered or "api_key" in lowered:
        raise SmokeFailure("map renderer contains a key-required marker")
    if "carto" in lowered or "cartocdn.com" in lowered:
        raise SmokeFailure("map renderer still references CARTO basemaps")
    if APPROVED_TILE_ENDPOINT not in module:
        raise SmokeFailure("map renderer is missing the approved keyless OpenStreetMap endpoint")


def _reachable_cell(compute: Mapping[str, Any]) -> str:
    cells = compute.get("cells")
    if not isinstance(cells, dict):
        raise SmokeFailure("GET /compute returned no cell map")
    for cell_id, pair in cells.items():
        if isinstance(cell_id, str) and isinstance(pair, (list, tuple)) and len(pair) == 2:
            # /compute's second value is the scheduled/realistic result.  Request one cell that
            # is reachable under the currently served data rather than assuming a fixed grid id.
            if isinstance(pair[1], (int, float)) and not isinstance(pair[1], bool):
                return cell_id
    raise SmokeFailure("GET /compute returned no reachable cell")


def _check_routing(base: str, *, timeout: float, opener: Opener) -> None:
    lat, lon = DESTINATION
    query = {"lat": f"{lat:.4f}", "lon": f"{lon:.4f}"}
    compute = _json_get(base, "/compute", query=query, timeout=timeout, opener=opener)
    dest = compute.get("dest")
    if not (isinstance(dest, list) and len(dest) == 2):
        raise SmokeFailure("GET /compute returned an invalid destination")
    cell_id = _reachable_cell(compute)
    itinerary = _json_get(
        base,
        "/itinerary",
        query={"id": cell_id, "dlat": f"{lat:.4f}", "dlon": f"{lon:.4f}"},
        timeout=timeout,
        opener=opener,
    )
    if not isinstance(itinerary.get("total"), (int, float)) or isinstance(itinerary.get("total"), bool):
        raise SmokeFailure("GET /itinerary returned no route total")
    if not isinstance(itinerary.get("legs"), list) or not itinerary["legs"]:
        raise SmokeFailure("GET /itinerary returned no route legs")


def run(base_url: str = DEFAULT_BASE_URL, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
        opener: Opener = urlopen) -> None:
    """Run the public contract checks; raise :class:`SmokeFailure` on the first failure."""
    base = _base_url(base_url)
    if timeout <= 0:
        raise SmokeFailure("timeout must be positive")
    _check_health(base, timeout=timeout, opener=opener)
    _check_map_provider(base, timeout=timeout, opener=opener)
    _check_routing(base, timeout=timeout, opener=opener)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("SFCI_SMOKE_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS,
                        help="per-request timeout in seconds (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        run(args.base_url, timeout=args.timeout)
    except SmokeFailure as exc:
        print(f"production smoke failed: {exc}", file=sys.stderr)
        return 1
    print("production smoke passed: health, keyless basemap contract, compute, itinerary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
