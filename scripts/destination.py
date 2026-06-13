"""Single source of truth for the commute destination.

Resolution order (first match wins):
  1. DEST_LAT + DEST_LON env vars (explicit coordinates; DEST_LABEL optional)
  2. DEFAULT_ADDRESS env var -> geocoded once via Nominatim, cached in .dest_cache.json
  3. built-in fallback: geographic center of San Francisco

Set these in a (gitignored) `.env` at the repo root — see .env.example.
"""
import os
from core import config, geo

config.load_dotenv()

_FALLBACK = (37.7749, -122.4194, "San Francisco (set DEFAULT_ADDRESS in .env)")


def _resolve():
    if os.environ.get("DEST_LAT") and os.environ.get("DEST_LON"):
        return (float(os.environ["DEST_LAT"]), float(os.environ["DEST_LON"]),
                os.environ.get("DEST_LABEL") or os.environ.get("DEFAULT_ADDRESS")
                or "custom location")
    addr = os.environ.get("DEFAULT_ADDRESS")
    if addr:
        try:
            return geo.geocode(addr)
        except (LookupError, OSError, ValueError) as e:
            # RuntimeError, NOT SystemExit: SystemExit is a BaseException, so it sails past
            # the `except Exception` guard around server.py's best-effort default-workplace
            # resolution and kills the service on a cold-cache/no-network boot. CLI importers
            # (isochrone.py, route_map.py, ...) still fail loudly with a traceback.
            raise RuntimeError(f"Could not geocode DEFAULT_ADDRESS={addr!r}: {e}") from e
    return _FALLBACK


DEST_LAT, DEST_LON, DEST_LABEL = _resolve()
