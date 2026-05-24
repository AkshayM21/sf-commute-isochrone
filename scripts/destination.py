"""Single source of truth for the commute destination.

Resolution order (first match wins):
  1. DEST_LAT + DEST_LON env vars (explicit coordinates; DEST_LABEL optional)
  2. DEFAULT_ADDRESS env var -> geocoded once via Nominatim, cached in .dest_cache.json
  3. built-in fallback: geographic center of San Francisco

Set these in a (gitignored) `.env` at the repo root — see .env.example.
"""
import os, json, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FALLBACK = (37.7749, -122.4194, "San Francisco (set DEFAULT_ADDRESS in .env)")


def _load_dotenv():
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _geocode(addr):
    cache = ROOT / ".dest_cache.json"
    store = json.loads(cache.read_text()) if cache.exists() else {}
    if addr in store:
        return tuple(store[addr])
    q = addr
    if "san francisco" not in q.lower() and ", ca" not in q.lower():
        q += ", San Francisco, CA"
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": q, "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "sf-commute-isochrone/1.0"})
    d = json.load(urllib.request.urlopen(req, timeout=12))
    if not d:
        raise SystemExit(f"Could not geocode DEFAULT_ADDRESS={addr!r}")
    res = (float(d[0]["lat"]), float(d[0]["lon"]), addr)
    store[addr] = list(res)
    cache.write_text(json.dumps(store, indent=2))
    return res


_load_dotenv()
if os.environ.get("DEST_LAT") and os.environ.get("DEST_LON"):
    DEST_LAT = float(os.environ["DEST_LAT"])
    DEST_LON = float(os.environ["DEST_LON"])
    DEST_LABEL = os.environ.get("DEST_LABEL") or os.environ.get("DEFAULT_ADDRESS") or "custom location"
elif os.environ.get("DEFAULT_ADDRESS"):
    DEST_LAT, DEST_LON, DEST_LABEL = _geocode(os.environ["DEFAULT_ADDRESS"])
else:
    DEST_LAT, DEST_LON, DEST_LABEL = _FALLBACK
