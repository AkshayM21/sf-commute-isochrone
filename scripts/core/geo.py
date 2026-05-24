"""Nominatim geocoding, shared by destination.py and the server's /geocode endpoint.

Results are cached in the gitignored .dest_cache.json (keyed by the raw query string).
The returned label is the input address, so a typed string round-trips unchanged.
"""
import json
import urllib.request
import urllib.parse
from . import config

_UA = "sf-commute-isochrone/1.0"
_CACHE = config.ROOT / ".dest_cache.json"


def _sf_query(q):
    """Bias an SF-local address toward San Francisco unless it already names a place."""
    low = q.lower()
    if "san francisco" in low or "sf" in low or ", ca" in low:
        return q
    return q + ", San Francisco, CA"


def geocode(addr, *, cache=True):
    """Geocode `addr` -> (lat, lon, label). label == addr (the input). Caches in
    .dest_cache.json. Raises LookupError if Nominatim returns nothing; network/parse
    errors propagate as urllib/ValueError for the caller to handle."""
    store = {}
    if cache and _CACHE.exists():
        store = json.loads(_CACHE.read_text())
        if addr in store:
            return tuple(store[addr])
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": _sf_query(addr), "format": "json", "limit": 1})
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    d = json.load(urllib.request.urlopen(req, timeout=12))
    if not d:
        raise LookupError(f"could not geocode {addr!r}")
    res = (float(d[0]["lat"]), float(d[0]["lon"]), addr)
    if cache:
        store[addr] = list(res)
        _CACHE.write_text(json.dumps(store, indent=2))
    return res
