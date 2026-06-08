"""Geocoding for destination.py and the server's /geocode + /autocomplete endpoints.

Default provider is **Photon** (https://photon.komoot.io/api) — free, no API key, biased
to San Francisco, and fast enough for type-ahead autocomplete. The provider is swappable
via the GEOCODER env var ("photon" default, "nominatim" fallback).

Two layers of caching, deliberately kept separate:
  * an in-memory bounded LRU (~512 entries, keyed by normalized query) wraps BOTH geocode
    and autocomplete so repeated/popular lookups never hit the upstream. This lives only in
    the process and is NOT persisted.
  * the gitignored .dest_cache.json persists ONLY destination.py's resolved workplace
    (geocode(..., cache=True)); the in-memory LRU never writes to it. That file's label is
    the input address, so a typed workplace round-trips unchanged across restarts.

Public API:
  geocode(q, *, cache=True) -> (lat, lon, label)   # top hit; raises LookupError if none
  autocomplete(q, limit=6)  -> [{"label", "lat", "lon"}, ...]   # type-ahead, <=limit
"""
import json
import os
import time
import threading
import urllib.request
import urllib.parse
from collections import OrderedDict
from . import config

# Compliant User-Agent on every upstream request (Nominatim requires one; Photon is polite).
_UA = "SF Commute Explorer (https://sfcommutemap.com)"

# Persistent destination cache (destination.py only). The in-memory LRU below never touches it.
_CACHE = config.ROOT / ".dest_cache.json"

# San Francisco geographic bias + bounding box, shared by both providers.
_SF_LAT, _SF_LON = 37.773, -122.42
_SF_BBOX = "-122.55,37.70,-122.34,37.84"   # minLon,minLat,maxLon,maxLat


def _geoapify_key():
    return os.environ.get("GEOAPIFY_KEY", "").strip()


def _provider():
    """Effective provider, lowercased: 'geoapify' (only if selected AND a key is present),
    else 'photon' (default) or 'nominatim'."""
    p = (os.environ.get("GEOCODER") or "photon").strip().lower()
    if p == "geoapify" and not _geoapify_key():
        return "photon"
    return p


def _dedup(results):
    """Drop duplicate hits (a provider can return the same place several times — e.g. Photon
    emits one OSM object under multiple tags). Keep the first (highest-ranked) occurrence;
    skip any later one colliding on upstream id, ~1m-rounded coord, OR normalized label."""
    seen_id, seen_coord, seen_label = set(), set(), set()
    out = []
    for r in results:
        oid = r.get("_id")
        has_id = bool(oid) and None not in oid
        coord = (round(r["lat"], 5), round(r["lon"], 5))
        label = " ".join(str(r["label"]).split()).casefold().rstrip(" ,.")
        if (has_id and oid in seen_id) or coord in seen_coord or label in seen_label:
            continue
        if has_id:
            seen_id.add(oid)
        seen_coord.add(coord)
        seen_label.add(label)
        out.append(r)
    return out


def _norm(q):
    """Normalize a query for cache keying: trim + collapse internal whitespace, casefold."""
    return " ".join(str(q).split()).casefold()


# ---- Bounded in-memory LRU (shared by geocode + autocomplete) --------------------------
# Keyed by (kind, provider, normalized-query, limit) so a geocode hit and an autocomplete
# hit for the same text don't collide, and switching GEOCODER at runtime stays correct.
# The lock makes move_to_end + popitem safe under concurrent Flask request threads (without
# it, a get racing with an eviction can KeyError on the `return _LRU[key]` after move_to_end).
_LRU_MAX = 512
_LRU = OrderedDict()
_LRU_LOCK = threading.Lock()


def _lru_get(key):
    with _LRU_LOCK:
        if key in _LRU:
            _LRU.move_to_end(key)
            return _LRU[key]
    return None


def _lru_put(key, value):
    with _LRU_LOCK:
        _LRU[key] = value
        _LRU.move_to_end(key)
        while len(_LRU) > _LRU_MAX:
            _LRU.popitem(last=False)


def _clear_lru():
    """Test/maintenance hook: drop the in-memory cache."""
    with _LRU_LOCK:
        _LRU.clear()


# ---- Global upstream rate-limit (ban protection) ---------------------------------------
# The LRU stops REPEAT lookups, and Flask-Limiter caps per-IP, but neither bounds the rate of
# UNIQUE queries hitting the provider across all visitors — which is how a public deploy gets the
# server IP banned (Nominatim's usage policy is a hard 1 req/s + identifying UA; Photon asks for
# fair use; Geoapify is metered by the key). A tiny global min-interval throttle keeps us polite.
# It only engages on an LRU MISS, so cached keystrokes/popular queries are never delayed.
_MIN_INTERVAL = {"nominatim": 1.0, "photon": 0.25, "geoapify": 0.0}   # seconds between upstream calls
_throttle_lock = threading.Lock()
_last_upstream = 0.0


def _throttle(provider):
    global _last_upstream
    iv = _MIN_INTERVAL.get(provider, 0.5)
    if iv <= 0:
        return
    with _throttle_lock:                       # serialize upstream calls to <= the provider's rate
        wait = _last_upstream + iv - time.monotonic()
        if wait > 0:
            time.sleep(min(wait, iv))          # cap the wait at one interval (clock-jump safe)
        _last_upstream = time.monotonic()


# ---- HTTP ------------------------------------------------------------------------------
def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    return json.load(urllib.request.urlopen(req, timeout=12))


# ---- Photon ----------------------------------------------------------------------------
def _photon_url(q, limit):
    params = {
        "q": q,
        "limit": int(limit),
        "lat": _SF_LAT,
        "lon": _SF_LON,
        "bbox": _SF_BBOX,
    }
    return "https://photon.komoot.io/api?" + urllib.parse.urlencode(params)


def _photon_label(props):
    """Build a human label from a Photon feature's properties. Joins, in order, the street
    address (housenumber + street), the place name, and the city — skipping blanks and
    de-duplicating (a POI named the same as its street shouldn't repeat). Falls back to
    state/postcode if nothing better is present."""
    props = props or {}
    house = (props.get("housenumber") or "").strip()
    street = (props.get("street") or "").strip()
    addr = (house + " " + street).strip() if (house or street) else ""
    parts, seen = [], set()
    for piece in (addr, (props.get("name") or "").strip(), (props.get("city") or "").strip()):
        if piece and piece.lower() not in seen:
            parts.append(piece)
            seen.add(piece.lower())
    if not parts:                      # nothing nameable; fall back to coarse fields
        for piece in ((props.get("state") or "").strip(),
                      (props.get("postcode") or "").strip(),
                      (props.get("country") or "").strip()):
            if piece and piece.lower() not in seen:
                parts.append(piece)
                seen.add(piece.lower())
    return ", ".join(parts)


def _photon_results(q, limit):
    """Raw Photon query -> list of {"label","lat","lon"} (already de-blanked). Network/parse
    errors propagate to the caller."""
    data = _get_json(_photon_url(q, limit))
    out = []
    for feat in (data or {}).get("features", []):
        geom = (feat or {}).get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feat.get("properties") or {}
        lon, lat = float(coords[0]), float(coords[1])
        out.append({"label": _photon_label(props) or q, "lat": lat, "lon": lon,
                    "_id": (props.get("osm_type"), props.get("osm_id"))})
    return out


# ---- Nominatim (fallback) --------------------------------------------------------------
def _nominatim_sf_query(q):
    """Bias an SF-local address toward San Francisco unless it already names a place."""
    low = q.lower()
    if "san francisco" in low or "sf" in low or ", ca" in low:
        return q
    return q + ", San Francisco, CA"


def _nominatim_results(q, limit):
    """Raw Nominatim query -> list of {"label","lat","lon"}."""
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": _nominatim_sf_query(q), "format": "json", "limit": int(limit)})
    data = _get_json(url)
    out = []
    for r in (data or []):
        out.append({"label": r.get("display_name") or q,
                    "lat": float(r["lat"]), "lon": float(r["lon"])})
    return out


# ---- Geoapify (default when GEOAPIFY_KEY is set) --------------------------------------
def _geoapify_results(q, limit, *, path="search"):
    """Raw Geoapify query -> [{label,lat,lon,_id}]. `path`='search' (forward) or
    'autocomplete' (type-ahead). SF-biased; needs GEOAPIFY_KEY. Errors propagate."""
    params = {"text": q, "limit": int(limit), "filter": "rect:" + _SF_BBOX,
              "bias": f"proximity:{_SF_LON},{_SF_LAT}", "format": "geojson",
              "apiKey": _geoapify_key()}
    url = f"https://api.geoapify.com/v1/geocode/{path}?" + urllib.parse.urlencode(params)
    data = _get_json(url)
    out = []
    for feat in (data or {}).get("features", []):
        p = feat.get("properties") or {}
        lat, lon = p.get("lat"), p.get("lon")
        if lat is None or lon is None:
            continue
        out.append({"label": p.get("formatted") or q, "lat": float(lat), "lon": float(lon),
                    "_id": ("geoapify", p.get("place_id"))})
    return out


def _provider_results(q, limit, *, kind="search"):
    """Dispatch a raw upstream query for the active provider -> list of result dicts.
    `kind` is 'search' or 'autocomplete' (only Geoapify has distinct endpoints)."""
    p = _provider()
    _throttle(p)                               # polite global rate-limit (LRU miss only)
    if p == "geoapify":
        return _geoapify_results(q, limit, path=("autocomplete" if kind == "autocomplete" else "search"))
    if p == "nominatim":
        return _nominatim_results(q, limit)
    return _photon_results(q, limit)


# ---- Public API ------------------------------------------------------------------------
def autocomplete(q, limit=6):
    """Type-ahead suggestions for `q`: a list of up to `limit` {"label","lat","lon"} dicts,
    best match first. Blank/whitespace input -> []. Backed by the in-memory LRU so repeated
    keystrokes/popular queries don't hit the upstream. Network/parse errors propagate."""
    norm = _norm(q)
    if len(norm) < 3:                          # too short to be useful; don't hit the upstream
        return []
    limit = max(1, int(limit))
    key = ("auto", _provider(), norm, limit)
    hit = _lru_get(key)
    if hit is not None:
        return [dict(r) for r in hit]            # defensive copy; callers may mutate
    raw = _provider_results(q.strip(), limit + 4, kind="autocomplete")
    clean = [{"label": r["label"], "lat": r["lat"], "lon": r["lon"]}
             for r in _dedup(raw)][:limit]
    _lru_put(key, clean)
    return [dict(r) for r in clean]


def geocode(addr, *, cache=True):
    """Geocode `addr` -> (lat, lon, label), best match first; raises LookupError if the
    provider returns nothing. Network/parse errors propagate (OSError / ValueError / KeyError).

    `cache=True` (destination.py path) consults and writes the persistent .dest_cache.json
    and keeps label == addr so a typed workplace round-trips unchanged. `cache=False` (the
    server's /geocode) returns the provider's human label and never touches that file. Both
    paths share the bounded in-memory LRU so repeated lookups skip the upstream."""
    addr = str(addr)
    if cache and _CACHE.exists():
        store = json.loads(_CACHE.read_text())
        if addr in store:
            return tuple(store[addr])

    norm = _norm(addr)
    key = ("geo", _provider(), norm, 1)
    hit = _lru_get(key)
    if hit is None:
        results = _provider_results(addr.strip(), 1)
        if not results:
            raise LookupError(f"could not geocode {addr!r}")
        top = results[0]
        hit = (top["lat"], top["lon"], top["label"])
        _lru_put(key, hit)
    lat, lon, label = hit

    if cache:
        # Preserve the historical round-trip contract: persisted label IS the input address.
        res = (lat, lon, addr)
        store = json.loads(_CACHE.read_text()) if _CACHE.exists() else {}
        store[addr] = list(res)
        _CACHE.write_text(json.dumps(store, indent=2))
        return res
    return (lat, lon, label)
