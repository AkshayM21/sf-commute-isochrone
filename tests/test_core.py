"""Fast, JVM-free unit tests for the pure-logic core (scripts/core/*).

These cover the *stable* modules so refactors can't silently regress the
canonical commute model, feed/route resolution, the SF grid, and geocoding:

    config  -- paths, commute constants, departure/window, .env parsing
    feeds   -- load_routes (feed-aware), route_name, pick_service_date
    grid    -- neighborhoods, build_grid, square_cells, attach_neighborhoods
    geo     -- autocomplete/geocode with the HTTP layer mocked (offline)

We deliberately do NOT import core.network or scripts.server -- they boot the
JVM (r5py). feeds/grid read the real GTFS zips and neighborhoods geojson in
data/, which is JVM-free and fast.

Run:  .venv/bin/python -m pytest tests/test_core.py -q
"""
import datetime as dt
import os
import sys
from pathlib import Path

import pytest

# Tests run from the repo root; make `from core import ...` work.
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from core import config, feeds, grid, geo  # noqa: E402


# --------------------------------------------------------------------------- #
# Shared fixtures: build the real (cheap) GTFS/grid artifacts once per module. #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def gtfs():
    paths = config.gtfs_paths()
    assert paths, "no GTFS feeds found under data/ -- cannot run feed tests"
    return paths


@pytest.fixture(scope="module")
def routes(gtfs):
    return feeds.load_routes(gtfs)


@pytest.fixture(scope="module")
def neigh():
    return grid.load_neighborhoods()


@pytest.fixture(scope="module")
def built_grid(neigh):
    # 200 m == the canonical finest resolution (~3068 cells).
    return grid.build_grid(neigh, 200)


# =========================================================================== #
# config.py                                                                   #
# =========================================================================== #
class TestConfig:
    def test_gtfs_paths_exist_and_include_caltrain(self):
        paths = config.gtfs_paths()
        assert paths, "expected at least one feed"
        for p in paths:
            assert isinstance(p, Path)
            assert p.exists(), f"gtfs_paths returned a non-existent path: {p}"
        names = {p.name for p in paths}
        # Caltrain must always be present (it's a committed feed in data/).
        assert config.CALTRAIN in names
        # Muni + BART round out the canonical 3-feed network.
        assert config.BART in names
        assert any("muni" in n.lower() for n in names)

    def test_gtfs_paths_network_order_muni_first(self):
        # Network order matters (Muni, BART, Caltrain). Muni leads, Caltrain trails.
        names = [p.name for p in config.gtfs_paths()]
        assert "muni" in names[0].lower()
        assert names[-1] == config.CALTRAIN

    def test_gtfs_paths_extra_resolves_under_data(self):
        # A bare name resolves under data/; non-existent extras are dropped silently.
        paths = config.gtfs_paths(extra=["does_not_exist_xyz.zip"])
        assert all(p.exists() for p in paths)
        assert not any("does_not_exist" in p.name for p in paths)

    def test_departure_is_835_on_given_date(self):
        d = dt.date(2026, 5, 20)
        dep = config.departure(d)
        assert isinstance(dep, dt.datetime)
        assert (dep.year, dep.month, dep.day) == (2026, 5, 20)
        assert (dep.hour, dep.minute) == (8, 35)
        assert (dep.second, dep.microsecond) == (0, 0)
        assert config.DEP_HM == (8, 35)

    def test_window_default_is_30_min(self, monkeypatch):
        monkeypatch.delenv("WINDOW_MIN", raising=False)
        assert config.window() == dt.timedelta(minutes=30)
        assert config.WINDOW_MIN == 30

    def test_window_honors_env_override(self, monkeypatch):
        monkeypatch.setenv("WINDOW_MIN", "45")
        assert config.window() == dt.timedelta(minutes=45)

    def test_constants_are_sane(self):
        assert config.WGS == "EPSG:4326"
        assert config.UTM == "EPSG:32610"          # UTM 10N for SF
        assert config.GRID_M == 200
        assert config.MAX_MIN == 75
        assert config.PERCENTILES == [5, 50]       # best-case + median
        # Walking speed must beat r5py's slow 3.6 km/h default but stay human.
        assert 4.0 <= config.WALK_KMH <= 6.0
        assert config.WALK_KMH == 4.8

    def test_path_helpers_point_under_data(self):
        assert config.osm_path() == config.DATA / config.OSM_FILE
        assert config.neigh_path() == config.DATA / config.NEIGH_FILE
        assert config.ROOT.name == "sf-commute-isochrone"


class TestLoadDotenv:
    """load_dotenv parsing, exercised against a temp .env via monkeypatched ROOT."""

    def _run(self, monkeypatch, tmp_path, body, preset=None):
        env_file = tmp_path / ".env"
        env_file.write_text(body)
        monkeypatch.setattr(config, "ROOT", tmp_path)
        # Start from a clean env for the keys we care about.
        for k in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HASHKEY"):
            monkeypatch.delenv(k, raising=False)
        if preset:
            for k, v in preset.items():
                monkeypatch.setenv(k, v)
        config.load_dotenv()

    def test_parses_basic_key_value(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, "AAA=hello\nBBB=world\n")
        assert os.environ["AAA"] == "hello"
        assert os.environ["BBB"] == "world"

    def test_skips_comments_and_blank_lines(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, "# a comment\n\n   \nAAA=kept\n")
        assert os.environ["AAA"] == "kept"

    def test_strips_surrounding_quotes(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path,
                  'AAA="double quoted"\nBBB=\'single quoted\'\n')
        assert os.environ["AAA"] == "double quoted"
        assert os.environ["BBB"] == "single quoted"

    def test_handles_export_prefix(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, 'export AAA=exported\n')
        assert os.environ["AAA"] == "exported"

    def test_does_not_override_existing_env(self, monkeypatch, tmp_path):
        # setdefault semantics: a value already in the environment wins.
        self._run(monkeypatch, tmp_path, "AAA=fromfile\n", preset={"AAA": "fromenv"})
        assert os.environ["AAA"] == "fromenv"

    def test_value_with_equals_keeps_rhs_intact(self, monkeypatch, tmp_path):
        # split('=', 1): only the first '=' separates key from value.
        self._run(monkeypatch, tmp_path, "AAA=a=b=c\n")
        assert os.environ["AAA"] == "a=b=c"

    def test_inline_hash_is_not_a_comment(self, monkeypatch, tmp_path):
        # An address may legitimately contain '#'; only a leading '#' is a comment.
        self._run(monkeypatch, tmp_path, 'AAA=123 Main St #4\n')
        assert os.environ["AAA"] == "123 Main St #4"

    def test_missing_file_is_noop(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "ROOT", tmp_path / "nope")
        config.load_dotenv()  # must not raise


# =========================================================================== #
# feeds.py                                                                     #
# =========================================================================== #
class TestFeeds:
    def test_load_routes_shape(self, routes, gtfs):
        # {feed_stem: {route_id: name}}, one entry per feed.
        assert set(routes.keys()) == {Path(p).stem for p in gtfs}
        for stem, d in routes.items():
            assert isinstance(d, dict) and d, f"{stem} has no routes"
            k, v = next(iter(d.items()))
            assert isinstance(k, str) and isinstance(v, str)

    def test_route_name_is_feed_aware(self, routes):
        # The headline invariant: route_id "8" collides across feeds and must
        # resolve per-feed (Muni "8" the bus vs BART "8" == "Red-N").
        muni = feeds.route_name("8", "muni_current", routes)
        bart = feeds.route_name("8", "bart_gtfs", routes)
        assert muni == "8"
        assert bart == "Red-N"
        assert muni != bart

    def test_route_name_normalizes_floatish_id(self, routes):
        # r5py hands ids back as floats; "8.0" must resolve like "8".
        assert feeds.route_name("8.0", "bart_gtfs", routes) == "Red-N"
        assert feeds.route_name(8.0, "bart_gtfs", routes) == "Red-N"
        assert feeds.route_name("8", "bart_gtfs", routes) == "Red-N"

    def test_route_name_nan_returns_none(self, routes):
        import math
        assert feeds.route_name(float("nan"), "bart_gtfs", routes) is None
        assert feeds.route_name(math.nan, "muni_current", routes) is None

    def test_route_name_unknown_id_falls_back_to_key(self, routes):
        # Missing id returns the normalized key itself, not a crash.
        assert feeds.route_name("999999", "bart_gtfs", routes) == "999999"
        # Unknown feed -> empty map -> key returned.
        assert feeds.route_name("8", "no_such_feed", routes) == "8"

    def test_route_name_non_numeric_id_passes_through(self, routes):
        # A non-float-ish id is used verbatim as the key.
        assert feeds.route_name("KT", "muni_current", routes) == \
            routes["muni_current"].get("KT", "KT")

    def test_pick_service_date_is_a_wednesday(self, gtfs):
        d = feeds.pick_service_date(gtfs)
        assert isinstance(d, dt.date)
        assert d.weekday() == 2, "service date must be a Wednesday"

    def test_picked_date_has_trips_in_every_feed(self, gtfs):
        d = feeds.pick_service_date(gtfs)
        ds = d.strftime("%Y%m%d")
        for p in gtfs:
            assert feeds._feed_has_trips(p, ds), \
                f"{Path(p).stem} has no trips on the picked date {ds}"

    def test_feed_has_trips_false_far_future(self, gtfs):
        # Well outside any GTFS validity window -> no service.
        assert feeds._feed_has_trips(gtfs[0], "20990101") is False


# =========================================================================== #
# grid.py                                                                      #
# =========================================================================== #
class TestGrid:
    def test_load_neighborhoods_wgs_nonempty(self, neigh):
        assert len(neigh) > 0
        assert str(neigh.crs).upper() == config.WGS
        assert "name" in neigh.columns
        assert "geometry" in neigh.columns

    def test_build_grid_count_and_ids(self, built_grid):
        # The 200 m grid extent follows the neighborhoods dataset (currently Realtor-based,
        # ~2999 cells). Assert a sane SF-sized range rather than an exact count, so a dataset
        # refresh doesn't break the test but an empty/blown-up grid still does.
        assert 2500 <= len(built_grid) <= 3300, \
            f"grid cell count {len(built_grid)} outside the expected SF range"
        # ids are strings "0".."N-1", contiguous and in order.
        ids = list(built_grid["id"])
        assert all(isinstance(i, str) for i in ids)
        assert ids == [str(i) for i in range(len(built_grid))]

    def test_build_grid_geometry_and_crs(self, built_grid):
        assert built_grid.geometry.name == "geometry"
        assert str(built_grid.crs).upper() == config.WGS
        assert (built_grid.geom_type == "Point").all()

    def test_square_cells_index_aligned_polygons(self, built_grid):
        cells = grid.square_cells(built_grid, 200)
        assert len(cells) == len(built_grid)
        assert list(cells.index) == list(built_grid.index)
        assert (cells.geom_type == "Polygon").all()
        assert str(cells.crs).upper() == config.WGS

    def test_square_cells_are_roughly_grid_m_on_a_side(self, built_grid):
        # Side length in metric UTM should be ~200 m (squares built in UTM).
        cells_utm = grid.square_cells(built_grid, 200).to_crs(config.UTM)
        minx, miny, maxx, maxy = cells_utm.iloc[0].bounds
        assert abs((maxx - minx) - 200) < 1.0
        assert abs((maxy - miny) - 200) < 1.0

    def test_attach_neighborhoods_adds_name_one_row_per_id(self, built_grid, neigh):
        cells = grid.square_cells(built_grid, 200)
        cells = cells.to_frame("geometry")
        cells["id"] = built_grid["id"].values
        cells = grid.__dict__["gpd"].GeoDataFrame(cells, geometry="geometry",
                                                  crs=cells.crs)
        attached = grid.attach_neighborhoods(cells, neigh)
        assert "name" in attached.columns
        # One row per cell id (the sjoin de-dups multi-match cells).
        assert attached["id"].nunique() == len(attached)
        assert len(attached) == len(built_grid)
        assert "index_right" not in attached.columns


# =========================================================================== #
# geo.py  (HTTP mocked -> deterministic + offline)                            #
# =========================================================================== #
def _photon_feature(lon, lat, **props):
    return {"geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props}


@pytest.fixture(autouse=True)
def _isolate_geo(monkeypatch):
    """Every geo test starts with a clean LRU, the default (photon) provider,
    and no persistent cache leaking in."""
    monkeypatch.delenv("GEOCODER", raising=False)
    geo._clear_lru()
    # Make the persistent .dest_cache.json invisible so geocode() tests are pure.
    monkeypatch.setattr(geo, "_CACHE", config.ROOT / ".test_nonexistent_cache.json")
    yield
    geo._clear_lru()


class TestGeoAutocomplete:
    def test_parses_photon_geojson(self, monkeypatch):
        payload = {"features": [
            _photon_feature(-122.3937, 37.7955, name="Ferry Building",
                            city="San Francisco"),
        ]}
        monkeypatch.setattr(geo, "_get_json", lambda url: payload)
        out = geo.autocomplete("ferry building")
        assert out == [{"label": "Ferry Building, San Francisco",
                        "lat": 37.7955, "lon": -122.3937}]
        assert set(out[0].keys()) == {"label", "lat", "lon"}

    def test_blank_query_returns_empty(self, monkeypatch):
        # Must short-circuit BEFORE any HTTP call.
        def boom(url):
            raise AssertionError("blank query should not hit the network")
        monkeypatch.setattr(geo, "_get_json", boom)
        assert geo.autocomplete("") == []
        assert geo.autocomplete("   \t  ") == []

    def test_lru_caches_second_call(self, monkeypatch):
        calls = {"n": 0}

        def counting(url):
            calls["n"] += 1
            return {"features": [_photon_feature(-122.4, 37.78, name="Cafe")]}
        monkeypatch.setattr(geo, "_get_json", counting)
        first = geo.autocomplete("cafe")
        second = geo.autocomplete("cafe")
        assert first == second
        assert calls["n"] == 1, "second identical query must be served from the LRU"

    def test_lru_returns_defensive_copy(self, monkeypatch):
        monkeypatch.setattr(geo, "_get_json",
                            lambda url: {"features": [_photon_feature(-122.4, 37.78, name="X")]})
        a = geo.autocomplete("x")
        a[0]["label"] = "MUTATED"
        b = geo.autocomplete("x")  # served from cache; must be unaffected
        assert b[0]["label"] == "X"

    def test_normalized_query_shares_cache(self, monkeypatch):
        calls = {"n": 0}

        def counting(url):
            calls["n"] += 1
            return {"features": [_photon_feature(-122.4, 37.78, name="Y")]}
        monkeypatch.setattr(geo, "_get_json", counting)
        geo.autocomplete("Ferry  Building")
        geo.autocomplete("  ferry building ")  # same after _norm
        assert calls["n"] == 1

    def test_respects_limit(self, monkeypatch):
        feats = [_photon_feature(-122.4 - i / 1000, 37.78, name=f"P{i}")
                 for i in range(10)]
        monkeypatch.setattr(geo, "_get_json", lambda url: {"features": feats})
        out = geo.autocomplete("p", limit=3)
        assert len(out) == 3

    def test_request_url_carries_sf_bbox_and_bias(self, monkeypatch):
        seen = {}

        def capture(url):
            seen["url"] = url
            return {"features": [_photon_feature(-122.4, 37.78, name="Z")]}
        monkeypatch.setattr(geo, "_get_json", capture)
        geo.autocomplete("z")
        url = seen["url"]
        assert "photon.komoot.io" in url
        # SF bias coordinates + the bounding box must be in the query.
        assert "lat=37.773" in url
        assert "lon=-122.42" in url
        assert "37.70" in url and "37.84" in url  # bbox lat extent
        assert "-122.55" in url and "-122.34" in url  # bbox lon extent

    @pytest.mark.xfail(strict=False, reason=(
        "Encodes the KNOWN autocomplete dedup bug: when Photon returns two "
        "identical features (same label/lat/lon), autocomplete passes both "
        "through unchanged -- it never de-dupes across results. _photon_label "
        "only de-dupes the parts WITHIN one label. The correct expectation is "
        "that identical suggestions are collapsed; this currently FAILS."))
    def test_results_are_deduped(self, monkeypatch):
        dup = _photon_feature(-122.3937, 37.7955, name="Ferry Building",
                              city="San Francisco")
        payload = {"features": [dup, dict(dup)]}
        monkeypatch.setattr(geo, "_get_json", lambda url: payload)
        out = geo.autocomplete("ferry building")
        keys = [(r["label"], r["lat"], r["lon"]) for r in out]
        assert len(keys) == len(set(keys)), (
            "autocomplete returned duplicate identical suggestions "
            f"(dedup bug): {keys}")


class TestPhotonLabel:
    def test_builds_address_name_city(self):
        label = geo._photon_label({"housenumber": "1", "street": "Market St",
                                   "name": "Ferry Building", "city": "San Francisco"})
        assert label == "1 Market St, Ferry Building, San Francisco"

    def test_dedupes_repeated_pieces_within_label(self):
        # A POI named the same as its street should not repeat.
        label = geo._photon_label({"name": "Market Street", "street": "Market Street",
                                   "city": "San Francisco"})
        assert label.count("Market Street") == 1

    def test_falls_back_to_coarse_fields_when_empty(self):
        label = geo._photon_label({"state": "California", "postcode": "94105"})
        assert "California" in label
        assert "94105" in label

    def test_empty_props_yields_empty_string(self):
        assert geo._photon_label({}) == ""
        assert geo._photon_label(None) == ""


class TestGeocode:
    def test_returns_lat_lon_label_from_provider(self, monkeypatch):
        monkeypatch.setattr(geo, "_get_json",
                            lambda url: {"features": [
                                _photon_feature(-122.3937, 37.7955,
                                                name="Ferry Building",
                                                city="San Francisco")]})
        lat, lon, label = geo.geocode("ferry building", cache=False)
        assert (lat, lon) == (37.7955, -122.3937)
        assert label == "Ferry Building, San Francisco"

    def test_raises_lookuperror_on_empty(self, monkeypatch):
        monkeypatch.setattr(geo, "_get_json", lambda url: {"features": []})
        with pytest.raises(LookupError):
            geo.geocode("nowhere at all zzz", cache=False)

    def test_geocode_uses_lru(self, monkeypatch):
        calls = {"n": 0}

        def counting(url):
            calls["n"] += 1
            return {"features": [_photon_feature(-122.4, 37.78, name="Q")]}
        monkeypatch.setattr(geo, "_get_json", counting)
        geo.geocode("q", cache=False)
        geo.geocode("q", cache=False)
        assert calls["n"] == 1


class TestProviderSwitch:
    def test_default_provider_is_photon(self, monkeypatch):
        monkeypatch.delenv("GEOCODER", raising=False)
        assert geo._provider() == "photon"

    def test_geocoder_env_switches_to_nominatim(self, monkeypatch):
        monkeypatch.setenv("GEOCODER", "nominatim")
        assert geo._provider() == "nominatim"

        seen = {}

        def capture(url):
            seen["url"] = url
            return [{"lat": "37.7955", "lon": "-122.3937",
                     "display_name": "Ferry Building, SF"}]
        monkeypatch.setattr(geo, "_get_json", capture)
        lat, lon, label = geo.geocode("123 Main St", cache=False)
        assert (lat, lon) == (37.7955, -122.3937)
        assert label == "Ferry Building, SF"
        # Nominatim path + SF bias applied to a bare local address.
        assert "nominatim.openstreetmap.org" in seen["url"]
        assert "San+Francisco" in seen["url"]

    def test_nominatim_autocomplete_parses_list(self, monkeypatch):
        monkeypatch.setenv("GEOCODER", "nominatim")
        monkeypatch.setattr(geo, "_get_json", lambda url: [
            {"lat": "37.78", "lon": "-122.41", "display_name": "A, SF"},
            {"lat": "37.79", "lon": "-122.42", "display_name": "B, SF"}])
        out = geo.autocomplete("a")
        assert out == [{"label": "A, SF", "lat": 37.78, "lon": -122.41},
                       {"label": "B, SF", "lat": 37.79, "lon": -122.42}]

    def test_provider_keys_cache_separately(self, monkeypatch):
        # Switching GEOCODER at runtime must NOT serve a stale cross-provider hit.
        calls = {"photon": 0, "nominatim": 0}

        def photon(url):
            calls["photon"] += 1
            return {"features": [_photon_feature(-122.4, 37.78, name="P")]}

        def nominatim(url):
            calls["nominatim"] += 1
            return [{"lat": "37.78", "lon": "-122.4", "display_name": "N"}]

        monkeypatch.delenv("GEOCODER", raising=False)
        monkeypatch.setattr(geo, "_get_json", photon)
        geo.autocomplete("same")
        monkeypatch.setenv("GEOCODER", "nominatim")
        monkeypatch.setattr(geo, "_get_json", nominatim)
        geo.autocomplete("same")
        assert calls["photon"] == 1 and calls["nominatim"] == 1
