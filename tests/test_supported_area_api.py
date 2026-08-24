"""Focused coverage for the API's graph-backed supported-area contract."""

import pytest


# Public, non-user fixtures spanning the graph boundary and common SF neighborhoods.
SUPPORTED = {
    "downtown": (37.7714, -122.4031),
    "mission": (37.7599, -122.4194),
    "sunset": (37.7520, -122.4940),
    "waterfront": (37.7950, -122.3920),
    "city_edge": (37.7080, -122.4860),
}
UNSUPPORTED = {
    "oakland": (37.8044, -122.2712),
    "berkeley": (37.8715, -122.2730),
    "daly_city": (37.6879, -122.4702),
    "water": (37.8400, -122.4200),
    "far": (38.0000, -122.0000),
}


def _outside(resp):
    assert resp.status_code == 422, resp.get_data(as_text=True)
    assert resp.get_json() == {
        "error": "outside_supported_area",
        "detail": "Location is outside the supported San Francisco walking area.",
    }


def test_graph_policy_matrix_includes_city_and_boundary_fixtures(server):
    for lat, lon in SUPPORTED.values():
        assert server._WG.supports_point(lon, lat, max_connector_m=300), (lat, lon)
    for lat, lon in UNSUPPORTED.values():
        assert not server._WG.supports_point(lon, lat, max_connector_m=300), (lat, lon)


@pytest.mark.parametrize("name", tuple(SUPPORTED))
def test_every_accepted_fixture_produces_reachable_cells(client, name):
    """The boundary predicate must never accept a destination the router cannot serve."""
    lat, lon = SUPPORTED[name]
    response = client.get(f"/compute?lat={lat}&lon={lon}")
    assert response.status_code == 200, response.get_data(as_text=True)
    cells = response.get_json()["cells"]
    assert any(
        isinstance(times, list) and any(value is not None for value in times)
        for times in cells.values()
    ), f"accepted supported-area fixture {name!r} produced no reachable cells"


@pytest.mark.parametrize("path", ("/compute", "/compute_exact"))
def test_compute_contract_rejects_unsupported_without_routing(client, server, monkeypatch, path):
    def fail(*_args, **_kwargs):
        raise AssertionError("unsupported coordinate reached routing")

    monkeypatch.setattr(server.sr, "compute_raptor", fail)
    lat, lon = UNSUPPORTED["oakland"]
    _outside(client.get(f"{path}?lat={lat}&lon={lon}"))


@pytest.mark.parametrize(
    "request_path",
    (
        "/variance?dlat=37.8044&dlon=-122.2712",
        "/attribution?dlat=37.8044&dlon=-122.2712",
        "/itinerary?id=not-a-cell&olat=37.7714&olon=-122.4031&dlat=37.8044&dlon=-122.2712",
    ),
)
def test_coordinate_endpoints_share_destination_policy(client, server, monkeypatch, request_path):
    def fail(*_args, **_kwargs):
        raise AssertionError("unsupported coordinate reached routing")

    monkeypatch.setattr(server.sr, "raptor_mc", fail)
    monkeypatch.setattr(server.sr, "raptor_attribution", fail)
    monkeypatch.setattr(server.sr, "itinerary_arriveby", fail)
    monkeypatch.setattr(server.sr, "itinerary_departafter", fail)
    _outside(client.get(request_path))


def test_itinerary_rejects_explicit_unsupported_origin_but_valid_cell_ids_remain_valid(
    client, server, monkeypatch
):
    # The destination is valid; the explicit origin is not, so no assembler may run.
    assembler_name = ("itinerary_departafter"
                      if server.RAPTOR_SEMANTIC == "departafter" else "itinerary_arriveby")
    monkeypatch.setattr(server.sr, assembler_name, lambda *_a, **_k: pytest.fail("routed"))
    lat, lon = UNSUPPORTED["oakland"]
    dlat, dlon = SUPPORTED["downtown"]
    _outside(client.get(
        f"/itinerary?olat={lat}&olon={lon}&dlat={dlat}&dlon={dlon}"
    ))

    cid = next(iter(server.ORIGIN_LL))
    monkeypatch.setattr(server.sr, assembler_name, lambda *_a, **_k: {"total": 1})
    ok = client.get(f"/itinerary?id={cid}&dlat={dlat}&dlon={dlon}")
    assert ok.status_code == 200, ok.get_data(as_text=True)


def test_geocode_rejects_upstream_unsupported_hit(client, server, monkeypatch):
    monkeypatch.setattr(
        server.geo, "geocode", lambda *_a, **_k: (*UNSUPPORTED["oakland"], "Oakland")
    )
    _outside(client.get("/geocode?q=oakland"))


def test_autocomplete_filters_unsupported_results_and_keeps_supported_order(
    client, server, monkeypatch
):
    supported_lat, supported_lon = SUPPORTED["mission"]
    outside_lat, outside_lon = UNSUPPORTED["berkeley"]
    monkeypatch.setattr(server.geo, "autocomplete", lambda *_a, **_k: [
        {"label": "Berkeley", "lat": outside_lat, "lon": outside_lon},
        {"label": "Mission", "lat": supported_lat, "lon": supported_lon},
        {"label": "NaN", "lat": float("nan"), "lon": supported_lon},
    ])
    resp = client.get("/autocomplete?q=bay")
    assert resp.status_code == 200
    assert resp.get_json()["results"] == [{
        "label": "Mission", "lat": supported_lat, "lon": supported_lon,
    }]


@pytest.mark.parametrize("query", ("lat=nan&lon=-122.4", "lat=&lon=-122.4", "lat=foo&lon=-122.4"))
def test_malformed_or_nonfinite_coordinates_remain_400(client, query):
    resp = client.get(f"/compute?{query}")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "bad_request"
