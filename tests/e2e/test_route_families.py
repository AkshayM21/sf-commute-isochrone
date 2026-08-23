"""Bounded live-server tests for route-family truth and responsive route inspector state.

The responsive assertions use the inspector controller's data attributes rather than retired
breakpoint classes.  Route identity checks remain deliberately independent of presentation.
"""

import os
from urllib.parse import quote

import pytest

from conftest import BASE_URL
from route_family_hotspots import (
    ARTIFACT_PATH, DEFAULT_SEED, PARETO_DIMENSIONS, PUBLIC_DESTINATIONS, SAVED_HOTSPOT,
    _api_get, _rank_hotspots, _route_slots, _speed_query,
    _stable_rank, _transit_legs, assert_api_matches_dom, dom_family_snapshot,
    measure_family_card, open_destination, origin_container_point, route_metrics, scan_hotspots,
    validate_route_response, write_artifact,
)


def _enabled(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _catalog_subset():
    count = max(1, min(int(os.environ.get("ROUTE_FAMILY_HOTSPOT_DESTS", "6")), len(PUBLIC_DESTINATIONS)))
    return tuple(sorted(PUBLIC_DESTINATIONS,
                        key=lambda item: _stable_rank(DEFAULT_SEED, "destination", item["slug"]))[:count])


def _scan_speeds():
    speeds = tuple(value.strip() for value in os.environ.get(
        "ROUTE_FAMILY_HOTSPOT_SPEEDS", "slow,med,fast").split(",") if value.strip())
    assert speeds and set(speeds).issubset({"slow", "med", "fast"}), speeds
    return speeds


def _cell_container_point(page, cell_id):
    found = page.evaluate("""id => { let target=null; layer.eachLayer(candidate => {
      if(String(candidate.feature?.properties?.id)===String(id)) target=candidate; });
      if(!target) return false; map.setView(target.getBounds().getCenter(),13,{animate:false}); return true; }""",
                          cell_id)
    assert found, f"Leaflet layer did not contain saved cell {cell_id}"
    page.wait_for_function("""id => { let target=null; layer.eachLayer(candidate => {
      if(String(candidate.feature?.properties?.id)===String(id)) target=candidate; });
      if(!target || !map?._loaded) return false;
      const point=map.latLngToContainerPoint(target.getBounds().getCenter());
      const rect=document.getElementById('map').getBoundingClientRect();
      return point.x>=0 && point.y>=0 && point.x<=rect.width && point.y<=rect.height;
    }""", arg=cell_id, timeout=5_000)
    return page.evaluate("""id => { let target=null; layer.eachLayer(candidate => {
      if(String(candidate.feature?.properties?.id)===String(id)) target=candidate; });
      const point=map.latLngToContainerPoint(target.getBounds().getCenter());
      const rect=document.getElementById('map').getBoundingClientRect();
      return {x:rect.left+point.x,y:rect.top+point.y}; }""", cell_id)


def _expand_all_route_disclosures(page):
    toggle = page.locator("#all-routes-toggle")
    if not (page.locator("#allroutes").count() and toggle.count() and toggle.is_visible()):
        return
    if not page.locator("#allroutes").evaluate("el => el.open"):
        toggle.click()
    page.wait_for_function("() => document.getElementById('allroutes')?.open")
    page.wait_for_function("() => document.querySelectorAll('#all-routes-panel .route-choice').length > 0")


def _scan_with_page(new_context, destinations, speeds, per_config):
    context = new_context(viewport={"width": 900, "height": 700})
    artifact = scan_hotspots(context.request, BASE_URL, destinations=destinations, speeds=speeds,
                             seed=DEFAULT_SEED, per_config=per_config)
    path = write_artifact(artifact)
    errors = [f"{row['destination']['slug']}/{row['speed']}/{row['cell_id']}: {error}"
              for row in artifact["hotspots"] for error in row["errors"]]
    assert artifact["hotspots"], f"hotspot scan produced no candidates; artifact={path}"
    assert not errors, f"route-family API false advertising; artifact={path}:\n" + "\n".join(errors)
    return artifact, path


def test_hotspot_ranking_retains_pareto_extremes_before_scalar_fill():
    def hotspot(cell_id, score, **dimensions):
        metrics = {dimension: 0 for dimension in PARETO_DIMENSIONS}
        metrics.update(dimensions)
        metrics["score"] = score
        return {"destination": {"slug": "fixture"}, "speed": "med", "cell_id": cell_id,
                "metrics": metrics, "errors": []}

    route_rich = hotspot("route-rich", 50, routes=8, families=4, branches=7,
                         max_branches_per_family=3, max_options_per_family=4, max_transfers=2,
                         unique_lines=6, unique_services=7, label_chars=40, longest_label=20,
                         time_spread=8, geometry_points=100, service_branch_cross_product=20)
    label_geometry = hotspot("label-geometry", 15, routes=2, families=1, branches=1,
                             max_branches_per_family=1, max_options_per_family=2, unique_lines=2,
                             unique_services=2, label_chars=300, longest_label=180, time_spread=20,
                             geometry_points=500, service_branch_cross_product=2)
    dominated = hotspot("route-dominated", 35, routes=6, families=3, branches=5,
                        max_branches_per_family=2, max_options_per_family=3, max_transfers=1,
                        unique_lines=5, unique_services=6, label_chars=30, longest_label=15,
                        time_spread=7, geometry_points=90, service_branch_cross_product=15)
    label_dominated = hotspot("label-dominated", 10, routes=1, families=1, branches=1,
                              max_branches_per_family=1, max_options_per_family=1, unique_lines=1,
                              unique_services=1, label_chars=200, longest_label=100, time_spread=10,
                              geometry_points=300, service_branch_cross_product=1)
    shuffled = [dominated, label_dominated, label_geometry, route_rich]
    ranked = _rank_hotspots(shuffled, seed=101, limit=3)
    assert [row["cell_id"] for row in ranked] == ["route-rich", "label-geometry", "route-dominated"]
    assert [row["pareto_frontier"] for row in ranked] == [True, True, False]
    assert tuple(ranked[0]["pareto_vector"]) == PARETO_DIMENSIONS
    assert [row["cell_id"] for row in _rank_hotspots(list(reversed(shuffled)), seed=101, limit=3)] == [
        row["cell_id"] for row in ranked]
    assert [row["cell_id"] for row in _rank_hotspots(shuffled, seed=101, limit=2)] == [
        "route-rich", "label-geometry"]


def test_route_family_hotspot_sampler_smoke(new_context):
    artifact, path = _scan_with_page(new_context, (SAVED_HOTSPOT["destination"],), ("med",), 4)
    assert path == ARTIFACT_PATH and artifact["health"]["engine"] == "raptor"
    assert all(row["metrics"]["families"] >= 1 for row in artifact["hotspots"])


def test_saved_hotspot_walk_speed_is_monotone(new_context):
    context = new_context(viewport={"width": 800, "height": 600})
    destination, origin, totals = SAVED_HOTSPOT["destination"], SAVED_HOTSPOT["origin"], {}
    for speed in ("slow", "med", "fast"):
        body = _api_get(context.request, BASE_URL,
                        f"/itinerary?olat={origin['lat']}&olon={origin['lon']}&dlat={destination['lat']}"
                        f"&dlon={destination['lon']}{_speed_query(speed)}&pin=1")
        assert not validate_route_response(body, destination), speed
        totals[speed] = int((body.get("typical") or {}).get("total", body["total"]))
    assert totals["slow"] >= totals["med"] >= totals["fast"], totals


def test_saved_medium_shared_corridor_retains_transfer_finish(new_context):
    context = new_context(viewport={"width": 800, "height": 600})
    destination = SAVED_HOTSPOT["destination"]
    body = _api_get(context.request, BASE_URL,
                    f"/itinerary?id={quote(SAVED_HOTSPOT['cell_id'])}&dlat={destination['lat']}"
                    f"&dlon={destination['lon']}&speed=med&pin=1")
    assert not validate_route_response(body, destination)
    by_family = {}
    for route in [body, *(body.get("alts") or [])]:
        family = route.get("family") or {}
        by_family.setdefault(str(family.get("key") or ""), {"meta": family, "routes": []})["routes"].append(route)
    assert any(
        len(value["meta"].get("services") or []) >= 2
        and len({str(service.get("key") or "") for service in value["meta"].get("services") or []})
            == len(value["meta"].get("services") or [])
        and any((route.get("branch") or {}).get("kind") == "transit"
                and len(_transit_legs(_route_slots(route)[1])) >= 2 for route in value["routes"])
        for value in by_family.values()
    )


WALK_SPEED_REGRESSION_CASES = (
    ("market-982", "982", {"slug": "market", "lat": 37.7942095, "lon": -122.3947452}),
    ("market-869", "869", {"slug": "market", "lat": 37.7942095, "lon": -122.3947452}),
    ("mission-bay-982", "982", PUBLIC_DESTINATIONS[2]),
    ("mission-bay-325", "325", PUBLIC_DESTINATIONS[2]),
    ("mission-bay-1219", "1219", PUBLIC_DESTINATIONS[2]),
    ("city-hall-1507", "1507", PUBLIC_DESTINATIONS[3]),
    ("city-hall-2080", "2080", PUBLIC_DESTINATIONS[3]),
    ("city-hall-2401", "2401", PUBLIC_DESTINATIONS[3]),
)


def test_walk_speed_matrix_is_monotone(new_context):
    """Historically dense public cells remain monotone at every supported walking pace."""
    context = new_context(viewport={"width": 800, "height": 600})
    failures = []
    for label, cell_id, destination in WALK_SPEED_REGRESSION_CASES:
        totals = {}
        for speed in ("slow", "med", "fast"):
            body = _api_get(context.request, BASE_URL,
                            f"/itinerary?id={quote(cell_id)}&dlat={destination['lat']}&dlon={destination['lon']}"
                            f"{_speed_query(speed)}&pin=1")
            assert not validate_route_response(body, destination), f"{label}/{speed}"
            totals[speed] = int((body.get("typical") or {}).get("total", body["total"]))
        if not totals["slow"] >= totals["med"] >= totals["fast"]:
            failures.append(f"{label}: {totals}")
    assert not failures, "walk-speed monotonicity failures:\n" + "\n".join(failures)


@pytest.mark.skipif(not _enabled("ROUTE_FAMILY_HOTSPOT_SCAN"),
                    reason="set ROUTE_FAMILY_HOTSPOT_SCAN=1 for the broader seeded SF scan")
def test_route_family_hotspot_sampler_broad(new_context):
    destinations, speeds = _catalog_subset(), _scan_speeds()
    per_config = max(1, min(int(os.environ.get("ROUTE_FAMILY_HOTSPOT_PER_CONFIG", "5")), 5))
    assert len(destinations) * len(speeds) <= 18 and len(destinations) * len(speeds) * per_config <= 90
    artifact, path = _scan_with_page(new_context, destinations, speeds, per_config)
    assert len(artifact["configs"]) <= 18 and len(artifact["hotspots"]) <= 90, path


MOBILE_CONTEXT = {
    "viewport": {"width": 390, "height": 844}, "has_touch": True, "is_mobile": True,
    "device_scale_factor": 3,
    "user_agent": ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                   "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"),
}


def _state(page):
    state = page.locator("#pincard").evaluate("""el => ({
      open:el.classList.contains('open'), layout:el.dataset.layoutCapability, surface:el.dataset.surface,
      planOpen:el.dataset.planOpen, presentation:el.dataset.presentation,
      sheetContent:el.dataset.sheetContent, sheetSnap:el.dataset.sheetSnap,
      dragging:el.dataset.dragging, selected:el.dataset.selectedChoiceKey || null })""")
    assert state["open"], state
    for name in ("layout", "surface", "planOpen", "presentation"):
        assert state[name] is not None, state
    return state


def _wait_state(page, **expected):
    attributes = {"layout": "layoutCapability", "surface": "surface", "planOpen": "planOpen",
                  "presentation": "presentation", "sheetContent": "sheetContent",
                  "sheetSnap": "sheetSnap", "dragging": "dragging"}
    page.wait_for_function("""([expected, attributes]) => { const el=document.getElementById('pincard');
      return !!el && Object.entries(expected).every(([key,value]) => el.dataset[attributes[key]]===value); }""",
                           arg=[expected, attributes], timeout=8_000)
    return _state(page)


def _click(page, selector, *, touch=False):
    target = page.locator(selector)
    assert target.count(), f"missing semantic inspector control: {selector}"
    (target.tap if touch else target.click)()


def _rects(page, *selectors):
    return page.evaluate("""selectors => Object.fromEntries(selectors.map(selector => {
      const r=document.querySelector(selector)?.getBoundingClientRect();
      return [selector,r&&{left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width,height:r.height}];
    }))""", list(selectors))


def _wait_for_final_pin(page):
    # Peek intentionally hides the choice scroller visually, so mobile only requires the final
    # route rows to be mounted. Browse/Expanded tests assert their visibility after snapping.
    page.wait_for_selector("#pincard.open #route-choices-panel .route-choice", state="attached", timeout=30_000)
    page.wait_for_function("() => routePin!=null && BDCACHE.get(routePin)?._pin===true", timeout=30_000)
    page.wait_for_function("() => !document.getElementById('pincard').classList.contains('pinloading')",
                           timeout=30_000)


def _open_saved_inspector(page, *, touch):
    open_destination(page, BASE_URL, SAVED_HOTSPOT["destination"], SAVED_HOTSPOT["speed"])
    point = _cell_container_point(page, SAVED_HOTSPOT["cell_id"]) if touch else origin_container_point(
        page, SAVED_HOTSPOT["origin"])
    if touch:
        page.touchscreen.tap(point["x"], point["y"])
        page.wait_for_selector("#touchpeek.open #peekinspect", timeout=30_000)
        page.wait_for_function("() => !document.getElementById('peekbody').innerText.includes('Loading commute')",
                               timeout=30_000)
        page.locator("#peekinspect").tap()
    else:
        page.mouse.click(point["x"] + 1, point["y"] + 1)
    _wait_for_final_pin(page)


def _select_other_route(page, *, touch=False):
    row = page.locator("#route-choices-panel .route-choice[aria-pressed='false']").first
    if not row.count():
        _expand_all_route_disclosures(page)
        row = page.locator("#route-choices-panel .route-choice[aria-pressed='false']").first
    key = row.get_attribute("data-key")
    assert key, "saved hotspot needs an unselected alternative"
    (row.tap if touch else row.click)()
    page.wait_for_function("key => document.getElementById('pincard').dataset.selectedChoiceKey===key", arg=key,
                           timeout=5_000)
    page.wait_for_function("key => DRAWN?.key===key", arg=key, timeout=5_000)
    return key


@pytest.mark.parametrize("context_args,touch", [
    pytest.param({"viewport": {"width": 1280, "height": 800}}, False, id="desktop"),
    pytest.param(MOBILE_CONTEXT, True, id="mobile"),
])
def test_saved_hotspot_routes_match_api_and_keep_choice_identity(new_context, context_args, touch):
    context = new_context(**context_args)
    page = context.new_page()
    _open_saved_inspector(page, touch=touch)
    if touch:
        _click(page, '[data-show-choices]', touch=True)
        _wait_state(page, sheetSnap="browse", sheetContent="choices")
    cell_id, destination, speed = page.evaluate("() => routePin"), SAVED_HOTSPOT["destination"], SAVED_HOTSPOT["speed"]
    body = _api_get(context.request, BASE_URL,
                    f"/itinerary?id={quote(str(cell_id))}&dlat={destination['lat']}&dlon={destination['lon']}"
                    f"{_speed_query(speed)}&pin=1")
    assert not validate_route_response(body, destination)
    _expand_all_route_disclosures(page)
    snapshot, crowding = dom_family_snapshot(page), measure_family_card(page)
    assert_api_matches_dom(body, snapshot)
    assert crowding["families"] == route_metrics(body)["families"]
    assert crowding["branches"] == route_metrics(body)["branches"]
    assert crowding["route_choice_rows"] == crowding["route_options"]
    assert crowding["recommended_rows"] == 1 and len(crowding["selected_keys"]) == 1
    assert crowding["nested_interactives"] == 0 and crowding["horizontal_overflow"] <= 1
    assert not crowding["clipped_labels"], crowding
    selected_key = _select_other_route(page, touch=touch)
    assert _state(page)["selected"] == selected_key


def test_desktop_plan_sidecar_tray_scrolls_and_remaps_across_resize(new_context):
    """Plan is closed by default, then remains requested through sidecar/tray remapping."""
    context = new_context(viewport={"width": 1440, "height": 820})
    page = context.new_page()
    _open_saved_inspector(page, touch=False)
    _wait_state(page, layout="wide-sidecar", surface="routes", presentation="expanded", planOpen="false")
    assert not page.locator("#route-plan-panel").is_visible()
    toggle = page.locator("#pincard .pin-head [data-route-plan-control]")
    assert page.locator("[data-route-plan-toggle]").count() == 1
    assert toggle.get_attribute("aria-expanded") == "false"
    assert toggle.get_attribute("aria-controls") == "route-plan-panel"
    selected_key = _select_other_route(page)
    _expand_all_route_disclosures(page)
    page.wait_for_function("""() => { const el=document.getElementById('route-choices-panel');
      return el.scrollHeight>el.clientHeight; }""")
    page.evaluate("() => document.getElementById('route-choices-panel').scrollTop=80")
    choices_scroll = page.evaluate("() => document.getElementById('route-choices-panel').scrollTop")
    assert choices_scroll > 0
    _click(page, "#pincard .pin-head [data-route-plan-control]")
    _wait_state(page, planOpen="true")
    assert toggle.get_attribute("aria-expanded") == "true"
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == selected_key
    wide = _rects(page, "#pincard", "#route-choices-panel", "#route-plan-panel", "#map")
    card, choices, plan, map_rect = wide["#pincard"], wide["#route-choices-panel"], wide["#route-plan-panel"], wide["#map"]
    assert plan["width"] < choices["width"] and plan["right"] <= choices["left"] + 1, wide
    assert abs(plan["bottom"] - card["bottom"]) <= 4 and map_rect["left"] <= plan["left"] < choices["left"], wide
    page.locator("#route-directions summary").click()
    page.wait_for_function("""() => { const panel=document.getElementById('route-plan-panel');
      return document.getElementById('route-directions')?.open && panel.scrollHeight>panel.clientHeight; }""")
    page.evaluate("() => document.getElementById('route-plan-panel').scrollTop=40")
    plan_scroll = page.evaluate("() => document.getElementById('route-plan-panel').scrollTop")
    assert plan_scroll > 0

    # A selection made while Plan is open must update the plan in place without destroying
    # disclosure state or either independently scrolled pane.
    next_row = page.locator("#route-choices-panel .route-choice[aria-pressed='false']").first
    next_key = next_row.get_attribute("data-key")
    assert next_key and next_key != selected_key
    next_row.evaluate("el => el.click()")
    page.wait_for_function("key => document.getElementById('pincard').dataset.selectedChoiceKey===key",
                           arg=next_key)
    page.wait_for_function("key => document.getElementById('route-plan-panel')?.dataset.selectedChoiceKey===key",
                           arg=next_key)
    _wait_state(page, planOpen="true")
    assert page.locator("#route-directions").evaluate("el => el.open")
    page.wait_for_function("expected => document.getElementById('route-choices-panel').scrollTop===expected",
                           arg=choices_scroll)
    page.wait_for_function("expected => document.getElementById('route-plan-panel').scrollTop===expected",
                           arg=plan_scroll)
    selected_key = next_key

    _click(page, "#pincard .pin-head [data-route-plan-control]")
    _wait_state(page, planOpen="false")
    page.wait_for_function("expected => document.getElementById('route-choices-panel').scrollTop===expected",
                           arg=choices_scroll)
    _click(page, "#pincard .pin-head [data-route-plan-control]")
    _wait_state(page, planOpen="true")
    page.wait_for_function("expected => document.getElementById('route-plan-panel').scrollTop===expected",
                           arg=plan_scroll)

    page.set_viewport_size({"width": 900, "height": 820})
    _wait_state(page, layout="single-card", planOpen="true")
    tray = _rects(page, "#pincard", "#route-choices-panel", "#route-plan-panel")
    card, choices, plan = tray["#pincard"], tray["#route-choices-panel"], tray["#route-plan-panel"]
    assert abs(plan["top"] - choices["bottom"]) <= 2, tray
    assert plan["height"] < card["height"] and choices["height"] > plan["height"], tray
    assert _state(page)["selected"] == selected_key
    page.wait_for_function("expected => document.getElementById('route-choices-panel').scrollTop===expected",
                           arg=choices_scroll)
    page.wait_for_function("expected => document.getElementById('route-plan-panel').scrollTop===expected",
                           arg=plan_scroll)
    page.set_viewport_size({"width": 1440, "height": 820})
    _wait_state(page, layout="wide-sidecar", planOpen="true")
    assert _state(page)["selected"] == selected_key
    page.wait_for_function("expected => document.getElementById('route-choices-panel').scrollTop===expected",
                           arg=choices_scroll)
    page.wait_for_function("expected => document.getElementById('route-plan-panel').scrollTop===expected",
                           arg=plan_scroll)


def test_desktop_map_focus_settings_and_escape_restore_then_unpin(new_context):
    context = new_context(viewport={"width": 1440, "height": 820})
    page = context.new_page()
    _open_saved_inspector(page, touch=False)
    selected_key = _select_other_route(page)
    _click(page, "[data-route-plan-toggle]")
    _wait_state(page, planOpen="true")
    page.locator("#route-directions summary").click()
    page.wait_for_function("() => document.getElementById('route-directions')?.open")
    _click(page, "[data-map-focus-toggle]")
    _wait_state(page, presentation="map-focus", planOpen="true")
    assert page.locator("[data-map-focus-strip]").is_visible()
    assert not page.locator("#route-choices-panel").is_visible()
    _click(page, "[data-show-choices]")
    _wait_state(page, presentation="expanded", planOpen="true")
    assert _state(page)["selected"] == selected_key

    _click(page, "[data-settings-toggle]")
    _wait_state(page, surface="settings", planOpen="true")
    assert not page.locator("#route-choices-panel").is_visible()
    assert not page.locator("#route-plan-panel").is_visible()
    page.keyboard.press("Escape")
    _wait_state(page, surface="routes", planOpen="true")
    assert page.locator("#route-directions").evaluate("el => el.open")
    page.keyboard.press("Escape")
    _wait_state(page, planOpen="false")
    page.keyboard.press("Escape")
    page.wait_for_function("() => !document.getElementById('pincard').classList.contains('open')")


def test_mobile_sheet_is_map_first_and_preserves_plan_selection_settings_and_resize(new_context):
    context = new_context(**MOBILE_CONTEXT)
    page = context.new_page()
    _open_saved_inspector(page, touch=True)
    _wait_state(page, layout="bottom-sheet", surface="routes", planOpen="false", presentation="expanded",
                sheetContent="choices", sheetSnap="peek")
    assert page.locator("#pincard .pin-workspace").get_attribute("inert") is not None
    peek = _rects(page, "#map", "#pincard")
    assert peek["#map"]["top"] < peek["#pincard"]["top"] < peek["#map"]["bottom"], peek
    _click(page, '[data-show-choices]', touch=True)
    _wait_state(page, sheetContent="choices", sheetSnap="browse")
    assert page.locator("#route-choices-panel").is_visible()
    assert page.locator("#pincard .pin-workspace").get_attribute("inert") is None
    browse = _rects(page, "#map", "#pincard", "#route-choices-panel")
    assert browse["#map"]["top"] < browse["#pincard"]["top"] < browse["#map"]["bottom"], browse
    selected_key = _select_other_route(page, touch=True)
    _click(page, '[data-sheet-content-toggle="plan"]', touch=True)
    _wait_state(page, planOpen="true", sheetContent="plan", sheetSnap="browse")
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == selected_key
    assert not page.locator("#route-directions").evaluate("el => el.open")
    _click(page, '[data-sheet-snap-action="expanded"]', touch=True)
    _wait_state(page, sheetContent="plan", sheetSnap="expanded")
    _click(page, '[data-sheet-snap-action="browse"]', touch=True)
    _wait_state(page, sheetContent="plan", sheetSnap="browse")
    _click(page, '[data-sheet-content-toggle="choices"]', touch=True)
    _wait_state(page, planOpen="false", sheetContent="choices", sheetSnap="browse")
    selected_key = _select_other_route(page, touch=True)
    _click(page, '[data-sheet-content-toggle="plan"]', touch=True)
    _wait_state(page, planOpen="true", sheetContent="plan", sheetSnap="browse")
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == selected_key
    _click(page, '[data-sheet-snap-action="expanded"]', touch=True)
    _wait_state(page, planOpen="true", sheetContent="plan", sheetSnap="expanded")
    _click(page, "[data-settings-toggle]", touch=True)
    _wait_state(page, surface="settings", sheetSnap="browse")
    assert not page.locator("#route-plan-panel").is_visible()
    _click(page, "[data-settings-return]", touch=True)
    _wait_state(page, surface="routes", planOpen="true", sheetContent="plan", sheetSnap="expanded")
    assert _state(page)["selected"] == selected_key
    _click(page, '[data-sheet-snap-action="peek"]', touch=True)
    _wait_state(page, sheetContent="plan", sheetSnap="peek")
    page.set_viewport_size({"width": 320, "height": 568})
    _wait_state(page, layout="bottom-sheet", planOpen="true", sheetContent="plan", sheetSnap="peek")
    assert _state(page)["selected"] == selected_key
    compact = _rects(page, "[data-sheet-handle]", "#pinx", "[data-show-choices]")
    for selector, rect in compact.items():
        assert rect and rect["left"] >= 0 and rect["right"] <= 320 and rect["top"] >= 0 and rect["bottom"] <= 568, compact
        assert rect["height"] >= 43.5, compact
    _click(page, "[data-show-choices]", touch=True)
    _wait_state(page, sheetContent="choices", sheetSnap="browse")
    page.wait_for_function("""() => {
      const top=document.querySelector('[data-sheet-handle]')?.getBoundingClientRect().top;
      return Number.isFinite(top) && Math.abs(top-sheetMetrics().snaps.browse)<3;
    }""")
    plan_action = _rects(page, '[data-sheet-content-toggle="plan"]')['[data-sheet-content-toggle="plan"]']
    assert plan_action and 0 <= plan_action["left"] < plan_action["right"] <= 320
    assert 0 <= plan_action["top"] < plan_action["bottom"] <= 568 and plan_action["height"] >= 44


def test_mobile_sheet_handle_drag_snaps_without_synthetic_click_toggle(new_context):
    """A real handle drag owns sheet movement; its ensuing click cannot undo the snap."""
    context = new_context(**MOBILE_CONTEXT)
    page = context.new_page()
    _open_saved_inspector(page, touch=True)
    _click(page, "[data-show-choices]", touch=True)
    _wait_state(page, sheetSnap="browse")
    handle = page.locator("[data-sheet-handle]")
    page.wait_for_function("""() => {
      const top=document.querySelector('[data-sheet-handle]')?.getBoundingClientRect().top;
      return Number.isFinite(top) && Math.abs(top-sheetMetrics().snaps.browse)<3;
    }""")
    box = handle.bounding_box()
    assert box and box["height"] >= 44
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x, y + 260, steps=6)
    page.mouse.up()
    page.wait_for_timeout(300)
    state = _state(page)
    assert state["sheetSnap"] == "peek" and state["dragging"] == "false", {
        "state": state, "metrics": page.evaluate("() => sheetMetrics()"), "handle": box,
    }
    handle.click()
    _wait_state(page, sheetSnap="browse")


def test_mobile_speed_change_cancels_stale_pin_enrichment(new_context):
    """The unrelated request-cancellation contract survives the sheet replacement."""
    context = new_context(**MOBILE_CONTEXT)
    page = context.new_page()
    requests, errors = [], []
    page.on("request", lambda request: requests.append(request.url) if "/itinerary?" in request.url else None)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    open_destination(page, BASE_URL, SAVED_HOTSPOT["destination"], SAVED_HOTSPOT["speed"])
    point = _cell_container_point(page, SAVED_HOTSPOT["cell_id"])
    page.touchscreen.tap(point["x"], point["y"])
    page.wait_for_selector("#touchpeek.open #peekinspect", timeout=30_000)
    page.evaluate("""() => { const realFetch=window.fetch.bind(window); let held=false;
      window.__releaseOldPin=null; window.__oldPinSettled=false; window.fetch=(input, init) => {
        const url=String(input); if(!held && url.includes('/itinerary?') && url.includes('pin=1') && !url.includes('speed=')) {
          held=true; return new Promise((resolve,reject) => { window.__releaseOldPin=() => {
            const replay={...(init||{})}; delete replay.signal; realFetch(input,replay).then(resolve,reject)
              .finally(() => setTimeout(() => { window.__oldPinSettled=true; },100)); }; }); }
        return realFetch(input,init); }; }""")
    page.locator("#peekinspect").tap()
    page.wait_for_function("() => typeof window.__releaseOldPin==='function'", timeout=5_000)
    _click(page, "[data-show-choices]", touch=True)
    _wait_state(page, sheetSnap="browse")
    _click(page, "[data-settings-toggle]", touch=True)
    _wait_state(page, surface="settings")
    page.locator('#speed [data-v="fast"]').tap()
    page.wait_for_function("""() => walkspeed==='fast' && routePin!=null && BDCACHE.get(routePin)?._pin===true &&
      !document.getElementById('pincard').classList.contains('pinloading')""", timeout=45_000)
    page.evaluate("() => { window.__fastPin=BDCACHE.get(routePin); window.__releaseOldPin(); }")
    page.wait_for_function("() => window.__oldPinSettled", timeout=30_000)
    assert page.evaluate("() => walkspeed==='fast' && BDCACHE.get(routePin)===window.__fastPin && window.__fastPin._pin")
    _click(page, "[data-settings-return]", touch=True)
    _wait_state(page, surface="routes")
    page.wait_for_function("() => document.activeElement?.matches('[data-settings-toggle]')")
    assert len([url for url in requests if "pin=1" in url]) == 2, requests
    assert not errors, errors
