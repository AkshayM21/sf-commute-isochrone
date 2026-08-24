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
    _transit_legs, assert_api_matches_dom, assert_inspector_health, dom_family_snapshot,
    measure_family_card, open_destination, origin_container_point, route_metrics, scan_hotspots,
    validate_route_response, write_artifact,
)


def _enabled(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _catalog_subset():
    count = max(1, min(int(os.environ.get("ROUTE_FAMILY_HOTSPOT_DESTS", "6")), len(PUBLIC_DESTINATIONS)))
    return tuple(sorted(PUBLIC_DESTINATIONS, key=lambda item: item["slug"])[:count])


def _scan_speeds():
    speeds = tuple(value.strip() for value in os.environ.get(
        "ROUTE_FAMILY_HOTSPOT_SPEEDS", "slow,med,fast").split(",") if value.strip())
    assert speeds and set(speeds).issubset({"slow", "med", "fast"}), speeds
    return speeds


def _cell_container_point(page, cell_id):
    found = page.evaluate("""id => { const {cellLayer,map}=window.__SFCI_E2E__; let target=null; cellLayer.eachLayer(candidate => {
      if(String(candidate.feature?.properties?.id)===String(id)) target=candidate; });
      if(!target) return false; map.setView(target.getBounds().getCenter(),13,{animate:false}); return true; }""",
                          cell_id)
    assert found, f"Leaflet layer did not contain saved cell {cell_id}"
    page.wait_for_function("""id => { const {cellLayer,map}=window.__SFCI_E2E__; let target=null; cellLayer.eachLayer(candidate => {
      if(String(candidate.feature?.properties?.id)===String(id)) target=candidate; });
      if(!target || !map?._loaded) return false;
      const point=map.latLngToContainerPoint(target.getBounds().getCenter());
      const rect=document.getElementById('map').getBoundingClientRect();
      return point.x>=0 && point.y>=0 && point.x<=rect.width && point.y<=rect.height;
    }""", arg=cell_id, timeout=5_000)
    return page.evaluate("""id => { const {cellLayer,map}=window.__SFCI_E2E__; let target=null; cellLayer.eachLayer(candidate => {
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


def _wait_camera_stable(page):
    """Wait until Leaflet's center and zoom have remained unchanged for 400 ms."""
    page.evaluate("""async () => {
      const map=window.__SFCI_E2E__.map;
      const read=()=>{const c=map.getCenter();return {lat:c.lat,lng:c.lng,zoom:map.getZoom()};};
      const same=(a,b)=>Math.abs(a.lat-b.lat)<1e-9&&Math.abs(a.lng-b.lng)<1e-9&&a.zoom===b.zoom;
      let previous=read(),stableSince=performance.now();const deadline=stableSince+3500;
      while(performance.now()<deadline){
        await new Promise(resolve=>setTimeout(resolve,50));
        const current=read(),now=performance.now();
        if(!same(previous,current))stableSince=now;
        previous=current;
        if(now-stableSince>=400)return;
      }
      throw new Error('Leaflet camera did not settle');
    }""")


def _camera(page):
    _wait_camera_stable(page)
    return page.evaluate("""() => { const map=window.__SFCI_E2E__.map,c=map.getCenter();
      return {lat:c.lat,lng:c.lng,zoom:map.getZoom()}; }""")


def _assert_camera_same(page, expected):
    actual = _camera(page)
    assert actual["zoom"] == expected["zoom"], {"expected": expected, "actual": actual}
    assert abs(actual["lat"] - expected["lat"]) < 1e-7, {"expected": expected, "actual": actual}
    assert abs(actual["lng"] - expected["lng"]) < 1e-7, {"expected": expected, "actual": actual}


def _visible_plan_keys(page):
    return page.evaluate("""() => [...document.querySelectorAll('[data-route-plan-for]')]
      .filter(el => { const r=el.getBoundingClientRect(),s=getComputedStyle(el);
        return s.display!=='none'&&s.visibility!=='hidden'&&r.width>0&&r.height>0; })
      .map(el => el.dataset.routePlanFor)""")


def _plan_action(page, key):
    actions = page.locator("[data-route-plan-for]")
    keys = actions.evaluate_all("els => els.map(el => el.dataset.routePlanFor)")
    assert key in keys, {"missing_plan_key": key, "available": keys}
    return actions.nth(keys.index(key))


def _open_mobile_browse(page):
    _click(page, "[data-sheet-handle]", touch=True)
    _wait_state(page, sheetSnap="browse", sheetContent="choices")
    page.wait_for_selector("#route-choices-panel .route-choice", state="visible")


def _wait_for_final_pin(page):
    # Peek intentionally hides the choice scroller visually, so mobile only requires the final
    # route rows to be mounted. Browse/Expanded tests assert their visibility after snapping.
    page.wait_for_selector("#pincard.open #route-choices-panel .route-choice", state="attached", timeout=30_000)
    page.wait_for_function("() => { const e=window.__SFCI_E2E__; return e.routePin!=null && e.breakdownCache.get(e.routePin)?._pin===true; }", timeout=30_000)
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
    page.wait_for_function("key => window.__SFCI_E2E__?.drawn?.key===key", arg=key, timeout=5_000)
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
        _open_mobile_browse(page)
    cell_id, destination, speed = page.evaluate("() => window.__SFCI_E2E__.routePin"), SAVED_HOTSPOT["destination"], SAVED_HOTSPOT["speed"]
    body = _api_get(context.request, BASE_URL,
                    f"/itinerary?id={quote(str(cell_id))}&dlat={destination['lat']}&dlon={destination['lon']}"
                    f"{_speed_query(speed)}&pin=1")
    assert not validate_route_response(body, destination)
    _expand_all_route_disclosures(page)
    snapshot, crowding = dom_family_snapshot(page), measure_family_card(page)
    assert_api_matches_dom(body, snapshot)
    assert_inspector_health(crowding)
    assert crowding["families"] == route_metrics(body)["families"]
    assert crowding["branches"] == route_metrics(body)["branches"]
    assert crowding["route_choice_rows"] == crowding["route_options"]
    assert crowding["recommended_rows"] == 1 and len(crowding["selected_keys"]) == 1
    assert crowding["nested_interactives"] == 0 and crowding["horizontal_overflow"] <= 1
    assert not crowding["clipped_labels"], crowding
    selected_key = _select_other_route(page, touch=touch)
    assert _state(page)["selected"] == selected_key


def test_desktop_route_local_plan_is_directions_only_and_preserves_camera(new_context):
    """Only the selected desktop route owns Plan; selecting and opening it never reframes."""
    context = new_context(viewport={"width": 1440, "height": 820})
    page = context.new_page()
    _open_saved_inspector(page, touch=False)
    _wait_state(page, layout="wide-sidecar", surface="routes", presentation="expanded", planOpen="false")
    assert not page.locator("#route-plan-panel").is_visible()
    assert page.locator("[data-route-plan-control], [data-route-plan-toggle]").count() == 0
    selected_key = _state(page)["selected"]
    assert _visible_plan_keys(page) == [selected_key]
    camera = _camera(page)

    selected_key = _select_other_route(page)
    assert _visible_plan_keys(page) == [selected_key]
    _assert_camera_same(page, camera)

    _plan_action(page, selected_key).click()
    _wait_state(page, planOpen="true")
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == selected_key
    assert page.locator("#route-plan-panel").is_visible()
    wide = _rects(page, "#pincard", "#route-choices-panel", "#route-plan-panel", "#map")
    card, choices, plan, map_rect = wide["#pincard"], wide["#route-choices-panel"], wide["#route-plan-panel"], wide["#map"]
    assert plan["width"] < choices["width"] and plan["right"] <= choices["left"] + 1, wide
    assert abs(plan["bottom"] - card["bottom"]) <= 4 and map_rect["left"] <= plan["left"] < choices["left"], wide
    assert page.locator("#route-plan-panel details, #route-plan-panel .plan-facts").count() == 0
    assert page.locator("#route-plan-panel .plan-note, #route-plan-panel .plan-timing").count() == 0
    assert page.locator("#route-plan-panel .bd .foot, #route-plan-panel .foot").count() == 0
    assert page.locator("#route-plan-panel .route-directions > li").count() > 0
    assert page.locator("#route-plan-panel .plan-google").is_visible()
    _assert_camera_same(page, camera)

    _click(page, "#route-plan-panel [data-route-plan-close]")
    _wait_state(page, planOpen="false")
    assert _plan_action(page, selected_key).is_visible()
    _assert_camera_same(page, camera)


def test_narrow_desktop_plan_toggles_inline_without_replacing_choices(new_context):
    context = new_context(viewport={"width": 900, "height": 820})
    page = context.new_page()
    _open_saved_inspector(page, touch=False)
    _wait_state(page, layout="single-card", surface="routes", planOpen="false")
    assert page.locator("#route-choices-panel").is_visible()
    selected_key = _state(page)["selected"]
    assert _visible_plan_keys(page) == [selected_key]
    camera = _camera(page)

    _plan_action(page, selected_key).click()
    _wait_state(page, layout="single-card", planOpen="true")
    assert page.locator("#route-choices-panel").is_visible()
    assert page.locator("#route-plan-panel").is_visible()
    tray = _rects(page, "#pincard", "#route-choices-panel", "#route-plan-panel")
    card, choices, plan = tray["#pincard"], tray["#route-choices-panel"], tray["#route-plan-panel"]
    assert abs(plan["top"] - choices["bottom"]) <= 2, tray
    assert plan["height"] < card["height"] and choices["height"] > plan["height"], tray
    _assert_camera_same(page, camera)

    # A requested Plan survives responsive capability remapping. Reuse this loaded fixture so the
    # resize contract does not pay for another destination compute and pinned-route enrichment.
    page.set_viewport_size({"width": 390, "height": 844})
    state = _wait_state(page, layout="bottom-sheet", planOpen="true", sheetContent="plan")
    assert state["sheetSnap"] in {"browse", "expanded"}, state
    assert page.locator("#route-plan-panel").is_visible()
    assert page.locator("#route-plan-panel .plan-google").is_visible()

    page.evaluate("""() => { const pane=document.getElementById('route-plan-panel');
      pane.scrollTop=pane.scrollHeight-pane.clientHeight; }""")
    page.wait_for_function("""() => { const pane=document.getElementById('route-plan-panel');
      return pane.scrollHeight-pane.clientHeight-pane.scrollTop<=1; }""")
    flow = page.evaluate("""() => { const pane=document.getElementById('route-plan-panel'),
      last=pane.querySelector('.route-directions > li:last-child'),footer=pane.querySelector('.plan-footer');
      const p=pane.getBoundingClientRect(),l=last.getBoundingClientRect(),f=footer.getBoundingClientRect();
      return {pane:{top:p.top,bottom:p.bottom},last:{top:l.top,bottom:l.bottom},
        footer:{top:f.top,bottom:f.bottom},scroll:pane.scrollTop,max:pane.scrollHeight-pane.clientHeight}; }""")
    assert flow["last"]["top"] >= flow["pane"]["top"] - 1, flow
    assert flow["footer"]["top"] >= flow["last"]["bottom"] - 1, flow
    assert flow["footer"]["bottom"] <= flow["pane"]["bottom"] + 1, flow

    # Reuse the responsive-remap fixture for a short landscape audit. The sheet's actual box must
    # equal the controller's declared snap height; Choices must remain reachable without creating
    # document-level overflow, and keyboard snap heights must increase monotonically.
    page.set_viewport_size({"width": 568, "height": 320})
    state = _wait_state(page, layout="bottom-sheet", planOpen="true", sheetContent="plan")
    assert state["sheetSnap"] in {"browse", "expanded"}, state
    landscape_handle = page.locator("[data-sheet-handle]")
    landscape_handle.focus()
    page.keyboard.press("End")
    _wait_state(page, sheetSnap="expanded", sheetContent="plan")
    page.wait_for_function("""key => { const card=document.getElementById('pincard');
      return Math.abs(card.getBoundingClientRect().height-window.__SFCI_E2E__.sheetMetrics().visible[key])<=1; }""", arg="expanded")
    page.evaluate("""() => { const pane=document.getElementById('route-plan-panel');
      pane.scrollTop=pane.scrollHeight-pane.clientHeight; }""")
    page.wait_for_function("""() => { const pane=document.getElementById('route-plan-panel');
      return pane.scrollHeight-pane.clientHeight-pane.scrollTop<=1; }""")
    landscape_plan = page.evaluate("""() => { const card=document.getElementById('pincard'),
      pane=document.getElementById('route-plan-panel'),last=pane.querySelector('.route-directions > li:last-child'),
      footer=pane.querySelector('.plan-footer'),cr=card.getBoundingClientRect(),lr=last.getBoundingClientRect(),
      fr=footer.getBoundingClientRect(),expected=window.__SFCI_E2E__.sheetMetrics().visible[card.dataset.sheetSnap];
      return {actual:cr.height,expected,lastBottom:lr.bottom,footerTop:fr.top,footerBottom:fr.bottom,
        paneBottom:pane.getBoundingClientRect().bottom,overflow:pane.scrollWidth-pane.clientWidth,
        documentOverflow:document.documentElement.scrollWidth-innerWidth}; }""")
    assert abs(landscape_plan["actual"] - landscape_plan["expected"]) <= 1, landscape_plan
    assert landscape_plan["footerTop"] >= landscape_plan["lastBottom"] - 1, landscape_plan
    assert landscape_plan["footerBottom"] <= landscape_plan["paneBottom"] + 1, landscape_plan
    assert landscape_plan["overflow"] <= 1 and landscape_plan["documentOverflow"] <= 1, landscape_plan

    page.keyboard.press("Escape")
    _wait_state(page, planOpen="false", sheetContent="choices", sheetSnap="browse")
    handle = page.locator("[data-sheet-handle]")
    handle.focus()

    def snap_height(key, *, press=None):
        if press:
            page.keyboard.press(press)
        _wait_state(page, sheetSnap=key)
        page.wait_for_function("""key => { const card=document.getElementById('pincard');
          return Math.abs(card.getBoundingClientRect().height-window.__SFCI_E2E__.sheetMetrics().visible[key])<=1; }""", arg=key)
        measured = page.evaluate("""key => { const card=document.getElementById('pincard'),
          pane=document.getElementById('route-choices-panel'),r=card.getBoundingClientRect();
          return {actual:r.height,expected:window.__SFCI_E2E__.sheetMetrics().visible[key],bottom:r.bottom,
            overflow:pane.scrollWidth-pane.clientWidth,documentOverflow:document.documentElement.scrollWidth-innerWidth}; }""", key)
        assert abs(measured["actual"] - measured["expected"]) <= 1, measured
        assert abs(measured["bottom"] - 320) <= 1, measured
        assert measured["overflow"] <= 1 and measured["documentOverflow"] <= 1, measured
        return measured["actual"]

    browse_height = snap_height("browse")
    expanded_height = snap_height("expanded", press="End")
    browse_again = snap_height("browse", press="ArrowDown")
    peek_height = snap_height("peek", press="Home")
    assert abs(browse_again - browse_height) <= 1
    assert peek_height < browse_height < expanded_height

    page.keyboard.press("End")
    _wait_state(page, sheetSnap="expanded")
    page.evaluate("""() => { const pane=document.getElementById('route-choices-panel');
      pane.scrollTop=pane.scrollHeight-pane.clientHeight; }""")
    page.wait_for_function("""() => { const pane=document.getElementById('route-choices-panel');
      return pane.scrollHeight-pane.clientHeight-pane.scrollTop<=1; }""")
    page.evaluate("""() => { const pane=document.getElementById('route-choices-panel'),
      visible=el=>el.getClientRects().length&&getComputedStyle(el).display!=='none',
      actions=[...pane.querySelectorAll('button,summary,a[href]')].filter(visible),
      p=pane.getBoundingClientRect(),last=actions.at(-1).getBoundingClientRect();
      pane.scrollTop=Math.max(0,pane.scrollTop+(last.bottom-p.bottom)); }""")
    page.wait_for_function("""() => { const pane=document.getElementById('route-choices-panel'),
      visible=el=>el.getClientRects().length&&getComputedStyle(el).display!=='none',
      actions=[...pane.querySelectorAll('button,summary,a[href]')].filter(visible),
      p=pane.getBoundingClientRect(),last=actions.at(-1).getBoundingClientRect();
      return last.top>=p.top-1&&last.bottom<=p.bottom+1; }""")
    choices_end = page.evaluate("""() => { const pane=document.getElementById('route-choices-panel'),
      visible=el=>el.getClientRects().length&&getComputedStyle(el).display!=='none',
      actions=[...pane.querySelectorAll('button,summary,a[href]')].filter(visible),
      p=pane.getBoundingClientRect(),last=actions.at(-1).getBoundingClientRect();
      return {paneTop:p.top,paneBottom:p.bottom,lastTop:last.top,lastBottom:last.bottom}; }""")
    assert choices_end["lastTop"] >= choices_end["paneTop"] - 1, choices_end
    assert choices_end["lastBottom"] <= choices_end["paneBottom"] + 1, choices_end


def test_desktop_map_focus_settings_and_escape_restore_then_unpin(new_context):
    context = new_context(viewport={"width": 1440, "height": 820})
    page = context.new_page()
    _open_saved_inspector(page, touch=False)
    selected_key = _select_other_route(page)
    _plan_action(page, selected_key).click()
    _wait_state(page, planOpen="true")

    # Returning before the 260 ms Focus-map fit delay must cancel that pending fit. Focus itself may
    # reframe; the assertion begins only after Choices has already been restored.
    rapid = page.evaluate("""() => { const started=performance.now();
      document.querySelector('[data-map-focus-toggle]').click();
      document.querySelector('[data-show-choices]').click();
      const map=window.__SFCI_E2E__.map,c=map.getCenter();return {elapsed:performance.now()-started,lat:c.lat,lng:c.lng,zoom:map.getZoom()}; }""")
    assert rapid["elapsed"] < 260, rapid
    _wait_state(page, presentation="expanded", planOpen="true")
    delayed = page.evaluate("""async () => { await new Promise(resolve=>setTimeout(resolve,420));
      const map=window.__SFCI_E2E__.map,c=map.getCenter();return {lat:c.lat,lng:c.lng,zoom:map.getZoom()}; }""")
    assert delayed["zoom"] == rapid["zoom"], {"after_exit": rapid, "delayed": delayed}
    assert abs(delayed["lat"] - rapid["lat"]) < 1e-7, {"after_exit": rapid, "delayed": delayed}
    assert abs(delayed["lng"] - rapid["lng"]) < 1e-7, {"after_exit": rapid, "delayed": delayed}

    _click(page, "[data-map-focus-toggle]")
    _wait_state(page, presentation="map-focus", planOpen="true")
    assert page.locator("[data-map-focus-strip]").is_visible()
    assert not page.locator("#route-choices-panel").is_visible()
    assert page.locator("#pincard #pinx").is_visible()
    assert page.locator("#pincard [data-close-pin]").count() == 0
    assert not page.evaluate("""() => [...document.querySelectorAll('#pincard button')].some(button => {
      const r=button.getBoundingClientRect(),s=getComputedStyle(button);
      return s.display!=='none'&&r.width>0&&r.height>0&&button.textContent.trim().includes('Close'); })""")
    close_rect = _rects(page, "#pinx", "[data-map-focus-strip]")
    assert close_rect["#pinx"]["left"] >= close_rect["[data-map-focus-strip]"]["left"]
    assert close_rect["#pinx"]["right"] <= close_rect["[data-map-focus-strip]"]["right"]
    assert not page.evaluate("""() => { const close=document.getElementById('pinx').getBoundingClientRect();
      return [...document.querySelectorAll('[data-map-focus-strip] button')].some(button => {
        const r=button.getBoundingClientRect();return r.width>0&&r.height>0&&
          Math.max(0,Math.min(close.right,r.right)-Math.max(close.left,r.left))*
          Math.max(0,Math.min(close.bottom,r.bottom)-Math.max(close.top,r.top))>1; }); }""")
    page.locator("[data-show-choices]").first.click()
    _wait_state(page, presentation="expanded", planOpen="true")
    assert _state(page)["selected"] == selected_key

    _click(page, "[data-settings-toggle]")
    _wait_state(page, surface="settings", planOpen="true")
    assert not page.locator("#route-choices-panel").is_visible()
    assert not page.locator("#route-plan-panel").is_visible()
    page.keyboard.press("Escape")
    _wait_state(page, surface="routes", planOpen="true")
    assert page.locator("#route-plan-panel .route-directions").is_visible()
    page.keyboard.press("Escape")
    _wait_state(page, planOpen="false")
    page.keyboard.press("Escape")
    page.wait_for_function("() => !document.getElementById('pincard').classList.contains('open')")


def test_desktop_settings_return_stays_visible_and_restores_camera_state_and_focus(new_context):
    context = new_context(viewport={"width": 1440, "height": 820})
    page = context.new_page()
    _open_saved_inspector(page, touch=False)
    selected_key = _select_other_route(page)
    camera = _camera(page)
    _click(page, "[data-settings-toggle]")
    _wait_state(page, surface="settings")
    page.evaluate("() => { const panel=document.getElementById('panel');panel.scrollTop=panel.scrollHeight; }")
    page.wait_for_function("() => document.getElementById('panel').scrollTop>0")
    back = page.locator("[data-settings-return]")
    assert back.is_visible()
    back_rect = back.bounding_box()
    assert back_rect and 0 <= back_rect["y"] < back_rect["y"] + back_rect["height"] <= 820

    back.click()
    _wait_state(page, surface="routes", planOpen="false")
    assert _state(page)["selected"] == selected_key
    page.wait_for_function("() => document.activeElement?.matches('[data-settings-toggle]')")
    _assert_camera_same(page, camera)


@pytest.mark.parametrize("viewport", [
    pytest.param({"width": 390, "height": 844}, id="iphone"),
    pytest.param({"width": 320, "height": 568}, id="compact"),
])
def test_mobile_route_local_plan_sheet_states_camera_and_bottom_reachability(new_context, viewport):
    context_args = {**MOBILE_CONTEXT, "viewport": viewport}
    context = new_context(**context_args)
    page = context.new_page()
    _open_saved_inspector(page, touch=True)
    _wait_state(page, layout="bottom-sheet", surface="routes", planOpen="false", presentation="expanded",
                sheetContent="choices", sheetSnap="peek")
    assert page.locator("[data-route-plan-control], [data-route-plan-toggle]").count() == 0
    assert page.locator(".pin-view, .pin-peek-actions, [data-sheet-content-toggle], [data-sheet-snap-action]").count() == 0
    assert page.locator("#pincard .pin-workspace").get_attribute("inert") is not None
    peek = _rects(page, "#map", "#pincard")
    assert peek["#map"]["top"] < peek["#pincard"]["top"] < peek["#map"]["bottom"], peek
    camera = _camera(page)

    _open_mobile_browse(page)
    assert page.locator("#route-choices-panel").is_visible()
    assert page.locator("#pincard .pin-workspace").get_attribute("inert") is None
    browse = _rects(page, "#map", "#pincard", "#route-choices-panel")
    assert browse["#map"]["top"] < browse["#pincard"]["top"] < browse["#map"]["bottom"], browse
    assert page.evaluate("""() => [...document.querySelectorAll('#route-choices-panel .route-choice-card')]
      .filter(card => getComputedStyle(card).display!=='none' && card.getClientRects().length)
      .every(card => { const action=card.querySelector('[data-route-plan-for]');
        return action && getComputedStyle(action).display!=='none' && action.getBoundingClientRect().height>=44; })""")

    target = page.locator("#route-choices-panel .route-choice[aria-pressed='false']").first
    target_key = target.get_attribute("data-key")
    assert target_key
    page.evaluate("""key => { const pane=document.getElementById('route-choices-panel');
      const card=document.querySelector(`[data-choice-card-key="${CSS.escape(key)}"]`);
      pane.scrollTop=Math.max(0,card.offsetTop-pane.offsetTop); }""", target_key)
    plan_action = _plan_action(page, target_key)
    if viewport["width"] == 390:
        plan_action.focus()
        page.keyboard.press("Enter")
    else:
        plan_action.tap()
    _wait_state(page, planOpen="true", sheetContent="plan", sheetSnap="browse")
    page.wait_for_function("key => document.getElementById('pincard').dataset.selectedChoiceKey===key && window.__SFCI_E2E__?.drawn?.key===key",
                           arg=target_key)
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == target_key
    assert page.locator("#route-plan-panel details").count() == 0
    assert page.locator("#route-plan-panel .route-directions > li").count() > 0
    _assert_camera_same(page, camera)

    if viewport["width"] == 390:
        page.wait_for_function("""() => { const active=document.activeElement,
          plan=document.getElementById('route-plan-panel'),r=active?.getBoundingClientRect();
          return !!active&&plan?.contains(active)&&r.width>0&&r.height>0&&
            getComputedStyle(active).visibility!=='hidden'&&getComputedStyle(active).display!=='none'; }""")
        focused = page.evaluate("""() => { const active=document.activeElement,
          plan=document.getElementById('route-plan-panel'),r=active?.getBoundingClientRect();
          return {inside:!!active&&plan.contains(active),visible:!!r&&r.width>0&&r.height>0&&
            getComputedStyle(active).visibility!=='hidden'&&getComputedStyle(active).display!=='none',
            label:active?.getAttribute('aria-label')||active?.textContent?.trim()||active?.tagName}; }""")
        assert focused["inside"] and focused["visible"], focused
        page.keyboard.press("Escape")
        _wait_state(page, planOpen="false", sheetContent="choices", sheetSnap="browse")
        returned = _plan_action(page, target_key)
        page.wait_for_function("""key => document.activeElement?.dataset.routePlanFor===key""", arg=target_key)
        assert returned.is_visible()
        assert returned.evaluate("el => document.activeElement===el")
        page.keyboard.press("Enter")
        _wait_state(page, planOpen="true", sheetContent="plan", sheetSnap="browse")

    # Reach the real scroll boundary; do not use scroll_into_view, which could hide broken sheet
    # geometry by moving an element independently of the pane's usable viewport.
    page.evaluate("""() => { const pane=document.getElementById('route-plan-panel');
      pane.scrollTop=pane.scrollHeight-pane.clientHeight; }""")
    page.wait_for_function("""() => { const pane=document.getElementById('route-plan-panel');
      return pane.scrollHeight-pane.clientHeight-pane.scrollTop<=1; }""")
    reach = page.evaluate("""() => { const pane=document.getElementById('route-plan-panel'),
      last=pane.querySelector('.route-directions > li:last-child'),link=pane.querySelector('.plan-google');
      const p=pane.getBoundingClientRect(),l=last.getBoundingClientRect(),g=link.getBoundingClientRect();
      return {pane:{top:p.top,bottom:p.bottom},last:{top:l.top,bottom:l.bottom},link:{top:g.top,bottom:g.bottom},
        max:pane.scrollHeight-pane.clientHeight,scroll:pane.scrollTop}; }""")
    assert reach["last"]["top"] >= reach["pane"]["top"] - 1, reach
    assert reach["last"]["bottom"] <= reach["link"]["top"] + 1, reach
    assert reach["link"]["bottom"] <= reach["pane"]["bottom"] + 1, reach

    _click(page, "#route-plan-panel [data-route-plan-close]", touch=True)
    _wait_state(page, planOpen="false", sheetContent="choices", sheetSnap="browse")
    handle = page.locator("[data-sheet-handle]")
    handle.focus()
    page.keyboard.press("End")
    _wait_state(page, sheetSnap="expanded")
    page.keyboard.press("Home")
    _wait_state(page, sheetSnap="peek")
    page.keyboard.press("ArrowUp")
    _wait_state(page, sheetSnap="browse")
    page.keyboard.press("ArrowUp")
    _wait_state(page, sheetSnap="expanded")
    page.keyboard.press("ArrowDown")
    _wait_state(page, sheetSnap="browse")
    _assert_camera_same(page, camera)


def test_mobile_sheet_handle_drag_snaps_without_synthetic_click_toggle(new_context):
    """A real handle drag owns sheet movement; its ensuing click cannot undo the snap."""
    context = new_context(**MOBILE_CONTEXT)
    page = context.new_page()
    _open_saved_inspector(page, touch=True)
    _open_mobile_browse(page)
    handle = page.locator("[data-sheet-handle]")
    page.wait_for_function("""() => {
      const top=document.querySelector('[data-sheet-handle]')?.getBoundingClientRect().top;
      return Number.isFinite(top) && Math.abs(top-window.__SFCI_E2E__.sheetMetrics().snaps.browse)<3;
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
        "state": state, "metrics": page.evaluate("() => window.__SFCI_E2E__.sheetMetrics()"), "handle": box,
    }
    handle.click()
    _wait_state(page, sheetSnap="browse")


def test_mobile_sheet_drag_preserves_choices_scroll_context_after_route_selection(new_context):
    """Resizing the sheet must not pull a browsing user back to their selected route."""
    context = new_context(**MOBILE_CONTEXT)
    page = context.new_page()
    _open_saved_inspector(page, touch=True)
    _open_mobile_browse(page)
    camera = _camera(page)
    selected_key = _select_other_route(page, touch=True)
    _expand_all_route_disclosures(page)

    # Pick a distinct, lower card and place it at a repeatable position within the Choices
    # viewport. This models selecting one route, then continuing to browse other alternatives.
    context_key = page.evaluate("""selected => {
      const pane=document.getElementById('route-choices-panel');
      const cards=[...pane.querySelectorAll('.route-choice-card')]
        .filter(card => card.dataset.choiceCardKey!==selected)
        .sort((a,b) => a.offsetTop-b.offsetTop);
      const selectedCard=pane.querySelector(`[data-choice-card-key="${CSS.escape(selected)}"]`);
      const lower=cards.find(card => selectedCard && card.offsetTop-selectedCard.offsetTop>Math.max(96,pane.clientHeight*.35)) || cards.at(-1);
      if(!lower)return null;
      const paneRect=pane.getBoundingClientRect(),cardRect=lower.getBoundingClientRect();
      const max=Math.max(0,pane.scrollHeight-pane.clientHeight);
      pane.scrollTop=Math.max(0,Math.min(max,cardRect.top-paneRect.top+pane.scrollTop-pane.clientHeight*.38));
      return lower.dataset.choiceCardKey;
    }""", selected_key)
    assert context_key and context_key != selected_key

    def choices_context(key):
        return page.evaluate("""key => {
          const pane=document.getElementById('route-choices-panel');
          const card=pane.querySelector(`[data-choice-card-key="${CSS.escape(key)}"]`);
          const selected=pane.querySelector(`[data-choice-card-key="${CSS.escape(document.getElementById('pincard').dataset.selectedChoiceKey)}"]`);
          const p=pane.getBoundingClientRect(),r=card?.getBoundingClientRect(),s=selected?.getBoundingClientRect();
          return {key,selectedKey:document.getElementById('pincard').dataset.selectedChoiceKey,
            scroll:pane.scrollTop,max:pane.scrollHeight-pane.clientHeight,
            anchor:r&&r.top-p.top,visible:!!r&&r.bottom>p.top+2&&r.top<p.bottom-2,
            selectedVisible:!!s&&s.bottom>p.top+2&&s.top<p.bottom-2};
        }""", key)

    page.wait_for_function("""key => { const pane=document.getElementById('route-choices-panel'),
      card=pane.querySelector(`[data-choice-card-key="${CSS.escape(key)}"]`),p=pane.getBoundingClientRect(),r=card?.getBoundingClientRect();
      return pane.scrollTop>80&&!!r&&r.bottom>p.top+2&&r.top<p.bottom-2; }""", arg=context_key)
    before_up = choices_context(context_key)
    assert before_up["max"] > 120 and before_up["visible"], before_up

    def drag_handle(delta_y, expected_snap):
        handle = page.locator("[data-sheet-handle]")
        box = handle.bounding_box()
        assert box and box["height"] >= 44
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        # Space the moves far enough apart to exercise distance-based snapping, not the
        # controller's intentionally separate fast-flick-to-Peek behavior.
        for step in range(1, 9):
            page.mouse.move(x, y + delta_y * step / 8)
            page.wait_for_timeout(90)
        page.mouse.up()
        page.wait_for_timeout(300)
        _wait_state(page, sheetSnap=expected_snap, sheetContent="choices", planOpen="false")

    drag_handle(-260, "expanded")
    after_up = choices_context(context_key)
    assert after_up["selectedKey"] == selected_key and after_up["visible"], {"before": before_up, "after": after_up}
    assert abs(after_up["scroll"] - before_up["scroll"]) <= 2, {"before": before_up, "after": after_up}
    assert abs(after_up["anchor"] - before_up["anchor"]) <= 4, {"before": before_up, "after": after_up}

    # Repeat from the other direction. Re-center the same non-selected card where Browse can
    # still display it after collapse so a clamped bottom boundary cannot mask a scroll reset.
    page.evaluate("""key => { const pane=document.getElementById('route-choices-panel'),
      card=pane.querySelector(`[data-choice-card-key="${CSS.escape(key)}"]`),p=pane.getBoundingClientRect(),r=card.getBoundingClientRect(),
      max=Math.max(0,pane.scrollHeight-pane.clientHeight); pane.scrollTop=Math.max(0,Math.min(max,r.top-p.top+pane.scrollTop-150)); }""", context_key)
    page.wait_for_function("""key => { const pane=document.getElementById('route-choices-panel'),
      card=pane.querySelector(`[data-choice-card-key="${CSS.escape(key)}"]`),p=pane.getBoundingClientRect(),r=card?.getBoundingClientRect();
      return pane.scrollTop>80&&!!r&&r.bottom>p.top+2&&r.top<p.bottom-2; }""", arg=context_key)
    before_down = choices_context(context_key)
    drag_handle(260, "browse")
    after_down = choices_context(context_key)
    assert after_down["selectedKey"] == selected_key and after_down["visible"], {"before": before_down, "after": after_down}
    assert abs(after_down["scroll"] - before_down["scroll"]) <= 2, {"before": before_down, "after": after_down}
    assert abs(after_down["anchor"] - before_down["anchor"]) <= 4, {"before": before_down, "after": after_down}
    _assert_camera_same(page, camera)


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
    _open_mobile_browse(page)
    _click(page, "[data-settings-toggle]", touch=True)
    _wait_state(page, surface="settings")
    page.locator('#speed [data-v="fast"]').tap()
    page.wait_for_function("""() => { const e=window.__SFCI_E2E__; return e.walkSpeed==='fast' && e.routePin!=null && e.breakdownCache.get(e.routePin)?._pin===true &&
      !document.getElementById('pincard').classList.contains('pinloading'); }""", timeout=45_000)
    page.evaluate("() => { const e=window.__SFCI_E2E__; window.__fastPin=e.breakdownCache.get(e.routePin); window.__releaseOldPin(); }")
    page.wait_for_function("() => window.__oldPinSettled", timeout=30_000)
    assert page.evaluate("() => { const e=window.__SFCI_E2E__; return e.walkSpeed==='fast' && e.breakdownCache.get(e.routePin)===window.__fastPin && window.__fastPin._pin; }")
    _click(page, "[data-settings-return]", touch=True)
    _wait_state(page, surface="routes")
    page.wait_for_function("() => document.activeElement?.matches('[data-settings-toggle]')")
    assert len([url for url in requests if "pin=1" in url]) == 2, requests
    assert not errors, errors
