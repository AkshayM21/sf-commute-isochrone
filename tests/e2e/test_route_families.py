"""Bounded live-server tests for route-family discovery and the recommendation-first inspector.

The default run scans one public destination and replays one saved hotspot at desktop, tight
desktop, and mobile sizes.  Set ROUTE_FAMILY_HOTSPOT_SCAN=1 for the broader seeded SF catalog.
"""

import os
from urllib.parse import quote

import pytest

from conftest import BASE_URL, shot
from route_family_hotspots import (
    ARTIFACT_PATH,
    DEFAULT_SEED,
    PARETO_DIMENSIONS,
    PUBLIC_DESTINATIONS,
    SAVED_HOTSPOT,
    _api_get,
    _speed_query,
    _stable_rank,
    _route_slots,
    _rank_hotspots,
    _transit_legs,
    assert_api_matches_dom,
    dom_family_snapshot,
    measure_family_card,
    open_destination,
    origin_container_point,
    route_metrics,
    scan_hotspots,
    validate_route_response,
    write_artifact,
)


def _enabled(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _catalog_subset():
    requested = int(os.environ.get("ROUTE_FAMILY_HOTSPOT_DESTS", "6"))
    requested = max(1, min(requested, len(PUBLIC_DESTINATIONS)))
    ordered = sorted(PUBLIC_DESTINATIONS,
                     key=lambda item: _stable_rank(DEFAULT_SEED, "destination", item["slug"]))
    return tuple(ordered[:requested])


def _scan_speeds():
    raw = os.environ.get("ROUTE_FAMILY_HOTSPOT_SPEEDS", "slow,med,fast")
    speeds = tuple(value.strip() for value in raw.split(",") if value.strip())
    assert speeds and set(speeds).issubset({"slow", "med", "fast"}), speeds
    return speeds


def _cell_container_point(page, cell_id):
    """Center an exact Leaflet grid cell and return a real-screen tap point inside it."""
    found = page.evaluate(
        """id => {
          let target=null;
          layer.eachLayer(candidate=>{
            if(String(candidate.feature?.properties?.id)===String(id))target=candidate;
          });
          if(!target)return false;
          map.setView(target.getBounds().getCenter(),13,{animate:false});
          return true;
        }""",
        cell_id,
    )
    assert found, f"Leaflet layer did not contain saved cell {cell_id}"
    page.wait_for_timeout(450)
    return page.evaluate(
        """id => {
          let target=null;
          layer.eachLayer(candidate=>{
            if(String(candidate.feature?.properties?.id)===String(id))target=candidate;
          });
          const point=map.latLngToContainerPoint(target.getBounds().getCenter());
          const rect=document.getElementById("map").getBoundingClientRect();
          return {x:rect.left+point.x,y:rect.top+point.y};
        }""",
        cell_id,
    )


def _expand_all_route_disclosures(page):
    """Materialize the one flat Additional disclosure before exhaustive DOM/API checks."""
    toggle = page.locator("#all-routes-toggle")
    if (not page.locator("#allroutes").count() or not toggle.count()
            or not toggle.is_visible()):
        return
    if not page.locator("#allroutes").evaluate("el => el.open"):
        toggle.click()
    page.wait_for_function(
        "() => document.getElementById('allroutes')?.open"
    )
    page.wait_for_function(
        "() => document.querySelectorAll('#all-routes-panel .route-choice').length > 0"
    )


def _open_route_plan(page, *, touch=False):
    """Open the selected route plan through the surface appropriate to this viewport."""
    plan = page.locator("#route-plan-panel")
    if plan.is_visible():
        return
    cta = page.locator("#view-route-plan")
    control = cta if cta.is_visible() else page.locator('#pinview [data-view="plan"]')
    (control.tap if touch else control.click)()
    page.wait_for_function(
        "() => document.getElementById('route-plan-panel') && "
        "getComputedStyle(document.getElementById('route-plan-panel')).display !== 'none'"
    )


def _open_route_choices(page, *, touch=False):
    """Return a narrow/mobile inspector from detail or map to the choice surface."""
    choices = page.locator("#route-choices-panel")
    if choices.is_visible():
        return
    control = page.locator('#pinview [data-view="choices"]')
    (control.tap if touch else control.click)()
    page.wait_for_function(
        "() => !document.body.classList.contains('pin-map-view') && "
        "!document.body.classList.contains('pin-plan-view') && "
        "getComputedStyle(document.getElementById('route-choices-panel')).display !== 'none'"
    )


def _scan_with_page(new_context, destinations, speeds, per_config):
    context = new_context(viewport={"width": 900, "height": 700})
    page = context.new_page()
    artifact = scan_hotspots(
        context.request, BASE_URL, destinations=destinations, speeds=speeds,
        seed=DEFAULT_SEED, per_config=per_config,
    )
    path = write_artifact(artifact)
    errors = [
        f"{row['destination']['slug']}/{row['speed']}/{row['cell_id']}: {error}"
        for row in artifact["hotspots"] for error in row["errors"]
    ]
    assert artifact["hotspots"], f"hotspot scan produced no candidates; artifact={path}"
    assert not errors, f"route-family API false advertising; artifact={path}:\n" + "\n".join(errors)
    return artifact, path


def test_hotspot_ranking_retains_pareto_extremes_before_scalar_fill():
    """A rare label/geometry extreme survives even when its scalar score is lower."""
    def hotspot(cell_id, score, **dimensions):
        metrics = {dimension: 0 for dimension in PARETO_DIMENSIONS}
        metrics.update(dimensions)
        metrics["score"] = score
        return {
            "destination": {"slug": "fixture"},
            "speed": "med",
            "cell_id": cell_id,
            "metrics": metrics,
            "errors": [],
        }

    route_rich = hotspot(
        "route-rich", 50, routes=8, families=4, branches=7,
        max_branches_per_family=3, max_options_per_family=4, max_transfers=2,
        unique_lines=6, unique_services=7, label_chars=40, longest_label=20,
        time_spread=8, geometry_points=100, service_branch_cross_product=20,
    )
    label_geometry_extreme = hotspot(
        "label-geometry", 15, routes=2, families=1, branches=1,
        max_branches_per_family=1, max_options_per_family=2, unique_lines=2,
        unique_services=2, label_chars=300, longest_label=180, time_spread=20,
        geometry_points=500, service_branch_cross_product=2,
    )
    route_dominated = hotspot(
        "route-dominated", 35, routes=6, families=3, branches=5,
        max_branches_per_family=2, max_options_per_family=3, max_transfers=1,
        unique_lines=5, unique_services=6, label_chars=30, longest_label=15,
        time_spread=7, geometry_points=90, service_branch_cross_product=15,
    )
    label_dominated = hotspot(
        "label-dominated", 10, routes=1, families=1, branches=1,
        max_branches_per_family=1, max_options_per_family=1, unique_lines=1,
        unique_services=1, label_chars=200, longest_label=100, time_spread=10,
        geometry_points=300, service_branch_cross_product=1,
    )
    shuffled = [route_dominated, label_dominated, label_geometry_extreme, route_rich]

    ranked = _rank_hotspots(shuffled, seed=101, limit=3)
    assert [row["cell_id"] for row in ranked] == [
        "route-rich", "label-geometry", "route-dominated",
    ]
    assert [row["pareto_frontier"] for row in ranked] == [True, True, False]
    assert tuple(ranked[0]["pareto_vector"]) == PARETO_DIMENSIONS

    # Neither input dictionary/list order nor a tight replay budget may discard a lower-scoring
    # frontier extreme in favor of a scalar-strong but dominated case.
    reranked = _rank_hotspots(list(reversed(shuffled)), seed=101, limit=3)
    assert [row["cell_id"] for row in reranked] == [row["cell_id"] for row in ranked]
    bounded = _rank_hotspots(shuffled, seed=101, limit=2)
    assert [row["cell_id"] for row in bounded] == ["route-rich", "label-geometry"]


def test_route_family_hotspot_sampler_smoke(new_context):
    """One bounded compute -> variance -> pin scan always runs and emits the JSON artifact."""
    artifact, path = _scan_with_page(
        new_context, destinations=(SAVED_HOTSPOT["destination"],), speeds=("med",), per_config=4,
    )
    assert path == ARTIFACT_PATH
    assert artifact["health"]["engine"] == "raptor"
    assert all(row["metrics"]["families"] >= 1 for row in artifact["hotspots"])
    assert all(row["metrics"]["branches"] >= row["metrics"]["families"]
               for row in artifact["hotspots"])


def test_saved_hotspot_walk_speed_is_monotone(new_context):
    """The same public OD must not get slower when the configured walking pace gets faster."""
    context = new_context(viewport={"width": 800, "height": 600})
    destination = SAVED_HOTSPOT["destination"]
    origin = SAVED_HOTSPOT["origin"]
    totals = {}
    for speed in ("slow", "med", "fast"):
        body = _api_get(
            context.request, BASE_URL,
            f"/itinerary?olat={origin['lat']}&olon={origin['lon']}"
            f"&dlat={destination['lat']}&dlon={destination['lon']}"
            f"{_speed_query(speed)}&pin=1",
        )
        errors = validate_route_response(body, destination)
        assert not errors, f"{speed}: {errors}"
        totals[speed] = int((body.get("typical") or {}).get("total", body["total"]))
    assert totals["slow"] >= totals["med"] >= totals["fast"], totals


def test_saved_medium_shared_corridor_retains_transfer_finish(new_context):
    """The saved Dolores-area OD keeps its structural shared-corridor transfer alternative.

    This intentionally asserts no concrete route names.  The contract is that a boarding family
    advertises more than one service discovered on its corridor and that at least one branch in
    that same family finishes with another transit ride.  It need not be the primary family: the
    regression was a useful alternate corridor disappearing because branch closure ran too early.
    """
    context = new_context(viewport={"width": 800, "height": 600})
    destination = SAVED_HOTSPOT["destination"]
    body = _api_get(
        context.request, BASE_URL,
        f"/itinerary?id={quote(SAVED_HOTSPOT['cell_id'])}"
        f"&dlat={destination['lat']}&dlon={destination['lon']}&speed=med&pin=1",
    )
    errors = validate_route_response(body, destination)
    assert not errors, errors

    options = [body, *(body.get("alts") or [])]
    by_family = {}
    for route in options:
        family = route.get("family") or {}
        by_family.setdefault(str(family.get("key") or ""),
                             {"meta": family, "routes": []})["routes"].append(route)
    qualifying = []
    for value in by_family.values():
        services = value["meta"].get("services") or []
        unique_services = {str(service.get("key") or "") for service in services}
        transfer_finishes = [
            route for route in value["routes"]
            if (route.get("branch") or {}).get("kind") == "transit"
            and len(_transit_legs(_route_slots(route)[1])) >= 2
        ]
        if len(services) >= 2 and len(unique_services) == len(services) and transfer_finishes:
            qualifying.append(value)
    assert qualifying, (
        "saved medium-speed OD lost its generic shared-corridor transfer-finish alternative")


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
    """A fixed set of historically dense cells must be monotone across all three walk speeds."""
    context = new_context(viewport={"width": 800, "height": 600})
    failures = []
    for label, cell_id, destination in WALK_SPEED_REGRESSION_CASES:
        totals = {}
        for speed in ("slow", "med", "fast"):
            body = _api_get(
                context.request, BASE_URL,
                f"/itinerary?id={quote(cell_id)}&dlat={destination['lat']}"
                f"&dlon={destination['lon']}{_speed_query(speed)}&pin=1",
            )
            errors = validate_route_response(body, destination)
            assert not errors, f"{label}/{speed}: {errors}"
            totals[speed] = int((body.get("typical") or {}).get("total", body["total"]))
        if not totals["slow"] >= totals["med"] >= totals["fast"]:
            failures.append(f"{label}: {totals}")
    assert not failures, "walk-speed monotonicity failures:\n" + "\n".join(failures)


@pytest.mark.skipif(not _enabled("ROUTE_FAMILY_HOTSPOT_SCAN"),
                    reason="set ROUTE_FAMILY_HOTSPOT_SCAN=1 for the broader seeded SF scan")
def test_route_family_hotspot_sampler_broad(new_context):
    """Opt-in citywide scan, kept below live compute/variance/itinerary rate limits by default."""
    per_config = int(os.environ.get("ROUTE_FAMILY_HOTSPOT_PER_CONFIG", "5"))
    per_config = max(1, min(per_config, 5))
    destinations, speeds = _catalog_subset(), _scan_speeds()
    assert len(destinations) * len(speeds) <= 18, (
        "requested scan exceeds the bounded 20/min variance budget; lower "
        "ROUTE_FAMILY_HOTSPOT_DESTS or ROUTE_FAMILY_HOTSPOT_SPEEDS")
    assert len(destinations) * len(speeds) * per_config <= 90, (
        "requested scan exceeds the bounded pinned-itinerary budget")
    artifact, path = _scan_with_page(
        new_context, destinations=destinations, speeds=speeds, per_config=per_config,
    )
    assert len(artifact["configs"]) <= 18, (
        f"scan exceeded the bounded 20/min variance budget; artifact={path}")
    assert len(artifact["hotspots"]) <= 90, (
        f"scan exceeds the bounded itinerary budget; artifact={path}")


VIEWPORTS = (
    pytest.param({"viewport": {"width": 1280, "height": 800}}, False, id="desktop"),
    pytest.param({"viewport": {"width": 900, "height": 700}}, False, id="tight-desktop"),
    pytest.param({
        "viewport": {"width": 390, "height": 844},
        "has_touch": True,
        "is_mobile": True,
        "device_scale_factor": 3,
        "user_agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
        ),
    }, True, id="mobile"),
)


@pytest.mark.parametrize("context_args,touch", VIEWPORTS)
def test_saved_hotspot_family_card_matches_api_and_fits(new_context, context_args, touch):
    """Replay a public hotspot and audit the recommendation-first route inspector."""
    context = new_context(**context_args)
    page = context.new_page()
    destination = SAVED_HOTSPOT["destination"]
    origin = SAVED_HOTSPOT["origin"]
    speed = SAVED_HOTSPOT["speed"]
    open_destination(page, BASE_URL, destination, speed)
    point = origin_container_point(page, origin)

    # Desktop first checks that the hover card advertises exactly the alternatives available to a
    # non-pinned lookup. Pinning may add planned branch alternatives, so that later card is checked
    # against its own pin=1 response instead of assuming both sets are identical.
    if not touch:
        page.mouse.move(point["x"] + 1, point["y"] + 1)
        page.wait_for_selector(".leaflet-tooltip.tt .bd", timeout=30_000)
        hover_id = page.evaluate("() => hoverCell")
        hover_body = _api_get(
            context.request, BASE_URL,
            f"/itinerary?id={quote(str(hover_id))}&dlat={destination['lat']}&dlon={destination['lon']}"
            f"{_speed_query(speed)}",
        )
        hover_errors = validate_route_response(hover_body, destination)
        assert not hover_errors, hover_errors
        chips = page.locator(".leaflet-tooltip.tt .altfoot .altchip")
        assert chips.count() == len(hover_body.get("alts") or []), (
            "hover card advertised a different alternative count than its API response")
        page.mouse.click(point["x"] + 1, point["y"] + 1)
    else:
        page.touchscreen.tap(point["x"] + 1, point["y"] + 1)

        # Touch has a deliberate two-stage interaction: the first tap is a persistent preview,
        # and only its explicit action opens the full-screen route inspector.
        page.wait_for_selector("#touchpeek.open #peekinspect", timeout=30_000)
        assert page.locator("#touchpeek").get_attribute("aria-hidden") == "false"
        assert not page.locator("#pincard").evaluate("el => el.classList.contains('open')")
        peek_sizes = page.locator("#touchpeek button").evaluate_all(
            "els => els.map(el => { const r=el.getBoundingClientRect(); return Math.min(r.width,r.height); })"
        )
        assert peek_sizes and min(peek_sizes) >= 43.5, peek_sizes
        page.locator("#peekinspect").tap()

    page.wait_for_selector("#pincard.open #route-choices-panel .route-recommendations .route-choice", timeout=30_000)
    page.wait_for_selector("#pincard.open #route-plan-panel", state="attached", timeout=30_000)
    # The inspector intentionally paints its cached/lightweight response immediately, then swaps in
    # the complete pinned branch set. Compare the final DOM to pin=1 only after that enrichment has
    # committed; waiting on the unrelated settings-recompute class can race this transition.
    page.wait_for_function(
        "() => routePin!=null && BDCACHE.get(routePin)?._pin===true",
        timeout=30_000,
    )
    page.wait_for_function(
        "() => !document.getElementById('pincard').classList.contains('pinloading')",
        timeout=30_000,
    )
    page.wait_for_timeout(650)  # allow the one-shot union fit to settle before geometry checks
    cell_id = page.evaluate("() => routePin")
    body = _api_get(
        context.request, BASE_URL,
        f"/itinerary?id={quote(str(cell_id))}&dlat={destination['lat']}&dlon={destination['lon']}"
        f"{_speed_query(speed)}&pin=1",
    )
    api_errors = validate_route_response(body, destination)
    assert not api_errors, api_errors
    metrics = route_metrics(body)
    assert metrics["routes"] >= 2, f"saved hotspot lost all alternatives: {metrics}"
    viewport_width = context_args["viewport"]["width"]
    shot(page, f"route_inspector_v2_{viewport_width}_choices")
    if viewport_width in {1280, 390}:
        page.evaluate("() => { themePref='dark'; applyTheme(); }")
        page.wait_for_timeout(250)
        shot(page, f"route_inspector_v2_{viewport_width}_choices_dark")
        page.evaluate("() => { themePref='auto'; applyTheme(); }")

    # Expand the remaining-choice surface before comparing the disjoint union of recommendation,
    # practical, and remaining rows with the API and measuring whether its last route is reachable.
    if page.locator("#allroutes").count():
        _expand_all_route_disclosures(page)
    snapshot = dom_family_snapshot(page)
    assert_api_matches_dom(body, snapshot)
    if context_args["viewport"]["width"] >= 1240:
        assert snapshot["choices_visible"] and snapshot["plan_visible"], snapshot
    else:
        assert snapshot["choices_visible"] and not snapshot["plan_visible"], snapshot
    crowding = measure_family_card(page)
    assert crowding["families"] == metrics["families"]
    assert crowding["branches"] == metrics["branches"]
    assert crowding["route_options"] == metrics["routes"]
    assert crowding["route_choice_rows"] == crowding["route_options"]
    assert crowding["recommended_rows"] == 1
    assert len(crowding["selected_keys"]) == 1
    assert crowding["nested_interactives"] == 0
    assert crowding["horizontal_overflow"] <= 1, crowding
    assert crowding["document_horizontal_overflow"] <= 1, {
        "overflow": crowding["document_horizontal_overflow"],
        "offenders": crowding["offscreen_right"],
    }
    assert not crowding["clipped_labels"], crowding
    assert crowding["scroll_reaches_last"], crowding
    assert crowding["close_visible"], crowding
    assert crowding["card_in_view"], crowding
    assert crowding["panel_overlap_px2"] <= 1, crowding
    assert crowding["legend_overlap_px2"] <= 1, crowding
    if touch:
        assert crowding["min_target_px"] >= 43.5, crowding

    # Every input path uses the exact route key advertised by one flat route-choice row. Desktop
    # covers transient pointer preview, keyboard focus preview, and click lock; touch covers tap.
    choice = page.locator("#pincard .route-choice").last
    expected_key = choice.get_attribute("data-choice-key")
    expected_family = choice.get_attribute("data-family")
    expected_branch = choice.get_attribute("data-branch")
    expected_time = choice.locator(".route-time").evaluate(
        "el => (el.childNodes[0]?.textContent||'').trim().replace(/m$/, ' min')"
    )
    drawn_matches = (
        "([key,family,branch]) => DRAWN && DRAWN.multi && DRAWN.key===key "
        "&& DRAWN.famKey===family && DRAWN.branchKey===branch"
    )
    if touch:
        choice.tap()
        page.wait_for_function(
            "([key,family,branch]) => selKey===key && DRAWN && DRAWN.multi && DRAWN.key===key "
            "&& DRAWN.famKey===family && DRAWN.branchKey===branch",
            arg=[expected_key, expected_family, expected_branch], timeout=5_000,
        )
    else:
        choice.hover()
        page.wait_for_function(
            drawn_matches, arg=[expected_key, expected_family, expected_branch], timeout=5_000,
        )
        choice.focus()
        page.wait_for_function(
            drawn_matches, arg=[expected_key, expected_family, expected_branch], timeout=5_000,
        )
        choice.click()
        page.wait_for_function(
            "([key,family,branch]) => selKey===key && DRAWN && DRAWN.multi && DRAWN.key===key "
            "&& DRAWN.famKey===family && DRAWN.branchKey===branch",
            arg=[expected_key, expected_family, expected_branch], timeout=5_000,
        )
    assert choice.get_attribute("aria-pressed") == "true"
    _open_route_plan(page, touch=touch)
    shot(page, f"route_inspector_v2_{viewport_width}_plan")
    if viewport_width in {1280, 390}:
        page.evaluate("() => { themePref='dark'; applyTheme(); }")
        page.wait_for_timeout(250)
        shot(page, f"route_inspector_v2_{viewport_width}_plan_dark")
        page.evaluate("() => { themePref='auto'; applyTheme(); }")
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == expected_key
    assert page.locator('#route-plan-panel [data-fact="time"] .route-fact-value').inner_text().strip() == expected_time
    directions = page.locator("#route-directions")
    (directions.locator("summary").tap if touch else directions.locator("summary").click)()
    page.wait_for_function("() => document.getElementById('route-directions')?.open")

    # Detail disclosure is an intentional reader state, not something a selection refresh is
    # allowed to erase. Select the recommendation when it differs from the expert row above.
    _open_route_choices(page, touch=touch)
    alternative = page.locator("#route-choices-panel .route-recommendations .route-choice")
    alternative_key = alternative.get_attribute("data-choice-key")
    if alternative_key != expected_key:
        (alternative.tap if touch else alternative.click)()
        page.wait_for_function("key => selKey===key", arg=alternative_key, timeout=5_000)
        _open_route_plan(page, touch=touch)
        assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == alternative_key
        assert page.locator("#route-directions").evaluate("el => el.open")


def test_mobile_inspector_reuses_preview_and_keeps_one_stable_selection(new_context):
    """Touch preview, inspect, route taps, and view switches form one stable interaction.

    This is deliberately a behavioral test rather than a screenshot test.  It catches the costly
    mobile failure mode where Inspect replaced a useful first-tap preview with a blank loading
    sheet, and where focus+click event overlap drew the same large route layer more than once.
    """
    context = new_context(**VIEWPORTS[-1].values[0])
    page = context.new_page()
    itinerary_requests = []
    request_failures = []
    http_errors = []
    browser_errors = []

    def remember_request(request):
        if "/itinerary?" in request.url:
            itinerary_requests.append(request.url)

    def remember_request_failure(request):
        if "/itinerary?" in request.url:
            request_failures.append((request.url, request.failure))

    def remember_response(response):
        if "/itinerary?" in response.url and response.status >= 400:
            http_errors.append((response.url, response.status))

    page.on("request", remember_request)
    page.on("requestfailed", remember_request_failure)
    page.on("response", remember_response)
    page.on(
        "console",
        lambda message: browser_errors.append(("console", message.text))
        if message.type == "error" else None,
    )
    page.on("pageerror", lambda error: browser_errors.append(("pageerror", str(error))))

    destination = SAVED_HOTSPOT["destination"]
    open_destination(page, BASE_URL, destination, SAVED_HOTSPOT["speed"])
    point = _cell_container_point(page, SAVED_HOTSPOT["cell_id"])
    page.touchscreen.tap(point["x"], point["y"])

    # The first tap is a durable preview.  It must finish in place and remain the only open sheet
    # until the user explicitly asks to inspect routes.
    page.wait_for_selector("#touchpeek.open #peekinspect", timeout=30_000)
    page.wait_for_function(
        "() => !document.getElementById('peekbody').innerText.includes('Loading commute')",
        timeout=30_000,
    )
    preview = page.evaluate(
        """() => ({
          id:String(touchFeature?.properties?.id),
          title:document.querySelector("#peekbody .peek-title")?.textContent.trim()||"",
          text:document.getElementById("peekbody").innerText,
          open:document.getElementById("touchpeek").classList.contains("open"),
          hidden:document.getElementById("touchpeek").getAttribute("aria-hidden"),
          pinOpen:document.getElementById("pincard").classList.contains("open")
        })"""
    )
    assert preview["id"] == SAVED_HOTSPOT["cell_id"]
    assert preview["title"] and preview["open"] and preview["hidden"] == "false"
    assert not preview["pinOpen"]
    assert "Loading commute" not in preview["text"]
    assert len(itinerary_requests) == 1, itinerary_requests
    assert "pin=1" not in itinerary_requests[0]

    # Inspect must synchronously reuse that useful cached route.  A richer pin=1 response may
    # enhance it in the background, but the user never sees a blank/loading replacement.
    page.locator("#peekinspect").tap()
    immediate = page.evaluate(
        """() => ({
          open:document.getElementById("pincard").classList.contains("open"),
          text:document.getElementById("pinbody").innerText,
          rows:document.querySelectorAll("#pincard .route-choice").length,
          summary:!!document.querySelector("#pincard .pin-summary")
        })"""
    )
    assert immediate["open"] and immediate["summary"]
    assert immediate["rows"] >= 1
    assert preview["title"] in immediate["text"]
    assert "loading route" not in immediate["text"].lower()

    # Allow the optional enhancement request to be scheduled.  If present, wait until its result
    # has been committed before instrumenting draws so the background render cannot pollute counts.
    page.wait_for_timeout(750)
    enhancement_requests = itinerary_requests[1:]
    assert len(enhancement_requests) <= 1, itinerary_requests
    assert not enhancement_requests or "pin=1" in enhancement_requests[0], itinerary_requests
    if enhancement_requests:
        page.wait_for_function(
            "() => routePin!=null && BDCACHE.get(routePin)?._pin===true",
            timeout=30_000,
        )
    page.wait_for_function(
        "() => !document.getElementById('pincard').classList.contains('pinloading')",
        timeout=30_000,
    )
    page.wait_for_timeout(650)
    request_count_after_inspect = len(itinerary_requests)

    # Prefer a remaining-choice row deep in the sheet. If this feed/config legitimately fits every
    # branch into the featured section, use its last non-primary row instead.
    more_toggle = page.locator("#all-routes-toggle")
    if more_toggle.count():
        _expand_all_route_disclosures(page)
        choice = page.locator("#all-routes-panel .route-choice").last
    else:
        choice = page.locator("#pincard .route-choice:not(.recommended)").last
    expected_key = choice.get_attribute("data-choice-key")
    assert expected_key and choice.get_attribute("aria-pressed") == "false"

    page.evaluate(
        """() => {
          const original=drawSelected;
          window.__e2eRouteDraws=0;
          drawSelected=function(){
            window.__e2eRouteDraws+=1;
            return original.apply(this,arguments);
          };
        }"""
    )
    choice.tap()
    page.wait_for_function(
        "key => selKey===key && previewKey===null && DRAWN && DRAWN.key===key",
        arg=expected_key,
        timeout=5_000,
    )
    assert page.evaluate("() => window.__e2eRouteDraws") == 1

    # Retapping the already-selected row is a no-op: selection, route layers, and summary remain
    # stable without rebuilding hundreds of SVG layers.
    page.evaluate("() => { window.__e2eRouteDraws=0; }")
    for _ in range(3):
        choice.tap()
    assert page.evaluate("() => window.__e2eRouteDraws") == 0
    selected = page.evaluate(
        """() => ({
          key:selKey,
          drawn:DRAWN?.key||null,
          layers:routeLayer.getLayers().length,
          pressed:[...new Set([...document.querySelectorAll("#pincard .route-choice[aria-pressed='true']")]
            .map(row=>row.dataset.key))]
        })"""
    )
    assert selected["key"] == selected["drawn"] == expected_key
    assert selected["layers"] > 0
    assert selected["pressed"] == [expected_key]

    # Mobile keeps comparison and directions distinct: the sticky CTA opens a readable plan, and
    # a later choice change must preserve an intentionally expanded directions disclosure.
    _open_route_plan(page, touch=True)
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == expected_key
    page.locator("#route-directions summary").tap()
    page.wait_for_function("() => document.getElementById('route-directions')?.open")
    _open_route_choices(page, touch=True)
    recommended = page.locator("#route-choices-panel .route-recommendations .route-choice")
    recommended_key = recommended.get_attribute("data-choice-key")
    assert recommended_key and recommended_key != expected_key
    recommended.tap()
    page.wait_for_function(
        "key => selKey===key && selectionLocked && DRAWN && DRAWN.key===key",
        arg=recommended_key, timeout=5_000,
    )
    _open_route_plan(page, touch=True)
    assert page.locator("#route-plan-panel").get_attribute("data-selected-choice-key") == recommended_key
    assert page.locator("#route-directions").evaluate("el => el.open")
    expected_key = recommended_key

    # Compact Map mode preserves the locked route.  Start Choices from scrollTop=0 so the selected
    # expert row can only become visible if the view-switch restoration explicitly finds it.
    page.evaluate("() => { window.__e2eRouteDraws=0; }")
    page.locator('#pinview [data-view="map"]').tap()
    page.wait_for_function("() => document.body.classList.contains('pin-map-view')")
    page.wait_for_function(
        """() => {
          if(!DRAWN||!routeLayer.getLayers().length)return false;
          const choice=findChoice(selKey);
          const points=(choice?optLegs(choice.o):[]).flatMap(leg=>leg.pts||[])
            .map(point=>map.latLngToContainerPoint(L.latLng(point[0],point[1])));
          if(!points.length)return false;
          const left=Math.min(...points.map(point=>point.x));
          const right=Math.max(...points.map(point=>point.x));
          const top=Math.min(...points.map(point=>point.y));
          const bottom=Math.max(...points.map(point=>point.y));
          const card=document.getElementById('pincard').getBoundingClientRect();
          return left>=8 && right<=innerWidth-8 && top>=8 && bottom<=card.top-8;
        }""",
        timeout=5_000,
    )
    assert page.locator('#pinview [data-view="map"]').get_attribute("aria-pressed") == "true"
    page.evaluate("() => { document.getElementById('route-choices-panel').scrollTop=0; }")
    map_state = page.evaluate(
        "() => ({key:selKey,drawn:DRAWN?.key||null,layers:routeLayer.getLayers().length})"
    )
    assert map_state["key"] == map_state["drawn"] == expected_key
    assert map_state["layers"] > 0

    page.locator('#pinview [data-view="choices"]').tap()
    page.wait_for_function(
        """key => {
          if(document.body.classList.contains("pin-map-view"))return false;
          const body=document.getElementById("route-choices-panel");
          const row=[...document.querySelectorAll("#pincard .route-choice")]
            .find(candidate=>candidate.dataset.key===key);
          if(!row)return false;
          const bodyRect=body.getBoundingClientRect(),rowRect=row.getBoundingClientRect();
          const head=document.querySelector("#pincard .pin-head");
          const visibleTop=Math.max(bodyRect.top,head?.getBoundingClientRect().bottom||bodyRect.top);
          return rowRect.top>=visibleTop-1 && rowRect.bottom<=bodyRect.bottom+1;
        }""",
        arg=expected_key,
        timeout=5_000,
    )
    final_state = page.evaluate(
        """() => ({
          key:selKey,
          drawn:DRAWN?.key||null,
          layers:routeLayer.getLayers().length,
          draws:window.__e2eRouteDraws,
          pressed:[...new Set([...document.querySelectorAll("#pincard .route-choice[aria-pressed='true']")]
            .map(row=>row.dataset.key))]
        })"""
    )
    assert final_state["key"] == final_state["drawn"] == expected_key
    assert final_state["layers"] > 0
    assert final_state["draws"] == 0
    assert final_state["pressed"] == [expected_key]
    assert page.locator('#pinview [data-view="choices"]').get_attribute("aria-pressed") == "true"

    assert len(itinerary_requests) == request_count_after_inspect, itinerary_requests
    assert not request_failures, request_failures
    assert not http_errors, http_errors
    assert not browser_errors, browser_errors


def test_mobile_adjust_then_new_cell_owns_one_preview_and_survives_resize(new_context):
    """A map tap from Adjust replaces the old pin instead of stacking two sheets."""
    context = new_context(**VIEWPORTS[-1].values[0])
    page = context.new_page()
    destination = SAVED_HOTSPOT["destination"]
    open_destination(page, BASE_URL, destination, SAVED_HOTSPOT["speed"])

    first = _cell_container_point(page, SAVED_HOTSPOT["cell_id"])
    page.touchscreen.tap(first["x"], first["y"])
    page.wait_for_selector("#touchpeek.open #peekinspect", timeout=30_000)
    page.locator("#peekinspect").tap()
    page.wait_for_selector("#pincard.open .route-choice", timeout=30_000)
    page.wait_for_function(
        "() => pendingFitId==null && !document.getElementById('pincard').classList.contains('pinloading')",
        timeout=30_000,
    )
    page.wait_for_timeout(350)

    page.locator("#pinadjust").tap()
    page.wait_for_function(
        "() => document.body.classList.contains('route-adjusting') "
        "&& document.getElementById('panel').classList.contains('open')"
    )

    # The expanded controls deliberately leave a small strip of map visible. Put a known reachable
    # neighboring cell's actual polygon center in that strip, then exercise the real touch path.
    second_id = "1915"
    positioned = page.evaluate(
        """id => {
          let target=null;layer.eachLayer(candidate=>{
            if(String(candidate.feature?.properties?.id)===String(id))target=candidate;
          });
          if(!target||val(id)==null)return false;
          map.setView(target.getBounds().getCenter(),13,{animate:false});
          map.panBy([0,Math.max(0,map.getSize().y/2-92)],{animate:false});
          return true;
        }""",
        second_id,
    )
    assert positioned, f"replacement cell {second_id} was not reachable"
    page.wait_for_timeout(350)
    second = page.evaluate(
        """id => {
          let target=null;layer.eachLayer(candidate=>{
            if(String(candidate.feature?.properties?.id)===String(id))target=candidate;
          });
          const point=map.latLngToContainerPoint(target.getBounds().getCenter());
          const rect=document.getElementById('map').getBoundingClientRect();
          return {id:String(id),x:rect.left+point.x,y:rect.top+point.y};
        }""",
        second_id,
    )
    assert second, "no reachable replacement cell was visible above the Adjust sheet"
    page.touchscreen.tap(second["x"], second["y"])
    page.wait_for_function(
        """id => routePin==null && touchFeature && String(touchFeature.properties.id)===id &&
          document.getElementById('touchpeek').classList.contains('open') &&
          !document.body.classList.contains('route-pinned') &&
          !document.body.classList.contains('route-adjusting') &&
          !document.getElementById('panel').classList.contains('open') &&
          !document.getElementById('pincard').classList.contains('open')""",
        arg=second["id"],
        timeout=30_000,
    )

    # Inspect the replacement, then tighten the viewport while it is open. The route card and its
    # primary view/close controls must remain reachable at 320px rather than inheriting Adjust's
    # display:none state.
    page.locator("#peekinspect").tap()
    page.wait_for_selector("#pincard.open .route-choice", timeout=30_000)
    page.set_viewport_size({"width": 320, "height": 568})
    page.wait_for_timeout(300)
    card = page.locator("#pincard").evaluate(
        """el => { const r=el.getBoundingClientRect(); return {
          width:r.width,height:r.height,display:getComputedStyle(el).display,
          close:!!el.querySelector('#pinx'),views:el.querySelectorAll('#pinview button').length
        }; }"""
    )
    assert card["display"] != "none" and card["width"] > 0 and card["height"] > 0, card
    assert card["close"] and card["views"] == 3, card


def test_mobile_speed_change_cancels_stale_pin_enrichment(new_context):
    """A delayed old-speed pin response cannot overwrite the final fast-walk card."""
    context = new_context(**VIEWPORTS[-1].values[0])
    page = context.new_page()
    itinerary_requests = []
    browser_errors = []
    page.on("request", lambda request: itinerary_requests.append(request.url)
            if "/itinerary?" in request.url else None)
    page.on("pageerror", lambda error: browser_errors.append(str(error)))
    page.on("console", lambda message: browser_errors.append(message.text)
            if message.type == "error" else None)

    destination = SAVED_HOTSPOT["destination"]
    open_destination(page, BASE_URL, destination, SAVED_HOTSPOT["speed"])
    point = _cell_container_point(page, SAVED_HOTSPOT["cell_id"])
    page.touchscreen.tap(point["x"], point["y"])
    page.wait_for_selector("#touchpeek.open #peekinspect", timeout=30_000)
    page.wait_for_function(
        "() => !document.getElementById('peekbody').innerText.includes('Loading commute')",
        timeout=30_000,
    )

    # Hold the first old-speed pin=1 response at the browser fetch boundary. The later fast-speed
    # recompute must complete independently; releasing this stale response afterward exercises the
    # progressive upgrader's cancellation token, not merely a conveniently ordered network run.
    page.evaluate(
        """() => {
          const realFetch=window.fetch.bind(window);
          window.__heldOldPin=false;
          window.__oldPinSettled=false;
          window.__releaseOldPin=null;
          window.fetch=(input,init) => {
            const url=String(input);
            if(!window.__heldOldPin && url.includes('/itinerary?') &&
                url.includes('pin=1') && !url.includes('speed=')){
              window.__heldOldPin=true;
              return new Promise((resolve,reject) => {
                window.__releaseOldPin=() => {
                  const replayInit={...(init||{})};delete replayInit.signal;
                  return realFetch(input,replayInit).then(
                  response => {
                    resolve(response);
                    // The stale response's JSON/cache callbacks are promise microtasks.  A later
                    // timer is therefore a deterministic observation point after they have run.
                    setTimeout(() => { window.__oldPinSettled=true; }, 100);
                  },
                  error => {
                    reject(error);
                    setTimeout(() => { window.__oldPinSettled=true; }, 100);
                  });
                };
              });
            }
            return realFetch(input,init);
          };
        }"""
    )
    page.locator("#peekinspect").tap()
    page.wait_for_function("() => typeof window.__releaseOldPin==='function'", timeout=5_000)
    assert page.locator("#pincard").evaluate("el => el.classList.contains('open')")

    page.locator("#pinadjust").tap()
    page.wait_for_function("() => document.body.classList.contains('route-adjusting')")
    page.locator('#speed [data-v="fast"]').tap()
    page.wait_for_function(
        """() => walkspeed==='fast' && routePin!=null && BDCACHE.get(routePin)?._pin===true &&
          !document.getElementById('pincard').classList.contains('pinloading')""",
        timeout=45_000,
    )
    before_release = page.evaluate(
        """() => {
          window.__fastPinObject=BDCACHE.get(routePin);
          return {id:String(routePin),speed:walkspeed,
            settings:document.querySelector('.pin-context span')?.textContent||'',
            pending:!!BDCACHE.get(routePin)._pinPending,pin:!!BDCACHE.get(routePin)._pin};
        }"""
    )
    assert before_release["id"] == SAVED_HOTSPOT["cell_id"]
    assert before_release["speed"] == "fast" and before_release["pin"]
    assert not before_release["pending"]
    assert "Fast walk" in before_release["settings"]

    page.evaluate("() => window.__releaseOldPin()")
    page.wait_for_function(
        "() => window.__oldPinSettled && window.__fastPinObject===BDCACHE.get(routePin)",
        timeout=30_000,
    )
    assert page.evaluate(
        "() => walkspeed==='fast' && window.__fastPinObject===BDCACHE.get(routePin) && BDCACHE.get(routePin)._pin"
    )

    pin_requests = [url for url in itinerary_requests if "pin=1" in url]
    assert len(pin_requests) == 2, itinerary_requests
    assert "speed=fast" in pin_requests[0] and "speed=" not in pin_requests[1], pin_requests
    assert not browser_errors, browser_errors
