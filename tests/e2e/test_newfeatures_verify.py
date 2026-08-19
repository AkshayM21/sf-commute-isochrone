"""
Ad-hoc verification specs for the new work items (run against PORT=8765, RAPTOR engine):
  1. Pinned corner card replaces the on-cell popup (+ fitBounds, no leaflet popup on map).
  2. Route-family diagram model: hover draws ONE clean primary; pin opens authoritative family +
     branch cards and draws every returned route; focusing a card dims unrelated routes while
     hovering OTHER map cells does NOTHING.
  3. Footer About + How-it-works modals.

Drives the running server at $E2E_BASE_URL. Screenshots land in tests/e2e/screens/.
NOT part of the committed suite (pytest.ini testpaths excludes it); run explicitly:
  E2E_BASE_URL=http://127.0.0.1:8765 ../../.venv/bin/python -m pytest test_newfeatures_verify.py
The conftest set_address/wait_for_fast_map helpers assume the legacy R5 server (#dest shows
'fast ~Nms'); under RAPTOR #dest has no 'fast', so we set the address + wait locally here.
"""
import pytest

from conftest import fresh_load, find_colored_cell_hover, shot

COMPUTE_TIMEOUT = 25_000
ADDR_MARKET = "1 Market St"


@pytest.fixture
def page(new_context):
    ctx = new_context(viewport={"width": 1280, "height": 800})
    return ctx.new_page()


def set_addr_raptor(page, addr):
    """Set a workplace and wait for the map under the RAPTOR engine: the neighborhood list
    populates and #dest carries the typed label (RAPTOR #dest has no 'fast ~Nms')."""
    page.fill("#addr", addr)
    page.wait_for_timeout(400)
    page.keyboard.press("Escape")
    page.wait_for_function("() => !document.getElementById('ac').classList.contains('open')", timeout=3_000)
    page.click("#go")
    page.wait_for_function(
        "() => document.querySelectorAll('#list .nb').length > 0",
        timeout=COMPUTE_TIMEOUT,
    )


def wait_variance(page):
    # /variance is fetched progressively after the map paints; give it time to build + paint.
    page.wait_for_timeout(5000)


def test_pin_corner_card_no_popup(page):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    fresh_load(page)
    set_addr_raptor(page, ADDR_MARKET)
    wait_variance(page)

    hit = find_colored_cell_hover(page)
    assert hit, "no colored cell found to hover"
    page.wait_for_selector(".leaflet-tooltip.tt .bd", timeout=COMPUTE_TIMEOUT)
    shot(page, "nf_01_hover_route_alts")

    page.mouse.click(*hit)
    page.wait_for_selector("#pincard.open .bd", timeout=COMPUTE_TIMEOUT)
    assert page.is_visible("#pincard"), "pin card should be visible after click"
    assert page.query_selector(".leaflet-popup") is None, "an on-map Leaflet popup leaked (should be the corner card)"
    shot(page, "nf_02_pinned_corner_card")

    # Esc unpins.
    page.keyboard.press("Escape")
    page.wait_for_function("() => !document.getElementById('pincard').classList.contains('open')", timeout=4000)

    # Re-pin then close via the × button.
    page.mouse.click(*hit)
    page.wait_for_selector("#pincard.open", timeout=COMPUTE_TIMEOUT)
    page.click("#pinx")
    page.wait_for_function("() => !document.getElementById('pincard').classList.contains('open')", timeout=4000)

    assert not errors, f"console/page errors: {errors}"


# An alt-rich cell (Richmond): origin lat/lon that /variance gives >=2 alternatives for, under
# the "1 Market St" workplace. Pinning this deterministically (vs a flaky pixel scan over a
# revealed-tooltip hint) exercises the compare-list row swap.
ALT_CELL_LL = (37.80075, -122.46369)


def _cp_of(page, lat, lon):
    """Viewport pixel of a lat/lon (recenters the map there so it's on-screen + left of the card)."""
    page.evaluate("([lat,lon]) => { map.setView([lat,lon], 13); return null; }", [lat, lon])
    page.wait_for_timeout(500)
    return page.evaluate(
        "([lat,lon]) => { const p = map.latLngToContainerPoint(L.latLng(lat,lon));"
        " const r = document.getElementById('map').getBoundingClientRect();"
        " return {x: r.x+p.x, y: r.y+p.y}; }", [lat, lon])


def test_compare_family_focus(page):
    """Pin an alt-rich cell, then focus and lock an authoritative family/branch card."""
    fresh_load(page)
    set_addr_raptor(page, ADDR_MARKET)
    wait_variance(page)

    cp = _cp_of(page, *ALT_CELL_LL)
    page.mouse.move(cp["x"] + 1, cp["y"] + 1)
    page.wait_for_selector(".leaflet-tooltip.tt .bd", timeout=COMPUTE_TIMEOUT)
    page.wait_for_timeout(300)
    af = page.query_selector(".leaflet-tooltip.tt .altfoot")
    if not (af and af.is_visible()):
        pytest.skip("target cell reported no alternatives (GTFS feed window may have shifted)")
    shot(page, "nf_02b_hover_primary_with_hint")

    page.mouse.click(cp["x"] + 1, cp["y"] + 1)
    page.wait_for_selector("#pincard.open .cmp .family", timeout=COMPUTE_TIMEOUT)
    page.wait_for_timeout(500)
    families = page.query_selector_all("#pincard .cmp .family")
    branches = page.query_selector_all("#pincard .cmp .branch")
    options = page.evaluate("() => compareList.length")
    assert options >= 2, f"expected >=2 route options, got {options}"
    assert families and branches, "family diagram omitted its family/branch controls"
    assert page.query_selector("#pincard .cmp .strip") is None, "retired route strips returned"
    drawn0 = page.evaluate("() => DRAWN&&DRAWN.multi?{f:DRAWN.famKey,b:DRAWN.branchKey}:null")
    assert drawn0, "pin should draw the multi-route family diagram"
    shot(page, "nf_03_pinned_compare_list")

    target = branches[-1]
    expected_family = target.get_attribute("data-family")
    expected_branch = target.get_attribute("data-branch")
    target.hover()
    page.wait_for_timeout(400)
    drawn1 = page.evaluate("() => DRAWN&&DRAWN.multi?{f:DRAWN.famKey,b:DRAWN.branchKey}:null")
    assert drawn1 == {"f": expected_family, "b": expected_branch}, drawn1
    assert target.evaluate("el => el.classList.contains('foc')"), "hovered branch should be focused"
    shot(page, "nf_04_row_hover_swap")

    # Click the branch to lock the lens; moving away from the card keeps that lens locked.
    target.click()
    page.wait_for_timeout(250)
    box = page.eval_on_selector("#map", "el => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }")
    page.mouse.move(int(box["x"] + 160), int(box["y"] + box["h"] - 160))
    page.wait_for_timeout(400)
    locked = page.evaluate("() => DRAWN&&DRAWN.multi?{f:DRAWN.famKey,b:DRAWN.branchKey}:null")
    assert locked == drawn1, f"locked family lens should persist after mouse-away ({drawn1!r} -> {locked!r})"


def test_no_hover_while_pinned(page):
    """While a route is pinned, hovering OTHER squares does NOTHING: no sticky tooltip, no DRAWN
    change (user request #1)."""
    fresh_load(page)
    set_addr_raptor(page, ADDR_MARKET)
    wait_variance(page)
    hit = find_colored_cell_hover(page)
    assert hit, "no colored cell found to hover"
    page.mouse.click(*hit)
    page.wait_for_selector("#pincard.open", timeout=COMPUTE_TIMEOUT)
    page.wait_for_timeout(400)
    before = page.evaluate("() => (typeof DRAWN!=='undefined'&&DRAWN)?JSON.stringify({c:DRAWN.id,col:DRAWN.identityColor}):null")
    # hover lower-left map cells, clear of the top-right pin card
    box = page.eval_on_selector("#map", "el => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }")
    ox, oy = int(box["x"] + 160), int(box["y"] + box["h"] - 160)
    page.mouse.move(ox, oy)
    page.wait_for_timeout(200)
    page.mouse.move(ox + 40, oy - 40)
    page.wait_for_timeout(200)
    tt = page.query_selector(".leaflet-tooltip.tt")
    assert not (tt and tt.is_visible()), "a sticky tooltip opened while pinned"
    after = page.evaluate("() => (typeof DRAWN!=='undefined'&&DRAWN)?JSON.stringify({c:DRAWN.id,col:DRAWN.identityColor}):null")
    assert before == after, f"DRAWN changed by hovering another cell while pinned ({before} -> {after})"


def test_footer_modals(page):
    fresh_load(page)
    page.click("#aboutbtn")
    page.wait_for_selector("#aboutmodal.open", timeout=4000)
    txt = page.inner_text("#aboutmodal .card")
    assert "Akshay Manglik" in txt
    assert "akshaymanglik.com" in page.inner_html("#aboutmodal .card")
    shot(page, "nf_05_about_modal")
    page.click("#aboutx")
    page.wait_for_function("() => !document.getElementById('aboutmodal').classList.contains('open')", timeout=4000)

    page.click("#howbtn")
    page.wait_for_selector("#howmodal.open", timeout=4000)
    how = page.inner_text("#howmodal .card")
    assert "How it works" in how
    assert "data sources" in how.lower()  # h3 is uppercased via CSS; inner_text reflects that
    for src in ("511 SF Bay Open Data", "OpenStreetMap", "USGS 3DEP", "DataSF", "CARTO"):
        assert src in page.inner_html("#howmodal .card"), f"missing data source: {src}"
    shot(page, "nf_06_how_modal")
    page.keyboard.press("Escape")
    page.wait_for_function("() => !document.getElementById('howmodal').classList.contains('open')", timeout=4000)


def test_footer_modals_dark_and_light(page):
    fresh_load(page)
    page.evaluate("() => { document.querySelector('#theme button[data-v=\"dark\"]').click(); }")
    page.click("#howbtn")
    page.wait_for_selector("#howmodal.open", timeout=4000)
    shot(page, "nf_07_how_modal_dark")
    page.keyboard.press("Escape")
    page.evaluate("() => { document.querySelector('#theme button[data-v=\"light\"]').click(); }")
    page.wait_for_timeout(200)
    page.click("#aboutbtn")
    page.wait_for_selector("#aboutmodal.open", timeout=4000)
    shot(page, "nf_08_about_modal_light")
