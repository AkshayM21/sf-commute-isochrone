"""
Desktop end-to-end specs (viewport 1280x800) for the SF Commute Explorer.

Specs 1-10 from the brief. Run against the live server at $E2E_BASE_URL (default
http://127.0.0.1:8000). Each test is isolated via fresh_load() (localStorage + hash
cleared). Tests assert current user-visible behavior for the graph-native RAPTOR app.
"""
import time
import pytest
from playwright.sync_api import expect

from conftest import (
    ADDR_MARKET, ADDR_FERRY, BASE_URL, HEAVY_TIMEOUT, COMPUTE_TIMEOUT,
    fresh_load, set_address, wait_for_fast_map, find_colored_cell_hover, shot,
)


@pytest.fixture
def page(new_context):
    """Desktop page on a 1280x800 context with downloads + clipboard enabled."""
    ctx = new_context(
        viewport={"width": 1280, "height": 800},
        accept_downloads=True,
        permissions=["clipboard-read", "clipboard-write"],
    )
    return ctx.new_page()


# ---- Spec 1: Load ----------------------------------------------------------------------
def test_01_load_no_errors_prompt(page):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    fresh_load(page)

    # Prompt to enter a workplace is visible pre-address.
    assert page.is_visible("#prompt"), "intro prompt should be visible before an address is set"
    expect(page.locator("#prompt")).to_contain_text("workplace")

    # The retired approximate/exact split has no Refine surface anywhere in the app.
    assert page.locator("#refinebox").count() == 0
    assert page.locator("#refine").count() == 0

    # Map + legend present, no JS errors.
    assert page.is_visible("#map")
    assert page.is_visible("#legend")
    shot(page, "desktop_01_load")
    assert errors == [], f"console/page errors on load: {errors}"


def test_01b_returning_visitor_gets_restore_shell_without_onboarding_flash(page):
    """A saved commute is truthful from first paint even while compute is deliberately held."""
    page.add_init_script(
        """localStorage.setItem('wp_v1', JSON.stringify({
          lat:37.7714154,lon:-122.4030885,label:'650 Townsend St'
        }));"""
    )
    held = []
    page.route("**/compute?*", lambda route: held.append(route))
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.documentElement.dataset.startup === 'restoring' && "
        "document.querySelector('#prompt .ob-restoring') && "
        "getComputedStyle(document.querySelector('#prompt .ob-restoring')).display !== 'none'"
    )
    for _ in range(40):
        if held:
            break
        page.wait_for_timeout(25)
    assert held, "startup compute was not intercepted"
    assert page.locator("#prompt .ob-restoring").is_visible()
    assert not page.locator("#prompt .ob-onboarding").is_visible()
    expect(page.locator("#prompt .ob-restoring")).to_contain_text("Opening your commute")
    shot(page, "desktop_01b_restoring")

    held.pop(0).continue_()
    page.wait_for_function("() => document.querySelectorAll('#list .nb').length > 0", timeout=COMPUTE_TIMEOUT)
    page.wait_for_function("() => getComputedStyle(document.getElementById('prompt')).display === 'none'")


def test_01c_failed_startup_restore_offers_retry_and_recovers(page):
    """A failed saved-workplace restore stays actionable, then Retry starts a fresh compute."""
    page.add_init_script(
        """localStorage.setItem('wp_v1', JSON.stringify({
          lat:37.7714154,lon:-122.4030885,label:'650 Townsend St'
        }));"""
    )
    attempts = []

    def fail_once(route):
        attempts.append(route.request.url)
        if len(attempts) == 1:
            route.fulfill(status=503, content_type="text/plain", body="temporarily unavailable")
        else:
            route.continue_()

    page.route("**/compute?*", fail_once)
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.documentElement.dataset.startup === 'error' && "
        "document.getElementById('prompt').dataset.state === 'error' && "
        "getComputedStyle(document.getElementById('startupretry')).display !== 'none'"
    )
    expect(page.locator("#startup-copy")).to_contain_text("couldn’t open this commute")
    assert page.locator("#startupretry").is_visible()
    assert page.locator("#startupchange").is_visible()
    assert len(attempts) == 1

    page.click("#startupretry")
    # A warm local compute can move through `restoring` between animation frames. The durable
    # contract is that Retry starts exactly one fresh request and reaches the ready surface.
    page.wait_for_function("() => document.querySelectorAll('#list .nb').length > 0", timeout=COMPUTE_TIMEOUT)
    page.wait_for_function(
        "() => document.documentElement.dataset.startup === 'ready' && "
        "getComputedStyle(document.getElementById('prompt')).display === 'none'"
    )
    assert len(attempts) == 2, attempts


def test_01d_change_workplace_after_failed_restore_returns_to_focused_onboarding(page):
    """Change workplace abandons the failed restore without leaving error-state UI behind."""
    page.add_init_script(
        """localStorage.setItem('wp_v1', JSON.stringify({
          lat:37.7714154,lon:-122.4030885,label:'650 Townsend St'
        }));"""
    )
    page.route(
        "**/compute?*",
        lambda route: route.fulfill(
            status=503, content_type="text/plain", body="temporarily unavailable"),
    )
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_function(
        "() => document.documentElement.dataset.startup === 'error' && "
        "document.getElementById('prompt').dataset.state === 'error'"
    )

    page.click("#startupchange")
    page.wait_for_function(
        """() => document.documentElement.dataset.startup === 'onboarding' &&
          document.getElementById('prompt').dataset.state === 'onboarding' &&
          getComputedStyle(document.querySelector('#prompt .ob-onboarding')).display !== 'none' &&
          getComputedStyle(document.querySelector('#prompt .ob-restoring')).display === 'none' &&
          document.activeElement === document.getElementById('obaddr')"""
    )
    assert not page.locator("#startupretry").is_visible()
    assert not page.locator("#startupchange").is_visible()
    assert page.locator("#obaddr").input_value() == ""
    assert page.locator("#dest").inner_text().strip() == ""


# ---- Spec 2: Autocomplete (incl. dedup flag) -------------------------------------------
def test_02_autocomplete_opens_navigates_selects(page):
    fresh_load(page)
    page.fill("#addr", ADDR_FERRY)  # "ferry build"

    # #ac opens with .ac-item suggestions.
    page.wait_for_selector("#ac.open .ac-item", timeout=10_000)
    items = page.eval_on_selector_all("#ac .ac-item", "els => els.map(e => e.textContent.trim())")
    assert len(items) > 0, "autocomplete should produce suggestions for 'ferry build'"
    shot(page, "desktop_02_autocomplete")

    # ArrowDown highlights the first item (.hl).
    page.focus("#addr")
    page.keyboard.press("ArrowDown")
    hl = page.eval_on_selector_all("#ac .ac-item.hl", "els => els.length")
    assert hl == 1, "ArrowDown should highlight exactly one item"

    # Enter selects the highlighted item -> map computes.
    page.keyboard.press("Enter")
    wait_for_fast_map(page)
    assert page.eval_on_selector_all("#list .nb", "els => els.length") > 0


def test_02b_autocomplete_dedup_FLAG(page):
    """Autocomplete suggestions remain deduplicated by their visible place identity."""
    fresh_load(page)
    page.fill("#addr", ADDR_FERRY)
    page.wait_for_selector("#ac.open .ac-item", timeout=10_000)
    labels = page.eval_on_selector_all("#ac .ac-item", "els => els.map(e => e.textContent.trim())")
    dupes = [l for l in set(labels) if labels.count(l) > 1]
    assert not dupes, (
        f"BUG: autocomplete shows duplicate suggestions {dupes} "
        f"(full list: {labels}). Dedup the /autocomplete results by label (and/or lat,lon)."
    )


# ---- Spec 3: Set + fast map ------------------------------------------------------------
def test_03_set_fast_map(page):
    fresh_load(page)
    set_address(page, ADDR_MARKET, via="go")

    dest = page.inner_text("#dest").strip()
    assert dest, "#dest should retain the selected workplace label after the map renders"
    assert ADDR_MARKET in dest, "#dest should echo the typed workplace label"

    assert page.locator("#refinebox").count() == 0
    assert page.locator("#refine").count() == 0

    # Neighborhood list populated, prompt dismissed.
    assert page.eval_on_selector_all("#list .nb", "els => els.length") > 0
    assert not page.is_visible("#prompt"), "intro prompt should hide after a workplace is set"
    expect(page.locator("#nbcount")).to_contain_text("min")
    shot(page, "desktop_03_fastmap")


# ---- Spec 4: Retired approximate-map surface -----------------------------------------
def test_04_no_refine_surface(page):
    fresh_load(page)
    assert page.locator("#refinebox").count() == 0
    assert page.locator("#refine").count() == 0
    set_address(page, ADDR_MARKET, via="go")
    assert page.locator("#refinebox").count() == 0
    assert page.locator("#refine").count() == 0


# ---- Spec 5: Hover breakdown -----------------------------------------------------------
def test_05_hover_breakdown_tooltip(page):
    fresh_load(page)
    set_address(page, ADDR_MARKET, via="go")

    hit = find_colored_cell_hover(page)
    assert hit is not None, "could not find a colored cell to hover (no .tt tooltip opened)"
    page.mouse.move(*hit)
    page.wait_for_selector(".bd", timeout=8_000)
    # Wait for the async /itinerary fetch to fill in real legs (not the 'loading route…' stub).
    page.wait_for_function(
        "() => { const e = document.querySelector('.bd'); return e && !e.textContent.includes('loading route'); }",
        timeout=8_000,
    )
    bd = page.inner_text(".bd")
    assert "min" in bd, f"breakdown tooltip should show a time + legs, got: {bd!r}"
    # A real trip shows at least one leg chip (walk / wait / line).
    legs = page.eval_on_selector_all(".bd .leg", "els => els.length")
    assert legs > 0, "breakdown should render at least one leg chip"
    shot(page, "desktop_05_hover_breakdown")


# ---- Spec 6: Color-by-line -------------------------------------------------------------
def test_06_color_by_line(page):
    """Primary-line mode returns attribution, recolors cells, and populates its legend."""
    fresh_load(page)
    set_address(page, ADDR_MARKET, via="go")

    with page.expect_response(
        lambda response: "/attribution?" in response.url,
        timeout=HEAVY_TIMEOUT,
    ) as response_info:
        page.click("#cmode button[data-v='line']")
    response = response_info.value
    assert response.ok, f"/attribution failed with HTTP {response.status}"
    attribution = response.json()
    assert attribution, "/attribution should map at least one reachable cell to a transit line"

    page.wait_for_function(
        "() => document.getElementById('legtitle').textContent.includes('line')",
        timeout=COMPUTE_TIMEOUT,
    )
    page.wait_for_function(
        "() => document.querySelectorAll('#legend .sc > span > span').length > 0",
        timeout=COMPUTE_TIMEOUT,
    )
    shot(page, "desktop_06_color_by_line")
    assert page.locator("#legend .sc > span").count() > 0, (
        "color-by-line legend should list at least one returned line"
    )


# ---- Spec 7: Sliders -------------------------------------------------------------------
def test_07_sliders_recolor_and_filter(page):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    fresh_load(page)
    set_address(page, ADDR_MARKET, via="go")

    before_under = page.inner_text("#nbcount")

    # Move 'ideal' (sweet spot) — legend label should update.
    page.eval_on_selector("#ideal", "el => { el.value = 35; el.dispatchEvent(new Event('input', {bubbles:true})); }")
    assert page.inner_text("#idval") == "35", "ideal slider value label should update"

    # Move 'thr' (max commute filter) DOWN — fewer neighborhoods remain under it.
    page.eval_on_selector("#thr", "el => { el.value = 20; el.dispatchEvent(new Event('input', {bubbles:true})); }")
    assert page.inner_text("#thrval") == "20", "thr slider value label should update"
    time.sleep(0.3)
    after_under = page.inner_text("#nbcount")
    assert "20 min" in after_under, f"nbcount should reflect new threshold, got {after_under!r}"
    assert after_under != before_under, "lowering the max-commute filter should change the under-count"
    shot(page, "desktop_07_sliders")
    assert errors == [], f"slider interaction raised page errors: {errors}"


# ---- Spec 8: Export --------------------------------------------------------------------
def test_08_export_csv_and_copy(page):
    fresh_load(page)
    set_address(page, ADDR_MARKET, via="go")

    # Download CSV — intercept the download and inspect contents.
    with page.expect_download() as di:
        page.click("#dlcsv")
    dl = di.value
    assert dl.suggested_filename.endswith(".csv"), f"unexpected download name {dl.suggested_filename!r}"
    csv = open(dl.path()).read()
    assert csv.startswith("name,minutes"), f"CSV should have a 'name,minutes' header, got: {csv[:40]!r}"
    rows = [r for r in csv.splitlines() if r and not r.startswith("name,")]
    assert len(rows) > 0, "CSV should contain neighborhood rows"

    # Copy list -> clipboard. Frontend writes TAB-separated `name<TAB>minutes` rows.
    page.click("#copylist")
    expect(page.locator("#toast")).to_contain_text("Copied", timeout=4_000)
    clip = page.evaluate("() => navigator.clipboard.readText()")
    assert "\t" in clip, "copied list should be tab-separated name<TAB>minutes rows"
    first = clip.splitlines()[0].split("\t")
    assert len(first) == 2 and first[1].strip().isdigit(), f"each copied row = name<TAB>minutes, got {first!r}"
    shot(page, "desktop_08_export")


# ---- Spec 9: Permalink round-trip ------------------------------------------------------
def test_09_permalink_roundtrip(page):
    fresh_load(page)
    set_address(page, ADDR_MARKET, via="go")
    # Change sliders + a segment toggle so the hash captures non-default state.
    page.eval_on_selector("#ideal", "el => { el.value = 30; el.dispatchEvent(new Event('input', {bubbles:true})); }")
    page.eval_on_selector("#thr", "el => { el.value = 50; el.dispatchEvent(new Event('input', {bubbles:true})); }")
    page.click("#metric button[data-v='b']")  # Best-case

    # Hash sync is debounced (~300ms); wait until it contains wp= and the params.
    page.wait_for_function(
        "() => location.hash.includes('wp=') && location.hash.includes('ideal=30') "
        "&& location.hash.includes('thr=50') && location.hash.includes('metric=b')",
        timeout=5_000,
    )
    h = page.evaluate("() => location.hash")
    assert "cmode=time" in h, f"hash should include cmode, got {h!r}"

    # Reload (same URL incl. hash) -> state restored + map recomputed.
    page.reload(wait_until="domcontentloaded")
    wait_for_fast_map(page)
    assert page.input_value("#addr") == ADDR_MARKET, "address text should restore from permalink"
    assert page.input_value("#ideal") == "30", "ideal slider should restore"
    assert page.input_value("#thr") == "50", "thr slider should restore"
    assert "on" in (page.get_attribute("#metric button[data-v='b']", "class") or ""), \
        "Best-case metric should restore as active"
    shot(page, "desktop_09_permalink_restored")


# ---- Spec 10: How it works modal -------------------------------------------------------
def test_10_how_it_works_modal(page):
    fresh_load(page)

    # Open via #howlink.
    page.click("#howlink")
    assert page.eval_on_selector("#howmodal", "el => el.classList.contains('open')"), "modal should open"
    expect(page.locator("#howmodal .card")).to_contain_text("How it works")
    shot(page, "desktop_10_howmodal")

    # Close via the × button.
    page.click("#howx")
    assert not page.eval_on_selector("#howmodal", "el => el.classList.contains('open')"), "× should close modal"

    # Open + close via backdrop click.
    page.click("#howlink")
    assert page.eval_on_selector("#howmodal", "el => el.classList.contains('open')")
    box = page.eval_on_selector("#howmodal", "el => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y}; }")
    page.mouse.click(int(box["x"]) + 8, int(box["y"]) + 8)  # backdrop corner, away from card
    assert not page.eval_on_selector("#howmodal", "el => el.classList.contains('open')"), "backdrop click should close"

    # Open + close via Escape.
    page.click("#howlink")
    assert page.eval_on_selector("#howmodal", "el => el.classList.contains('open')")
    page.keyboard.press("Escape")
    assert not page.eval_on_selector("#howmodal", "el => el.classList.contains('open')"), "Escape should close modal"
