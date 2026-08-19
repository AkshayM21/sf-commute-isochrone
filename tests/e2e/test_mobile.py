"""
Mobile end-to-end specs (iPhone emulation: 390x844, touch, mobile UA) for the SF
Commute Explorer. Specs 11-14 from the brief — the user specifically doubts mobile
quality, so these assert the CORRECT mobile expectation and do not paper over jank.

The mobile context engages BOTH the responsive CSS layer (<=719px: bottom sheet) AND
the JS TOUCH path (no hover; tap a cell -> popup breakdown).
"""
import time
import pytest
from playwright.sync_api import expect

from conftest import (
    ADDR_MARKET, IPHONE, COMPUTE_TIMEOUT, HEAVY_TIMEOUT,
    fresh_load, set_address, wait_for_fast_map, find_colored_cell_tap, shot,
)


@pytest.fixture
def page(new_context):
    """iPhone-emulated page (touch + mobile UA + 390x844)."""
    ctx = new_context(**IPHONE)
    return ctx.new_page()


def _open_sheet(page):
    """Tap the handle and wait for #panel.open. Returns once the sheet is open."""
    page.tap("#sheetbtn")
    page.wait_for_function("() => document.getElementById('panel').classList.contains('open')", timeout=3_000)


# ---- Spec 11: Mobile layout + bottom-sheet toggle --------------------------------------
def test_11_mobile_layout_and_sheet_toggle(page):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    fresh_load(page)

    # The grab handle is visible at 390px; on desktop it is display:none.
    assert page.is_visible("#sheetbtn"), "#sheetbtn (sheet handle) must be visible on mobile"

    # Panel starts collapsed: not .open, and translated down so only the ~52px handle peeks.
    assert not page.eval_on_selector("#panel", "el => el.classList.contains('open')"), \
        "panel should start collapsed on mobile"
    geom = page.eval_on_selector(
        "#panel",
        "el => { const r = el.getBoundingClientRect(); return {top:r.top, vh:window.innerHeight, h:r.height}; }",
    )
    # Collapsed sheet: its visible top sits near the bottom (only the handle peeks above the fold).
    assert geom["top"] > geom["vh"] - 90, \
        f"collapsed sheet should leave only the handle peeking; panel.top={geom['top']}, vh={geom['vh']}"

    # Map fills the screen behind the sheet.
    mb = page.eval_on_selector("#map", "el => { const r = el.getBoundingClientRect(); return {w:r.width, h:r.height}; }")
    assert mb["w"] >= 380 and mb["h"] >= 800, f"map should fill the viewport, got {mb}"
    shot(page, "mobile_11_collapsed")

    # Tap handle -> opens.
    _open_sheet(page)
    shot(page, "mobile_11_open")

    # Tap handle again -> closes.
    page.tap("#sheetbtn")
    page.wait_for_function("() => !document.getElementById('panel').classList.contains('open')", timeout=3_000)
    assert not page.eval_on_selector("#panel", "el => el.classList.contains('open')"), "second tap should close the sheet"
    assert errors == [], f"page errors during mobile layout test: {errors}"


# ---- Spec 12: Address on mobile --------------------------------------------------------
def test_12_mobile_address_autocomplete_and_compute(page):
    fresh_load(page)
    _open_sheet(page)

    # Autocomplete should work inside the sheet.
    page.fill("#addr", "ferry build")
    page.wait_for_selector("#ac.open .ac-item", timeout=10_000)
    items = page.eval_on_selector_all("#ac .ac-item", "els => els.length")
    assert items > 0, "autocomplete dropdown should be usable on mobile"
    shot(page, "mobile_12_autocomplete")

    # Tap the first suggestion -> map computes.
    page.tap("#ac .ac-item")
    wait_for_fast_map(page)
    assert page.inner_text("#dest").strip(), (
        "selecting a mobile suggestion should retain the destination label after computing the map")
    assert page.eval_on_selector_all("#list .nb", "els => els.length") > 0
    shot(page, "mobile_12_computed")


# ---- Spec 13: Tap-to-breakdown (the key mobile interaction) ----------------------------
def test_13_mobile_tap_to_breakdown(page):
    fresh_load(page)

    # Confirm the JS TOUCH branch is active (this is what makes tap, not hover, drive the
    # breakdown — see index.html onEachFeature).
    assert page.evaluate("() => TOUCH === true"), "TOUCH must be true under iPhone emulation"

    _open_sheet(page)
    set_address(page, ADDR_MARKET, via="go", tap=True)

    # Close the sheet so the map is fully tappable.
    page.tap("#sheetbtn")
    page.wait_for_function("() => !document.getElementById('panel').classList.contains('open')", timeout=3_000)

    # First tap gives a compact, touch-native preview instead of depending on hover or a tiny popup.
    hit = find_colored_cell_tap(page)
    assert hit is not None, "tapping the map did not open the commute preview"
    page.wait_for_selector("#touchpeek.open #peekinspect", timeout=8_000)
    page.wait_for_function(
        """() => { const e=document.getElementById('peekbody');
          return e && !e.textContent.includes('Loading commute'); }""",
        timeout=8_000,
    )
    preview = page.inner_text("#peekbody")
    assert "min" in preview, f"tap preview should show a commute time, got: {preview!r}"

    # Inspect promotes the cached preview immediately to the full route sheet, then enriches it.
    page.tap("#peekinspect")
    page.wait_for_function(
        """() => document.getElementById('pincard').classList.contains('open') &&
          document.querySelectorAll('#pinbody .route-row').length > 0""",
        timeout=30_000,
    )
    assert "min" in page.inner_text("#pinbody"), "route sheet should retain the commute time"

    # Confirm it relies on neither hover nor a small Leaflet popup.
    assert page.query_selector(".leaflet-tooltip.tt") is None, \
        "mobile must use the touch preview, not a hover tooltip"
    assert page.query_selector(".leaflet-popup") is None, \
        "mobile should use the touch preview and route sheet, not a tiny map popup"
    shot(page, "mobile_13_tap_breakdown")


# ---- Spec 14: Controls reachable inside the opened sheet -------------------------------
def test_14_mobile_controls_reachable(page):
    fresh_load(page)
    _open_sheet(page)
    set_address(page, ADDR_MARKET, via="go", tap=True)

    # Legacy R5 exposes Refine; RAPTOR is already exact and hides the no-op control.
    refine = page.locator("#refine")
    if page.is_visible("#refinebox"):
        refine.scroll_into_view_if_needed()
        assert refine.is_visible() and refine.is_enabled(), (
            "legacy Refine should be reachable + enabled in the sheet")
    else:
        assert not refine.is_visible() and not refine.is_enabled(), (
            "RAPTOR's hidden Refine control should remain inert")

    # Color-by-line toggle reachable + togglable.
    line_btn = page.locator("#cmode button[data-v='line']")
    line_btn.scroll_into_view_if_needed()
    assert line_btn.is_visible(), "color-by-line toggle should be reachable"
    line_btn.tap()
    assert "on" in (page.get_attribute("#cmode button[data-v='line']", "class") or ""), \
        "tapping Primary line should activate it"
    # reset to Time so the rest of the test is on the time view
    page.tap("#cmode button[data-v='time']")

    # Sliders reachable + adjustable.
    page.eval_on_selector("#ideal", "el => { el.value = 30; el.dispatchEvent(new Event('input', {bubbles:true})); }")
    assert page.inner_text("#idval") == "30", "ideal slider should be adjustable on mobile"
    page.locator("#thr").scroll_into_view_if_needed()
    assert page.locator("#thr").is_visible(), "max-commute slider should be reachable"

    # Export controls reachable + usable.
    page.locator("#dlcsv").scroll_into_view_if_needed()
    assert page.locator("#dlcsv").is_visible() and page.locator("#copylist").is_visible(), \
        "export buttons should be reachable in the sheet"
    with page.expect_download() as di:
        page.tap("#dlcsv")
    assert di.value.suggested_filename.endswith(".csv"), "CSV export should work on mobile"
    shot(page, "mobile_14_controls")

    # Legend must not obscure the map controls: on mobile it sits above the collapsed-sheet
    # handle (CSS bottom:60px) and must NOT overlap it.
    page.tap("#sheetbtn")  # close the sheet
    page.wait_for_function("() => !document.getElementById('panel').classList.contains('open')", timeout=3_000)
    page.wait_for_timeout(400)  # let the .28s sheet transition settle before measuring geometry
    legend = page.eval_on_selector("#legend", "el => { const r = el.getBoundingClientRect(); return {bottom:r.bottom, top:r.top}; }")
    handle = page.eval_on_selector("#sheetbtn", "el => { const r = el.getBoundingClientRect(); return {top:r.top}; }")
    assert legend["bottom"] <= handle["top"] + 2, \
        f"legend (bottom={legend['bottom']}) should sit above the collapsed sheet handle (top={handle['top']}), not over it"
    shot(page, "mobile_14_legend_clear")
