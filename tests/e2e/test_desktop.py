"""
Desktop end-to-end specs (viewport 1280x800) for the SF Commute Explorer.

Specs 1-10 from the brief. Run against the live server at $E2E_BASE_URL (default
http://127.0.0.1:8000). Each test is isolated via fresh_load() (localStorage + hash
cleared). Tests assert current user-visible behavior for both the featured RAPTOR app and the
remaining legacy R5-only refine surface.
"""
import time
import pytest
from playwright.sync_api import expect

from conftest import (
    ADDR_MARKET, ADDR_FERRY, HEAVY_TIMEOUT, COMPUTE_TIMEOUT,
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
def test_01_load_no_errors_prompt_refine_disabled(page):
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    fresh_load(page)

    # Prompt to enter a workplace is visible pre-address.
    assert page.is_visible("#prompt"), "intro prompt should be visible before an address is set"
    expect(page.locator("#prompt")).to_contain_text("workplace")

    # Refine is disabled until a fast map has been computed.
    assert page.get_attribute("#refine", "disabled") is not None, "Refine must be disabled pre-address"

    # Map + legend present, no JS errors.
    assert page.is_visible("#map")
    assert page.is_visible("#legend")
    shot(page, "desktop_01_load")
    assert errors == [], f"console/page errors on load: {errors}"


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

    # Legacy R5 exposes an approximate-map refine pass. RAPTOR is already exact and deliberately
    # hides/disables that control, so both served configurations get an honest assertion.
    if page.is_visible("#refinebox"):
        assert page.get_attribute("#refine", "disabled") is None, (
            "legacy Refine must enable after the first map")
    else:
        assert page.get_attribute("#refine", "disabled") is not None, (
            "RAPTOR's hidden no-op Refine control must stay disabled")

    # Neighborhood list populated, prompt dismissed.
    assert page.eval_on_selector_all("#list .nb", "els => els.length") > 0
    assert not page.is_visible("#prompt"), "intro prompt should hide after a workplace is set"
    expect(page.locator("#nbcount")).to_contain_text("min")
    shot(page, "desktop_03_fastmap")


# ---- Spec 4: Opt-in Refine -------------------------------------------------------------
def test_04_refine_is_opt_in_then_exact(page):
    fresh_load(page)
    set_address(page, ADDR_MARKET, via="go")

    if not page.is_visible("#refinebox"):
        pytest.skip("RAPTOR /compute is already exact; the legacy approximate-map refiner is absent")

    # It must NOT auto-run: immediately after fast map, #dest still says 'fast', not 'exact'.
    assert "exact" not in page.inner_text("#dest"), "exact refine must NOT run automatically"

    # Record the busy-chip text the instant Refine is clicked, via a MutationObserver, so the
    # assertion is not flaky when /compute_exact is cached and returns near-instantly (the
    # busy chip can flash faster than a polled is-visible check).
    page.evaluate(
        """() => {
            window.__busySeen = [];
            const b = document.getElementById('busy');
            const rec = () => { if (b.style.display !== 'none' && b.textContent) window.__busySeen.push(b.textContent); };
            new MutationObserver(rec).observe(b, {attributes:true, childList:true, subtree:true, characterData:true});
            rec();
        }"""
    )
    page.click("#refine")

    # Exact result must land: #dest shows 'exact ...s'. Generous timeout — the exact pass is
    # 14-34s when cold, instant when the result cache is warm.
    page.wait_for_function(
        "() => document.querySelector('#dest').textContent.includes('exact')",
        timeout=HEAVY_TIMEOUT,
    )
    dest = page.inner_text("#dest")
    assert "exact" in dest and dest.rstrip().endswith("s"), f"expected 'exact ...s', got {dest!r}"

    # The busy ("refining (exact)…") indicator should appear during the pass. On a warm
    # result-cache the exact pass can complete within a single microtask, faster than a
    # MutationObserver batch — so a miss here is not a regression; we record it as info but
    # only hard-fail if the chip never carried the refining text on a SLOW (cold) pass.
    busy_seen = page.evaluate("() => window.__busySeen || []")
    cold = "exact 0.0s" not in dest  # cold pass takes seconds; warm cache reports ~0.0s
    if cold:
        assert any("refining" in t for t in busy_seen), \
            f"a 'refining (exact)…' busy indicator should appear during a cold refine; saw {busy_seen}"
    # The chip must be cleared once done regardless.
    expect(page.locator("#busy")).to_be_hidden()
    shot(page, "desktop_04_refine_exact")


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

    page.click("#cmode button[data-v='line']")
    # Legend title flips immediately (this part works).
    page.wait_for_function(
        "() => document.getElementById('legtitle').textContent.includes('line')",
        timeout=COMPUTE_TIMEOUT,
    )
    # Wait for the heavy /attribution build to COMPLETE rather than for ATTR to become
    # non-empty (which never happens given the bug). loadAttribution() shows the busy chip
    # ('mapping lines…') for the duration and hides it on completion; once it's hidden the
    # request has returned and ATTR holds whatever the server sent. This makes the test fail
    # fast + precisely instead of burning the full timeout.
    page.wait_for_function(
        "() => document.getElementById('busy').textContent.includes('mapping lines')",
        timeout=COMPUTE_TIMEOUT,
    )
    page.wait_for_function(
        "() => document.getElementById('busy').style.display === 'none'",
        timeout=HEAVY_TIMEOUT,
    )
    attr_n = page.evaluate("() => Object.keys(ATTR).length")
    line_n = page.evaluate("() => Object.keys(LINECOLOR).length")
    shot(page, "desktop_06_color_by_line")
    assert attr_n > 0, (
        "BUG: color-by-line broken — /attribution returned 0 cells, so NO cells recolor by "
        "line. The legend title flips to 'Primary transit line per area' but the map and "
        "legend never update (verified directly: /attribution yields {} for multiple "
        "destinations after a 36-169s build)."
    )
    assert line_n > 0, "color-by-line legend should list at least one line"


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
