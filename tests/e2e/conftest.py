"""
Shared fixtures + helpers for the SF Commute Explorer end-to-end browser suite.

These tests drive the ALREADY-RUNNING JVM-free Flask + RAPTOR + walk-graph server at
http://127.0.0.1:8000. They do not boot their own application process. Run the server first:

    .venv/bin/python scripts/server.py

then run the suite (see tests/e2e/README in the docstring of run.sh / the report).

Design notes / why things look the way they do:
  * Leaflet cells are CANVAS-rendered geoJSON (preferCanvas:true) — there is NO per-cell
    DOM. To "click/hover a cell" we drive real mouse/touch events at map pixel coords and
    detect the resulting Leaflet tooltip/popup. `find_colored_cell_hover` scans a grid of
    points until one opens the `.bd` breakdown — this is robust to map recentering.
  * The page's JS state lives in MODULE scope (`let TT={}` etc.), NOT on `window`. So we
    detect "map computed" via the user-visible destination label and neighborhood list
    populating — exactly what a user sees — rather than poking globals. The RAPTOR backend
    deliberately omits the legacy implementation-timing string ("fast ~Nms").
  * PRIVACY: tests only ever type neutral public addresses ("ferry building" / "1 Market
    St"). Never the user's saved workplace. Every test starts from a clean slate
    (localStorage cleared, location.hash cleared) for isolation.
"""
import os
import pathlib
import pytest
from playwright.sync_api import Page, expect

BASE_URL = os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000")
SCREENS = pathlib.Path(__file__).parent / "screens"
SCREENS.mkdir(exist_ok=True)

# Neutral, public addresses only (privacy invariant).
ADDR_MARKET = "1 Market St"
ADDR_FERRY = "ferry build"

# How long a fast /compute may take end-to-end (geocode + R5 reverse tree). Generous.
COMPUTE_TIMEOUT = 25_000
# /compute_exact + /attribution are heavy (14-36s observed); give them real headroom.
HEAVY_TIMEOUT = 75_000


def shot(page: Page, name: str):
    """Save a screenshot into tests/e2e/screens/ and return its absolute path."""
    p = SCREENS / f"{name}.png"
    page.screenshot(path=str(p), full_page=False)
    return str(p)


def fresh_load(page: Page):
    """Navigate to the app and hard-reset client state for test isolation.

    Clears localStorage (the app persists the last workplace in `wp_v1`) and
    location.hash (permalink) BEFORE the boot script can act on them, by clearing
    then reloading so boot() runs against a clean slate.
    """
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.evaluate("() => { try { localStorage.clear(); } catch (e) {} location.hash = ''; }")
    page.reload(wait_until="domcontentloaded")
    # The boot IIFE + Leaflet init are synchronous on load; the legend/prompt are present.
    page.wait_for_selector("#panel", state="attached")
    return page


def set_address(page: Page, addr: str, *, via="go", tap=False):
    """Type an address into #addr and trigger a compute.

    via="go"    -> fill the box and click/tap the Set button (#go)
    via="enter" -> fill the box and press Enter (falls back to /geocode of typed text)
    Waits until the first map lands (#dest has the selected label and the neighborhood
    list populates), i.e. the user-visible "computed" signal.
    """
    page.fill("#addr", addr)
    if via == "go":
        # Typing fires the autocomplete dropdown (250ms debounce + fetch), which can overlay
        # #go and intercept the click. Let it settle, then dismiss it (Escape closes #ac
        # without selecting) so we exercise the Set-button -> /geocode path deterministically.
        page.wait_for_timeout(400)
        page.keyboard.press("Escape")
        page.wait_for_function("() => !document.getElementById('ac').classList.contains('open')", timeout=3_000)
        if tap:
            page.tap("#go")
        else:
            page.click("#go")
    elif via == "enter":
        page.focus("#addr")
        page.keyboard.press("Enter")
    wait_for_fast_map(page)


def wait_for_fast_map(page: Page):
    """Block until the first /compute map has visibly rendered.

    A populated neighborhood list is the durable completion signal. ``#dest`` must retain a
    non-empty selected-place label; RAPTOR intentionally does not expose the legacy timing text.
    """
    page.wait_for_function(
        """() => document.querySelector('#dest').textContent.trim().length > 0 &&
          document.querySelectorAll('#list .nb').length > 0""",
        timeout=COMPUTE_TIMEOUT,
    )


def map_box(page: Page):
    return page.eval_on_selector(
        "#map", "el => { const r = el.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height}; }"
    )


def find_colored_cell_hover(page: Page):
    """Move the mouse across a grid of map pixels until a Leaflet hover tooltip
    (.leaflet-tooltip.tt, the desktop breakdown) opens over a colored cell.
    Returns (x, y) of the hit, or None if nothing was found.

    Desktop only (touch devices skip hover by design — see index.html onEachFeature)."""
    box = map_box(page)
    x0, y0 = int(box["x"]), int(box["y"])
    w, h = int(box["w"]), int(box["h"])
    # Sweep the central band where SF cells live; step coarse-to-fine.
    for x in range(x0 + 220, x0 + min(w - 200, 1000), 36):
        for y in range(y0 + 130, y0 + min(h - 120, 700), 36):
            page.mouse.move(x, y)
            if page.query_selector(".leaflet-tooltip.tt"):
                return (x, y)
    return None


def find_colored_cell_tap(page: Page):
    """Tap across the map until the touch-first commute preview opens over a colored cell.

    Touch intentionally has no hover or tiny Leaflet popup: the first tap opens ``#touchpeek``
    with a lightweight cached itinerary, and its explicit Inspect action opens the full route
    sheet. A tap on ocean/empty map opens nothing. Returns the hit point or ``None``.
    """
    import time
    box = map_box(page)
    x0, y0 = int(box["x"]), int(box["y"])
    w, h = int(box["w"]), int(box["h"])
    for x in range(x0 + 50, x0 + w - 40, 28):
        for y in range(y0 + 150, y0 + h - 200, 28):
            page.touchscreen.tap(x, y)
            time.sleep(0.12)
            if page.query_selector("#touchpeek.open"):
                # Cell hit — let the lightweight preview fetch land.
                try:
                    page.wait_for_function(
                        """() => { const e=document.getElementById('peekbody');
                          return e && !e.textContent.includes('Loading commute'); }""",
                        timeout=4_000,
                    )
                    return (x, y)
                except Exception:
                    # Preview opened but failed to fill; let the caller report the precise state.
                    return (x, y)
    return None


@pytest.fixture
def console_errors(page: Page):
    """Collect console errors + page errors over the life of a test."""
    errors = []
    page.on("console", lambda m: errors.append(("console:" + m.type, m.text)) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(("pageerror", str(e))))
    return errors


# ---- viewport fixtures ------------------------------------------------------------------
@pytest.fixture
def desktop_context_args():
    return {
        "viewport": {"width": 1280, "height": 800},
        "accept_downloads": True,
        "permissions": ["clipboard-read", "clipboard-write"],
    }


# iPhone-ish: 390x844, touch, mobile UA so the responsive (<=719px) layer + TOUCH path engage.
IPHONE = {
    "viewport": {"width": 390, "height": 844},
    "has_touch": True,
    "is_mobile": True,
    "device_scale_factor": 3,
    "user_agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    ),
    "accept_downloads": True,
    "permissions": ["clipboard-read", "clipboard-write"],
}
