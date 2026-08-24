"""Behavioral browser coverage for the supported-area replacement/startup contract.

The real app shell runs in Chromium while coordinate responses are intercepted.  This keeps the
tests deterministic and exercises the actual localStorage, permalink, DOM accessibility, race,
and transactional map-state behavior without asking the routing engine to model invalid places.
"""
import json
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import expect

from conftest import BASE_URL, fresh_load


VALID = {"lat": 37.7714154, "lon": -122.4030885, "label": "650 Townsend St"}
SECOND = {"lat": 37.7946, "lon": -122.3950, "label": "Ferry Building"}
OUTSIDE = {"lat": 37.8044, "lon": -122.2712, "label": "Oakland"}
OUTSIDE_BODY = {
    "error": "outside_supported_area",
    "detail": "Location is outside the supported San Francisco walking area.",
}


@pytest.fixture
def page(new_context):
    context = new_context(viewport={"width": 1280, "height": 800})
    return context.new_page()


def _lat(route):
    return float(parse_qs(urlparse(route.request.url).query)["lat"][0])


def _success(route, point):
    route.fulfill(
        status=200,
        content_type="application/json",
        body=json.dumps({"cells": {"0": 12}, "dest": [point["lat"], point["lon"]], "ms": 1}),
    )


def _outside(route):
    route.fulfill(status=422, content_type="application/json", body=json.dumps(OUTSIDE_BODY))


def _mock_variance(page):
    page.route(
        "**/variance?*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"realistic": {}, "variance": {"0": {"frag": 0}}}),
        ),
    )


def _start_workplace(page, point):
    page.evaluate(
        "point => { void window.__SFCI_E2E__.setWorkplace(point.lat, point.lon, point.label); }",
        point,
    )


def _wait_for_destination(page, label):
    expect(page.locator("#dest")).to_have_text(label)


def test_fresh_unsupported_workplace_returns_actionable_onboarding(page):
    _mock_variance(page)
    page.route("**/compute?*", _outside)
    fresh_load(page)

    _start_workplace(page, OUTSIDE)

    expect(page.locator("#onboarding-error")).to_be_visible()
    expect(page.locator("#onboarding-error")).to_contain_text("supported San Francisco walking area")
    expect(page.locator("#obaddr")).to_have_attribute("aria-invalid", "true")
    expect(page.locator("#obaddr")).to_have_attribute("aria-describedby", "onboarding-error")
    assert page.evaluate("localStorage.getItem('wp_v1')") is None
    assert not page.locator("#startupretry").is_visible()


def test_replacement_422_preserves_current_destination_and_saved_workplace(page):
    attempts = []

    def compute(route):
        attempts.append(_lat(route))
        if abs(attempts[-1] - VALID["lat"]) < 1e-6:
            _success(route, VALID)
        else:
            _outside(route)

    _mock_variance(page)
    page.route("**/compute?*", compute)
    fresh_load(page)
    _start_workplace(page, VALID)
    _wait_for_destination(page, VALID["label"])
    saved_before = page.evaluate("localStorage.getItem('wp_v1')")

    _start_workplace(page, OUTSIDE)
    expect(page.locator("#workplace-error")).to_be_visible()
    expect(page.locator("#addr")).to_have_attribute("aria-invalid", "true")
    expect(page.locator("#addr")).to_have_attribute("aria-describedby", "workplace-error")
    _wait_for_destination(page, VALID["label"])
    assert page.evaluate("localStorage.getItem('wp_v1')") == saved_before
    assert len(attempts) == 2


def test_unsupported_saved_workplace_is_removed_and_not_retried(page):
    attempts = []
    page.add_init_script(
        f"localStorage.setItem('wp_v1', JSON.stringify({json.dumps(OUTSIDE)}));"
    )
    page.route("**/compute?*", lambda route: (attempts.append(route.request.url), _outside(route)))
    page.goto(BASE_URL, wait_until="domcontentloaded")

    expect(page.locator("#onboarding-error")).to_be_visible()
    assert page.evaluate("localStorage.getItem('wp_v1')") is None
    assert len(attempts) == 1
    assert not page.locator("#startupretry").is_visible()


def test_unsupported_permalink_is_cleaned_then_saved_workplace_restores_once(page):
    attempts = []
    page.add_init_script(
        f"localStorage.setItem('wp_v1', JSON.stringify({json.dumps(VALID)}));"
    )

    def compute(route):
        attempts.append(_lat(route))
        if abs(attempts[-1] - OUTSIDE["lat"]) < 1e-6:
            _outside(route)
        else:
            _success(route, VALID)

    _mock_variance(page)
    page.route("**/compute?*", compute)
    bad_hash = f"#wp={OUTSIDE['lat']},{OUTSIDE['lon']},Oakland&sp=med"
    page.goto(BASE_URL + "/" + bad_hash, wait_until="domcontentloaded")

    _wait_for_destination(page, VALID["label"])
    page.wait_for_timeout(400)  # allow the accepted saved destination's debounced syncHash
    assert attempts == [OUTSIDE["lat"], VALID["lat"]]
    assert str(OUTSIDE["lat"]) not in page.url
    assert json.loads(page.evaluate("localStorage.getItem('wp_v1')"))["label"] == VALID["label"]
    assert not page.locator("#onboarding-error").is_visible()


def test_stale_unsupported_response_cannot_overwrite_newer_success(page):
    held = []
    attempts = []

    def compute(route):
        attempts.append(_lat(route))
        if abs(attempts[-1] - OUTSIDE["lat"]) < 1e-6:
            held.append(route)
        else:
            _success(route, SECOND)

    _mock_variance(page)
    page.route("**/compute?*", compute)
    fresh_load(page)
    _start_workplace(page, OUTSIDE)
    page.wait_for_function("() => document.getElementById('busy').style.display === 'block'")
    _start_workplace(page, SECOND)
    _wait_for_destination(page, SECOND["label"])

    assert len(held) == 1
    _outside(held.pop())
    page.wait_for_timeout(100)
    _wait_for_destination(page, SECOND["label"])
    assert not page.locator("#workplace-error").is_visible()
    assert not page.locator("#onboarding-error").is_visible()
    assert json.loads(page.evaluate("localStorage.getItem('wp_v1')"))["label"] == SECOND["label"]


@pytest.mark.parametrize(
    ("status", "content_type", "body"),
    [
        (400, "text/plain", "not json"),
        (200, "application/json", "{malformed"),
    ],
)
def test_non_json_and_malformed_compute_failures_keep_generic_startup_recovery(
    page, status, content_type, body
):
    page.add_init_script(
        f"localStorage.setItem('wp_v1', JSON.stringify({json.dumps(VALID)}));"
    )
    page.route(
        "**/compute?*",
        lambda route: route.fulfill(status=status, content_type=content_type, body=body),
    )
    page.goto(BASE_URL, wait_until="domcontentloaded")

    expect(page.locator("#startup-copy")).to_contain_text("couldn’t open this commute")
    assert page.locator("#startupretry").is_visible()
    assert json.loads(page.evaluate("localStorage.getItem('wp_v1')"))["label"] == VALID["label"]
    assert not page.locator("#onboarding-error").is_visible()
