"""Unit tests for the dependency-free production smoke contract."""

from __future__ import annotations

import json
from urllib.parse import urlsplit, parse_qs

import pytest

from scripts import production_smoke as smoke


class _Response:
    def __init__(self, body, status=200):
        self.body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, limit=-1):
        return self.body if limit < 0 else self.body[:limit]

    def getcode(self):
        return self.status


def _fake_opener(responses, seen):
    def opener(request, timeout):
        url = request.full_url
        seen.append((url, timeout))
        path = urlsplit(url).path
        if path not in responses:
            raise AssertionError(f"unexpected request path: {path}")
        body, status = responses[path]
        return _Response(body, status)

    return opener


def _healthy_responses():
    module = (
        'export const BASE_TILES = {dark: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", '
        'light: "https://tile.openstreetmap.org/{z}/{x}/{y}.png"};'
    )
    compute = {"dest": [37.7955, -122.3937], "cells": {"cell-a": [18, 21]}}
    itinerary = {"total": 21, "legs": [{"mode": "transit", "line": "1"}]}
    return {
        "/livez": (json.dumps({"ok": True}), 200),
        "/readyz": (json.dumps({"ok": True}), 200),
        "/static/map-renderer.mjs": (module, 200),
        "/compute": (json.dumps(compute), 200),
        "/itinerary": (json.dumps(itinerary), 200),
    }


def test_run_checks_health_module_and_one_reachable_itinerary():
    seen = []
    smoke.run("https://example.test", timeout=3, opener=_fake_opener(_healthy_responses(), seen))
    assert [urlsplit(url).path for url, _ in seen] == [
        "/livez", "/readyz", "/static/map-renderer.mjs", "/compute", "/itinerary"
    ]
    itinerary_query = parse_qs(urlsplit(seen[-1][0]).query)
    assert itinerary_query["id"] == ["cell-a"]
    assert all(timeout == 3 for _, timeout in seen)


def test_watermarked_or_carto_module_fails_without_printing_body(capsys):
    responses = _healthy_responses()
    responses["/static/map-renderer.mjs"] = (
        "https://cartodb-basemaps-a.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png "
        "API KEY REQUIRED",
        200,
    )
    with pytest.raises(smoke.SmokeFailure, match="key-required marker"):
        smoke.run("https://example.test", opener=_fake_opener(responses, []))
    assert "API KEY REQUIRED" not in capsys.readouterr().err


def test_missing_keyless_endpoint_fails():
    responses = _healthy_responses()
    responses["/static/map-renderer.mjs"] = ("export const BASE_TILES = {};", 200)
    with pytest.raises(smoke.SmokeFailure, match="approved keyless OpenStreetMap"):
        smoke.run("https://example.test", opener=_fake_opener(responses, []))


def test_unreachable_compute_does_not_guess_a_cell():
    responses = _healthy_responses()
    responses["/compute"] = (json.dumps({"dest": [37.7955, -122.3937], "cells": {"a": [None, None]}}), 200)
    with pytest.raises(smoke.SmokeFailure, match="no reachable cell"):
        smoke.run("https://example.test", opener=_fake_opener(responses, []))


def test_failures_redact_response_body_and_do_not_fetch_tiles(capsys):
    responses = _healthy_responses()
    responses["/readyz"] = ("private feed path /srv/secret", 503)
    with pytest.raises(smoke.SmokeFailure, match="HTTP 503"):
        smoke.run("https://example.test", opener=_fake_opener(responses, []))
    captured = capsys.readouterr()
    assert "/srv/secret" not in captured.out + captured.err
