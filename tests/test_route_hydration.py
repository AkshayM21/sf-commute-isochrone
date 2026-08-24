"""Direct contracts for the pure route hydration seam."""

import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from core import route_hydration  # noqa: E402


def _journey(label="A", total=12):
    return {
        "total": total,
        "xfers": 0,
        "legs": [{"mode": "transit", "line": label, "min": total}],
        "geom": [{"mode": "transit", "line": label, "pts": [[1.0, 2.0], [1.1, 2.1]]}],
    }


def test_journey_payload_is_independent_and_keeps_schema():
    source = _journey()
    result = route_hydration.journey_payload(source)
    assert result["total"] == source["total"]
    assert result["legs"] == source["geom"]
    assert "geom" not in result
    assert result is not source
    result["legs"][0]["line"] = "changed"
    result["legs"][0]["pts"][0][0] = 99
    assert source["geom"][0]["line"] == "A"
    assert source["geom"][0]["pts"][0][0] == 1.0


def test_hydrate_planned_proxy_does_not_mutate_live_selection_handle():
    branch = {"stop": 4, "raw": ["raw"], "home": 7, "jt": object(), "total": 13}
    option = {"line": "A", "_needs_hydration": True, "_branch": branch}
    calls = []

    def formatter(*args, **kwargs):
        calls.append((args, kwargs))
        return _journey("A", 13)

    result = route_hydration.hydrate_planned_proxy(
        option, formatter=formatter, transit_predicate=lambda legs: bool(legs),
        cell=3, provider="provider")
    assert result["typical"]["total"] == 13
    assert result["best"]["total"] == 13
    assert result["typical"]["legs"][0]["pts"] == [[1.0, 2.0], [1.1, 2.1]]
    assert result["best"]["legs"][0]["pts"] == [[1.0, 2.0], [1.1, 2.1]]
    assert result["_branch"] is branch
    assert option == {"line": "A", "_needs_hydration": True, "_branch": branch}
    assert calls[0][0][:2] == (3, 4)
    result["typical"]["legs"][0]["line"] = "changed"
    assert option.get("typical") is None


def test_hydrate_planned_proxy_drops_invalid_or_walk_only_route():
    option = {"line": "A", "_needs_hydration": True, "_branch": {"stop": 1}}
    assert route_hydration.hydrate_planned_proxy(
        option, formatter=lambda *args, **kwargs: None,
        transit_predicate=lambda legs: True, cell=0, provider=None) is None
    assert route_hydration.hydrate_planned_proxy(
        option, formatter=lambda *args, **kwargs: _journey(),
        transit_predicate=lambda legs: False, cell=0, provider=None) is None


def test_alternative_payload_uses_traced_label_and_copies_geometry():
    typical = _journey("A", 10)
    best = _journey("A", 9)
    option, signature = route_hydration.alternative_payload(
        "chip", typical, best, via_stop=8,
        route_label=lambda geom: geom[0]["line"],
        trace_signature=lambda geom: tuple((x["line"],) for x in geom),
        transit_predicate=lambda legs: bool(legs))
    assert option["line"] == "A"
    assert option["chip_line"] == "chip"
    assert option["via_stop"] == 8
    assert signature == ((("A",),), (("A",),))
    assert option["typical"]["legs"][0]["pts"] == [[1.0, 2.0], [1.1, 2.1]]
    assert option["best"]["legs"][0]["pts"] == [[1.0, 2.0], [1.1, 2.1]]
    option["typical"]["legs"][0]["line"] = "changed"
    option["best"]["legs"][0]["pts"][0][0] = 99
    assert typical["geom"][0]["line"] == "A"
    assert best["geom"][0]["pts"][0][0] == 1.0


def test_reliability_projection_is_absent_safe_and_non_aliasing():
    primary = {"line": "A"}
    alternatives = [{"line": "B"}, {"line": "C"}]
    p, alts = route_hydration.apply_reliability(
        primary, alternatives, {"prim": (17, 4), "alts": [(18, 5), None]}, metric="r")
    assert p["real"] == 17 and p["frag"] == 4
    assert alts[0]["real"] == 18 and alts[0]["frag"] == 5
    assert "frag" not in alts[1]
    assert "real" not in primary and "frag" not in alternatives[0]
    p2, a2 = route_hydration.apply_reliability(primary, alternatives, None, metric="b")
    assert p2 == primary and a2 == alternatives
    try:
        route_hydration.apply_reliability(primary, alternatives, None, metric="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid reliability metric must fail closed")


def test_departafter_reliability_clamps_and_aligns_missing_values():
    primary = {"line": "A"}
    alternatives = [{"line": "B"}, {"line": "C"}]
    prim_frag, frags, p, alts = route_hydration.apply_departafter_reliability(
        primary, alternatives, {"prim_frag": -3, "alt_frags": [6]}, metric="r")
    assert prim_frag == 0
    assert frags == [6, None]
    assert p["frag"] == 0 and alts[0]["frag"] == 6
    assert "frag" not in primary and "frag" not in alternatives[1]
