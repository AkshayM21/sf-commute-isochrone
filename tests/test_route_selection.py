import os
import sys


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from core.route_selection import SelectionOps, alt_dominates, prune_dominated_alts, select_diverse_alts


def _ops():
    def family(option, keys=None):
        return (keys or {}).get(id(option), option.get("family", "walk"))

    def legs(option):
        return option.get("legs", ())

    return SelectionOps(
        family_key=family,
        branch_key=lambda option, _family: option.get("branch", "only"),
        choice_bucket=lambda option: option.get("choice", option.get("line")),
        choice_equivalent=lambda left, right: left.get("choice", left.get("line"))
        == right.get("choice", right.get("line")),
        quality_rank=lambda option: (option["total"], option.get("walk", 0), option.get("line", "")),
        total=lambda option: option["total"],
        exact_seconds=lambda option: option.get("exact", option["total"] * 60),
        access_walk_min=lambda option: option.get("access", option.get("walk", 0)),
        transfers=lambda option: option.get("transfers", 0),
        final_walk_min=lambda option: option.get("final", option.get("walk", 0)),
        physical_walk_min=lambda option: option.get("walk", 0),
        fragility=lambda option: option.get("fragility", 0),
        transit_legs=legs,
        leg_name=lambda leg: leg.get("name", ""),
        service_meta=lambda leg: {"key": leg["service"], "name": leg["name"]},
        discover_family_keys=lambda options: {id(option): option.get("family", "walk") for option in options},
    )


def _route(line, *, family="A", branch="T", total=20, walk=3, choice=None, service=None):
    return {
        "line": line,
        "family": family,
        "branch": branch,
        "total": total,
        "walk": walk,
        "choice": choice or line,
        "legs": ([{"name": line, "service": service or line}] if line else []),
    }


def test_pareto_dominance_stays_within_family_and_branch():
    ops = _ops()
    better = _route("A", total=20, walk=3)
    worse = _route("A-detour", total=22, walk=5)
    assert alt_dominates(better, worse, {id(better): "A", id(worse): "A"}, ops=ops)
    assert not alt_dominates(better, _route("B", family="B"), {}, ops=ops)


def test_prune_collapses_equivalent_choice_and_keeps_faster_winner():
    ops = _ops()
    slow = _route("A", total=24, choice="same")
    fast = _route("A", total=21, choice="same")
    kept = prune_dominated_alts([slow, fast], (), {id(slow): "A", id(fast): "A"}, ops=ops)
    assert kept == [fast]


def test_select_reserves_distinct_families_before_extra_branches():
    ops = _ops()
    options = [
        _route("A-fast", family="A", branch="direct", total=20),
        _route("A-tail", family="A", branch="tail", total=21),
        _route("B", family="B", branch="direct", total=22),
    ]
    selected = select_diverse_alts(options, cap=2, ops=ops)
    assert {option["family"] for option in selected} == {"A", "B"}


def test_select_complete_family_adds_near_tie_sibling_without_new_family():
    ops = _ops()
    primary = _route("A-primary", family="A", branch="direct", total=20)
    sibling = _route("A-sibling", family="A", branch="tail", total=22)
    far_family = _route("B", family="B", branch="direct", total=30)
    selected = select_diverse_alts(
        [sibling, far_family], cap=1, primary=primary,
        complete_selected_families=True, ops=ops,
    )
    assert [option["line"] for option in selected] == ["A-sibling", "B"]
