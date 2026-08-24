import io
import zipfile

from core.transfer_rules import (
    PathwayEdge,
    StopKey,
    parse_transfer_rules,
)
from core.transfer_rules import _integer


def _feed(files, name="muni"):
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as z:
        for filename, contents in files.items():
            z.writestr(filename, contents)
    raw.seek(0)
    return zipfile.ZipFile(raw), name


def _parse(files, name="muni"):
    z, filename = _feed(files, name)
    try:
        return parse_transfer_rules(z, filename)
    finally:
        z.close()


def test_missing_optional_tables_are_valid_and_stop_ids_are_namespaced():
    rules = _parse({"stops.txt": "stop_id\nA\n"}, "Muni-2026")
    assert rules.feed == "Muni-2026"
    assert rules.rules == ()
    assert rules.route_scoped_rules == ()
    assert rules.pathway_edges == ()


def test_unconditional_minimum_and_prohibition_are_explicit_and_deduplicated():
    rules = _parse({
        "transfers.txt": (
            "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n"
            "A,B,2,45\n"
            "A,B,2,90\n"
            "A,B,0,not-a-number\n"
            "B,C,3,0\n"
            "B,C,3,30\n"
        )
    })
    a, b, c = StopKey("muni", "A"), StopKey("muni", "B"), StopKey("muni", "C")
    assert rules.min_transfer_seconds[(a, b)] == 90
    assert rules.prohibited_pairs == frozenset({(b, c)})
    assert [(r.from_stop, r.to_stop, r.min_transfer_seconds, r.prohibited)
            for r in rules.rules] == [(a, b, 90, False), (b, c, 0, True)]


def test_route_and_trip_scoped_rules_never_become_global():
    rules = _parse({
        "transfers.txt": (
            "from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id\n"
            "A,B,2,30,R1,R2,,\n"
            "A,B,2,45,R1,R2,,\n"
            "A,B,2,75,R3,R4,,\n"
            "A,B,3,0,RX,RY,,\n"
            "A,B,3,0,RX,RY,,\n"
            "A,B,3,0,,,T1,T2\n"
        )
    })
    assert rules.min_transfer_seconds == {}
    assert rules.prohibited_pairs == frozenset()
    assert len(rules.route_scoped_rules) == 4
    assert len(rules.scoped_prohibitions) == 2
    scoped_mins = {
        (r.from_route_id, r.to_route_id): r.min_transfer_seconds
        for r in rules.route_scoped_rules if r.min_transfer_seconds is not None
    }
    assert scoped_mins == {("R1", "R2"): 45, ("R3", "R4"): 75}
    assert {(r.from_route_id, r.to_route_id, r.from_trip_id, r.to_trip_id)
            for r in rules.scoped_prohibitions} == {
                ("RX", "RY", None, None), (None, None, "T1", "T2")
            }


def test_linked_trip_types_four_and_five_require_trip_scope_and_never_globalize():
    rules = _parse({
        "transfers.txt": (
            "from_stop_id,to_stop_id,transfer_type,min_transfer_time,from_route_id,to_route_id,from_trip_id,to_trip_id\n"
            "A,B,4,90,R1,R2,T1,T2\n"
            "A,B,5,120,R3,R4,T3,T4\n"
            "B,C,4,60,RX,RY,,T5\n"
            "C,D,5,60,R7,R8,,\n"
            "D,E,4,60,,, ,\n"
        )
    })
    assert rules.min_transfer_seconds == {}
    assert rules.prohibited_pairs == frozenset()
    assert [(rule.transfer_type, rule.from_trip_id, rule.to_trip_id,
             rule.min_transfer_seconds, rule.prohibited)
            for rule in rules.route_scoped_rules] == [
                ("4", "T1", "T2", None, False),
                ("5", "T3", "T4", None, False),
            ]


def test_pathways_emit_directed_edges_and_reverse_for_bidirectional_rows():
    rules = _parse({
        "pathways.txt": (
            "pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,length,traversal_time\n"
            "p1,A,B,1,1,20,40\n"
            "p2,B,C,2,0,30,55\n"
            "p3,C,D,3,1,70,\n"
        )
    }, "bart")
    a, b = StopKey("bart", "A"), StopKey("bart", "B")
    c, d = StopKey("bart", "C"), StopKey("bart", "D")
    assert rules.pathway_edges == (
        PathwayEdge("p1", a, b, 40, "1", 20.0, False),
        PathwayEdge("p1", b, a, 40, "1", 20.0, True),
        PathwayEdge("p2", b, c, 55, "2", 30.0, False),
        PathwayEdge("p3", c, d, None, "3", 70.0, False),
        PathwayEdge("p3", d, c, None, "3", 70.0, True),
    )
    assert rules.min_transfer_seconds == {}
    assert rules.rules == ()


def test_pathway_duplicates_choose_shortest_edge_but_global_pair_rule_is_strongest_constraint():
    rules = _parse({
        "pathways.txt": (
            "pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,length,traversal_time\n"
            "same,A,B,1,0,20,\n"
            "same,A,B,1,0,20,40\n"
            "same,A,B,1,0,20,25\n"
            "other,A,B,1,0,20,40\n"
        ),
        "transfers.txt": "from_stop_id,to_stop_id,transfer_type,min_transfer_time\nA,B,2,60\n",
    })
    a, b = StopKey("muni", "A"), StopKey("muni", "B")
    assert rules.pathway_edges[0].pathway_id == "other"
    assert rules.pathway_edges[1].pathway_id == "same"
    assert rules.pathway_edges[1].traversal_seconds == 25
    # The 40-second pathway is a physical edge cost.  The 60-second transfer
    # row is the fixed GTFS constraint; they are intentionally not combined.
    assert rules.min_transfer_seconds[(a, b)] == 60


def test_malformed_rows_and_negative_times_are_ignored_without_negative_output():
    rules = _parse({
        "transfers.txt": (
            "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n"
            "A,B,2,-1\n"
            "A,C,wat,15\n"
            ",D,2,15\n"
            "D,D,2,15\n"
            "E,F,3,-4\n"
            "G,H,0,10\n"
        ),
        "pathways.txt": (
            "pathway_id,from_stop_id,to_stop_id,pathway_mode,is_bidirectional,length,traversal_time\n"
            "neg,I,J,1,0,-10,-5\n"
            "flag,K,L,1,2,10,5\n"
            "ok,M,N,1,0,10,0\n"
        ),
    })
    assert all(r.min_transfer_seconds >= 0 for r in rules.rules)
    assert all(e.traversal_seconds is None or e.traversal_seconds >= 0
               for e in rules.pathway_edges)
    negative = next(e for e in rules.pathway_edges if e.pathway_id == "neg")
    assert negative.traversal_seconds is None
    assert negative.length_meters is None
    assert all(e.pathway_id != "flag" for e in rules.pathway_edges)
    assert rules.min_transfer_seconds == {
        (StopKey("muni", "G"), StopKey("muni", "H")): 10,
    }


def test_bom_headers_and_whitespace_are_tolerated():
    rules = _parse({
        "transfers.txt": (
            "\ufefffrom_stop_id, to_stop_id , transfer_type , min_transfer_time\n"
            " A , B , 2 , 12 \n"
        )
    })
    assert rules.min_transfer_seconds == {
        (StopKey("muni", "A"), StopKey("muni", "B")): 12
    }


def test_same_pair_global_prohibition_wins_while_minimum_remains_diagnostic():
    rules = _parse({
        "transfers.txt": (
            "from_stop_id,to_stop_id,transfer_type,min_transfer_time\n"
            "A,B,2,30\n"
            "A,B,3,0\n"
        )
    })
    a, b = StopKey("muni", "A"), StopKey("muni", "B")
    assert rules.prohibited_pairs == frozenset({(a, b)})
    assert rules.min_transfer_seconds[(a, b)] == 30
    assert rules.rules[0].prohibited is True
    assert rules.rules[0].min_transfer_seconds == 30


def test_integer_parser_rejects_overflow_without_escaping():
    assert _integer("9" * 4000) is None
