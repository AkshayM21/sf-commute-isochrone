# Graph-native RAPTOR

This repository serves a graph-native reverse range-RAPTOR engine. GTFS feeds are
parsed into compact route/pattern arrays, the OSM/DEM pedestrian graph supplies
access and transfer costs, and one cached reverse search produces the map,
breakdown, and route alternatives for a workplace.

## Runtime workflow

1. `scripts/core/raptor_build.py` parses the available GTFS feeds and publishes a
   validated, freshness-aware structural cache.
2. `scripts/build_walk_graph.py` builds a directed, slope-aware OSM pedestrian
   graph from the clipped extract and elevation raster.
3. `scripts/bake_walk_access.py` derives cell-to-stop access data directly from
   that graph and the current grid.
4. `scripts/core/raptor_engine.py` runs the reverse range search and its planned
   depart-after read-off. `scripts/core/raptor_journey*.py` reconstructs the
   selected path from the same cached tree.
5. `scripts/core/server_raptor.py` exposes one cached tree to `/compute`,
   `/compute_exact`, `/itinerary`, variance, and route-family endpoints.

The production path has one source of truth: the transit structures, access graph,
transfer rules, and route reconstruction all use the same feed freshness metadata.
There is no alternate routing runtime or exporter workflow.

## Semantics and invariants

- Depart-after is the served scheduled semantic. The painted minute is anchored at
  the actual boarding-window grid point, so a visible wait is never charged twice.
- Arrive-by remains available as a graph-engine semantic for callers that need it;
  both semantics share the same route structures and walking graph.
- A route breakdown and its map value come from one journey tree. Their leg totals
  must agree exactly, and route geometry is derived from the same stop/transfer
  back-pointers.
- Walking speed scales access, transfer, egress, and walk-only legs consistently.
- Freshness checks use readable source size/mtime metadata. A changed feed or graph
  input invalidates the corresponding cache and rebuilds it.

## Historical benchmark record

The following conclusions are retained from the pre-purge local benchmark record.
They describe one historical feed snapshot and machine, not a release guarantee:

- Full-grid route computation was measured in roughly 105–130 ms for a fresh
  workplace after warm-up.
- The custom reverse search removed the old per-cell heavy-job bottleneck and made
  map, hover, and route inspection agree within roughly 1–2 minutes in that record.
- The walk graph's grade-aware costs intentionally diverged on steep streets while
  preserving the flat-distance mechanics used during the original calibration.

Refresh the feeds, rebuild the graph and access artifacts, and run the current test
suite before making a new performance claim. See `README.md` and `scripts/setup.sh`
for the only supported build and run workflow.

## Development checks

```bash
.venv/bin/python -m pytest tests/ --ignore=tests/e2e -q
node --test tests/test_viz.mjs
```

The graph-native tests use small structural fixtures where possible, so they do not
require downloaded production data. Data-backed checks skip with an actionable bake
message when the local artifacts are absent.
