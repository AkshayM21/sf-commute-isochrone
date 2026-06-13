"""Shared core for the SF commute-isochrone tools.

One source of truth for paths, the commute model, GTFS feeds, the SF grid, routing
engines, and geocoding — so the live server and the offline analyses can never drift
apart (different feed lists, dates, departure times, or route-name logic).

Submodules:
  config          paths + canonical commute model + gtfs_paths() + load_dotenv()
  geo             address geocode()/autocomplete(); provider via GEOCODER env
                  (default Geoapify when GEOAPIFY_KEY set, else Photon;
                  Nominatim only via explicit GEOCODER=nominatim)
  feeds           GTFS helpers: load_routes, route_name, route_shapes,
                  active_service_ids, pick_service_date
  grid            SF grid: load_neighborhoods, build_grid, square_cells,
                  attach_neighborhoods
  network         R5: build_network, travel_time_matrix, routing_template (imports r5py)
  raptor          reverse range-RAPTOR: numpy reference kernels + numba dispatch + MC
  raptor_build    GTFS -> flat CSR RAPTOR structures, disk-cached (load_or_build)
  raptor_numba    nogil njit port of the raptor hot path (LOCKSTEP with raptor.py)
  raptor_engine   JVM-free engine API: depart-after p5/p50 + arrive-by + montecarlo
  raptor_journey  back-pointer journey reconstruction (hover breakdown, color-by-line)
  raptor_golden   golden-oracle alignment helpers (id-keyed purewalk_aligned; numpy-only)
  r5_extract      shared R5 travel-time-matrix parsers (tt_col, egress/purewalk extract;
                  pandas lazy, NO r5py — used by both server legacy path and the oracle)
  walk            hill-aware JVM-free walk router over data/walk_graph.npz (scipy Dijkstra)

`network` is the only submodule that imports r5py (and thus starts the JVM); everything
else stays JVM-free — never import r5py from core/raptor* or core/walk. Convention:
pandas (~44 MB) is imported LAZILY (module-level `pd = None` + a `_pd()` accessor in
feeds/raptor_build) so the lean server boot never pulls it; don't add top-level
pandas/geopandas imports to the JVM-free boot/engine path.
"""
