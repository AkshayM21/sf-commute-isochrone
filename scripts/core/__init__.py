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
  raptor          reverse range-RAPTOR: numpy reference kernels + numba dispatch + MC
  raptor_build    GTFS -> flat CSR RAPTOR structures, disk-cached (load_or_build)
  raptor_numba    nogil njit port of the raptor hot path (LOCKSTEP with raptor.py)
  raptor_engine   graph-native engine API: depart-after p5/p50 + arrive-by + montecarlo
  raptor_journey  back-pointer journey reconstruction (hover breakdown, color-by-line)
  walk            hill-aware graph-native walk router over data/walk_graph.npz (scipy Dijkstra)

The production path is graph-native from feed parsing through walking, RAPTOR, and route
reconstruction. pandas (~44 MB) is imported lazily (module-level `pd = None` + a `_pd()`
accessor in feeds/raptor_build) so the lean server boot never pulls it; don't add top-level
pandas/geopandas imports to the boot/engine path.
"""
