"""Shared core for the SF commute-isochrone tools.

One source of truth for paths, the commute model, GTFS feeds, the SF grid, and R5
network/routing setup — so the live server and the offline analyses can never drift
apart (different feed lists, dates, departure times, or route-name logic).

Submodules:
  config   paths + canonical commute model + gtfs_paths() + load_dotenv()
  geo      Nominatim geocoding (no heavy deps)
  feeds    GTFS helpers: load_routes, route_name, route_shapes, pick_service_date
  grid     SF grid: load_neighborhoods, build_grid, square_cells, attach_neighborhoods
  network  R5: build_network, travel_time_matrix, routing_template  (imports r5py)

`network` is the only submodule that imports r5py (and thus starts the JVM); the rest
stay light so importing config/feeds/grid for a quick task is cheap.
"""
