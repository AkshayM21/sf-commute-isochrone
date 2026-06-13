"""R5 network construction and routing-request templates.

Importing this imports r5py and starts the in-process JVM, so keep it out of the import
path of light/offline tasks.
"""
import datetime as dt
from r5py import TransportNetwork, TravelTimeMatrix, TransportMode
from r5py.r5.regional_task import RegionalTask
from . import config

MAX_INT32 = (2 ** 31) - 1                       # R5's "unreachable" sentinel
MODES = [TransportMode.TRANSIT, TransportMode.WALK]
DEFAULT_MAX_RIDES = 8                            # R5's default cap on public-transport rides
                                                # (rides = transfers + 1); unset == today's behavior


def build_network(gtfs_paths):
    """Warm R5 TransportNetwork from the SF OSM extract + the given GTFS feeds."""
    return TransportNetwork(str(config.osm_path()), [str(p) for p in gtfs_paths])


def _common(dep, max_rides=DEFAULT_MAX_RIDES, window=None):
    """The shared routing parameters (the canonical commute model) for both the matrix
    and per-origin RegionalTask paths, so they can never drift. ``max_rides`` caps the
    number of public-transport rides (rides = transfers + 1); the default equals R5's own
    default so an unset value reproduces today's behavior exactly. ``window`` (a timedelta)
    overrides the canonical departure window — None keeps config.window(); diagnostic probes
    pass timedelta(minutes=1) for single-departure routing."""
    return dict(
        departure=dep,
        departure_time_window=config.window() if window is None else window,
        max_time=dt.timedelta(minutes=config.MAX_MIN),
        transport_modes=MODES,
        percentiles=config.PERCENTILES,
        speed_walking=config.WALK_KMH,
        max_public_transport_rides=max_rides)


def travel_time_matrix(net, origins, destinations, dep, *, snap_to_network=True,
                       max_rides=DEFAULT_MAX_RIDES):
    """Door-to-door travel-time matrix (best-case + realistic percentiles)."""
    return TravelTimeMatrix(net, origins=origins, destinations=destinations,
                            snap_to_network=snap_to_network, **_common(dep, max_rides))


def walk_time_matrix(net, origins, destinations, dep, max_min, *, snap_to_network=True):
    """WALK-only door-to-door matrix (minutes) — used to derive the RAPTOR engine's
    per-workplace egress (W->stops) and pure-walk (W->cells) at the canonical walk speed."""
    return TravelTimeMatrix(net, origins=origins, destinations=destinations,
                            departure=dep, transport_modes=[TransportMode.WALK],
                            speed_walking=config.WALK_KMH,
                            max_time=dt.timedelta(minutes=max_min),
                            snap_to_network=snap_to_network)


def routing_template(net, dest, dep, *, paths=False, n_paths=8,
                     max_rides=DEFAULT_MAX_RIDES, window=None):
    """A reusable RegionalTask routing TO `dest` (callers set .origin per request, then
    clone with copy.copy for thread-safe per-origin routing). paths=True records detailed
    leg paths for itinerary breakdowns. ``max_rides`` caps public-transport rides;
    ``window`` overrides the departure window (see ``_common``). RegionalTask.__init__
    already runs the destinations setter (building the FreeFormPointSet before the
    transport_modes linkage prewarm) — do NOT re-assign t.destinations afterwards, it
    would rebuild the point set AFTER the prewarm."""
    t = RegionalTask(net, origin=None, destinations=dest,
                     **_common(dep, max_rides, window=window))
    if paths:
        t._regional_task.includePathResults = True
        t._regional_task.nPathsPerTarget = n_paths
    return t
