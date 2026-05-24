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


def build_network(gtfs_paths):
    """Warm R5 TransportNetwork from the SF OSM extract + the given GTFS feeds."""
    return TransportNetwork(str(config.osm_path()), [str(p) for p in gtfs_paths])


def _common(dep):
    """The shared routing parameters (the canonical commute model) for both the matrix
    and per-origin RegionalTask paths, so they can never drift."""
    return dict(
        departure=dep,
        departure_time_window=config.window(),
        max_time=dt.timedelta(minutes=config.MAX_MIN),
        transport_modes=MODES,
        percentiles=config.PERCENTILES,
        speed_walking=config.WALK_KMH)


def travel_time_matrix(net, origins, destinations, dep, *, snap_to_network=True):
    """Door-to-door travel-time matrix (best-case + realistic percentiles)."""
    return TravelTimeMatrix(net, origins=origins, destinations=destinations,
                            snap_to_network=snap_to_network, **_common(dep))


def routing_template(net, dest, dep, *, paths=False, n_paths=8):
    """A reusable RegionalTask routing TO `dest` (callers set .origin per request, then
    clone with copy.copy for thread-safe per-origin routing). paths=True records detailed
    leg paths for itinerary breakdowns."""
    t = RegionalTask(net, origin=None, destinations=dest, **_common(dep))
    t.destinations = dest
    if paths:
        t._regional_task.includePathResults = True
        t._regional_task.nPathsPerTarget = n_paths
    return t
