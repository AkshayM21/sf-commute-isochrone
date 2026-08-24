"""Build the workplace-independent JSON map bundle.

The live server and the staged data refresh use the same builder.  It deliberately
does not carry any cache identity beyond the direct source metadata the runtime already
validates (feed sizes/mtimes, grid resolution, and neighborhood size/mtime).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from . import config


def source_metadata(path: str | Path) -> dict[str, Any] | None:
    try:
        p = Path(path)
        st = p.stat()
    except OSError:
        return None
    return {
        "name": p.name,
        "size": int(st.st_size),
        "mtime_ns": config.portable_mtime_ns(st),
    }


def build_static_bundle(output: str | Path, gtfs, *, grid_m: int, service_date=None,
                        source_mtimes=None) -> Mapping[str, Any]:
    """Build and atomically publish ``server_static.json`` under the active data root.

    ``gtfs`` is an iterable of feed paths.  Imports of geopandas/grid/feeds stay inside
    this build-only seam, preserving the lean server boot when a valid bundle exists.
    """
    import geopandas as gpd
    from . import feeds, grid

    gtfs = list(gtfs)
    if source_mtimes is None:
        from . import raptor_build
        source_mtimes = raptor_build._source_mtimes(gtfs)
    if service_date is None:
        service_date = feeds.pick_service_date(gtfs)

    neighborhoods = grid.load_neighborhoods()
    points = grid.build_grid(neighborhoods, int(grid_m))[['id', 'geometry']]
    origin_ll = {str(r.id): (float(r.geometry.y), float(r.geometry.x))
                 for r in points.itertuples()}
    cells = gpd.GeoDataFrame({'id': points['id'].values},
                             geometry=grid.square_cells(points, int(grid_m)).values,
                             crs=config.WGS)
    cells = grid.attach_neighborhoods(cells, neighborhoods)
    cells_geojson = json.loads(cells.to_json())
    for feature in cells_geojson['features']:
        feature['properties'] = {
            'id': feature['properties']['id'],
            'n': feature['properties'].get('name'),
        }

    grid_meta = source_metadata(config.neigh_path())
    if grid_meta is None:
        raise FileNotFoundError(f'grid source disappeared: {config.neigh_path()}')
    payload = {
        'source_mtimes': [list(v) for v in source_mtimes],
        'grid_m': int(grid_m),
        'grid_source_name': grid_meta['name'],
        'grid_source_size': grid_meta['size'],
        'grid_source_mtime_ns': grid_meta['mtime_ns'],
        'svc_date': service_date.strftime('%Y%m%d'),
        'origin_ll': {k: [v[0], v[1]] for k, v in origin_ll.items()},
        'cells': cells_geojson,
        'lines': feeds.route_shapes(gtfs),
    }

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp',
                                     dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, separators=(',', ':'))
            handle.flush()
            os.fsync(handle.fileno())
        # Parse the closed temporary file before publication.  This catches an interrupted
        # serialization without ever replacing a previously valid bundle.
        json.loads(Path(temporary).read_text(encoding='utf-8'))
        os.replace(temporary, destination)
    except Exception:
        try:
            Path(temporary).unlink()
        except OSError:
            pass
        raise
    return payload
