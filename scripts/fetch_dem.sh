#!/usr/bin/env bash
# Fetch the SF elevation DEM (USGS 3DEP, ~10 m, free, no key) for the hill-aware walk router.
# Writes data/dem_sf.tif (EPSG:4326, float32 metres). Gitignored like the other data files.
# Re-run only if the bbox/grid changes; build_walk_graph.py folds the DEM into walk_graph.npz.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
# bbox covers the whole SF neighborhoods grid (lon_min,lat_min,lon_max,lat_max).
# Bottom is 37.685, not 37.69: walk-graph nodes reach lat 37.6895, and build_walk_graph.py
# now hard-fails on nodes outside the DEM bounds (don't rely on server-side pixel snapping).
BBOX="-122.55,37.685,-122.35,37.84"
Q="bbox=${BBOX}&bboxSR=4326&imageSR=4326&size=1800,1700&format=tiff&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image"
echo "fetching SF DEM -> data/dem_sf.tif"
curl -s "${URL}?${Q}" -o data/dem_sf.tif -w 'http %{http_code}  %{size_download} bytes\n'
echo "done."
