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

OUT="data/dem_sf.tif"
TMP="$OUT.part"
# Never leave a partial download behind (mv below makes this a no-op on success).
trap 'rm -f "$TMP" 2>/dev/null || true' EXIT

echo "fetching SF DEM -> $OUT"
# -f: fail on HTTP errors instead of saving the error body; retries for flaky USGS.
# Download to a .part and rename only after validation, so an interrupted or bogus
# download can never masquerade as a present-and-valid DEM (same pattern as fetch_511.sh).
if ! curl -fsS --retry 4 --retry-delay 3 --retry-all-errors \
     --connect-timeout 25 --max-time 600 \
     -o "$TMP" -w 'http %{http_code}  %{size_download} bytes\n' \
     "${URL}?${Q}"; then
  echo "ERROR: DEM download failed (curl error). $OUT was NOT written." >&2
  exit 1
fi

# The ArcGIS ImageServer returns JSON error bodies WITH HTTP 200, which -f can't catch.
# A real GeoTIFF starts with the TIFF magic: II*\0 (little-endian) or MM\0* (big-endian).
magic="$(head -c 4 "$TMP" | od -An -tx1 | tr -d ' \n')"
if [ "$magic" != "49492a00" ] && [ "$magic" != "4d4d002a" ]; then
  echo "ERROR: download is not a TIFF (magic: ${magic:-empty}). First bytes:" >&2
  head -c 300 "$TMP" >&2 || true; echo >&2
  echo "$OUT was NOT written." >&2
  exit 1
fi

mv "$TMP" "$OUT"
echo "done."
