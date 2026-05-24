#!/usr/bin/env bash
# One-command setup for the SF commute isochrone tool.
#
# Installs the Python environment and downloads every input dataset the pipeline
# needs into data/ (all gitignored). Safe to re-run: existing venv and data
# files are left alone; only missing pieces are created/downloaded.
#
#   bash scripts/setup.sh
#
# Prerequisites (checked below): Java 21, uv, osmium-tool, and a free 511.org
# API token. See README.md for details.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

# SF bounding box (W,S,E,N) used to clip the NorCal extract down to the city.
SF_BBOX="-122.5250,37.7000,-122.3500,37.8350"

say()  { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  OK  %s\n' "$*"; }
err()  { printf '  ERROR: %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
say "Checking prerequisites"

missing=0

# Java 21 (r5py needs a JDK 21 runtime)
if command -v java >/dev/null 2>&1; then
  # `java -version` prints to stderr; capture both streams.
  jver="$(java -version 2>&1 | head -n1)"
  jmajor="$(printf '%s' "$jver" | sed -nE 's/.*version "([0-9]+).*/\1/p')"
  if [ "${jmajor:-0}" -ge 21 ] 2>/dev/null; then
    ok "java ($jver)"
  else
    err "Java 21+ required, found: $jver"
    err "  Install a JDK 21, e.g.  brew install openjdk@21"
    missing=1
  fi
else
  err "java not found. Install a JDK 21, e.g.  brew install openjdk@21"
  missing=1
fi

# uv (Python env + package manager)
if command -v uv >/dev/null 2>&1; then
  ok "uv ($(uv --version 2>/dev/null || echo present))"
else
  err "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh"
  err "  (or:  brew install uv)"
  missing=1
fi

# osmium (clips the OSM extract to SF)
if command -v osmium >/dev/null 2>&1; then
  ok "osmium ($(osmium --version 2>/dev/null | head -n1 || echo present))"
else
  err "osmium not found. Install it:  brew install osmium-tool"
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  err "Resolve the prerequisites above and re-run scripts/setup.sh."
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------
say "Python environment ($VENV)"
if [ -x "$PY" ]; then
  ok "venv already exists"
else
  echo "  creating venv with uv (Python 3.12)..."
  uv venv --python 3.12 "$VENV"
  ok "venv created"
fi

echo "  installing/updating Python dependencies..."
uv pip install --python "$PY" \
  r5py geopandas folium mapclassify shapely requests pandas numpy branca contextily flask
ok "dependencies installed"

# ---------------------------------------------------------------------------
# 3. Load .env and require the 511 token
# ---------------------------------------------------------------------------
say "511.org API token"
if [ -f "$ROOT/.env" ]; then
  # Export simple KEY=VALUE lines from .env without executing the file.
  set -a
  # shellcheck disable=SC1090
  . "$ROOT/.env"
  set +a
  ok "loaded .env"
else
  echo "  no .env found (copy .env.example to .env to configure the destination)"
fi

if [ -z "${API511_TOKEN:-}" ]; then
  err "API511_TOKEN is not set."
  err "  Get a free token at https://511.org/open-data/token and either:"
  err "    - add  API511_TOKEN=your-token  to $ROOT/.env, or"
  err "    - export API511_TOKEN=your-token  before running this script."
  exit 2
fi
ok "API511_TOKEN present"

# ---------------------------------------------------------------------------
# 4. Data downloads (idempotent)
# ---------------------------------------------------------------------------
mkdir -p "$DATA"

# Validate a zip contains a GTFS routes.txt (catches HTML/error bodies).
valid_gtfs_zip () {
  local f="$1"
  [ -s "$f" ] && unzip -l "$f" >/dev/null 2>&1 && unzip -l "$f" | grep -q "routes.txt"
}

# Validate a file is non-empty JSON/GeoJSON.
valid_geojson () {
  local f="$1"
  [ -s "$f" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if (d.get('features') or d.get('type')) else 1)" "$f" >/dev/null 2>&1
}

# --- OpenStreetMap (NorCal -> clip to SF) --------------------------------
say "OpenStreetMap (data/osm_sf.pbf)"
if [ -s "$DATA/osm_sf.pbf" ]; then
  ok "osm_sf.pbf already present"
else
  if [ ! -s "$DATA/norcal.osm.pbf" ]; then
    echo "  downloading Geofabrik NorCal extract (~640 MB)..."
    curl -fL --retry 4 --retry-delay 3 --retry-all-errors \
      --connect-timeout 25 --max-time 1800 \
      -o "$DATA/norcal.osm.pbf" \
      "https://download.geofabrik.de/north-america/us/california/norcal-latest.osm.pbf"
  else
    ok "norcal.osm.pbf already present (skipping download)"
  fi
  if [ ! -s "$DATA/norcal.osm.pbf" ]; then
    err "NorCal extract download failed or is empty."
    exit 1
  fi
  echo "  clipping NorCal to SF bbox ($SF_BBOX)..."
  osmium extract -b "$SF_BBOX" "$DATA/norcal.osm.pbf" -o "$DATA/osm_sf.pbf" --overwrite
  if [ ! -s "$DATA/osm_sf.pbf" ]; then
    err "osmium extract produced an empty osm_sf.pbf."
    exit 1
  fi
  ok "osm_sf.pbf created ($(ls -lh "$DATA/osm_sf.pbf" | awk '{print $5}'))"
fi

# --- BART GTFS -----------------------------------------------------------
say "BART GTFS (data/bart_gtfs.zip)"
if valid_gtfs_zip "$DATA/bart_gtfs.zip"; then
  ok "bart_gtfs.zip already present"
else
  echo "  downloading BART GTFS..."
  curl -fL --retry 4 --retry-delay 3 --retry-all-errors \
    --connect-timeout 25 --max-time 240 \
    -o "$DATA/bart_gtfs.zip" \
    "https://www.bart.gov/dev/schedules/google_transit.zip"
  if valid_gtfs_zip "$DATA/bart_gtfs.zip"; then
    ok "bart_gtfs.zip downloaded ($(ls -lh "$DATA/bart_gtfs.zip" | awk '{print $5}'))"
  else
    err "BART download is not a valid GTFS zip. First bytes:"
    head -c 200 "$DATA/bart_gtfs.zip" >&2; echo >&2
    exit 1
  fi
fi

# --- Muni GTFS (via 511) -------------------------------------------------
say "Muni GTFS (data/muni_current.zip via 511.org)"
if valid_gtfs_zip "$DATA/muni_current.zip"; then
  ok "muni_current.zip already present"
else
  echo "  fetching current Muni feed from 511.org..."
  API511_TOKEN="$API511_TOKEN" bash "$ROOT/scripts/fetch_511.sh"
  if valid_gtfs_zip "$DATA/muni_current.zip"; then
    ok "muni_current.zip ready ($(ls -lh "$DATA/muni_current.zip" | awk '{print $5}'))"
  else
    err "Muni feed is not a valid GTFS zip after fetch_511.sh."
    exit 1
  fi
fi

# --- SF neighborhoods (DataSF) -------------------------------------------
say "SF neighborhoods (data/sf_neighborhoods.geojson via DataSF)"
if valid_geojson "$DATA/sf_neighborhoods.geojson"; then
  ok "sf_neighborhoods.geojson already present"
else
  echo "  downloading DataSF 'SF Find Neighborhoods' (gfpk-269f)..."
  curl -fL --retry 4 --retry-delay 3 --retry-all-errors \
    --connect-timeout 25 --max-time 120 \
    -o "$DATA/sf_neighborhoods.geojson" \
    'https://data.sfgov.org/resource/gfpk-269f.geojson?$limit=300'
  if valid_geojson "$DATA/sf_neighborhoods.geojson"; then
    ok "sf_neighborhoods.geojson downloaded ($(ls -lh "$DATA/sf_neighborhoods.geojson" | awk '{print $5}'))"
  else
    err "Neighborhoods download is not valid GeoJSON. First bytes:"
    head -c 200 "$DATA/sf_neighborhoods.geojson" >&2; echo >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# 5. Next steps
# ---------------------------------------------------------------------------
say "Setup complete"
cat <<EOF
Everything is installed and all input data is in data/.

Next steps:
  1. Configure your destination (if you haven't already):
       cp .env.example .env      # then edit .env
       # set DEFAULT_ADDRESS=...  (an address to geocode)
       #   or DEST_LAT / DEST_LON / DEST_LABEL (explicit coordinates)

  2. Run the live interactive server:
       $PY scripts/server.py
       # then open http://127.0.0.1:8000

  3. Or generate the static outputs in out/:
       $PY scripts/isochrone.py --tag full
       $PY scripts/isochrone.py --gtfs muni_current.zip --tag munionly
       $PY scripts/make_interactive.py     # builds out/commute_explorer.html

See README.md for full usage and CLI flags.
EOF
