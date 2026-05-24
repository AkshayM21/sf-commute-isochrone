#!/usr/bin/env bash
# Fetch CURRENT Muni (and optionally BART) GTFS from the authoritative 511.org
# Bay Area regional API. Requires a free token: https://511.org/open-data/token
#
# Usage:  API511_TOKEN=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx bash scripts/fetch_511.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"

# Pick up API511_TOKEN from a (gitignored) .env if not already set in the env.
if [ -z "${API511_TOKEN:-}" ] && [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ROOT/.env"
  set +a
fi

TOKEN="${API511_TOKEN:-${1:-}}"
if [ -z "$TOKEN" ]; then echo "ERROR: set API511_TOKEN env var, add it to .env, or pass token as arg 1"; exit 2; fi

fetch_op () {
  local op="$1" out="$2"
  echo "=== 511 operator=$op -> $out ==="
  # 511 returns a zip of GTFS for the operator
  curl -fsSL --retry 4 --retry-delay 3 --retry-all-errors --connect-timeout 25 --max-time 240 \
       -o "$DATA/$out" \
       "https://api.511.org/transit/datafeeds?api_key=${TOKEN}&operator_id=${op}"
  # 511 sometimes wraps zip with BOM; validate
  if unzip -l "$DATA/$out" >/dev/null 2>&1 && unzip -l "$DATA/$out" | grep -q "routes.txt"; then
    echo "OK $out ($(ls -lh "$DATA/$out" | awk '{print $5}'))"
    unzip -p "$DATA/$out" calendar.txt 2>/dev/null | head -3 || true
  else
    echo "FAILED to get valid zip for $op. First bytes:"; head -c 300 "$DATA/$out"; echo; exit 1
  fi
}

fetch_op SF muni_current.zip   # SFMTA Muni
# BART already fetched from bart.gov; uncomment to also pull from 511:
# fetch_op BA bart_511.zip
echo "DONE"
