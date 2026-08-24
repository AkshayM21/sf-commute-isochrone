#!/usr/bin/env bash
# Fetch CURRENT Muni + Caltrain (and optionally BART) GTFS from the authoritative
# 511.org Bay Area regional API. Requires a free token:
# https://511.org/open-data/token
#
# Usage:  API511_TOKEN=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx bash scripts/fetch_511.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"

# Read a single KEY's value from .env WITHOUT executing the file. Sourcing .env
# breaks on multi-word unquoted values (e.g. DEFAULT_ADDRESS=1 Ferry Building),
# so we extract only the one key we need: tolerate a leading `export `, optional
# surrounding whitespace, and surrounding single/double quotes.
env_value () {
  local key="$1" file="$ROOT/.env"
  [ -f "$file" ] || return 0
  sed -nE "s/^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=[[:space:]]*//p" "$file" \
    | tail -1 | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}

# Pick up API511_TOKEN from a (gitignored) .env if not already set in the env.
if [ -z "${API511_TOKEN:-}" ]; then
  API511_TOKEN="$(env_value API511_TOKEN)"
fi

TOKEN="${API511_TOKEN:-${1:-}}"
if [ -z "$TOKEN" ]; then echo "ERROR: set API511_TOKEN env var, add it to .env, or pass token as arg 1"; exit 2; fi

mkdir -p "$DATA"

# Clean up any leftover partial downloads on exit.
trap 'rm -f "$DATA"/*.part 2>/dev/null || true' EXIT

# A zip is a valid GTFS feed if it lists a (top-level or nested) routes.txt.
valid_gtfs_zip () {
  local f="$1"
  [ -s "$f" ] && unzip -t "$f" >/dev/null 2>&1 \
    && unzip -Z1 "$f" | grep -E '(^|/)routes\.txt$' >/dev/null
}

fetch_op () {
  local op="$1" out="$2"
  echo "=== 511 operator=$op -> $out ==="
  # 511 returns a zip of GTFS for the operator. Download to a temp .part and
  # rename only on success so an interrupted download never masquerades as done.
  # The URL (which carries api_key) is fed via a stdin config file (`curl -K -`)
  # so the token never appears on the curl argv (visible to any user via ps).
  # Inside curl-config double quotes, `"` and `\` are meta — reject tokens carrying
  # them (511 tokens are UUIDs; anything else means a paste error) instead of
  # silently mangling the URL.
  case "$TOKEN" in *[\"\\]*)
    echo "ERROR: API511_TOKEN contains a quote/backslash — check .env" >&2; exit 2;;
  esac
  printf 'url = "https://api.511.org/transit/datafeeds?api_key=%s&operator_id=%s"\n' \
         "$TOKEN" "$op" \
    | curl -fsSL --retry 4 --retry-delay 3 --retry-all-errors --connect-timeout 25 --max-time 240 \
           -o "$DATA/$out.part" -K - \
    && mv "$DATA/$out.part" "$DATA/$out"
  # 511 sometimes wraps zip with BOM; validate
  if valid_gtfs_zip "$DATA/$out"; then
    echo "OK $out ($(du -h "$DATA/$out" | awk '{print $1}'))"
    unzip -p "$DATA/$out" calendar.txt 2>/dev/null | head -3 || true
  else
    echo "FAILED to get valid zip for $op. First bytes:"; head -c 300 "$DATA/$out" 2>/dev/null; echo; exit 1
  fi
}

fetch_op SF muni_current.zip   # SFMTA Muni
fetch_op CT caltrain.zip       # Caltrain
# BART already fetched from bart.gov; uncomment to also pull from 511:
# fetch_op BA bart_511.zip
echo "DONE"
