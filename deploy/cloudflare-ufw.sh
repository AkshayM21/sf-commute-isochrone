#!/usr/bin/env bash
# Lock origin ports 80/443 to Cloudflare IP ranges only.
#
# Without this, anyone who learns the Oracle public IP (via CT logs, port scan, DNS history)
# can bypass Cloudflare entirely and hit the origin directly — defeating the WAF, the CDN cache,
# the DDoS protection, and the rate limiter. This script pulls the current CF IP ranges (v4+v6)
# and rewrites ufw so port 80/443 ONLY accept connections from those.
#
# Designed to be idempotent / re-runnable. The cloudflare-ufw.timer runs it weekly so we follow
# CF's IP changes automatically.

set -euo pipefail

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Cloudflare publishes these forever at these stable URLs.
curl -fsS --retry 3 --max-time 15 https://www.cloudflare.com/ips-v4 > "$TMP/v4"
curl -fsS --retry 3 --max-time 15 https://www.cloudflare.com/ips-v6 > "$TMP/v6"

# Sanity: refuse to wipe rules if we got an empty list (CF outage, DNS, whatever).
total_cidrs=$(grep -cv '^[[:space:]]*$' "$TMP/v4" "$TMP/v6" | awk -F: '{s+=$2}END{print s}')
[[ "$total_cidrs" -ge 10 ]] || { echo "[cf-ufw] aborting: only $total_cidrs CIDRs fetched — bad list?" >&2; exit 1; }

# Delete any prior CF rules (tagged via the rule comment we set below).
while read -r num; do
  ufw --force delete "$num" >/dev/null 2>&1 || true
done < <(ufw status numbered | awk -F'[][]' '/\[CF\]/ {print $2}' | sort -rn)

# Add new CF rules — proto tcp, ports 80 and 443, tagged "[CF] HTTP/S" for the next sweep.
while read -r cidr; do
  [[ -n "$cidr" ]] && ufw allow proto tcp from "$cidr" to any port 80,443 comment "[CF] HTTP/S" >/dev/null
done < "$TMP/v4"
while read -r cidr; do
  [[ -n "$cidr" ]] && ufw allow proto tcp from "$cidr" to any port 80,443 comment "[CF] HTTP/S" >/dev/null
done < "$TMP/v6"

# Remove any blanket allow on 80/443 left over from install.sh's bootstrap rules.
ufw delete allow 80/tcp 2>/dev/null || true
ufw delete allow 443/tcp 2>/dev/null || true

logger -t cloudflare-ufw "refreshed: $total_cidrs CF CIDRs allowed on 80/443; blanket rules removed"
echo "[cf-ufw] OK ($total_cidrs CIDRs)"
