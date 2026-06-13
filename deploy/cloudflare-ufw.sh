#!/usr/bin/env bash
# Lock origin ports 80/443 to Cloudflare IP ranges only (firewalld implementation for OL9).
#
# Without this, anyone who learns the Oracle public IP (via CT logs, port scan, DNS history)
# can bypass Cloudflare entirely and hit the origin directly — defeating the WAF, the CDN cache,
# the DDoS protection, and the rate limiter. This script pulls the current CF IP ranges (v4+v6)
# and rewrites a dedicated "cloudflare" firewalld zone so HTTP/S ONLY answers from those sources.
#
# How the zone routing works on firewalld: incoming packets whose source IP matches a zone's
# `--list-sources` get evaluated against THAT zone's allowed services/ports, even if the
# interface is bound to a different zone. So:
#   - public  zone (bound to enp0s6): allows SSH only — never http/https.
#   - cloudflare zone (no interface):  allows http+https; sources = the CF IP ranges.
# Result: 80/443 only reach Caddy when they come from a CF IP. Everything else gets dropped
# silently by the public zone's default-deny.
#
# Designed to be idempotent / re-runnable. The cloudflare-ufw.timer runs it weekly so we follow
# CF's IP changes automatically. (File still named *-ufw.* for repo-history compat; the script
# itself uses firewall-cmd, not ufw.)

set -euo pipefail

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Cloudflare publishes these forever at these stable URLs.
curl -fsS --retry 3 --max-time 15 https://www.cloudflare.com/ips-v4 > "$TMP/v4"
curl -fsS --retry 3 --max-time 15 https://www.cloudflare.com/ips-v6 > "$TMP/v6"

# Sanity: refuse to touch rules unless BOTH lists are CIDR-shaped. A line count alone is not
# enough — a transparent proxy / captive portal / CF HTML error served with a 200 would pass it,
# and then the remove-then-add below could strand the permanent zone with zero sources.
# Require >=10 valid CIDRs AND every non-blank line to be a CIDR.
v4_ok=$(grep -Ec '^([0-9]{1,3}\.){3}[0-9]{1,3}/[0-9]{1,2}$' "$TMP/v4" || true)
v6_ok=$(grep -Ec '^[0-9A-Fa-f:]+/[0-9]{1,3}$' "$TMP/v6" || true)
total_cidrs=$((v4_ok + v6_ok))
nonblank=$(grep -cv '^[[:space:]]*$' "$TMP/v4" "$TMP/v6" | awk -F: '{s+=$2}END{print s}')
[[ "$total_cidrs" -ge 10 && "$total_cidrs" -eq "$nonblank" ]] \
  || { echo "[cf-fw] aborting: $total_cidrs valid CIDRs of $nonblank non-blank lines — bad list?" >&2; exit 1; }

# 1) Ensure the cloudflare zone exists, with http+https allowed and no interfaces bound.
#    --permanent writes to /etc/firewalld/zones; we --reload at the end to apply atomically.
if ! firewall-cmd --permanent --get-zones | tr ' ' '\n' | grep -qx cloudflare; then
  firewall-cmd --permanent --new-zone=cloudflare
fi
firewall-cmd --permanent --zone=cloudflare --set-target=DROP   # source-not-matched → drop
firewall-cmd --permanent --zone=cloudflare --add-service=http   2>/dev/null || true
firewall-cmd --permanent --zone=cloudflare --add-service=https  2>/dev/null || true

# 2) Diff the zone's source list against the freshly-fetched CF CIDRs, then ADD new sources
#    FIRST and REMOVE stale ones AFTER. Order matters: a transient firewall-cmd failure
#    mid-script then leaves the permanent zone holding a superset (briefly over-permissive to
#    retired CF ranges) instead of an empty zone — which would mean a dark site at the next
#    --reload/reboot, since public has no http/https. Also truly idempotent: an unchanged
#    list touches nothing instead of churning every source weekly.
sort -u "$TMP/v4" "$TMP/v6" | grep -v '^[[:space:]]*$' > "$TMP/want"
firewall-cmd --permanent --zone=cloudflare --list-sources 2>/dev/null \
  | tr ' ' '\n' | grep -v '^[[:space:]]*$' | sort -u > "$TMP/have" || true
comm -23 "$TMP/want" "$TMP/have" > "$TMP/add"   # in want, not in have
comm -13 "$TMP/want" "$TMP/have" > "$TMP/del"   # in have, not in want
while read -r cidr; do
  [[ -n "$cidr" ]] || continue
  firewall-cmd --permanent --zone=cloudflare --add-source="$cidr" >/dev/null
done < "$TMP/add"
while read -r cidr; do
  [[ -n "$cidr" ]] || continue
  firewall-cmd --permanent --zone=cloudflare --remove-source="$cidr" >/dev/null
done < "$TMP/del"

# 3) Make sure the PUBLIC zone (the one bound to the interface) does NOT have http/https.
#    Leftovers from install.sh's bootstrap would defeat the CF-only lockdown.
firewall-cmd --permanent --zone=public --remove-service=http   2>/dev/null || true
firewall-cmd --permanent --zone=public --remove-service=https  2>/dev/null || true

# 4) Apply.
firewall-cmd --reload

logger -t cloudflare-ufw "refreshed: $total_cidrs CF CIDRs allowed on http/https in 'cloudflare' zone; public zone narrowed to SSH-only"
echo "[cf-fw] OK ($total_cidrs CIDRs in zone 'cloudflare'; public zone has: $(firewall-cmd --zone=public --list-services))"
