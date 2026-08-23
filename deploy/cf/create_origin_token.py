#!/usr/bin/env python3
"""Create a narrowly scoped Cloudflare API token for Origin CA and DNS operations.

Uses the existing CF_USER_API_TOKEN to authenticate the token-create call.
Required existing permission: 'API Tokens: Edit' on the calling token.

If the call succeeds, the new token value is written to .env as CF_ORIGIN_TOKEN."""
from __future__ import annotations
import json, pathlib, sys, urllib.request, urllib.error

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

# Permission group IDs (from /user/tokens/permission_groups). Stable UUIDs across accounts.
# Origin CA cert minting (POST /client/v4/certificates) is authorized by
# "Account: SSL and Certificates Write", NOT by what's confusingly named "Origin Write" (that one
# is for Cloudflare Workers' origin-rewrite rules — same word, different feature). CF buried this
# rename in 2023 and the dashboard's permission picker doesn't always surface the right one.
PG_SSL_CERTS_WRITE = "db37e5f1cb1a4e1aabaef8deaea43575"  # "Account: SSL and Certificates Write"
PG_ZONE_READ       = "c8fed203ed3043cba015a93ad1616f1f"  # "Zone Read"  (for hostname-belongs-to-zone validation)
PG_DNS_WRITE       = "4755a26eedb94da69e1066d98aa820be"  # "DNS Write" (for the A-record flip after Caddy is up)

ACCOUNT_ID = None  # filled from .env


def load_env():
    out = {}
    if not ENV_FILE.exists():
        return out
    for line in ENV_FILE.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("=")
            out[k.strip()] = v.strip()
    return out


def post(url, token, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"_raw": "(non-JSON)"}


def main():
    env = load_env()
    caller = env.get("CF_USER_API_TOKEN")
    account = env.get("CF_ACCOUNT_ID")
    zone_id = env.get("CF_ZONE_ID", "").strip()
    if not (caller and account and zone_id):
        print("ERR: need CF_USER_API_TOKEN + CF_ACCOUNT_ID + CF_ZONE_ID in .env", file=sys.stderr)
        return 1

    # Two policies because SSL/Certs Write is account-scoped while Zone Read / DNS Write are
    # zone-scoped — different "resources" shape. We pin to the explicitly configured zone.
    body = {
        "name": "commute-map-deploy",
        "policies": [
            {   # account-scope: Origin CA cert minting (POST /certificates)
                "effect": "allow",
                "resources": {f"com.cloudflare.api.account.{account}": "*"},
                "permission_groups": [{"id": PG_SSL_CERTS_WRITE}],
            },
            {   # zone-scope: Zone Read + DNS Write, pinned to the configured zone
                "effect": "allow",
                "resources": {f"com.cloudflare.api.account.zone.{zone_id}": "*"},
                "permission_groups": [{"id": PG_ZONE_READ}, {"id": PG_DNS_WRITE}],
            },
        ],
    }
    code, resp = post("https://api.cloudflare.com/client/v4/user/tokens", caller, body)
    print(f"HTTP {code}")
    if not resp.get("success"):
        print(json.dumps(resp, indent=2))
        return 2

    new_value = resp["result"]["value"]
    new_id = resp["result"]["id"]
    print(f"created token id={new_id} (value written to .env as CF_ORIGIN_TOKEN)")

    # Persist to .env (no echo of the value)
    lines = ENV_FILE.read_text().splitlines()
    out_lines = [ln for ln in lines if not ln.startswith("CF_ORIGIN_TOKEN=")]
    out_lines.append(f"CF_ORIGIN_TOKEN={new_value}")
    ENV_FILE.write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
