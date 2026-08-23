#!/usr/bin/env python3
"""Upsert apex and ``www`` DNS A records for the configured Cloudflare zone.

Reads:
  CF_GLOBAL_KEY + CF_EMAIL  (preferred; works for everything)
  OR CF_ORIGIN_TOKEN / CF_API_TOKEN (Bearer fallback)
  CF_ZONE_ID  (required; Cloudflare zone ID)

Usage:
  python3 deploy/cf/dns.py <origin-ip>   # apex + www both -> <origin-ip>, proxied

Idempotent: looks up existing records and PUTs if found, POSTs if not."""
from __future__ import annotations
import ipaddress
import json, pathlib, sys, urllib.request, urllib.error

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
def load_env():
    out = {}
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return out
    for line in env_file.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("=")
            out[k.strip()] = v.strip()
    return out


def cf_request(method, url, headers, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"_raw": "(non-JSON)"}
    except urllib.error.URLError as e:
        # Network-level failure (DNS, timeout, no route): synthesize a CF-shaped error body
        # so both call sites handle it via the same falsy-`success` path instead of a traceback.
        return 0, {"success": False, "errors": [{"message": str(e.reason)}]}


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <origin-ip>", file=sys.stderr); return 1
    new_ip = sys.argv[1]
    try:
        if ipaddress.ip_address(new_ip).version != 4:
            raise ValueError("DNS A records require an IPv4 address")
    except ValueError as error:
        print(f"ERR: invalid origin IPv4 address {new_ip!r}: {error}", file=sys.stderr)
        return 1

    env = load_env()
    zone = env.get("CF_ZONE_ID", "").strip()
    if not zone:
        print("ERR: need CF_ZONE_ID in .env (the Cloudflare zone ID to update)", file=sys.stderr)
        return 1
    if env.get("CF_GLOBAL_KEY") and env.get("CF_EMAIL"):
        auth = {"X-Auth-Email": env["CF_EMAIL"], "X-Auth-Key": env["CF_GLOBAL_KEY"]}
        src = "CF_GLOBAL_KEY"
    elif env.get("CF_ORIGIN_TOKEN"):
        auth = {"Authorization": f"Bearer {env['CF_ORIGIN_TOKEN']}"}; src = "CF_ORIGIN_TOKEN"
    elif env.get("CF_API_TOKEN"):
        auth = {"Authorization": f"Bearer {env['CF_API_TOKEN']}"}; src = "CF_API_TOKEN"
    else:
        print("ERR: no CF credentials in .env", file=sys.stderr); return 1
    print(f"[dns] auth={src}  zone={zone[:12]}...  target_ip={new_ip}")

    zone_url = f"https://api.cloudflare.com/client/v4/zones/{zone}"
    code, zone_response = cf_request("GET", zone_url, auth)
    if not zone_response.get("success") or not zone_response.get("result", {}).get("name"):
        print(f"ERR: could not read configured CF_ZONE_ID ({code})", file=sys.stderr)
        print(json.dumps(zone_response.get("errors", zone_response), indent=2), file=sys.stderr)
        return 2
    zone_name = zone_response["result"]["name"]
    base = f"{zone_url}/dns_records"
    # The two records we want to upsert. proxied=True (orange-cloud) is critical for origin
    # lockdown; without it, traffic can bypass Cloudflare and hit the host directly.
    desired = [
        {"type": "A", "name": zone_name,          "content": new_ip, "ttl": 1, "proxied": True},
        {"type": "A", "name": f"www.{zone_name}", "content": new_ip, "ttl": 1, "proxied": True},
    ]

    # List existing A records once
    code, resp = cf_request("GET", f"{base}?type=A&per_page=100", auth)
    if not resp.get("success"):
        print(f"ERR: list failed ({code})", file=sys.stderr)
        print(json.dumps(resp, indent=2), file=sys.stderr); return 2
    by_name = {r["name"]: r for r in resp.get("result", [])}
    print(f"[dns] existing A records: {sorted(by_name.keys())}")

    failed = False
    for d in desired:
        if d["name"] in by_name:
            existing = by_name[d["name"]]
            if existing.get("content") == d["content"] and existing.get("proxied") == d["proxied"]:
                print(f"[dns]   {d['name']:30}  unchanged (already -> {d['content']}, proxied)")
                continue
            code, r = cf_request("PUT", f"{base}/{existing['id']}", auth, d)
            ok = r.get("success")
            print(f"[dns]   {d['name']:30}  UPDATED  ({existing.get('content')} -> {d['content']})  ok={ok}")
            if not ok: print(json.dumps(r.get("errors"), indent=2))
        else:
            code, r = cf_request("POST", base, auth, d)
            ok = r.get("success")
            print(f"[dns]   {d['name']:30}  CREATED  -> {d['content']}  ok={ok}")
            if not ok: print(json.dumps(r.get("errors"), indent=2))
        failed |= not ok

    # Nonzero exit on any failed upsert — the deploy runbook chains this script, and a green
    # exit with DNS still pointing at the old IP would let the operator lock down a dark site.
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
