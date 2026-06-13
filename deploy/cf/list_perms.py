#!/usr/bin/env python3
"""List every available API-token permission group, grep for anything Origin CA flavored.
Cloudflare renames these periodically; we want the canonical current name so we can tell the
user EXACTLY what to click in the dashboard."""
from __future__ import annotations
import json, pathlib, urllib.request, urllib.error

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_env():
    out = {}
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("=")
            out[k.strip()] = v.strip()
    return out


def get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {"_raw": "(non-JSON body)"}
        return e.code, body


def show(label, code, body, needle):
    print(f"\n=== {label} (HTTP {code}) ===")
    if not body.get("success"):
        for err in body.get("errors", []):
            print(f"  ERROR {err.get('code')}: {err.get('message')}")
        return
    found = []
    for pg in body.get("result", []):
        name = pg.get("name", "")
        if needle.lower() in name.lower():
            found.append(pg)
    if not found:
        print(f"  (no '{needle}'-matching permission groups visible)")
        # show a sample of what we DO see
        all_names = sorted(set(pg.get("name","") for pg in body.get("result", [])))
        print(f"  ({len(all_names)} groups total. sample of names containing 'CA', 'Cert', or 'Origin':)")
        for n in all_names:
            if any(k in n.lower() for k in ("ca", "cert", "origin", "ssl")):
                print(f"     - {n}")
    else:
        for pg in found:
            print(f"  - name: {pg.get('name')}")
            print(f"    id:   {pg.get('id')}")
            scopes = pg.get("scopes", [])
            print(f"    scopes: {scopes}")


def main():
    env = load_env()
    account = env["CF_ACCOUNT_ID"]
    user_token = env.get("CF_USER_API_TOKEN")
    acct_token = env.get("CF_API_TOKEN")

    for label, token, url in [
        ("USER perms (via cfut_* token)",  user_token,
         "https://api.cloudflare.com/client/v4/user/tokens/permission_groups"),
        ("ACCOUNT perms (via cfat_* token)",  acct_token,
         f"https://api.cloudflare.com/client/v4/accounts/{account}/tokens/permission_groups"),
    ]:
        if not token:
            print(f"\n=== {label} (skipped: no token) ==="); continue
        code, body = get(url, token)
        show(label, code, body, needle="origin")


if __name__ == "__main__":
    import sys; sys.exit(main())
