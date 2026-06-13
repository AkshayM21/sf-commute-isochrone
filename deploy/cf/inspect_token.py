#!/usr/bin/env python3
"""Show what the CF_API_TOKEN in .env actually has access to. Used to diagnose 1016
'User is not authorized' errors before blaming the request."""
from __future__ import annotations
import json, pathlib, sys, urllib.error, urllib.request

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
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
        except Exception:
            body = {"_raw": "(non-JSON body)"}
        return e.code, body
    except urllib.error.URLError as e:
        # network down / DNS failure — diagnostic tool should report, not traceback
        return 0, {"success": False, "errors": [{"message": f"network error: {e.reason}"}]}


def main():
    try:
        env = load_env()
    except FileNotFoundError:
        print(f"ERR: no .env at {REPO_ROOT / '.env'} — this script reads CF credentials from it",
              file=sys.stderr)
        return 1
    # Prefer the user-scoped token (cfut_*) for Origin CA work; fall back to the account token.
    # This is the diagnostic for misconfigured credentials, so missing keys must print a clear
    # message, not a KeyError traceback.
    token = env.get("CF_USER_API_TOKEN") or env.get("CF_API_TOKEN")
    account = env.get("CF_ACCOUNT_ID", "")
    if not token:
        print("ERR: no Bearer token in .env (need CF_USER_API_TOKEN or CF_API_TOKEN; "
              "note this script inspects Bearer tokens — CF_GLOBAL_KEY is not inspectable here)",
              file=sys.stderr)
        return 1
    print(f"(using token: {'CF_USER_API_TOKEN' if env.get('CF_USER_API_TOKEN') else 'CF_API_TOKEN'})")
    print()

    probes = [
        ("/user/tokens/verify (user-scoped)",
         "https://api.cloudflare.com/client/v4/user/tokens/verify"),
        ("/zones (what zones can I see?)",
         "https://api.cloudflare.com/client/v4/zones"),
        ("/certificates (Origin CA list — needs origin_ca:edit)",
         "https://api.cloudflare.com/client/v4/certificates"),
    ]
    if account:
        probes.insert(1, (f"/accounts/{account[:8]}.../tokens/verify (account-scoped)",
                          f"https://api.cloudflare.com/client/v4/accounts/{account}/tokens/verify"))
    else:
        print("(skipping account-scoped /accounts/.../tokens/verify probe: CF_ACCOUNT_ID not in .env)")
        print()

    for label, url in probes:
        print(f"=== {label} ===")
        code, body = get(url, token)
        print(f"  HTTP {code}")
        if "result" in body and isinstance(body["result"], list):
            for item in body["result"][:5]:
                if "name" in item:
                    print(f"    - {item['name']:25}  id={item.get('id','?')}  status={item.get('status','?')}")
                else:
                    print(f"    - {json.dumps(item)[:120]}")
        elif "result" in body and isinstance(body["result"], dict) and body["result"]:
            for k, v in list(body["result"].items())[:6]:
                print(f"    {k}: {v}")
        if not body.get("success", True):
            for err in body.get("errors", []):
                print(f"    ERROR {err.get('code')}: {err.get('message')}")
        print()

if __name__ == "__main__":
    sys.exit(main())
