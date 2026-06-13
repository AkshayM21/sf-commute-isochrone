#!/usr/bin/env python3
"""Mint a Cloudflare Origin CA certificate for sfcommutemap.com (+ wildcard).

Reads CF_API_TOKEN from the repo-root .env (gitignored). The token never appears in
argv — only in the Authorization header sent over TLS to Cloudflare. The signed cert
is written to deploy/cf/origin-cert.pem alongside the key. From there, `scp` it to the
box and reference both from the Caddyfile.

Why Origin CA over Let's Encrypt:
  - 15-year validity (5475 days) — no renewal cron, no expiry-fire-drills.
  - No HTTP-01 / DNS-01 challenge required — works on a box where port 80 is locked
    down to Cloudflare-only from day one.
  - Trusted ONLY by Cloudflare's edge — perfect for the CF-fronted origin model,
    and never useful to anyone who bypasses CF.

Usage:
  python3 deploy/cf/mint.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
CSR_PATH = REPO_ROOT / "deploy/cf/origin.csr"
CERT_PATH = REPO_ROOT / "deploy/cf/origin-cert.pem"

HOSTNAMES = ["sfcommutemap.com", "www.sfcommutemap.com"]   # apex + www; no wildcard
# (wildcard requires Zone:Read on the user token to validate; we don't need it for this deploy.
# If we ever add other subdomains we can re-mint or extend the token's scope.)
VALIDITY_DAYS = 5475  # 15 years — Origin CA max


def load_env() -> dict[str, str]:
    """Tiny .env parser — KEY=VALUE per line, no quotes/exports/multiline. Matches our
    `.env` shape (mirrored by /etc/sfci.env on the box)."""
    out: dict[str, str] = {}
    for line in ENV_FILE.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip()
    return out


def main() -> int:
    env = load_env()
    # Auth preference for the Origin CA endpoint (which is mysteriously strict — Bearer tokens
    # with the documented permission groups still 1016 in mid-2026, so we fall through to the
    # Global API Key which is the legacy admin credential that's always worked):
    #   1. CF_GLOBAL_KEY + CF_EMAIL — Global API Key (cfk_* in the new format). Most powerful;
    #      bypasses the permission-group plumbing entirely.
    #   2. CF_ORIGIN_TOKEN / CF_USER_API_TOKEN / CF_API_TOKEN — Bearer tokens (kept as future-
    #      forward fallback in case CF fixes the permission propagation).
    if env.get("CF_GLOBAL_KEY") and env.get("CF_EMAIL"):
        auth_headers = {"X-Auth-Email": env["CF_EMAIL"], "X-Auth-Key": env["CF_GLOBAL_KEY"]}
        token_src = "CF_GLOBAL_KEY"
    else:
        for key in ("CF_ORIGIN_TOKEN", "CF_USER_API_TOKEN", "CF_API_TOKEN"):
            if env.get(key):
                auth_headers = {"Authorization": f"Bearer {env[key]}"}
                token_src = key
                break
        else:
            print("ERR: no CF credentials in .env (need CF_GLOBAL_KEY+CF_EMAIL or CF_*_TOKEN)", file=sys.stderr)
            return 1
    print(f"[mint] using {token_src}")
    if not CSR_PATH.exists():
        print(f"ERR: CSR not found at {CSR_PATH} — run openssl req first", file=sys.stderr)
        return 1

    csr = CSR_PATH.read_text()
    body = json.dumps({
        "hostnames": HOSTNAMES,
        "requested_validity": VALIDITY_DAYS,
        "request_type": "origin-rsa",
        "csr": csr,
    }).encode()

    # Origin CA API endpoint (account-scoped). The Cloudflare docs show this as
    # https://api.cloudflare.com/client/v4/certificates — it's an account-level resource
    # despite the unqualified path. Auth: Bearer token with Origin CA: Edit permission.
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/certificates",
        data=body,
        method="POST",
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print(f"ERR: HTTP {e.code} from Cloudflare:", file=sys.stderr)
        print(e.read().decode(), file=sys.stderr)
        return 2

    if not resp.get("success"):
        print("ERR: Cloudflare rejected the CSR:", file=sys.stderr)
        print(json.dumps(resp.get("errors", []), indent=2), file=sys.stderr)
        return 3

    cert_pem = resp["result"]["certificate"]
    expires = resp["result"].get("expires_on", "?")
    cert_id = resp["result"].get("id", "?")

    CERT_PATH.write_text(cert_pem)
    CERT_PATH.chmod(0o644)
    print(f"[mint] wrote {CERT_PATH} ({len(cert_pem)} bytes)")
    print(f"[mint] cert id: {cert_id}")
    print(f"[mint] expires: {expires}")
    print(f"[mint] hostnames: {', '.join(HOSTNAMES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
