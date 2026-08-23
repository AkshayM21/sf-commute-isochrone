# Production deployment guide

This runbook documents the project's current Oracle Linux 9 (`dnf`/systemd) deployment for
`sfcommutemap.com` behind Cloudflare. It uses placeholders for the SSH host and credentials so
those values never enter the repository. Forks using another distribution or public hostname must
adapt `deploy/install.sh`, `deploy/Caddyfile`, the Origin CA hostnames, and DNS together.

## Prerequisites

- An Oracle Linux 9 host you can reach as `<user@host>` with `sudo` access.
- The Cloudflare zone for `sfcommutemap.com`, with `CF_ZONE_ID` available only in a local `.env`.
- A local `.env` based on [`.env.example`](../.env.example). It is ignored by Git and is never
  copied by `push.sh`.

## 1. Copy the deployable source

```bash
deploy/push.sh <user@host>
```

The script syncs runtime source and locally present data. It excludes credentials, `.git`,
agent/session state, caches, screenshots, test artifacts, raw source extracts, and Cloudflare
certificate tooling. Remote `/opt/sfci/data` is protected from deletion when a clean checkout has
no local data; refresh feeds and bakes as a deliberate operation rather than an incidental code
push.

If an older deployment predates these exclusions, remove legacy development metadata only after
confirming the active certificate and key already live under `/etc/caddy`:

```bash
ssh <user@host> 'sudo rm -rf /opt/sfci/.git /opt/sfci/.impeccable /opt/sfci/.agents /opt/sfci/deploy/cf'
```

## 2. Install on the host

```bash
ssh <user@host> 'sudo bash /opt/sfci/deploy/install.sh'
```

The installer creates the service user and virtual environment, installs systemd units, starts
the application on `localhost:8000`, and writes `/etc/sfci.env` if it does not already exist.

## 3. Configure runtime secrets on the host

```bash
ssh <user@host> 'sudo vi /etc/sfci.env'
ssh <user@host> 'sudo systemctl restart sfci'
```

Set only the credentials and provider settings your deployment needs, such as `API511_TOKEN`,
`GEOAPIFY_KEY`, and (when using Nominatim publicly) `GEOCODER_USER_AGENT`. Keep this file
root-owned and mode 600. Verify the local service without exposing it publicly:

```bash
ssh <user@host> 'curl -fsS http://localhost:8000/healthz'
```

## 4. Configure Cloudflare HTTPS and DNS

Create an Origin CA certificate in the Cloudflare dashboard for the hostnames configured in your
Caddyfile. Transfer the certificate and private key to the host through a secure channel, install
them under `/etc/caddy/` with restrictive permissions, then start Caddy. Do not commit or rsync
the private key.

Set Cloudflare SSL/TLS mode to **Full (strict)**. To update the apex and `www` A records through
the helper, add an explicit zone ID and an appropriately scoped Cloudflare credential to your
local `.env`, then run:

```bash
python3 deploy/cf/dns.py <origin-ip>
```

`dns.py` refuses to run without `CF_ZONE_ID`, discovers the zone name through the API, and creates
or updates proxied apex and `www` records. Confirm the hostname is proxied before restricting
origin ingress to Cloudflare address ranges.

## 5. Verify and update

```bash
curl -fsSI https://<your-public-hostname>/
ssh <user@host> 'sudo systemctl status sfci caddy --no-pager'
```

For ordinary code updates, repeat the source copy and restart the app:

```bash
deploy/push.sh <user@host>
ssh <user@host> 'sudo systemctl restart sfci'
```

Use `journalctl -u sfci -f` and `journalctl -u caddy -f` on the host for diagnostics. For a
rollback, redeploy a known-good checked-out revision; avoid treating an uncommitted local tree as
a release artifact.
