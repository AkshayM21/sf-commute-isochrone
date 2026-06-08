# Deploy runbook — sfcommutemap.com

End-to-end from a freshly-provisioned Oracle Always Free A1.Flex box (Oracle Linux 9 ARM, 1 OCPU /
6 GB) to a live HTTPS site behind Cloudflare. Every step is idempotent — re-run safely.

## 0. Prereqs (one-time, already done if you've gotten this far)

- Oracle Always Free A1.Flex instance running with a public IP. (`deploy/oci_launch.sh` provisions it.)
- Your laptop's SSH key matches the public key uploaded at instance launch (you can already `ssh opc@<ip>`).
- Cloudflare account holds `sfcommutemap.com`, nameservers switched at Namecheap, propagation done.

## 1. Push the repo to the box

```
deploy/push.sh opc@<oracle-public-ip>
```

(Oracle Linux uses `opc` as the default user; Ubuntu uses `ubuntu`.)

Rsyncs the repo to `/opt/sfci/` excluding the venv, `__pycache__`, the personal `.dest_cache.json`,
the numba cache (rebuilt on the box), `out/`, and the test golden oracles. Keeps `.git` so you can
`git pull` for incremental updates later.

## 2. Run install on the box

```
ssh opc@<ip> 'sudo bash /opt/sfci/deploy/install.sh'
```

Does, idempotently:
- Adds the Caddy COPR + EPEL repos; installs `python3.12`, `caddy`, `stress-ng`, `ufw`, build tools.
- Creates the `sfci` system user + Python venv at `/opt/sfci/.venv`.
- Writes `/etc/sfci.env` with sensible defaults (you fill in tokens next).
- Installs the systemd units: `sfci`, `sfci-keepalive.timer`, `cloudflare-ufw.timer`.
- Initializes ufw allowing 22/80/443 from anywhere (cloudflare-ufw.timer narrows 80/443 later).
- Starts the Flask app on `localhost:8000`.
- Enables Caddy but doesn't start it yet (waits for the Origin CA cert in step 4).
- Sets up `logrotate` for the app logs.

Verify the Flask side is up locally on the box:

```
ssh opc@<ip> 'curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/'
# → HTTP 200
```

## 3. Fill in /etc/sfci.env with secrets

```
ssh opc@<ip> 'sudo vi /etc/sfci.env'
```

Paste your `API511_TOKEN` and `GEOAPIFY_KEY` from your laptop's `.env`. Save, then:

```
ssh opc@<ip> 'sudo systemctl restart sfci && sudo journalctl -u sfci -n 30 --no-pager'
```

You should see the `[boot] RAPTOR engine ON … WALK GRAPH ON` line.

## 4. Cloudflare Origin CA cert (end-to-end HTTPS without port 80 exposure)

This is what lets us drop port 80 entirely (no Let's Encrypt HTTP-01 challenge dependency) and
keep 80/443 locked to Cloudflare IPs only.

1. Cloudflare dashboard → SSL/TLS → **Origin Server** → **Create Certificate**.
2. Hostnames: `sfcommutemap.com, *.sfcommutemap.com`. Validity: **15 years** (default). Key: RSA.
3. Click Create. Cloudflare shows two text blocks. Save them locally as:
   - `origin-cert.pem` (the Origin Certificate)
   - `origin-key.pem` (the Private Key — **you only see this once**, copy carefully)
4. From your laptop, push them to the box and start Caddy:

```bash
scp origin-cert.pem origin-key.pem opc@<ip>:/tmp/
ssh opc@<ip> <<'REMOTE'
sudo install -m 644 /tmp/origin-cert.pem /etc/caddy/origin-cert.pem
sudo install -m 600 /tmp/origin-key.pem  /etc/caddy/origin-key.pem
sudo rm /tmp/origin-cert.pem /tmp/origin-key.pem
sudo chown caddy:caddy /etc/caddy/origin-cert.pem /etc/caddy/origin-key.pem
sudo systemctl start caddy
sudo systemctl status caddy --no-pager -l | head -15
REMOTE
```

Caddy should report `serving https`. Verify origin cert is presented:

```
ssh opc@<ip> 'echo | openssl s_client -connect localhost:443 -servername sfcommutemap.com 2>/dev/null | openssl x509 -noout -subject -issuer'
# subject = sfcommutemap.com, issuer = Cloudflare Origin CA
```

## 5. Point DNS at the box

In Cloudflare DNS:

- **Edit** the `A @` record → set Content to the **Oracle public IP**. Keep **Proxied** (orange cloud).
- **Edit** the `CNAME www` record → change Content from `parkingpage.namecheap.com` to `sfcommutemap.com`. Keep **Proxied**.

Cloudflare DNS changes propagate instantly (you own DNS now).

## 6. Set Cloudflare SSL mode to Full (strict)

Cloudflare → SSL/TLS → Overview → **Full (strict)**. This enforces that the origin cert is valid;
your Origin CA cert from step 4 passes immediately.

Also recommended (under SSL/TLS → Edge Certificates):
- ☑ Always Use HTTPS
- ☑ Automatic HTTPS Rewrites
- ☑ HTTP/3 (with QUIC)

## 7. Verify live

```
curl -s https://sfcommutemap.com/ -o /dev/null -w "HTTP %{http_code} in %{time_total}s\n"
# First hit: HTTP 200 in ~0.3 s.   Repeat hit: HTTP 200 in ~0.02 s (CF edge cache).
```

Open in a browser. Type an address. Hover a cell. Map should color in; tooltip should show the
breakdown; switching Realistic/Best-case should update the headline; Slow/Medium/Fast should hold
the map then snap once.

## 8. Lock down origin to Cloudflare-only

`cloudflare-ufw.timer` fires ~2 min after boot and weekly thereafter; it pulls CF's current IP
ranges and rewrites ufw so only those CIDRs can reach 80/443. To force it now:

```
ssh opc@<ip> 'sudo systemctl start cloudflare-ufw && sudo ufw status numbered | head -30'
```

After this, an attacker who learns the Oracle IP can't bypass Cloudflare to hit the origin
directly. They get nothing on 80/443. SSH (22) stays open from anywhere.

## 9. Anti-reclamation (already running)

`sfci-keepalive.timer` fires hourly, runs `stress-ng --cpu 2 --cpu-load 80 --timeout 600s` at
nice=19 (lowest priority — preempted instantly by real requests). Drives the box's 95th-percentile
CPU well above Oracle's 20% reclamation threshold so the instance won't get reclaimed for
inactivity.

Verify:

```
ssh opc@<ip> 'systemctl list-timers --all | grep sfci'
ssh opc@<ip> 'sudo journalctl -u sfci-keepalive -n 10 --no-pager'
```

## Updating the app later

```
# On your laptop, after making changes:
deploy/push.sh opc@<ip>
ssh opc@<ip> 'sudo systemctl restart sfci'
```

`push.sh` only rsyncs the diff. Restart cycles the Python process (~3 s downtime per restart;
Caddy returns 502 during that window). For zero-downtime in v2 we'd add gunicorn + systemd
socket activation; skipped for v1 because the audience is tiny and the restart is rare.

## Health checks

| Command | What you learn |
|---|---|
| `journalctl -u sfci -f` | live app logs |
| `journalctl -u caddy -f` | live HTTPS access log |
| `systemctl status sfci caddy sfci-keepalive.timer cloudflare-ufw.timer` | service health |
| `curl https://sfcommutemap.com/` (from anywhere) | end-to-end probe |
| Cloudflare dashboard → Analytics | edge traffic / cache hit ratio |
| `ssh opc@<ip> 'free -h'` | RSS sanity (sfci ~250 MB warm) |

## Rollback

```
ssh opc@<ip> 'cd /opt/sfci && git reset --hard HEAD~1 && sudo systemctl restart sfci'
```

(Works because `push.sh` includes `.git/`.)

## Component layout

```
/opt/sfci/                          # repo (rsynced via push.sh)
  scripts/server.py                 # Flask entrypoint
  .venv/                            # Python venv (created by install.sh)
  .numba_cache/                     # JIT cache (persisted across restarts; safe to delete)
  data/                             # bakes (~32 MB; speed up boot)
  deploy/                           # this directory
/etc/sfci.env                       # secrets + flags (root-owned, 600)
/etc/caddy/Caddyfile                # reverse proxy
/etc/caddy/origin-cert.pem          # CF Origin CA cert (15-year)
/etc/caddy/origin-key.pem
/etc/systemd/system/sfci.service
                  sfci-keepalive.{service,timer}
                  cloudflare-ufw.{service,timer}
/var/log/sfci/                      # app logs (logrotate weekly, 4 rotations)
/var/log/caddy/                     # access logs (Caddy-managed rotation)
```
