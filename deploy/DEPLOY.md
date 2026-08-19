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
the numba cache (rebuilt on the box), `out/`, and the test golden oracles. It also copies `.git`,
but ordinary updates and recovery use `deploy/push.sh` from the laptop; do not assume Git is
installed on the host.

## 2. Run install on the box

```
ssh opc@<ip> 'sudo bash /opt/sfci/deploy/install.sh'
```

Does, idempotently:
- Adds the Caddy COPR + EPEL repos; installs `python3.12`, `caddy`, `stress-ng`, `firewalld`, build tools.
- Creates the `sfci` system user + Python venv at `/opt/sfci/.venv`.
- Writes `/etc/sfci.env` with sensible defaults (you fill in tokens next).
- Installs the systemd units: `sfci`, `sfci-keepalive.timer`, `cloudflare-ufw.timer`.
- Initializes firewalld with 22/80/443 temporarily open (cloudflare-ufw.timer narrows 80/443 later).
- Starts the Flask app on `localhost:8000`.
- Enables Caddy but doesn't start it yet (waits for the Origin CA cert in step 4).
- Sets up `logrotate` for the app logs.
- Sends Caddy's JSON access records to systemd-journald (view with `journalctl -u caddy`).

Verify the Flask side is up locally on the box:

```
ssh opc@<ip> 'curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8000/'
# → HTTP 200
```

## 3. Fill in /etc/sfci.env with secrets

```
ssh opc@<ip> 'sudo vi /etc/sfci.env'
```

Paste your `API511_TOKEN` and, if desired, `GEOAPIFY_KEY` from your laptop's `.env`. For Oracle
production, set `GEOCODER=photon` even when retaining a `GEOAPIFY_KEY`: the host can reach Photon
but has timed out reaching `api.geoapify.com`, and without an explicit provider the key selects
Geoapify. Save, then:

```
ssh opc@<ip> 'sudo systemctl restart sfci && sudo journalctl -u sfci -n 30 --no-pager'
```

You should see the `[boot] RAPTOR engine ON … WALK GRAPH ON` line.

Verify provider reachability through the application, rather than merely checking that a key is
present:

```
ssh opc@<ip> 'curl -fsS "http://localhost:8000/geocode?q=Ferry%20Building"'
```

This must return a JSON result with `lat`, `lon`, and `label`; a 502 means the selected upstream
geocoder is not reachable from the host.

> **Existing boxes (installed before 2026-07-11):** the old `/etc/sfci.env` template pinned
> `RAPTOR_SEMANTIC=arriveby`, which predates the 2026-06-17 default flip to **departafter**
> (the R5-validated served map). install.sh preserves user edits, so re-running it will NOT
> fix this — do a one-time edit: delete the `RAPTOR_SEMANTIC=arriveby` line from
> `/etc/sfci.env` (the code default, departafter, then rules) and `sudo systemctl restart
> sfci`. Verify with `curl -s localhost:8000/healthz` → `"semantic": "departafter"`.
> Likewise delete any `DEFAULT_ADDRESS=` line — the served page stopped injecting a default
> workplace on 2026-06-13 (privacy invariant); the setting has no effect.

## 4. Cloudflare Origin CA cert (end-to-end HTTPS without port 80 exposure)

This is what lets us drop port 80 entirely (no Let's Encrypt HTTP-01 challenge dependency) and
keep 80/443 locked to Cloudflare IPs only.

1. Cloudflare dashboard → SSL/TLS → **Origin Server** → **Create Certificate**.
2. Hostnames: `sfcommutemap.com, www.sfcommutemap.com`. Keep this exact list synchronized with
   `deploy/Caddyfile`. Validity: **15 years** (default). Key: RSA. The co-hosted bus app uses its
   own `bus.sfcommutemap.com` vhost and dedicated Origin CA certificate.
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

`cloudflare-ufw.timer` fires ~2 min after boot and weekly thereafter; despite its historic name,
it refreshes a dedicated firewalld zone so only Cloudflare CIDRs can reach 80/443. To force it now:

```
ssh opc@<ip> 'sudo systemctl start cloudflare-ufw && sudo firewall-cmd --zone=public --list-services && sudo firewall-cmd --zone=cloudflare --list-sources'
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

`push.sh` only rsyncs the diff. Restart runs `scripts/warm_numba.py` through the unit's
`ExecStartPre` before accepting traffic, so changed compiled signatures are populated in the same
Numba cache on every ordinary redeploy rather than by the first visitor. Its public pin covers the
normal journey flow, and a separate tiny synthetic schedule warms the optional planned one-transfer
dispatcher even on a service day with no matching transfer topology. Caddy returns 502 during the
restart; its duration is warmup-dependent rather than a fixed ~3 seconds. For zero-downtime in v2
we'd add gunicorn + systemd socket activation;
skipped for v1 because the audience is tiny and the restart is rare.

When an update changes `deploy/Caddyfile`, apply it through `install.sh`, which preserves the
co-hosted bus app's fenced vhost. Then validate the merged live file and restart Caddy:

```
ssh opc@<ip> 'sudo bash /opt/sfci/deploy/install.sh && sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl restart caddy'
```

Do not install the sfci base Caddyfile directly over `/etc/caddy/Caddyfile`; doing so removes
co-hosted fenced vhosts such as `bus.sfcommutemap.com`.

The Caddyfile deliberately sets `admin off`; the packaged `ExecReload` posts to the default
admin endpoint (`localhost:2019`), so `systemctl reload caddy` is unavailable. A restart is
therefore the supported, validated configuration-apply path.

Caddy access records remain JSON and are retained by journald; inspect them with
`sudo journalctl -u caddy -f`.

## Health checks

| Command | What you learn |
|---|---|
| `journalctl -u sfci -f` | live app logs |
| `journalctl -u caddy -f` | live HTTPS access log |
| `systemctl status sfci caddy sfci-keepalive.timer cloudflare-ufw.timer` | service health |
| `ssh opc@<ip> 'curl -fsS "http://localhost:8000/geocode?q=Ferry%20Building"'` | application-to-geocoder reachability; JSON coordinates/label, not 502 |
| `curl https://sfcommutemap.com/` (from anywhere) | end-to-end probe |
| Cloudflare dashboard → Analytics | edge traffic / cache hit ratio |
| `ssh opc@<ip> 'free -h'` | RSS sanity (sfci ~250 MB warm) |

## Rollback

To roll back, restore the saved known-good file set or redeploy that release with
`deploy/push.sh`, then restart the service:

```
deploy/push.sh opc@<ip>
ssh opc@<ip> 'sudo systemctl restart sfci'
```

`push.sh` can rsync an uncommitted working tree; therefore do not use blind `HEAD~1` as a rollback
target. Keep a saved known-good export (or an identified clean local checkout) for recovery.

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
```
