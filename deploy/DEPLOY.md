# Production deployment guide

This runbook documents the project's current Oracle Linux 9 (`dnf`/systemd) deployment for
`sfcommutemap.com` behind Cloudflare. It uses placeholders for the SSH host and credentials so
those values never enter the repository. Forks using another distribution or public hostname must
adapt `deploy/install.sh`, `deploy/Caddyfile`, the Origin CA hostnames, and DNS together.

## Prerequisites

- An Oracle Linux 9 host you can reach as `<user@host>` with `sudo` access.
- A validated immutable data release under `/opt/sfci/data-releases/<service-date-or-timestamp>`
  and a `data-current` pointer created by the trusted data-refresh helper below. Code pushes never
  carry feeds, graphs, or RAPTOR bakes.
- Cloudflare Origin CA files already installed at `/etc/caddy/origin-cert.pem` and
  `/etc/caddy/origin-key.pem`; the installer validates the complete temporary Caddy configuration
  before changing the live one.
- An explicit public HTTPS smoke target. Every promotion must set `SFCI_PUBLIC_SMOKE_URL` to the
  proxied production origin (normally `https://sfcommutemap.com`); the installer has no HTTP or
  skip-smoke fallback.
- The Cloudflare zone for `sfcommutemap.com`, with `CF_ZONE_ID` available only in a local `.env`.
- A local `.env` based on [`.env.example`](../.env.example). It is ignored by Git and is never
  copied by `push.sh`.

## 1. Stage and promote a release

```bash
deploy/push.sh <user@host>
```

The script stages runtime source under `/opt/sfci/.incoming/<UTC timestamp>`. It excludes
credentials, `.git`, agent/session state, caches, screenshots, test artifacts, raw source
extracts, symlink-bearing agent worktrees, and Cloudflare certificate tooling. It then copies only
`install.sh`, `release_ops.sh`, and `data-refresh.sh` into a versioned root-owned mode-700 control
directory. Root never executes or sources a file from mutable `.incoming`.

Run the exact promotion command printed by `push.sh`:

```bash
ssh <user@host> 'sudo env SFCI_PUBLIC_SMOKE_URL=https://sfcommutemap.com /opt/sfci/.deploy-control/<release>/install.sh --release-id <release>'
```

The installer requires `flock` before bootstrap and holds one stable descriptor lock for the full
operation. It secures the repository parent and lock before use, rejects every payload symlink and
hardlinked regular file, and makes payload source root-owned/read-only before reading requirements,
units, Caddy configuration, or Python. Only the installer-created `.venv` is temporarily writable
by `sfci`; it is frozen before execution.

Candidate validation installs a runtime-only copy of the exact production `sfci.service` template
with only the release path, private port, and candidate-specific cache directory replaced. Both
candidate and live service execute
`scripts/bootstrap_server.py`, which binds liveness before importing the heavy runtime; there is no
pre-bind `ExecStartPre`. Candidate and production therefore share the same environment file,
runtime settings, limits, and sandbox without writing into the live service's caches. Before each
attempt the installer validates and removes only trusted timestamp-named candidate units left by
an interrupted prior install, then proves the private candidate port is unbound. The candidate
probes `/livez` promptly and `/readyz` after load, then validates root, compute, itinerary, and
variance response shapes. Its unit/cgroup and isolated cache are always stopped and removed. Only
after those checks pass may live units or Caddy change.

Post-cutover checks stay under rollback protection. The installer unconditionally runs the
Cloudflare firewalld refresh, proves that the source-only `cloudflare` zone has HTTP/S, and checks
every permanent and runtime zone for HTTP/S services, custom services, raw/ranged ports, rich
rules, forward rules, direct rules/passthroughs, policies, and ACCEPT targets. If closure cannot be
proved, it stops both Caddy and sfci. It then reloads Caddy, exercises the local proxy without
ambient proxy variables, and calls exactly `https://sfcommutemap.com` without proxies or redirects.
The independently recorded old `current` and `previous` pointers are restored exactly on any
failure; origin certificate/key ownership and modes, any pre-existing environment file, and the
old proxy/runtime metadata are also restored exactly. A first-install environment file is removed
if that install fails. Rollback and interrupted
recovery first remove public HTTP/S so they fail closed even if the old service must be restored.

The layout is intentionally simple:

```text
/opt/sfci/current  -> releases/<timestamp>
/opt/sfci/previous -> releases/<timestamp>
/opt/sfci/releases/<timestamp>/.venv
/opt/sfci/releases/<timestamp>/data -> ../../data-releases/<data-id>
/opt/sfci/data-current -> data-releases/<data-id>
/opt/sfci/data-previous -> data-releases/<data-id>
/opt/sfci/data-releases/<data-id>      # immutable feeds, graph, and bakes
```

The active release, rollback release, and one newest additional release are retained. Every data
release pinned by those retained code releases, plus `data-current` and `data-previous`, is
retained. Timestamped code and data stages abandoned for more than seven days are cleaned only
while holding the deployment lock; requested, active, rollback, and code-pinned data IDs are
preserved.

### Build and promote data separately

Build/rebuild all three named feeds (`muni_current.zip`, `bart_gtfs.zip`, and `caltrain.zip`),
`server_static.json`, `walk_graph.npz`, the neighborhood grid source, and `raptor_cache/` outside
every active data release. The existing fetch/build programs still resolve the checkout's `data/`
directory, so perform that build in an inactive checkout/data tree; never aim them at
`data-current` or a concrete `data-releases` directory. Upload the complete tree as root so the
remote SSH account never owns the stage (the source path below is illustrative):

```bash
ssh <user@host> 'sudo install -d -m 700 -o root -g root /opt/sfci/data-incoming/<data-id>'
rsync -av --delete --no-owner --no-group --rsync-path='sudo rsync' \
  <inactive-data-tree>/ <user@host>:/opt/sfci/data-incoming/<data-id>/
```

Use an eight-digit service date or fourteen-digit UTC timestamp and run the root-owned helper
printed by `push.sh`:

```bash
ssh <user@host> 'sudo /opt/sfci/.deploy-control/<code-release>/data-refresh.sh --promote-id <data-id>'
```

The helper takes the same deployment lock, rejects symlinks and hardlinks, freezes the staged tree
root-owned/read-only, and requires the three feeds plus the complete JSON/walk/RAPTOR/access/static
artifact set. Using the active immutable code release and virtualenv, it runs the canonical
readiness validator against the stage at midnight on the service date declared by the staged
bundle, before the configured departure time can roll the model to the following week. The service
date must be an exact Wednesday, every required feed must serve that exact modeled date, and every
derived artifact must match that service date and grid. The incoming and release directories must
share a filesystem so the move is atomic. Only then does it move
the stage into `data-releases` and atomically advance `data-current`, recording the old target in
`data-previous`. A root-only transaction record preserves the exact old pointers across signals,
process crashes, or host restarts; the next refresh recovers an unfinished switch before doing any
new promotion work. Transaction replacements, stage moves, and release-pointer directory updates
are durably synced before the next state transition. Running code remains pinned to its
concrete old data release. The next code release pins the new concrete target. The helper refuses
ordinary data promotion until the code host uses immutable release pointers.

### One-time migration from the legacy in-place host

If an older deployment has a real mutable `/opt/sfci/data`, first use the control helper printed by
`push.sh` while choosing a service date or timestamp. On a host that predates the installer, first
ensure its small validation/locking prerequisites are present:

```bash
ssh <user@host> 'sudo dnf install -y jq unzip util-linux'
ssh <user@host> 'sudo /opt/sfci/.deploy-control/<release>/data-refresh.sh --adopt-legacy YYYYMMDD'
```

This locked maintenance action records whether sfci/Caddy are active, stops readers, renames and
freezes the data tree as an immutable data release, creates `data-current` plus a compatibility
`data` link, and restores only services that were active. It never copies data into code.

Do not copy the legacy working tree, virtualenv, or credentials into a release. The code installer
snapshots all existing runtime unit files even when inactive, enabled/active state for services and
timers, logrotate state, and Caddy file/active state into root-only rollback storage. The legacy
working tree stays in place. A failed or interrupted first cutover removes the failed `current`
pointer and restores that exact state.

The stable legacy `/opt/sfci/deploy` directory is moved aside only inside the protected cutover so
the new release's firewall timer can resolve through `/opt/sfci/deploy -> current/deploy`; rollback
puts the directory back idempotently. After public HTTPS and firewall verification succeeds, the
old deploy helper snapshot is removed. Remove the remaining legacy development tree only after the
new release is healthy and you have separately confirmed it contains no externally managed data:

```bash
ssh <user@host> 'sudo rm -rf /opt/sfci/.git /opt/sfci/.impeccable /opt/sfci/.agents'
```

## 2. Resume an interrupted install

Once staging has moved into `releases/`, the original `.incoming` path no longer exists. The
installer prints the exact resumable path before doing expensive or live work:

```bash
ssh <user@host> 'sudo env SFCI_PUBLIC_SMOKE_URL=https://sfcommutemap.com /opt/sfci/.deploy-control/<release>/install.sh --release-id <release>'
```

A small transaction record survives an interruption during cutover. The next locked install first
restores the independently recorded old release (or the live legacy unit on the one-time
migration), stable deploy path, and proxy configuration. Recovery also removes public HTTP/S and
refuses to clear the transaction unless the origin is fail-closed. Transaction and rollback
artifacts must remain root-owned, canonical, and restrictive; malformed recovery state is rejected
instead of guessed.

## 3. Configure runtime secrets on the host

```bash
ssh <user@host> 'sudo vi /etc/sfci.env'
ssh <user@host> 'sudo systemctl restart sfci'
```

Set only the credentials and provider settings your deployment needs, such as `API511_TOKEN`,
`GEOAPIFY_KEY`, and (when using Nominatim publicly) `GEOCODER_USER_AGENT`. Keep this file
root-owned, single-link, and mode 600; do not set `PORT`. Origin certificate and key files must be
single-link regular files and are enforced as root:caddy mode 640. Verify the local service without
exposing it publicly:

```bash
ssh <user@host> 'curl -fsS http://localhost:8000/readyz'
```

## 4. Configure Cloudflare HTTPS and DNS

Create an Origin CA certificate in the Cloudflare dashboard for the hostnames configured in your
Caddyfile. Transfer the certificate and private key to the host through a secure channel and
install them under `/etc/caddy/` with restrictive permissions before the first release promotion.
Do not commit or rsync the private key.

Set Cloudflare SSL/TLS mode to **Full (strict)**. To update the apex and `www` A records through
the helper, add an explicit zone ID and an appropriately scoped Cloudflare credential to your
local `.env`, then run:

```bash
python3 deploy/cf/dns.py <origin-ip>
```

`dns.py` refuses to run without `CF_ZONE_ID`, discovers the zone name through the API, and creates
or updates proxied apex and `www` records. Confirm the hostname is proxied before restricting
origin ingress to Cloudflare address ranges.

The firewall updater parses every Cloudflare source as a strict IPv4 or IPv6 network and rejects
malformed, host-bit, wrong-family, or dangerously broad ranges before changing the zone. The
installer repeats that validation against both permanent and runtime firewalld state.

## 5. Verify, update, and roll back

```bash
curl -fsSI https://<your-public-hostname>/
ssh <user@host> 'sudo systemctl status sfci caddy --no-pager'
```

For ordinary code updates, repeat the source staging and run the printed promotion command. Do not
rsync into `/opt/sfci/current` and do not restart the service before the installer has completed
its candidate checks:

```bash
deploy/push.sh <user@host>
```

Use `journalctl -u sfci -f` and `journalctl -u caddy -f` on the host for diagnostics. For a
rollback, the normal post-cutover failure path restores `/opt/sfci/previous` automatically. For
an operator-requested rollback, switch the pointers only through a reviewed release operation and
then restart `sfci`; keep the old release directory until the replacement is healthy. Avoid
treating an uncommitted local tree as a release artifact.
