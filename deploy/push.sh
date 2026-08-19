#!/usr/bin/env bash
# Run on your laptop. Rsync the repo to /opt/sfci on the Oracle box, then SSH a couple of
# next-step suggestions. Re-runnable: subsequent calls only push the diff.
#
# Usage:
#   deploy/push.sh opc@<oracle-public-ip>            # Oracle Linux uses opc; Ubuntu uses ubuntu
#
# Excludes the venv, the numba caches (rebuilt on the box; includes tests/.nbcache_*), the
# personal .dest_cache, the test golden oracles, out/, and the big OSM source extracts
# (osm_sf.pbf + the ~640 MB norcal.osm.pbf setup.sh downloads) — everything the box doesn't
# need or shouldn't have (the box consumes the baked walk_graph.npz, not raw OSM).
#
# PRIVACY/SECRETS INVARIANT: .env, REPORT.md, Progress.md, Issues.md, REVIEW_REPORT.md AND
# deploy/cf/ are excluded (the .md session notes are operator-personal). deploy/cf/ holds the
# Cloudflare Origin CA PRIVATE KEY (origin-key.pem) plus operator-local helper scripts — the box
# never needs them (mint.py/dns.py run on the laptop; the cert/key reach /etc/caddy via the
# separate scp step in DEPLOY.md). NB: rsync --delete-after protects excluded paths, so if an
# old push already shipped deploy/cf/, clean it once: ssh <box> 'sudo rm -rf /opt/sfci/deploy/cf'

set -euo pipefail
TARGET="${1:-}"
[[ -n "$TARGET" ]] || { echo "Usage: $0 <user@oracle-public-ip>"; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Runs on EVERY push (not one-time): rsync runs as $USER, so it must own the tree to overwrite
# the sfci-owned files; ownership is restored to sfci right after the rsync below.
echo "[push] preparing /opt/sfci on $TARGET (mkdir + chown to \$USER so rsync can write)..."
ssh -o StrictHostKeyChecking=accept-new "$TARGET" \
  "sudo mkdir -p /opt/sfci && sudo chown -R \$USER:\$USER /opt/sfci"

echo "[push] rsyncing $REPO_ROOT -> $TARGET:/opt/sfci ..."
# Keep .git so the box can `git pull` for incremental updates later. Drop everything else
# big or local-only.
rsync -avzP --delete-after \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude 'REPORT.md' \
  --exclude 'Progress.md' \
  --exclude 'Issues.md' \
  --exclude 'REVIEW_REPORT.md' \
  --exclude '/deploy/cf/' \
  --exclude '.dest_cache.json' \
  --exclude '.nbc' \
  --exclude '.numba_cache' \
  --exclude '.nbcache*' \
  --exclude '/out/' \
  --exclude '/tests/raptor_golden/' \
  --exclude '/tests/__pycache__/' \
  --exclude '/data/osm_sf.pbf' \
  --exclude '/data/norcal.osm.pbf' \
  ./ "$TARGET:/opt/sfci/"

# Restore service-user ownership so the documented redeploy flow (push + restart, NO install.sh
# re-run) leaves sfci able to write /opt/sfci/data (ReadWritePaths in sfci.service — e.g. the
# RAPTOR CSR cache rebake). Guarded: the first push happens before install.sh creates the user.
echo "[push] restoring sfci ownership of /opt/sfci (if the service user exists)..."
ssh "$TARGET" 'if id sfci >/dev/null 2>&1; then sudo chown -R sfci:sfci /opt/sfci; fi'

cat <<EOF

[push] done.

Next steps on the box:
  1. ssh $TARGET 'sudo bash /opt/sfci/deploy/install.sh'      # system deps, venv, units
  2. ssh $TARGET 'sudo vi /etc/sfci.env'                       # paste API tokens
  3. ssh $TARGET 'sudo systemctl restart sfci'
  4. See deploy/DEPLOY.md sections 4-8 for Cloudflare Origin CA cert + DNS + lockdown.

To redeploy after code changes later:
  deploy/push.sh $TARGET
  ssh $TARGET 'sudo systemctl restart sfci'  # ExecStartPre warms changed Numba signatures
EOF
