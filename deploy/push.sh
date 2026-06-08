#!/usr/bin/env bash
# Run on your laptop. Rsync the repo to /opt/sfci on the Oracle box, then SSH a couple of
# next-step suggestions. Re-runnable: subsequent calls only push the diff.
#
# Usage:
#   deploy/push.sh opc@<oracle-public-ip>            # Oracle Linux uses opc; Ubuntu uses ubuntu
#
# Excludes the venv, the numba cache (rebuilt on the box), the personal .dest_cache, the test
# golden oracles, and out/ — everything the box doesn't need or shouldn't have.

set -euo pipefail
TARGET="${1:-}"
[[ -n "$TARGET" ]] || { echo "Usage: $0 <user@oracle-public-ip>"; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "[push] preparing /opt/sfci on $TARGET (one-time mkdir + chown)..."
ssh -o StrictHostKeyChecking=accept-new "$TARGET" \
  "sudo mkdir -p /opt/sfci && sudo chown -R \$USER:\$USER /opt/sfci"

echo "[push] rsyncing $REPO_ROOT -> $TARGET:/opt/sfci ..."
# Keep .git so the box can `git pull` for incremental updates later. Drop everything else
# big or local-only.
rsync -avzP --delete-after \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude '.dest_cache.json' \
  --exclude '.nbc' \
  --exclude '.numba_cache' \
  --exclude '/out/' \
  --exclude '/tests/raptor_golden/' \
  --exclude '/tests/__pycache__/' \
  --exclude '/data/osm_sf.pbf' \
  ./ "$TARGET:/opt/sfci/"

cat <<EOF

[push] done.

Next steps on the box:
  1. ssh $TARGET 'sudo bash /opt/sfci/deploy/install.sh'      # system deps, venv, units
  2. ssh $TARGET 'sudo vi /etc/sfci.env'                       # paste API tokens
  3. ssh $TARGET 'sudo systemctl restart sfci'
  4. See deploy/DEPLOY.md sections 4-8 for Cloudflare Origin CA cert + DNS + lockdown.

To redeploy after code changes later:
  deploy/push.sh $TARGET
  ssh $TARGET 'sudo systemctl restart sfci'
EOF
