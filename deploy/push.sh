#!/usr/bin/env bash
# Run from a trusted development machine. Rsync the deployable source tree to the host, then
# print the next-step commands. Re-runnable: subsequent calls only push the diff.
#
# Usage:
#   deploy/push.sh <user@host>
#
# Excludes development-only state, credentials, Git metadata, browser captures, test artifacts,
# session notes, and source extracts that the runtime does not need. Patterns use only rsync's
# portable include/exclude syntax and work with the macOS-bundled rsync. The remote data directory
# is protected from deletion so a clean source checkout cannot erase host-built feeds and bakes;
# locally present data files can still be updated deliberately.
#
# PRIVACY/SECRETS INVARIANT: credentials and Cloudflare certificate tooling stay on the trusted
# machine. ``--delete-after`` preserves excluded paths already present on the host; remove any
# legacy sensitive files from the host deliberately after confirming they are no longer needed.

set -euo pipefail
TARGET="${1:-}"
[[ -n "$TARGET" ]] || { echo "Usage: $0 <user@host>"; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Runs on every push: rsync needs ownership while writing. The service ownership is restored
# afterwards if the service user exists.
echo "[push] preparing /opt/sfci on $TARGET (mkdir + temporary write ownership)..."
ssh -o StrictHostKeyChecking=accept-new "$TARGET" \
  "sudo mkdir -p /opt/sfci && sudo chown -R \$USER:\$USER /opt/sfci"

echo "[push] rsyncing $REPO_ROOT -> $TARGET:/opt/sfci ..."
# Deploy the source payload, not a developer checkout. Unanchored privacy patterns apply at every
# depth; leading slashes are reserved for intentionally root-scoped runtime paths.
rsync -avzP --delete-after \
  --filter 'protect /data/***' \
  --exclude '.git/' \
  --exclude '.impeccable/' \
  --exclude '.agents/' \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude 'REPORT.md' \
  --exclude 'Progress.md' \
  --exclude 'Issues.md' \
  --exclude 'REVIEW_REPORT.md' \
  --exclude 'HANDOFF*.md' \
  --exclude '*_PLAN_*.md' \
  --exclude 'REVIEW*.md' \
  --exclude '*_DESIGN_*.md' \
  --exclude '*_QA_*.md' \
  --exclude '*_SCOPE.md' \
  --exclude '.plans/' \
  --exclude '/deploy/cf/' \
  --exclude '.dest_cache.json' \
  --exclude '.nbc' \
  --exclude '.numba_cache' \
  --exclude '.nbcache*' \
  --exclude '/out/' \
  --exclude 'artifacts/' \
  --exclude 'screenshots/' \
  --exclude '/tests/e2e/artifacts/' \
  --exclude '/tests/e2e/screens/' \
  --exclude '/tests/e2e/screenshots/' \
  --exclude '.pytest_cache/' \
  --exclude '/tests/raptor_golden/' \
  --exclude '/tests/__pycache__/' \
  --exclude '/data/osm_sf.pbf' \
  --exclude '/data/norcal.osm.pbf' \
  ./ "$TARGET:/opt/sfci/"

# Restore service-user ownership. Guarded because the first push can happen before install.sh
# creates that user.
echo "[push] restoring sfci ownership of /opt/sfci (if the service user exists)..."
ssh "$TARGET" 'if id sfci >/dev/null 2>&1; then sudo chown -R sfci:sfci /opt/sfci; fi'

cat <<EOF

[push] done.

Next steps on the box:
  1. ssh $TARGET 'sudo bash /opt/sfci/deploy/install.sh'      # system deps, venv, units
  2. ssh $TARGET 'sudo vi /etc/sfci.env'                       # set runtime configuration
  3. ssh $TARGET 'sudo systemctl restart sfci'
  4. See deploy/DEPLOY.md for Cloudflare HTTPS, DNS, and ingress configuration.

To redeploy after code changes later:
  deploy/push.sh $TARGET
  ssh $TARGET 'sudo systemctl restart sfci'  # ExecStartPre warms changed Numba signatures
EOF
