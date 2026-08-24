#!/usr/bin/env bash
# Run from a trusted development machine. Rsync the deployable source tree to a timestamped
# server-side staging directory, then print the install command. The installer validates and
# promotes that staged payload without ever rsyncing into the active release.
#
# Usage:
#   deploy/push.sh <user@host>
#
# Excludes development-only state, credentials, Git metadata, browser captures, test artifacts,
# session notes, and source extracts that the runtime does not need. Patterns use only rsync's
# portable include/exclude syntax and work with the macOS-bundled rsync. Runtime data is external
# to code releases and is never copied or deleted by a source push.
#
# PRIVACY/SECRETS INVARIANT: credentials and Cloudflare certificate tooling stay on the trusted
# machine. Staging is isolated from both the active release and the shared data directory.

set -euo pipefail
TARGET="${1:-}"
[[ -n "$TARGET" ]] || { echo "Usage: $0 <user@host>"; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
RELEASE_ID="$(date -u +%Y%m%d%H%M%S)"
REMOTE_ROOT="/opt/sfci"
REMOTE_STAGE="$REMOTE_ROOT/.incoming/$RELEASE_ID"

# Runs on every push: root owns the inactive stage from its creation. The remote rsync process also
# runs through sudo, so the SSH account never owns, traverses, or can keep a writable payload FD.
echo "[push] preparing $REMOTE_STAGE on $TARGET (isolated staging payload)..."
ssh -o StrictHostKeyChecking=accept-new "$TARGET" \
  "sudo test ! -L '$REMOTE_ROOT' && sudo install -d -m 755 -o root -g root '$REMOTE_ROOT' && \
   sudo test ! -L '$REMOTE_ROOT/.incoming' && sudo test ! -L '$REMOTE_ROOT/releases' && \
   sudo install -d -m 755 -o root -g root '$REMOTE_ROOT/.incoming' '$REMOTE_ROOT/releases' && \
   if sudo test -e '$REMOTE_STAGE'; then echo 'release id already exists; retry in the next second' >&2; exit 73; fi && \
   sudo install -d -m 700 -o root -g root '$REMOTE_STAGE'"

echo "[push] rsyncing $REPO_ROOT -> $TARGET:$REMOTE_STAGE ..."
# Deploy the source payload, not a developer checkout. Unanchored privacy patterns apply at every
# depth; leading slashes are reserved for intentionally root-scoped runtime paths.
rsync -avzP --delete --no-owner --no-group --rsync-path='sudo rsync' \
  --exclude '.git/' \
  --exclude '.claude/' \
  --exclude '.impeccable/' \
  --exclude '.agents/' \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '*.pem' \
  --exclude '*.key' \
  --exclude '*.csr' \
  --exclude '*.p12' \
  --exclude '*.pfx' \
  --exclude 'id_*' \
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
  --exclude '/data/***' \
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
  --exclude '/tests/__pycache__/' \
  --exclude '/data/osm_sf.pbf' \
  --exclude '/data/norcal.osm.pbf' \
  ./ "$TARGET:$REMOTE_STAGE/"

# Bootstrap only a tiny versioned control surface into a root-owned, non-writable directory. The
# same root transaction first takes ownership/search permission away from the upload account. Each
# control file is then streamed from this trusted machine directly into that directory as root;
# root never reads the mutable payload copy as executable installer code.
REMOTE_CONTROL="$REMOTE_ROOT/.deploy-control/$RELEASE_ID"
# These paths are intentionally expanded locally into one quoted remote bootstrap transaction.
# shellcheck disable=SC2029
ssh "$TARGET" \
  "sudo test ! -L '$REMOTE_ROOT' && sudo chown root:root '$REMOTE_ROOT' && sudo chmod 755 '$REMOTE_ROOT' && \
   sudo test -d '$REMOTE_STAGE' && sudo test ! -L '$REMOTE_STAGE' && \
   sudo chown root:root '$REMOTE_STAGE' && sudo chmod 700 '$REMOTE_STAGE' && \
   sudo test ! -L '$REMOTE_ROOT/.deploy-control' && \
   sudo install -d -m 700 -o root -g root '$REMOTE_ROOT/.deploy-control' && \
   sudo test ! -e '$REMOTE_CONTROL' && sudo test ! -L '$REMOTE_CONTROL' && \
   sudo install -d -m 700 -o root -g root '$REMOTE_CONTROL'"

stream_control_file() {
  local source="$1" destination="$2" mode="$3"
  # The root-only destination directory prevents a remote account from racing the direct stream.
  # shellcheck disable=SC2029
  ssh "$TARGET" \
    "sudo dd of='$REMOTE_CONTROL/$destination' status=none && \
     sudo chown root:root '$REMOTE_CONTROL/$destination' && \
     sudo chmod '$mode' '$REMOTE_CONTROL/$destination'" < "$source"
}
stream_control_file deploy/install.sh install.sh 500
stream_control_file deploy/release_ops.sh release_ops.sh 400
stream_control_file deploy/data-refresh.sh data-refresh.sh 500

cat <<EOF

[push] done.

Next steps on the box:
  1. ssh $TARGET 'sudo env SFCI_PUBLIC_SMOKE_URL=https://sfcommutemap.com $REMOTE_CONTROL/install.sh --release-id $RELEASE_ID'
                                                               # validate, smoke, cut over
  2. ssh $TARGET 'sudo vi /etc/sfci.env'                       # set runtime configuration
  3. ssh $TARGET 'sudo systemctl status sfci'
  4. See deploy/DEPLOY.md for Cloudflare HTTPS, DNS, and ingress configuration.

To redeploy after code changes later:
  deploy/push.sh $TARGET

If installation stops after staging moves into releases/, resume with:
  ssh $TARGET 'sudo env SFCI_PUBLIC_SMOKE_URL=https://sfcommutemap.com $REMOTE_CONTROL/install.sh --release-id $RELEASE_ID'

If this is a legacy host whose data is still a real /opt/sfci/data directory, run once before
the code promotion (choose a service date or UTC timestamp):
  ssh $TARGET 'sudo $REMOTE_CONTROL/data-refresh.sh --adopt-legacy YYYYMMDD'
EOF
