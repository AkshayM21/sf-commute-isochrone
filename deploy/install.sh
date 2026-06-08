#!/usr/bin/env bash
# Server-side install for the sfci box. Runs on the freshly-provisioned Oracle Always Free A1
# (Oracle Linux 9 ARM). Re-runnable / idempotent — safe to re-run after edits.
#
# Run order:
#   1) deploy/push.sh opc@<ip>     (from your laptop, rsyncs the repo to /opt/sfci)
#   2) ssh opc@<ip> 'sudo bash /opt/sfci/deploy/install.sh'      (this script)
#   3) edit /etc/sfci.env to add secrets
#   4) install the Cloudflare Origin CA cert (see deploy/DEPLOY.md)
#   5) systemctl start caddy

set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo"; exit 1; }

REPO_DIR="/opt/sfci"
USER_NAME="sfci"
PY_BIN="python3.12"

log() { printf '[install] %s\n' "$*"; }

[[ -d "$REPO_DIR" ]] || { echo "$REPO_DIR not found — run deploy/push.sh from your laptop first"; exit 1; }

# ---- 1. system packages -------------------------------------------------------------------
# python3.12 — repo Python is 3.9/3.11; we want 3.12 for current numba/numpy wheels.
# caddy + stress-ng — repos may not have these; the install.sh adds them via the official COPRs.
log "installing system packages..."
dnf install -y -q dnf-plugins-core

# Add the Caddy COPR (official)
if ! dnf repolist | grep -q '@copr:copr.fedorainfracloud.org:group_caddy'; then
  log "  enabling Caddy COPR..."
  dnf copr enable -y @caddy/caddy
fi
# stress-ng comes from EPEL
if ! rpm -q epel-release >/dev/null 2>&1; then
  log "  enabling EPEL..."
  dnf install -y -q epel-release
fi

dnf install -y -q \
  "$PY_BIN" "$PY_BIN"-pip "$PY_BIN"-devel \
  gcc gcc-c++ make \
  caddy \
  stress-ng \
  ufw \
  curl jq tar rsync logrotate

# ---- 2. dedicated user --------------------------------------------------------------------
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  log "creating $USER_NAME user..."
  useradd -r -d "$REPO_DIR" -s /usr/sbin/nologin "$USER_NAME"
fi
# Repo files owned by sfci so the systemd unit (User=sfci) can read them.
chown -R "$USER_NAME:$USER_NAME" "$REPO_DIR"

# ---- 3. Python venv + deps ---------------------------------------------------------------
if [[ ! -d "$REPO_DIR/.venv" ]]; then
  log "creating Python venv..."
  sudo -u "$USER_NAME" "$PY_BIN" -m venv "$REPO_DIR/.venv"
fi
log "installing/upgrading Python dependencies..."
sudo -u "$USER_NAME" "$REPO_DIR/.venv/bin/pip" install --quiet --upgrade pip wheel setuptools
if [[ -f "$REPO_DIR/requirements.txt" ]]; then
  sudo -u "$USER_NAME" "$REPO_DIR/.venv/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"
else
  # Fallback: the core minimum to run the JVM-free server. (Should already be in requirements.txt;
  # this is a safety net so install.sh succeeds even on a slimmed-down checkout.)
  log "  (no requirements.txt -- installing baseline set)"
  sudo -u "$USER_NAME" "$REPO_DIR/.venv/bin/pip" install --quiet \
    flask flask-limiter numpy scipy numba esy-osm-pbf rasterio python-dotenv
fi
# waitress: production WSGI server. server.py prefers it over Flask's dev server when present.
sudo -u "$USER_NAME" "$REPO_DIR/.venv/bin/pip" install --quiet waitress

# ---- 4. /etc/sfci.env (preserve any user edits on re-run) --------------------------------
if [[ ! -f /etc/sfci.env ]]; then
  log "writing /etc/sfci.env (edit to paste secrets)..."
  cat > /etc/sfci.env <<'EOF'
# sfci runtime env. systemd loads this for the sfci service. Owner: root, mode 600.
# After editing, run: sudo systemctl restart sfci

# Engine flags (all default ON; flip to 0 to fall back to legacy R5 path -- requires JVM)
USE_RAPTOR=1
USE_WALK_GRAPH=1
RAPTOR_SEMANTIC=arriveby
RAPTOR_MC=1

# Server
PORT=8000

# Numba threading -- 1 OCPU box; cap to avoid oversubscription
NUMBA_NUM_THREADS=2

# Secrets -- paste from your laptop's .env
API511_TOKEN=
GEOAPIFY_KEY=

# Default workplace shown to first-time visitors (server injects it as CFG.default_wp)
DEFAULT_ADDRESS=650 Townsend St, San Francisco, CA
EOF
  chmod 600 /etc/sfci.env
  chown root:root /etc/sfci.env
  log "  (!) /etc/sfci.env has empty API tokens. Edit before going live."
fi

# ---- 5. runtime-writable dirs -------------------------------------------------------------
# /var/cache/sfci is the NUMBA_CACHE_DIR (CacheDirectory= in the unit, but pre-create so the
# pre-warm script below can use it before systemd touches it).
mkdir -p /var/log/sfci /var/log/caddy /var/cache/sfci/numba "$REPO_DIR/data"
chown -R "$USER_NAME:$USER_NAME" /var/log/sfci /var/cache/sfci "$REPO_DIR/data"
chown -R caddy:caddy /var/log/caddy 2>/dev/null || true

# ---- 6. systemd units --------------------------------------------------------------------
log "installing systemd units..."
install -m 644 "$REPO_DIR/deploy/sfci.service"            /etc/systemd/system/sfci.service
install -m 644 "$REPO_DIR/deploy/sfci-keepalive.service"  /etc/systemd/system/sfci-keepalive.service
install -m 644 "$REPO_DIR/deploy/sfci-keepalive.timer"    /etc/systemd/system/sfci-keepalive.timer
install -m 644 "$REPO_DIR/deploy/cloudflare-ufw.service"  /etc/systemd/system/cloudflare-ufw.service
install -m 644 "$REPO_DIR/deploy/cloudflare-ufw.timer"    /etc/systemd/system/cloudflare-ufw.timer

# ---- 7. Caddy config ---------------------------------------------------------------------
mkdir -p /etc/caddy
install -m 644 "$REPO_DIR/deploy/Caddyfile" /etc/caddy/Caddyfile

# ---- 8. ufw bootstrap (cloudflare-ufw.timer narrows 80/443 to CF IPs after first run) ----
if ! ufw status | grep -q "Status: active"; then
  log "initializing ufw (allow 22/80/443, deny everything else)..."
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow 22/tcp     comment 'SSH'
  ufw allow 80/tcp     comment 'HTTP — narrowed to CF on first cloudflare-ufw run'
  ufw allow 443/tcp    comment 'HTTPS — narrowed to CF on first cloudflare-ufw run'
  ufw --force enable
fi

# ---- 9. log rotation ---------------------------------------------------------------------
cat > /etc/logrotate.d/sfci <<'EOF'
/var/log/sfci/*.log {
  weekly
  rotate 4
  compress
  delaycompress
  missingok
  notifempty
  create 640 sfci sfci
}
EOF

# ---- 10. enable & start ------------------------------------------------------------------
log "reloading systemd + enabling services..."
systemctl daemon-reload
systemctl enable --now sfci
systemctl enable --now sfci-keepalive.timer
systemctl enable --now cloudflare-ufw.timer
# Caddy: enable but don't start (needs the Origin CA cert installed first; see DEPLOY.md)
systemctl enable caddy

# ---- summary -----------------------------------------------------------------------------
log "DONE.  Status:"
systemctl --no-pager --plain status sfci sfci-keepalive.timer cloudflare-ufw.timer 2>&1 | head -25 || true

cat <<'EOF'

[install] Next steps:
  1. Edit /etc/sfci.env to paste API511_TOKEN + GEOAPIFY_KEY, then:
       sudo systemctl restart sfci
  2. Install Cloudflare Origin CA cert at /etc/caddy/origin-cert.pem + origin-key.pem
       (see deploy/DEPLOY.md section 4)
       sudo systemctl start caddy
  3. Point Cloudflare DNS at this box's public IP (proxied / orange-cloud).
  4. After ~2 min, cloudflare-ufw.timer fires automatically and narrows 80/443 to CF IPs only.
       To force it now: sudo systemctl start cloudflare-ufw

Tail logs:   sudo journalctl -u sfci -f
             sudo journalctl -u caddy -f
EOF
