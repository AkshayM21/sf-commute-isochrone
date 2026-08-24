#!/usr/bin/env bash
# Validate and atomically promote one timestamped sfci release on Oracle Linux 9.
# Source pushes stage code only; immutable data releases use the separate data-refresh workflow.

set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo" >&2; exit 1; }

REPO_DIR="/opt/sfci"
RELEASES_DIR="$REPO_DIR/releases"
INCOMING_DIR="$REPO_DIR/.incoming"
DATA_RELEASES_DIR="$REPO_DIR/data-releases"
DATA_CURRENT="$REPO_DIR/data-current"
TRANSACTION_FILE="$REPO_DIR/.deploy-transaction"
UNIT_ROLLBACK_DIR="$REPO_DIR/.deploy-unit-rollback"
DEPLOY_ROLLBACK_DIR="$REPO_DIR/.deploy-tree-rollback"
CADDYFILE="/etc/caddy/Caddyfile"
CADDY_ROLLBACK="$UNIT_ROLLBACK_DIR/Caddyfile"
LOGROTATE_FILE="/etc/logrotate.d/sfci"
SFCI_ENV="/etc/sfci.env"
ORIGIN_CERT="/etc/caddy/origin-cert.pem"
ORIGIN_KEY="/etc/caddy/origin-key.pem"
USER_NAME="sfci"
PY_BIN="python3.12"
PRODUCTION_PORT="8000"
CANDIDATE_PORT="18080"
RUNTIME_UNITS=(sfci.service sfci-keepalive.service sfci-keepalive.timer cloudflare-ufw.service cloudflare-ufw.timer)

usage() {
  cat >&2 <<'EOF'
Usage: SFCI_PUBLIC_SMOKE_URL=https://public-host install.sh --release-id YYYYMMDDHHMMSS

The release must exist at /opt/sfci/.incoming/<id> (first attempt) or
/opt/sfci/releases/<id> (resume after staging was promoted). The HTTPS public smoke URL is
required. Legacy in-place trees are never copied into a release; stage a clean payload with
deploy/push.sh.
EOF
  exit 2
}

[[ "${1:-}" == "--release-id" && "${2:-}" =~ ^[0-9]{14}$ && -z "${3:-}" ]] || usage
RELEASE_ID="$2"
ENV_MARKER="$REPO_DIR/.sfci-env-created-$RELEASE_ID"
PUBLIC_SMOKE_URL="${SFCI_PUBLIC_SMOKE_URL:-}"
PUBLIC_HOST="sfcommutemap.com"
[[ "$PUBLIC_SMOKE_URL" == "https://$PUBLIC_HOST" ]] || {
  echo "SFCI_PUBLIC_SMOKE_URL must be exactly https://$PUBLIC_HOST" >&2
  exit 2
}
PUBLIC_SMOKE_URL="${PUBLIC_SMOKE_URL%/}"

log() { printf '[install] %s\n' "$*"; }

# Secure the repository parent before consulting control paths or opening the lock. This closes an
# old user-writable host layout before any root-owned deployment material is read.
[[ ! -L "$REPO_DIR" ]] || { echo "$REPO_DIR must not be a symlink" >&2; exit 1; }
mkdir -p "$REPO_DIR"
chown root:root "$REPO_DIR"
chmod 755 "$REPO_DIR"
[[ "$(cd "$REPO_DIR" && pwd -P)" == "$REPO_DIR" ]] || {
  echo "$REPO_DIR does not resolve to its expected canonical path" >&2; exit 1
}

# This program and its helper must have been copied by push.sh into a versioned root-only control
# directory. Never execute or source the mutable payload under .incoming as root.
CONTROL_DIR="$(cd "$(dirname "$0")" && pwd -P)"
EXPECTED_CONTROL_DIR="$REPO_DIR/.deploy-control/$RELEASE_ID"
[[ "$CONTROL_DIR" == "$EXPECTED_CONTROL_DIR" && ! -L "$CONTROL_DIR" ]] || {
  echo "run the trusted installer at $EXPECTED_CONTROL_DIR/install.sh" >&2
  exit 1
}
[[ ! -L "$REPO_DIR/.deploy-control" &&
   "$(stat -c '%u:%g:%a' "$REPO_DIR/.deploy-control")" == "0:0:700" ]] || {
  echo "$REPO_DIR/.deploy-control must be a root-owned mode-700 real directory" >&2; exit 1
}
trusted_path_ok() {
  local path="$1" expected_mode="$2" actual
  [[ -f "$path" && ! -L "$path" ]] || return 1
  actual="$(stat -c '%u:%g:%a:%h' "$path")"
  [[ "$actual" == "0:0:$expected_mode:1" ]]
}
[[ "$(stat -c '%u:%g:%a' "$CONTROL_DIR")" == "0:0:700" ]] || {
  echo "$CONTROL_DIR must be root:root mode 700" >&2; exit 1
}
trusted_path_ok "$CONTROL_DIR/install.sh" 500 || {
  echo "trusted install.sh must be a root-owned, single-link regular file with mode 500" >&2; exit 1
}
trusted_path_ok "$CONTROL_DIR/release_ops.sh" 400 || {
  echo "trusted release_ops.sh must be a root-owned, single-link regular file with mode 400" >&2; exit 1
}

# Production has no best-effort PID-file fallback.
command -v flock >/dev/null 2>&1 || {
  echo "flock (util-linux) is required before deployment bootstrap" >&2
  exit 127
}
if [[ ! -e "$REPO_DIR/.deploy.lock" && ! -L "$REPO_DIR/.deploy.lock" ]]; then
  # O_EXCL/noclobber makes first-host creation converge on one inode when two installers race.
  (umask 077; set -o noclobber; : > "$REPO_DIR/.deploy.lock") 2>/dev/null || true
fi
[[ -f "$REPO_DIR/.deploy.lock" && ! -L "$REPO_DIR/.deploy.lock" &&
      "$(stat -c '%h' "$REPO_DIR/.deploy.lock")" == "1" ]] || {
  echo "deployment lock must be a regular single-link file" >&2; exit 1
}
chown root:root "$REPO_DIR/.deploy.lock"
chmod 600 "$REPO_DIR/.deploy.lock"
exec 9>"$REPO_DIR/.deploy.lock"
flock -n 9 || { echo "another sfci install is already running" >&2; exit 75; }

# Source only the already-verified root-owned control helper while holding the deployment lock.
# shellcheck source=release_ops.sh disable=SC1091
source "$CONTROL_DIR/release_ops.sh"

for managed_dir in "$RELEASES_DIR" "$INCOMING_DIR" "$DATA_RELEASES_DIR"; do
  [[ ! -L "$managed_dir" ]] || { echo "$managed_dir must not be a symlink" >&2; exit 1; }
  install -d -m 755 -o root -g root "$managed_dir"
  [[ "$(cd "$managed_dir" && pwd -P)" == "$managed_dir" ]] || {
    echo "$managed_dir escapes its expected canonical path" >&2
    exit 1
  }
done

INCOMING_RELEASE="$INCOMING_DIR/$RELEASE_ID"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
if [[ -e "$INCOMING_RELEASE" || -L "$INCOMING_RELEASE" ]]; then
  [[ -d "$INCOMING_RELEASE" && ! -L "$INCOMING_RELEASE" ]] || {
    echo "incoming release root must be a real directory, not a symlink" >&2
    exit 1
  }
  SOURCE_DIR="$INCOMING_RELEASE"
elif [[ -e "$RELEASE_DIR" || -L "$RELEASE_DIR" ]]; then
  [[ -d "$RELEASE_DIR" && ! -L "$RELEASE_DIR" ]] || {
    echo "resumable release root must be a real directory, not a symlink" >&2
    exit 1
  }
  SOURCE_DIR="$RELEASE_DIR"
else
  echo "release $RELEASE_ID is neither staged nor resumable" >&2
  exit 1
fi

NEW_TARGET="releases/$RELEASE_ID"

freeze_source_payload() {
  local source_canonical bad_owner
  # Take away the staging owner's last write/search permission before traversing payload content.
  # The parent is root-owned/non-writable, so the timestamped entry cannot be swapped during handoff.
  chown root:root "$SOURCE_DIR"
  chmod 700 "$SOURCE_DIR"
  source_canonical="$(cd "$SOURCE_DIR" && pwd -P)"
  [[ "$source_canonical" == "$SOURCE_DIR" ]] || {
    echo "release root resolves outside its timestamped staging path" >&2; return 1
  }
  if [[ "$SOURCE_DIR" == "$INCOMING_RELEASE" ]]; then
    [[ ! -e "$SOURCE_DIR/.venv" && ! -L "$SOURCE_DIR/.venv" &&
       ! -e "$SOURCE_DIR/data" && ! -L "$SOURCE_DIR/data" ]] || {
      echo "incoming payload may not contain .venv or data" >&2; return 1
    }
  fi
  # .venv and data are installer-created on resumable releases. Every pushed payload path must be
  # a single-link regular file or a real directory; links cannot smuggle host content into root
  # reads, package installation, unit installation, or Caddy validation.
  reject_code_payload_links "$SOURCE_DIR"
  chown -R -h root:root "$SOURCE_DIR"
  chmod -R a-w "$SOURCE_DIR"
  chmod 555 "$SOURCE_DIR"
  reject_code_payload_links "$SOURCE_DIR"
  bad_owner="$(find "$SOURCE_DIR" -xdev \
    \( -path "$SOURCE_DIR/.venv" -o -path "$SOURCE_DIR/data" \) -prune -o \
    \( ! -user root -o ! -group root -o -perm /222 \) -print -quit)"
  [[ -z "$bad_owner" ]] || {
    echo "payload did not freeze root-owned/read-only: $bad_owner" >&2; return 1
  }
}

install_runtime_units() {
  local release="$1"
  install -m 644 "$release/deploy/sfci.service"            /etc/systemd/system/sfci.service
  install -m 644 "$release/deploy/sfci-keepalive.service"  /etc/systemd/system/sfci-keepalive.service
  install -m 644 "$release/deploy/sfci-keepalive.timer"    /etc/systemd/system/sfci-keepalive.timer
  install -m 644 "$release/deploy/cloudflare-ufw.service"  /etc/systemd/system/cloudflare-ufw.service
  install -m 644 "$release/deploy/cloudflare-ufw.timer"    /etc/systemd/system/cloudflare-ufw.timer
}

unit_enabled_state() {
  local unit="$1" state
  state="$(systemctl is-enabled "$unit" 2>/dev/null || true)"
  [[ -n "$state" ]] || state="not-found"
  case "$state" in
    enabled|enabled-runtime|disabled|static|indirect|generated|transient|alias|\
      masked|masked-runtime|linked|linked-runtime|not-found) printf '%s\n' "$state" ;;
    *) echo "unsupported systemd enablement state for $unit: $state" >&2; return 1 ;;
  esac
}

snapshot_runtime_state() {
  local unit enabled_raw active scope
  if [[ -e "$UNIT_ROLLBACK_DIR" || -L "$UNIT_ROLLBACK_DIR" ]]; then
    [[ -d "$UNIT_ROLLBACK_DIR" && ! -L "$UNIT_ROLLBACK_DIR" &&
       "$(cd "$UNIT_ROLLBACK_DIR" && pwd -P)" == "$UNIT_ROLLBACK_DIR" &&
       "$(stat -c '%u:%g:%a' "$UNIT_ROLLBACK_DIR")" == "0:0:700" ]] || return 1
  fi
  rm -rf "$UNIT_ROLLBACK_DIR"
  install -d -m 700 -o root -g root "$UNIT_ROLLBACK_DIR"
  : > "$UNIT_ROLLBACK_DIR/states"
  chmod 600 "$UNIT_ROLLBACK_DIR/states"
  for unit in "${RUNTIME_UNITS[@]}"; do
    if [[ -e "/etc/systemd/system/$unit" || -L "/etc/systemd/system/$unit" ]]; then
      [[ ( -f "/etc/systemd/system/$unit" || -L "/etc/systemd/system/$unit" ) &&
         "$(stat -c '%u' "/etc/systemd/system/$unit")" == "0" ]] || return 1
      cp -a "/etc/systemd/system/$unit" "$UNIT_ROLLBACK_DIR/$unit"
    fi
    if [[ -e "/run/systemd/system/$unit" || -L "/run/systemd/system/$unit" ]]; then
      [[ ( -f "/run/systemd/system/$unit" || -L "/run/systemd/system/$unit" ) &&
         "$(stat -c '%u' "/run/systemd/system/$unit")" == "0" ]] || return 1
      cp -a "/run/systemd/system/$unit" "$UNIT_ROLLBACK_DIR/runtime-$unit"
    fi
    enabled_raw="$(unit_enabled_state "$unit")"
    active=0
    systemctl is-active --quiet "$unit" 2>/dev/null && active=1
    printf '%s %s %s\n' "$unit" "$active" "$enabled_raw" \
      >> "$UNIT_ROLLBACK_DIR/states"
  done
  for scope in etc run; do
    if [[ -e "/$scope/systemd/system/caddy.service" ||
          -L "/$scope/systemd/system/caddy.service" ]]; then
      [[ ( -f "/$scope/systemd/system/caddy.service" ||
           -L "/$scope/systemd/system/caddy.service" ) &&
         "$(stat -c '%u' "/$scope/systemd/system/caddy.service")" == "0" ]] || return 1
      cp -a "/$scope/systemd/system/caddy.service" "$UNIT_ROLLBACK_DIR/caddy-$scope.service"
    fi
  done
  if [[ -e "$LOGROTATE_FILE" || -L "$LOGROTATE_FILE" ]]; then
    [[ ( -f "$LOGROTATE_FILE" || -L "$LOGROTATE_FILE" ) &&
       "$(stat -c '%u' "$LOGROTATE_FILE")" == "0" ]] || return 1
    cp -a "$LOGROTATE_FILE" "$UNIT_ROLLBACK_DIR/logrotate.sfci"
    : > "$UNIT_ROLLBACK_DIR/logrotate-present"
  fi
}

validate_rollback_artifacts() {
  local deploy_kind="$1" caddy_had="$2" path name
  [[ -d "$UNIT_ROLLBACK_DIR" && ! -L "$UNIT_ROLLBACK_DIR" &&
     "$(cd "$UNIT_ROLLBACK_DIR" && pwd -P)" == "$UNIT_ROLLBACK_DIR" &&
     "$(stat -c '%u:%g:%a' "$UNIT_ROLLBACK_DIR")" == "0:0:700" ]] || return 1
  [[ -f "$UNIT_ROLLBACK_DIR/states" && ! -L "$UNIT_ROLLBACK_DIR/states" &&
     "$(stat -c '%u:%g:%a:%h' "$UNIT_ROLLBACK_DIR/states")" == "0:0:600:1" ]] || return 1
  [[ "$(wc -l < "$UNIT_ROLLBACK_DIR/states")" == "${#RUNTIME_UNITS[@]}" ]] || return 1
  for name in "${RUNTIME_UNITS[@]}"; do
    [[ "$(grep -c "^$name " "$UNIT_ROLLBACK_DIR/states")" == "1" ]] || return 1
  done
  for path in "$UNIT_ROLLBACK_DIR"/*; do
    [[ -e "$path" || -L "$path" ]] || continue
    name="${path##*/}"
    case "$name" in
      states|logrotate-present|logrotate.sfci|Caddyfile|sfci.service|sfci-keepalive.service|\
        sfci-keepalive.timer|cloudflare-ufw.service|cloudflare-ufw.timer|\
        runtime-sfci.service|runtime-sfci-keepalive.service|runtime-sfci-keepalive.timer|\
        runtime-cloudflare-ufw.service|runtime-cloudflare-ufw.timer|\
        caddy-etc.service|caddy-run.service) ;;
      *) return 1 ;;
    esac
    if [[ -L "$path" ]]; then
      [[ "$(stat -c '%u' "$path")" == "0" && "$(readlink "$path")" != *$'\n'* ]] || return 1
    else
      [[ -f "$path" && "$(stat -c '%u:%h' "$path")" == "0:1" ]] || return 1
    fi
  done
  if [[ "$caddy_had" == "1" ]]; then
    [[ -f "$CADDY_ROLLBACK" && ! -L "$CADDY_ROLLBACK" &&
       "$(stat -c '%u:%g:%a:%h' "$CADDY_ROLLBACK")" == "0:0:600:1" ]] || return 1
  else
    [[ ! -e "$CADDY_ROLLBACK" && ! -L "$CADDY_ROLLBACK" ]] || return 1
  fi
  if [[ "$deploy_kind" == "legacy_dir" &&
        ( -e "$DEPLOY_ROLLBACK_DIR" || -L "$DEPLOY_ROLLBACK_DIR" ) ]]; then
    [[ -d "$DEPLOY_ROLLBACK_DIR" && ! -L "$DEPLOY_ROLLBACK_DIR" &&
       "$(cd "$DEPLOY_ROLLBACK_DIR" && pwd -P)" == "$DEPLOY_ROLLBACK_DIR" &&
       "$(stat -c '%u:%g:%a' "$DEPLOY_ROLLBACK_DIR")" == "0:0:700" ]] || return 1
  elif [[ "$deploy_kind" != "legacy_dir" ]]; then
    [[ ! -e "$DEPLOY_ROLLBACK_DIR" && ! -L "$DEPLOY_ROLLBACK_DIR" ]] || return 1
  fi
}

# A committed deployment no longer needs rollback contents, but power loss may interrupt their
# deletion. Validate every remaining control path before resuming cleanup; unlike pending rollback
# validation, this deliberately permits a partially removed unit snapshot.
validate_committed_rollback_artifacts() {
  local deploy_kind="$1" path name
  if [[ -e "$UNIT_ROLLBACK_DIR" || -L "$UNIT_ROLLBACK_DIR" ]]; then
    [[ -d "$UNIT_ROLLBACK_DIR" && ! -L "$UNIT_ROLLBACK_DIR" &&
       "$(cd "$UNIT_ROLLBACK_DIR" && pwd -P)" == "$UNIT_ROLLBACK_DIR" &&
       "$(stat -c '%u:%g:%a' "$UNIT_ROLLBACK_DIR")" == "0:0:700" ]] || return 1
    for path in "$UNIT_ROLLBACK_DIR"/*; do
      [[ -e "$path" || -L "$path" ]] || continue
      name="${path##*/}"
      case "$name" in
        states|logrotate-present|logrotate.sfci|Caddyfile|sfci.service|sfci-keepalive.service|\
          sfci-keepalive.timer|cloudflare-ufw.service|cloudflare-ufw.timer|\
          runtime-sfci.service|runtime-sfci-keepalive.service|runtime-sfci-keepalive.timer|\
          runtime-cloudflare-ufw.service|runtime-cloudflare-ufw.timer|\
          caddy-etc.service|caddy-run.service) ;;
        *) return 1 ;;
      esac
      if [[ -L "$path" ]]; then
        [[ "$(stat -c '%u' "$path")" == "0" && "$(readlink "$path")" != *$'\n'* ]] || return 1
      else
        [[ -f "$path" && "$(stat -c '%u:%h' "$path")" == "0:1" ]] || return 1
      fi
    done
  fi
  if [[ "$deploy_kind" == "legacy_dir" &&
        ( -e "$DEPLOY_ROLLBACK_DIR" || -L "$DEPLOY_ROLLBACK_DIR" ) ]]; then
    [[ -d "$DEPLOY_ROLLBACK_DIR" && ! -L "$DEPLOY_ROLLBACK_DIR" &&
       "$(cd "$DEPLOY_ROLLBACK_DIR" && pwd -P)" == "$DEPLOY_ROLLBACK_DIR" &&
       "$(stat -c '%u:%g:%a' "$DEPLOY_ROLLBACK_DIR")" == "0:0:700" ]] || return 1
  elif [[ "$deploy_kind" != "legacy_dir" ]]; then
    [[ ! -e "$DEPLOY_ROLLBACK_DIR" && ! -L "$DEPLOY_ROLLBACK_DIR" ]] || return 1
  fi
}

cleanup_committed_rollback_artifacts() {
  local deploy_kind="$1"
  validate_committed_rollback_artifacts "$deploy_kind"
  rm -rf --one-file-system "$DEPLOY_ROLLBACK_DIR"
  durable_sync_path "$REPO_DIR"
  rm -rf --one-file-system "$UNIT_ROLLBACK_DIR"
  durable_sync_path "$REPO_DIR"
}

remove_deploy_transaction() {
  [[ -f "$TRANSACTION_FILE" && ! -L "$TRANSACTION_FILE" &&
     "$(stat -c '%u:%g:%a:%h' "$TRANSACTION_FILE")" == "0:0:600:1" ]] || return 1
  rm -f "$TRANSACTION_FILE"
  durable_sync_path "$REPO_DIR"
}

restore_runtime_state() {
  local unit active enabled_raw actual_raw restored_active
  [[ -f "$UNIT_ROLLBACK_DIR/states" ]] || return 1
  systemctl stop "${RUNTIME_UNITS[@]}" >/dev/null 2>&1 || true
  systemctl disable "${RUNTIME_UNITS[@]}" >/dev/null 2>&1 || true
  systemctl disable --runtime "${RUNTIME_UNITS[@]}" >/dev/null 2>&1 || true
  systemctl unmask "${RUNTIME_UNITS[@]}" >/dev/null 2>&1 || true
  systemctl unmask --runtime "${RUNTIME_UNITS[@]}" >/dev/null 2>&1 || true
  for unit in "${RUNTIME_UNITS[@]}"; do
    rm -f "/etc/systemd/system/$unit"
    rm -f "/run/systemd/system/$unit"
    if [[ -e "$UNIT_ROLLBACK_DIR/$unit" || -L "$UNIT_ROLLBACK_DIR/$unit" ]]; then
      cp -a "$UNIT_ROLLBACK_DIR/$unit" "/etc/systemd/system/$unit"
    fi
    if [[ -e "$UNIT_ROLLBACK_DIR/runtime-$unit" || -L "$UNIT_ROLLBACK_DIR/runtime-$unit" ]]; then
      cp -a "$UNIT_ROLLBACK_DIR/runtime-$unit" "/run/systemd/system/$unit"
    fi
  done
  systemctl daemon-reload
  while read -r unit active enabled_raw; do
    [[ " ${RUNTIME_UNITS[*]} " == *" $unit "* ]] || return 1
    [[ "$active" == "0" || "$active" == "1" ]] || return 1
    restored_active=0
    case "$enabled_raw" in
      enabled) systemctl enable "$unit" >/dev/null ;;
      enabled-runtime) systemctl enable --runtime "$unit" >/dev/null ;;
      masked)
        if [[ "$active" == "1" ]]; then
          systemctl unmask "$unit" >/dev/null
          systemctl daemon-reload
          systemctl start "$unit"
          restored_active=1
        fi
        systemctl mask "$unit" >/dev/null
        ;;
      masked-runtime)
        if [[ "$active" == "1" ]]; then
          systemctl unmask --runtime "$unit" >/dev/null
          systemctl daemon-reload
          systemctl start "$unit"
          restored_active=1
        fi
        systemctl mask --runtime "$unit" >/dev/null
        ;;
      disabled|static|indirect|generated|transient|alias|\
        linked|linked-runtime|not-found) ;;
      *) return 1 ;;
    esac
    if [[ "$active" == "1" && "$restored_active" == "0" ]]; then
      systemctl start "$unit"
    fi
    actual_raw="$(unit_enabled_state "$unit")"
    [[ "$actual_raw" == "$enabled_raw" ]] || {
      echo "could not restore exact enablement state for $unit: $enabled_raw -> $actual_raw" >&2
      return 1
    }
  done < "$UNIT_ROLLBACK_DIR/states"
  rm -f "$LOGROTATE_FILE"
  if [[ -f "$UNIT_ROLLBACK_DIR/logrotate-present" ]]; then
    [[ -e "$UNIT_ROLLBACK_DIR/logrotate.sfci" ||
       -L "$UNIT_ROLLBACK_DIR/logrotate.sfci" ]] || return 1
    cp -a "$UNIT_ROLLBACK_DIR/logrotate.sfci" "$LOGROTATE_FILE"
  fi
}

restore_caddy() {
  local had_file="$1" was_active="$2" enabled_state="$3" file_meta="$4" allow_start="${5:-1}"
  local file_uid file_gid file_mode restored_active=0
  if [[ "$had_file" == "1" ]]; then
    [[ "$file_meta" =~ ^[0-9]+:[0-9]+:[0-7]{3,4}$ ]] || return 1
    IFS=: read -r file_uid file_gid file_mode <<<"$file_meta"
    [[ -f "$CADDY_ROLLBACK" && ! -L "$CADDY_ROLLBACK" &&
       "$(stat -c '%u:%g:%a:%h' "$CADDY_ROLLBACK")" == "0:0:600:1" ]] || return 1
    rm -f "$CADDYFILE"
    install -m "$file_mode" -o "$file_uid" -g "$file_gid" "$CADDY_ROLLBACK" "$CADDYFILE"
    [[ "$(stat -c '%u:%g:%a:%h' "$CADDYFILE")" == "$file_uid:$file_gid:$file_mode:1" ]] || return 1
  else
    [[ -z "$file_meta" ]] || return 1
    rm -f "$CADDYFILE"
  fi
  systemctl stop caddy >/dev/null 2>&1 || true
  systemctl disable caddy >/dev/null 2>&1 || true
  systemctl disable --runtime caddy >/dev/null 2>&1 || true
  systemctl unmask caddy >/dev/null 2>&1 || true
  systemctl unmask --runtime caddy >/dev/null 2>&1 || true
  rm -f /etc/systemd/system/caddy.service /run/systemd/system/caddy.service
  if [[ -e "$UNIT_ROLLBACK_DIR/caddy-etc.service" ||
        -L "$UNIT_ROLLBACK_DIR/caddy-etc.service" ]]; then
    cp -a "$UNIT_ROLLBACK_DIR/caddy-etc.service" /etc/systemd/system/caddy.service
  fi
  if [[ -e "$UNIT_ROLLBACK_DIR/caddy-run.service" ||
        -L "$UNIT_ROLLBACK_DIR/caddy-run.service" ]]; then
    cp -a "$UNIT_ROLLBACK_DIR/caddy-run.service" /run/systemd/system/caddy.service
  fi
  systemctl daemon-reload
  case "$enabled_state" in
    enabled) systemctl enable caddy >/dev/null ;;
    enabled-runtime) systemctl enable --runtime caddy >/dev/null ;;
    masked)
      if [[ "$was_active" == "1" && "$allow_start" == "1" ]]; then
        systemctl unmask caddy >/dev/null
        systemctl daemon-reload
        systemctl restart caddy
        restored_active=1
      fi
      systemctl mask caddy >/dev/null
      ;;
    masked-runtime)
      if [[ "$was_active" == "1" && "$allow_start" == "1" ]]; then
        systemctl unmask --runtime caddy >/dev/null
        systemctl daemon-reload
        systemctl restart caddy
        restored_active=1
      fi
      systemctl mask --runtime caddy >/dev/null
      ;;
    disabled|static|indirect|generated|transient|alias|\
      linked|linked-runtime|not-found) ;;
    *) return 1 ;;
  esac
  [[ "$(unit_enabled_state caddy)" == "$enabled_state" ]] || return 1
  if [[ "$was_active" == "1" && "$allow_start" == "1" && "$restored_active" == "0" ]]; then
    systemctl restart caddy
  else
    systemctl stop caddy 2>/dev/null || true
  fi
}

origin_file_meta() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" && "$(stat -c '%u:%h' "$path")" == "0:1" ]] || return 1
  stat -c '%u:%g:%a' "$path"
}

restore_origin_permissions() {
  local cert_meta="$1" key_meta="$2" path meta uid gid mode
  for path in "$ORIGIN_CERT" "$ORIGIN_KEY"; do
    if [[ "$path" == "$ORIGIN_CERT" ]]; then meta="$cert_meta"; else meta="$key_meta"; fi
    [[ "$meta" =~ ^[0-9]+:[0-9]+:[0-7]{3,4}$ ]] || return 1
    [[ -f "$path" && ! -L "$path" && "$(stat -c '%h' "$path")" == "1" ]] || return 1
    IFS=: read -r uid gid mode <<<"$meta"
    chown "$uid:$gid" "$path"
    chmod "$mode" "$path"
    [[ "$(stat -c '%u:%g:%a:%h' "$path")" == "$uid:$gid:$mode:1" ]] || return 1
  done
}

enforce_origin_permissions() {
  local caddy_gid path
  caddy_gid="$(getent group caddy | cut -d: -f3)"
  [[ "$caddy_gid" =~ ^[0-9]+$ ]] || return 1
  for path in "$ORIGIN_CERT" "$ORIGIN_KEY"; do
    [[ -f "$path" && ! -L "$path" && "$(stat -c '%u:%h' "$path")" == "0:1" ]] || return 1
    chown "root:$caddy_gid" "$path"
    chmod 640 "$path"
    [[ "$(stat -c '%u:%g:%a:%h' "$path")" == "0:$caddy_gid:640:1" ]] || return 1
  done
}

remove_created_env() {
  local created="$1" target="$2" marker
  [[ "$created" == "0" || "$created" == "1" ]] || return 1
  (( created == 1 )) || return 0
  [[ "$target" =~ ^releases/([0-9]{14})$ ]] || return 1
  marker="$REPO_DIR/.sfci-env-created-${BASH_REMATCH[1]}"
  if [[ -e "$SFCI_ENV" || -L "$SFCI_ENV" ]]; then
    [[ -f "$SFCI_ENV" && ! -L "$SFCI_ENV" &&
       "$(stat -c '%u:%g:%a:%h' "$SFCI_ENV")" == "0:0:600:1" ]] || return 1
    rm -f "$SFCI_ENV"
  fi
  if [[ -e "$marker" || -L "$marker" ]]; then
    [[ -f "$marker" && ! -L "$marker" &&
       "$(stat -c '%u:%g:%a:%h' "$marker")" == "0:0:600:1" ]] || return 1
    rm -f "$marker"
  fi
}

remove_env_marker() {
  local target="$1" marker
  [[ "$target" =~ ^releases/([0-9]{14})$ ]] || return 1
  marker="$REPO_DIR/.sfci-env-created-${BASH_REMATCH[1]}"
  [[ -e "$marker" || -L "$marker" ]] || return 0
  [[ -f "$marker" && ! -L "$marker" &&
     "$(stat -c '%u:%g:%a:%h' "$marker")" == "0:0:600:1" ]] || return 1
  rm -f "$marker"
}

cleanup_orphan_env_marker() {
  local marker id count=0 found=""
  for marker in "$REPO_DIR"/.sfci-env-created-*; do
    [[ -e "$marker" || -L "$marker" ]] || continue
    id="${marker##*.sfci-env-created-}"
    [[ "$id" =~ ^[0-9]{14}$ && -f "$marker" && ! -L "$marker" &&
       "$(stat -c '%u:%g:%a:%h' "$marker")" == "0:0:600:1" ]] || return 1
    count=$((count + 1)); found="$id"
  done
  (( count <= 1 )) || return 1
  (( count == 1 )) || return 0
  validate_release_pointers "$REPO_DIR" || return 1
  if [[ "$(current_target "$REPO_DIR")" == "releases/$found" ]]; then
    remove_env_marker "releases/$found"
  else
    remove_created_env 1 "releases/$found"
  fi
}

service_list_has() {
  local list="$1" wanted="$2"
  [[ " $list " == *" $wanted "* ]]
}

port_token_exposes_web() {
  local token="$1" spec protocol first last web
  spec="${token%/*}"; protocol="${token##*/}"
  [[ "$protocol" == "tcp" ]] || return 1
  first="${spec%%-*}"; last="${spec##*-}"
  [[ "$first" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ ]] || return 0
  for web in 80 443; do
    (( first <= web && web <= last )) && return 0
  done
  return 1
}

text_mentions_web() {
  local text="$1"
  [[ "$text" =~ (^|[^[:alnum:]_-])https?([^[:alnum:]_-]|$) ||
     "$text" =~ (^|[^0-9])(80|443)([^0-9]|$) ]]
}

# firewalld versions disagree on whether direct passthroughs are rendered as ordinary shell
# arguments or as a Python list. Parse both forms exactly. The public-web classifier ignores only
# rules that are provably irrelevant to inbound TCP 80/443: non-web ports/protocols, denying
# targets, OUTPUT, or loopback/link-local IPv4 and loopback/link-local/ULA IPv6 destinations.
# RFC1918 IPv4 is deliberately not exempt because a cloud public address can NAT to it.
direct_rule_fields() {
  "$PY_BIN" - "$1" <<'PY'
import ast
import shlex
import sys

raw = sys.argv[1].strip()
family, separator, rest = raw.partition(" ")
try:
    if not separator:
        raise ValueError("missing direct-rule arguments")
    if rest.lstrip().startswith("["):
        parsed = ast.literal_eval(rest)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("invalid direct-rule argument list")
        fields = [family, *(str(item) for item in parsed)]
    else:
        fields = shlex.split(raw)
except (SyntaxError, ValueError):
    raise SystemExit(1)
sys.stdout.buffer.write(b"\0".join(field.encode() for field in fields))
sys.stdout.buffer.write(b"\0__SFCI_DIRECT_FIELDS_OK__\0")
PY
}

direct_rule_to_array() {
  local line="$1" field marker=0
  DIRECT_FIELDS=()
  while IFS= read -r -d '' field; do
    if [[ "$field" == "__SFCI_DIRECT_FIELDS_OK__" ]]; then
      marker=1
    else
      DIRECT_FIELDS+=("$field")
    fi
  done < <(direct_rule_fields "$line")
  [[ "$marker" == "1" ]]
}

direct_rule_is_public_web() {
  local line="$1"
  "$PY_BIN" - "$line" <<'PY'
import ast
import ipaddress
import shlex
import sys


def parse_fields(raw):
    family, separator, rest = raw.strip().partition(" ")
    if not separator:
        raise ValueError("missing direct-rule arguments")
    if rest.lstrip().startswith("["):
        parsed = ast.literal_eval(rest)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("invalid direct-rule argument list")
        return [family, *(str(item) for item in parsed)]
    return shlex.split(raw)


def next_value(tokens, index):
    return tokens[index + 1] if index + 1 < len(tokens) else ""


def port_spec_exposes_web(spec):
    for part in spec.lower().split(","):
        if part in {"http", "https"}:
            return True
        bounds = part.replace(":", "-").split("-", 1)
        if not all(value.isdigit() for value in bounds):
            return True
        first = int(bounds[0])
        last = int(bounds[-1])
        if first > last or any(first <= port <= last for port in (80, 443)):
            return True
    return False


def destination_is_nonroutable(tokens):
    for index, token in enumerate(tokens):
        if token not in {"-d", "--dst", "--destination"}:
            continue
        if index > 0 and tokens[index - 1] == "!":
            return False
        try:
            network = ipaddress.ip_network(next_value(tokens, index), strict=False)
        except ValueError:
            return False
        if network.version == 4:
            return network.is_loopback or network.is_link_local
        ula = ipaddress.ip_network("fc00::/7")
        return network.is_loopback or network.is_link_local or network.subnet_of(ula)
    return False


try:
    tokens = parse_fields(sys.argv[1])
except (SyntaxError, ValueError):
    raise SystemExit(0)

chain = ""
if "-A" in tokens:
    chain = next_value(tokens, tokens.index("-A"))
elif len(tokens) >= 4 and tokens[0] in {"ipv4", "ipv6", "eb"}:
    chain = tokens[2]
if chain == "OUTPUT" or "-N" in tokens or destination_is_nonroutable(tokens):
    raise SystemExit(1)

protocol = ""
for index, token in enumerate(tokens):
    if token in {"-p", "--protocol"}:
        protocol = next_value(tokens, index).lower()
if protocol and protocol != "tcp":
    raise SystemExit(1)

jump = ""
for index, token in enumerate(tokens):
    if token in {"-j", "--jump", "-g", "--goto"}:
        jump = next_value(tokens, index).upper()
    elif token.startswith(("--jump=", "--goto=")):
        jump = token.split("=", 1)[1].upper()
if not jump or jump in {"DROP", "REJECT", "RETURN"}:
    raise SystemExit(1)

port_specs = []
for index, token in enumerate(tokens):
    if token in {"--dport", "--dports", "--destination-port"}:
        port_specs.append(next_value(tokens, index))
    elif token.startswith(("--dport=", "--dports=", "--destination-port=")):
        port_specs.append(token.split("=", 1)[1])

# An accepting/jumping TCP rule without a destination-port restriction includes web traffic.
raise SystemExit(0 if not port_specs or any(port_spec_exposes_web(spec) for spec in port_specs) else 1)
PY
}

firewall_call() {
  local mode="$1"; shift
  if [[ "$mode" == "permanent" ]]; then
    firewall-cmd --permanent "$@"
  else
    firewall-cmd "$@"
  fi
}

service_exposes_web() {
  local mode="$1" service="$2" info token
  [[ "$service" == "http" || "$service" == "https" ]] && return 0
  info="$(firewall_call "$mode" --info-service="$service" 2>/dev/null)" || return 0
  while read -r token; do
    port_token_exposes_web "$token" && return 0
  done < <(awk '/^[[:space:]]*ports:/{for(i=2;i<=NF;i++) print $i}' <<<"$info")
  return 1
}

close_zone_web_ingress() {
  local mode="$1" zone="$2" preserve_cloudflare="${3:-1}"
  local service port rule services ports rich_rules forward_ports
  [[ "$zone" == "cloudflare" && "$preserve_cloudflare" == "1" ]] && return 0
  services="$(firewall_call "$mode" --zone="$zone" --list-services)" || return 1
  for service in $services; do
    [[ -n "$service" ]] || continue
    if service_exposes_web "$mode" "$service"; then
      firewall_call "$mode" --zone="$zone" --remove-service="$service" >/dev/null
    fi
  done
  ports="$(firewall_call "$mode" --zone="$zone" --list-ports)" || return 1
  for port in $ports; do
    [[ -n "$port" ]] || continue
    if port_token_exposes_web "$port"; then
      firewall_call "$mode" --zone="$zone" --remove-port="$port" >/dev/null
    fi
  done
  rich_rules="$(firewall_call "$mode" --zone="$zone" --list-rich-rules)" || return 1
  while IFS= read -r rule; do
    [[ -n "$rule" ]] || continue
    if text_mentions_web "$rule"; then
      firewall_call "$mode" --zone="$zone" --remove-rich-rule="$rule" >/dev/null
    fi
  done <<<"$rich_rules"
  forward_ports="$(firewall_call "$mode" --zone="$zone" --list-forward-ports)" || return 1
  while IFS= read -r rule; do
    [[ -n "$rule" ]] || continue
    if text_mentions_web "$rule"; then
      firewall_call "$mode" --zone="$zone" --remove-forward-port="$rule" >/dev/null
    fi
  done <<<"$forward_ports"
  if [[ "$zone" == "public" || "$preserve_cloudflare" == "0" ]]; then
    firewall_call "$mode" --zone="$zone" --set-target=DROP >/dev/null
  fi
}

close_direct_web_ingress() {
  local mode="$1" kind line lines
  for kind in rules passthroughs; do
    lines="$(firewall_call "$mode" --direct "--get-all-$kind" 2>/dev/null)" || return 1
    while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      direct_rule_is_public_web "$line" || continue
      direct_rule_to_array "$line" || return 1
      ((${#DIRECT_FIELDS[@]} > 0)) || return 1
      if [[ "$kind" == "rules" ]]; then
        firewall_call "$mode" --direct --remove-rule "${DIRECT_FIELDS[@]}" >/dev/null
      else
        firewall_call "$mode" --direct --remove-passthrough "${DIRECT_FIELDS[@]}" >/dev/null
      fi
    done <<<"$lines"
  done
}

close_non_cloudflare_web_ingress() {
  local mode="$1" zone zones
  zones="$(firewall_call "$mode" --get-zones)" || return 1
  [[ -n "$zones" ]] || return 1
  for zone in $zones; do
    [[ -n "$zone" ]] || continue
    close_zone_web_ingress "$mode" "$zone"
  done
  close_direct_web_ingress "$mode"
}

verify_no_non_cloudflare_web_ingress() {
  local mode="$1" skip_cloudflare="${2:-1}"
  local zone zones target service port rule services ports rich_rules forward_ports
  local direct_rules passthroughs policies policy policy_info policy_value
  zones="$(firewall_call "$mode" --get-zones)" || return 1
  [[ -n "$zones" ]] || return 1
  for zone in $zones; do
    [[ -n "$zone" ]] || continue
    [[ "$zone" == "cloudflare" && "$skip_cloudflare" == "1" ]] && continue
    target="$(firewall_call "$mode" --zone="$zone" --get-target)" || return 1
    [[ "$target" != "ACCEPT" ]] || return 1
    services="$(firewall_call "$mode" --zone="$zone" --list-services)" || return 1
    for service in $services; do
      [[ -n "$service" ]] || continue
      service_exposes_web "$mode" "$service" && return 1
    done
    ports="$(firewall_call "$mode" --zone="$zone" --list-ports)" || return 1
    for port in $ports; do
      [[ -n "$port" ]] || continue
      port_token_exposes_web "$port" && return 1
    done
    rich_rules="$(firewall_call "$mode" --zone="$zone" --list-rich-rules)" || return 1
    while IFS= read -r rule; do
      [[ -n "$rule" ]] || continue
      # Any remaining rich rule is an unprovable alternate ingress language (custom services,
      # ipsets, marks, or nested limits). Fail closed instead of assuming it cannot reach 80/443.
      return 1
    done <<<"$rich_rules"
    forward_ports="$(firewall_call "$mode" --zone="$zone" --list-forward-ports)" || return 1
    while IFS= read -r rule; do
      [[ -n "$rule" ]] || continue
      return 1
    done <<<"$forward_ports"
  done
  direct_rules="$(firewall_call "$mode" --direct --get-all-rules 2>/dev/null)" || return 1
  while IFS= read -r rule; do
    [[ -n "$rule" ]] || continue
    direct_rule_is_public_web "$rule" && return 1
  done <<<"$direct_rules"
  passthroughs="$(firewall_call "$mode" --direct --get-all-passthroughs 2>/dev/null)" || return 1
  while IFS= read -r rule; do
    [[ -n "$rule" ]] || continue
    direct_rule_is_public_web "$rule" && return 1
  done <<<"$passthroughs"
  policies="$(firewall_call "$mode" --get-policies 2>/dev/null)" || return 1
  for policy in $policies; do
    policy_info="$(firewall_call "$mode" --policy="$policy" --list-all 2>/dev/null)" || return 1
    target="$(awk -F: '/^[[:space:]]*target:/{sub(/^[^:]*:[[:space:]]*/, ""); print; exit}' \
      <<<"$policy_info")"
    [[ "$target" != "ACCEPT" ]] || return 1
    policy_value="$(awk -F: '/^[[:space:]]*services:/{sub(/^[^:]*:[[:space:]]*/, ""); print; exit}' \
      <<<"$policy_info")"
    for service in $policy_value; do service_exposes_web "$mode" "$service" && return 1; done
    policy_value="$(awk -F: '/^[[:space:]]*ports:/{sub(/^[^:]*:[[:space:]]*/, ""); print; exit}' \
      <<<"$policy_info")"
    for port in $policy_value; do port_token_exposes_web "$port" && return 1; done
    policy_value="$(awk -F: '/^[[:space:]]*(rich rules|forward-ports):/{sub(/^[^:]*:[[:space:]]*/, ""); print}' \
      <<<"$policy_info")"
    [[ -z "$policy_value" ]] || return 1
  done
}

verify_cloudflare_zone() {
  local mode="$1" zones services sources interfaces target source
  zones="$(firewall_call "$mode" --get-zones)" || return 1
  service_list_has "$zones" cloudflare || return 1
  services="$(firewall_call "$mode" --zone=cloudflare --list-services)" || return 1
  sources="$(firewall_call "$mode" --zone=cloudflare --list-sources)" || return 1
  interfaces="$(firewall_call "$mode" --zone=cloudflare --list-interfaces)" || return 1
  target="$(firewall_call "$mode" --zone=cloudflare --get-target)" || return 1
  command -v python3 >/dev/null 2>&1 || return 1
  for source in $sources; do
    python3 -c 'import ipaddress, sys
n = ipaddress.ip_network(sys.argv[1], strict=True)
minimum = 12 if n.version == 4 else 32
raise SystemExit(0 if n.prefixlen >= minimum else 1)' "$source" || return 1
  done
  service_list_has "$services" http && service_list_has "$services" https &&
    [[ -n "$sources" && -z "$interfaces" && "$target" == "DROP" ]]
}

stop_origin() {
  systemctl stop caddy >/dev/null 2>&1 || true
  systemctl stop sfci >/dev/null 2>&1 || true
}

fail_closed_firewall() {
  local ok=1 zones cloudflare_exists=0 cloudflare_safe=0
  systemctl enable --now firewalld || ok=0
  if (( ok == 1 )); then
    firewall-cmd --permanent --zone=public --add-service=ssh >/dev/null || ok=0
    firewall-cmd --zone=public --add-service=ssh >/dev/null || ok=0
    close_non_cloudflare_web_ingress permanent || ok=0
    firewall-cmd --reload >/dev/null || ok=0
    close_non_cloudflare_web_ingress runtime || ok=0
  fi
  if (( ok == 1 )); then
    zones="$(firewall_call permanent --get-zones)" || ok=0
    service_list_has "$zones" cloudflare && cloudflare_exists=1
  fi
  if (( ok == 1 && cloudflare_exists == 1 )); then
    if verify_cloudflare_zone permanent && verify_cloudflare_zone runtime; then
      cloudflare_safe=1
    else
      # A malformed cloudflare zone (for example, one bound to an interface) is not a trusted
      # exception. Remove every web exposure from it and prove all zones closed.
      close_zone_web_ingress permanent cloudflare 0 || ok=0
      firewall-cmd --reload >/dev/null || ok=0
      close_zone_web_ingress runtime cloudflare 0 || ok=0
    fi
  fi
  if (( ok == 1 && cloudflare_safe == 1 )); then
    verify_no_non_cloudflare_web_ingress permanent || ok=0
    verify_no_non_cloudflare_web_ingress runtime || ok=0
  elif (( ok == 1 )); then
    verify_no_non_cloudflare_web_ingress permanent 0 || ok=0
    verify_no_non_cloudflare_web_ingress runtime 0 || ok=0
  fi
  if (( ok == 0 )); then
    stop_origin
    return 1
  fi
}

verify_cloudflare_lockdown() {
  verify_cloudflare_zone permanent && verify_cloudflare_zone runtime &&
    verify_no_non_cloudflare_web_ingress permanent &&
    verify_no_non_cloudflare_web_ingress runtime
}

ensure_cloudflare_lockdown() {
  systemctl enable --now firewalld || { stop_origin; return 1; }
  firewall-cmd --permanent --zone=public --add-service=ssh >/dev/null || {
    stop_origin; return 1;
  }
  systemctl enable cloudflare-ufw.timer || { stop_origin; return 1; }
  # Never make this conditional on a pre-existing zone. The oneshot creates/refreshes the zone.
  systemctl start cloudflare-ufw.service || { stop_origin; return 1; }
  close_non_cloudflare_web_ingress permanent || { stop_origin; return 1; }
  firewall-cmd --reload >/dev/null || { stop_origin; return 1; }
  close_non_cloudflare_web_ingress runtime || { stop_origin; return 1; }
  systemctl start cloudflare-ufw.timer || { stop_origin; return 1; }
  verify_cloudflare_lockdown || { stop_origin; return 1; }
}

read_transaction_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$TRANSACTION_FILE"
}

validate_transaction_schema() {
  local key
  local keys=(phase old old_previous new legacy env_created caddy_had caddy_active caddy_enabled
    caddy_file_meta origin_cert_meta origin_key_meta deploy_kind deploy_target deploy_meta)
  [[ "$(wc -l < "$TRANSACTION_FILE")" == "${#keys[@]}" ]] || return 1
  for key in "${keys[@]}"; do
    [[ "$(grep -c "^$key=" "$TRANSACTION_FILE")" == "1" ]] || return 1
  done
  awk -F= '
    $1 !~ /^(phase|old|old_previous|new|legacy|env_created|caddy_had|caddy_active|caddy_enabled|caddy_file_meta|origin_cert_meta|origin_key_meta|deploy_kind|deploy_target|deploy_meta)$/ { bad=1 }
    END { exit bad }
  ' "$TRANSACTION_FILE"
}

write_deploy_transaction() {
  local phase="$1" temp="${TRANSACTION_FILE}.tmp.$$"
  [[ ! -e "$temp" && ! -L "$temp" ]] || return 1
  (umask 077
    set -o noclobber
    printf 'phase=%s\nold=%s\nold_previous=%s\nnew=%s\nlegacy=%s\nenv_created=%s\ncaddy_had=%s\ncaddy_active=%s\ncaddy_enabled=%s\ncaddy_file_meta=%s\norigin_cert_meta=%s\norigin_key_meta=%s\ndeploy_kind=%s\ndeploy_target=%s\ndeploy_meta=%s\n' \
      "$phase" "$OLD_TARGET" "$OLD_PREVIOUS_TARGET" "$NEW_TARGET" "$LEGACY_PRESENT" \
      "$ENV_CREATED" "$CADDY_HAD_FILE" "$CADDY_WAS_ACTIVE" "$CADDY_WAS_ENABLED" \
      "$CADDY_FILE_META" "$ORIGIN_CERT_META" "$ORIGIN_KEY_META" "$DEPLOY_PATH_KIND" \
      "$DEPLOY_PATH_TARGET" "$DEPLOY_PATH_META" > "$temp")
  chown root:root "$temp"
  chmod 600 "$temp"
  mv -f "$temp" "$TRANSACTION_FILE"
  durable_sync_path "$TRANSACTION_FILE"
  durable_sync_path "$REPO_DIR"
  [[ -f "$TRANSACTION_FILE" && ! -L "$TRANSACTION_FILE" &&
     "$(stat -c '%u:%g:%a:%h' "$TRANSACTION_FILE")" == "0:0:600:1" ]]
}

restore_deploy_path() {
  local kind="$1" target="$2" metadata="$3" uid gid mode
  case "$kind" in
    absent)
      [[ -z "$target" && -z "$metadata" ]] || return 1
      [[ ! -e "$REPO_DIR/deploy" || -L "$REPO_DIR/deploy" ]] || return 1
      rm -f "$REPO_DIR/deploy"
      ;;
    symlink)
      [[ "$target" == "current/deploy" && -z "$metadata" ]] || return 1
      if [[ -L "$REPO_DIR/deploy" && "$(readlink "$REPO_DIR/deploy")" == "$target" ]]; then
        return 0
      fi
      [[ ! -e "$REPO_DIR/deploy" || -L "$REPO_DIR/deploy" ]] || return 1
      rm -f "$REPO_DIR/deploy"
      atomic_symlink "$target" "$REPO_DIR/deploy"
      ;;
    legacy_dir)
      [[ -z "$target" && "$metadata" =~ ^[0-9]+:[0-9]+:[0-7]{3,4}$ ]] || return 1
      # Idempotent when failure happened after the transaction record but before the directory
      # was moved aside. Otherwise remove the cutover symlink and restore the saved directory.
      if [[ -d "$REPO_DIR/deploy" && ! -L "$REPO_DIR/deploy" &&
            ! -e "$DEPLOY_ROLLBACK_DIR" ]]; then
        return 0
      fi
      [[ ! -e "$REPO_DIR/deploy" || -L "$REPO_DIR/deploy" ]] || return 1
      rm -f "$REPO_DIR/deploy"
      [[ -d "$DEPLOY_ROLLBACK_DIR" && ! -L "$DEPLOY_ROLLBACK_DIR" ]] || return 1
      mv "$DEPLOY_ROLLBACK_DIR" "$REPO_DIR/deploy"
      IFS=: read -r uid gid mode <<<"$metadata"
      chown "$uid:$gid" "$REPO_DIR/deploy"
      chmod "$mode" "$REPO_DIR/deploy"
      [[ "$(stat -c '%u:%g:%a' "$REPO_DIR/deploy")" == "$uid:$gid:$mode" ]] || return 1
      ;;
    *) return 1 ;;
  esac
}

activate_deploy_path() {
  local kind="$1"
  case "$kind" in
    absent) ;;
    symlink) rm -f "$REPO_DIR/deploy" ;;
    legacy_dir)
      [[ ! -e "$DEPLOY_ROLLBACK_DIR" && ! -L "$DEPLOY_ROLLBACK_DIR" ]] || return 1
      mv "$REPO_DIR/deploy" "$DEPLOY_ROLLBACK_DIR"
      chown root:root "$DEPLOY_ROLLBACK_DIR"
      chmod 700 "$DEPLOY_ROLLBACK_DIR"
      ;;
    *) return 1 ;;
  esac
  atomic_symlink "current/deploy" "$REPO_DIR/deploy"
}

restore_runtime() {
  local old="$1" old_previous="$2" new="$3" legacy="$4" unit
  [[ "$legacy" == "0" || "$legacy" == "1" ]] || return 1
  if [[ "$legacy" == "1" && ! -f "$UNIT_ROLLBACK_DIR/states" ]]; then
    echo "legacy rollback requires its root-only runtime snapshot" >&2
    return 1
  fi
  restore_release_pointers_exact "$REPO_DIR" "$old" "$old_previous"
  if [[ -f "$UNIT_ROLLBACK_DIR/states" ]]; then
    restore_runtime_state
  else
    systemctl disable --now "${RUNTIME_UNITS[@]}" 2>/dev/null || true
    for unit in "${RUNTIME_UNITS[@]}"; do
      rm -f "/etc/systemd/system/$unit"
    done
    systemctl daemon-reload
  fi
}

recover_pending_cutover() {
  [[ -e "$TRANSACTION_FILE" || -L "$TRANSACTION_FILE" ]] || return 0
  [[ -f "$TRANSACTION_FILE" && ! -L "$TRANSACTION_FILE" &&
     "$(stat -c '%u:%g:%a:%h' "$TRANSACTION_FILE")" == "0:0:600:1" ]] || {
    stop_origin
    echo "unsafe pending deployment transaction; refusing recovery" >&2; return 1
  }
  validate_transaction_schema || {
    stop_origin
    echo "malformed pending deployment transaction; refusing recovery" >&2; return 1
  }
  local phase old old_previous new legacy env_created caddy_had caddy_active caddy_enabled
  local caddy_file_meta origin_cert_meta origin_key_meta deploy_kind deploy_target deploy_meta
  local recover_ok=1
  phase="$(read_transaction_value phase)"
  [[ "$phase" == "pending" || "$phase" == "committed" ]] || recover_ok=0
  old="$(read_transaction_value old)"; new="$(read_transaction_value new)"
  old_previous="$(read_transaction_value old_previous)"
  legacy="$(read_transaction_value legacy)"; caddy_had="$(read_transaction_value caddy_had)"
  env_created="$(read_transaction_value env_created)"
  caddy_active="$(read_transaction_value caddy_active)"
  caddy_enabled="$(read_transaction_value caddy_enabled)"
  caddy_file_meta="$(read_transaction_value caddy_file_meta)"
  origin_cert_meta="$(read_transaction_value origin_cert_meta)"
  origin_key_meta="$(read_transaction_value origin_key_meta)"
  deploy_kind="$(read_transaction_value deploy_kind)"
  deploy_target="$(read_transaction_value deploy_target)"
  deploy_meta="$(read_transaction_value deploy_meta)"
  [[ -z "$old" ]] || valid_release_target "$REPO_DIR" "$old" || recover_ok=0
  [[ -z "$old_previous" ]] || valid_release_target "$REPO_DIR" "$old_previous" || recover_ok=0
  [[ -n "$old" || -z "$old_previous" ]] || recover_ok=0
  valid_release_target "$REPO_DIR" "$new" || recover_ok=0
  [[ "$legacy" == "0" || "$legacy" == "1" ]] || recover_ok=0
  [[ "$env_created" == "0" || "$env_created" == "1" ]] || recover_ok=0
  [[ "$caddy_had" == "0" || "$caddy_had" == "1" ]] || recover_ok=0
  [[ "$caddy_active" == "0" || "$caddy_active" == "1" ]] || recover_ok=0
  [[ "$origin_cert_meta" =~ ^0:[0-9]+:[0-7]{3,4}$ &&
     "$origin_key_meta" =~ ^0:[0-9]+:[0-7]{3,4}$ ]] || recover_ok=0
  if [[ "$caddy_had" == "1" ]]; then
    [[ "$caddy_file_meta" =~ ^0:[0-9]+:[0-7]{3,4}$ ]] || recover_ok=0
  else
    [[ -z "$caddy_file_meta" ]] || recover_ok=0
  fi
  case "$caddy_enabled" in
    enabled|enabled-runtime|disabled|static|indirect|generated|transient|alias|masked|\
      masked-runtime|linked|linked-runtime|not-found) ;;
    *) recover_ok=0 ;;
  esac
  case "$deploy_kind" in
    absent) [[ -z "$deploy_target" && -z "$deploy_meta" ]] || recover_ok=0 ;;
    symlink) [[ "$deploy_target" == "current/deploy" && -z "$deploy_meta" ]] || recover_ok=0 ;;
    legacy_dir) [[ -z "$deploy_target" && "$deploy_meta" =~ ^[0-9]+:[0-9]+:[0-7]{3,4}$ ]] || recover_ok=0 ;;
    *) recover_ok=0 ;;
  esac
  if (( recover_ok == 1 )); then
    if [[ "$phase" == "committed" ]]; then
      validate_committed_rollback_artifacts "$deploy_kind" || recover_ok=0
    else
      validate_rollback_artifacts "$deploy_kind" "$caddy_had" || recover_ok=0
    fi
  fi
  if (( recover_ok == 0 )); then
    stop_origin
    echo "invalid pending deployment transaction; refusing to guess rollback state" >&2
    return 1
  fi

  if [[ "$phase" == "committed" ]]; then
    validate_release_pointers "$REPO_DIR" || { stop_origin; return 1; }
    [[ "$(current_target "$REPO_DIR")" == "$new" && -L "$REPO_DIR/deploy" &&
       "$(previous_target "$REPO_DIR")" == "$old" &&
       "$(readlink "$REPO_DIR/deploy")" == "current/deploy" ]] || {
      stop_origin; return 1
    }
    remove_env_marker "$new" || { stop_origin; return 1; }
    cleanup_committed_rollback_artifacts "$deploy_kind" || { stop_origin; return 1; }
    remove_deploy_transaction || { stop_origin; return 1; }
    return 0
  fi

  log "recovering interrupted cutover before starting a new deployment"
  local rollback_ok=1 firewall_closed=1
  fail_closed_firewall || { rollback_ok=0; firewall_closed=0; }
  restore_deploy_path "$deploy_kind" "$deploy_target" "$deploy_meta" || rollback_ok=0
  remove_created_env "$env_created" "$new" || rollback_ok=0
  restore_origin_permissions "$origin_cert_meta" "$origin_key_meta" || rollback_ok=0
  restore_runtime "$old" "$old_previous" "$new" "$legacy" || rollback_ok=0
  restore_caddy "$caddy_had" "$caddy_active" "$caddy_enabled" "$caddy_file_meta" \
    "$firewall_closed" || rollback_ok=0
  if (( firewall_closed == 1 )); then
    verify_no_non_cloudflare_web_ingress permanent || rollback_ok=0
    verify_no_non_cloudflare_web_ingress runtime || rollback_ok=0
  else
    stop_origin
  fi
  if (( rollback_ok == 1 )); then
    cleanup_committed_rollback_artifacts "$deploy_kind" || rollback_ok=0
    (( rollback_ok == 0 )) || remove_deploy_transaction || rollback_ok=0
  fi
  if (( rollback_ok == 0 )); then
    echo "interrupted cutover recovery failed closed but is incomplete; transaction retained" >&2
    return 1
  fi
}

recover_pending_cutover
cleanup_stale_committed_rollback() {
  [[ ! -e "$TRANSACTION_FILE" && ! -L "$TRANSACTION_FILE" ]] || return 0
  if [[ ! -e "$UNIT_ROLLBACK_DIR" && ! -L "$UNIT_ROLLBACK_DIR" &&
        ! -e "$DEPLOY_ROLLBACK_DIR" && ! -L "$DEPLOY_ROLLBACK_DIR" ]]; then
    return 0
  fi
  validate_release_pointers "$REPO_DIR" && [[ -n "$(current_target "$REPO_DIR")" ]] &&
    [[ -L "$REPO_DIR/deploy" && "$(readlink "$REPO_DIR/deploy")" == "current/deploy" ]] || {
      stop_origin
      echo "rollback artifacts exist without a provably committed current release" >&2
      return 1
    }
  # .deploy-tree-rollback only belongs to the one-time legacy migration. A root-owned trusted
  # directory beside a committed current/deploy pointer is residue from an interrupted cleanup.
  cleanup_committed_rollback_artifacts legacy_dir || {
    stop_origin
    echo "unsafe stale rollback artifacts require manual inspection" >&2
    return 1
  }
}
cleanup_stale_committed_rollback
cleanup_orphan_env_marker || {
  echo "unsafe or ambiguous first-install environment marker" >&2; exit 1
}

# Recovery must run before the resumable release is touched or compared with current: an
# interrupted cutover can legitimately leave the failed new pointer active until this point.
freeze_source_payload
validate_release_pointers "$REPO_DIR" || {
  echo "current/previous must be symlinks to existing timestamped directories under releases/" >&2
  exit 1
}
OLD_TARGET="$(current_target "$REPO_DIR")"
OLD_PREVIOUS_TARGET="$(previous_target "$REPO_DIR")"
[[ "$OLD_TARGET" != "$NEW_TARGET" ]] || {
  echo "release $RELEASE_ID is already active; stage a new timestamped release" >&2; exit 1
}
validate_data_pointers "$REPO_DIR" || {
  echo "$DATA_CURRENT must point to an immutable data-releases/<id>; run the trusted data-refresh helper" >&2
  exit 1
}
[[ ! -e "$REPO_DIR/.data-deploy-transaction" &&
   ! -L "$REPO_DIR/.data-deploy-transaction" ]] || {
  echo "recover the pending immutable-data transaction with data-refresh.sh before code deployment" >&2
  exit 1
}
PINNED_DATA_TARGET="$(data_current_target "$REPO_DIR")"
SHARED_DATA="$REPO_DIR/$PINNED_DATA_TARGET"
reject_data_payload_links "$SHARED_DATA"
bad_data_owner="$(find "$SHARED_DATA" -xdev \
  \( ! -user root -o ! -group root -o -perm /222 \) -print -quit)"
[[ -z "$bad_data_owner" ]] || {
  echo "pinned data release is not root-owned/read-only: $bad_data_owner" >&2; exit 1
}

# ---- System prerequisites ----------------------------------------------------------------
log "installing system packages"
dnf install -y -q dnf-plugins-core
if ! dnf repolist | grep -q 'copr:copr\.fedorainfracloud\.org:group_caddy:caddy'; then
  dnf copr enable -y @caddy/caddy
fi
if ! rpm -q epel-release >/dev/null 2>&1; then dnf install -y -q epel-release; fi
dnf install -y -q \
  "$PY_BIN" "$PY_BIN"-pip "$PY_BIN"-devel \
  gcc gcc-c++ make caddy stress-ng firewalld curl jq unzip tar rsync logrotate util-linux

if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  useradd -r -d "$REPO_DIR" -s /usr/sbin/nologin "$USER_NAME"
fi
install -d -m 750 -o "$USER_NAME" -g "$USER_NAME" \
  /var/log/sfci /var/cache/sfci /var/cache/sfci/numba /run/sfci

# ---- Promote and prepare the isolated code payload ----------------------------------------
if [[ "$SOURCE_DIR" == "$INCOMING_RELEASE" ]]; then
  [[ ! -e "$RELEASE_DIR" && ! -L "$RELEASE_DIR" ]] || {
    echo "release already exists: $RELEASE_DIR" >&2; exit 1
  }
  [[ "$(stat -c '%d' "$SOURCE_DIR")" == "$(stat -c '%d' "$RELEASES_DIR")" ]] || {
    echo ".incoming and releases must share one filesystem for atomic promotion" >&2
    exit 1
  }
  log "moving staged code to $RELEASE_DIR"
  mv "$SOURCE_DIR" "$RELEASE_DIR"
  SOURCE_DIR="$RELEASE_DIR"
  log "resume command: sudo env SFCI_PUBLIC_SMOKE_URL=$PUBLIC_SMOKE_URL $CONTROL_DIR/install.sh --release-id $RELEASE_ID"
else
  log "resuming $RELEASE_DIR"
fi
[[ ! -L "$RELEASE_DIR" && "$(cd "$RELEASE_DIR" && pwd -P)" == "$RELEASE_DIR" ]] || {
  echo "release root became non-canonical" >&2; exit 1
}

LEGACY_PRESENT=0
if [[ -z "$OLD_TARGET" ]]; then
  [[ -f "$REPO_DIR/scripts/server.py" ]] && LEGACY_PRESENT=1
  for legacy_unit in "${RUNTIME_UNITS[@]}"; do
    if [[ -e "/etc/systemd/system/$legacy_unit" || -L "/etc/systemd/system/$legacy_unit" ||
          -e "/run/systemd/system/$legacy_unit" || -L "/run/systemd/system/$legacy_unit" ]]; then
      LEGACY_PRESENT=1
    fi
  done
fi

direct_release_checks() {
  [[ -f "$RELEASE_DIR/scripts/server.py" ]] || { echo "missing scripts/server.py" >&2; return 1; }
  [[ -f "$RELEASE_DIR/scripts/bootstrap_server.py" ]] || {
    echo "missing scripts/bootstrap_server.py" >&2; return 1;
  }
  [[ -f "$SHARED_DATA/server_static.json" ]] || { echo "missing data/server_static.json" >&2; return 1; }
  [[ -f "$SHARED_DATA/walk_graph.npz" ]] || { echo "missing data/walk_graph.npz" >&2; return 1; }
  [[ -d "$SHARED_DATA/raptor_cache" ]] || { echo "missing data/raptor_cache" >&2; return 1; }
  [[ -x "$RELEASE_DIR/.venv/bin/python" ]] || { echo "release virtualenv is incomplete" >&2; return 1; }
}

# ---- Sandboxed candidate and representative route smoke ----------------------------------
CANDIDATE_UNIT="sfci-candidate-$RELEASE_ID.service"
CANDIDATE_UNIT_FILE="/run/systemd/system/$CANDIDATE_UNIT"
CANDIDATE_MARKER="/run/sfci/release-$RELEASE_ID-readonly.marker"
CADDY_CANDIDATE=""; BUS_VHOST_TMP=""; HTTP_BODY_TMP=""
CUTOVER_PENDING=0; ROLLBACK_RUNNING=0
CADDY_HAD_FILE=0; CADDY_WAS_ACTIVE=0; CADDY_WAS_ENABLED="not-found"
ENV_CREATED=0; CADDY_FILE_META=""; ORIGIN_CERT_META=""; ORIGIN_KEY_META=""
DEPLOY_PATH_KIND="absent"; DEPLOY_PATH_TARGET=""; DEPLOY_PATH_META=""

render_candidate_unit() {
  local template="$1" id="$2" isolated="$3" output="$4"
  local intermediate="${output}.cache"
  [[ "$id" =~ ^[0-9]{14}$ && ( "$isolated" == "0" || "$isolated" == "1" ) ]] || return 1
  sed -e "s|/opt/sfci/current|/opt/sfci/releases/$id|g" \
      -e "s|^Environment=PORT=$PRODUCTION_PORT$|Environment=PORT=$CANDIDATE_PORT|" \
      "$template" > "$output"
  if [[ "$isolated" == "1" ]]; then
    sed -e "s|^Environment=NUMBA_CACHE_DIR=/var/cache/sfci/numba$|Environment=NUMBA_CACHE_DIR=/var/cache/sfci-candidate-$id/numba|" \
        -e "s|^Environment=DEST_CACHE_FILE=/var/cache/sfci/dest_cache.json$|Environment=DEST_CACHE_FILE=/var/cache/sfci-candidate-$id/dest_cache.json|" \
        -e "s|^CacheDirectory=sfci$|CacheDirectory=sfci-candidate-$id|" \
        "$output" > "$intermediate"
    mv -f "$intermediate" "$output"
  fi
}

candidate_unit_is_trusted() {
  local unit="$1" id path fragment template isolated legacy trusted=0
  [[ "$unit" =~ ^sfci-candidate-([0-9]{14})\.service$ ]] || return 1
  id="${BASH_REMATCH[1]}"
  path="/run/systemd/system/$unit"
  [[ -f "$path" && ! -L "$path" &&
     "$(stat -c '%u:%g:%a:%h' "$path")" == "0:0:600:1" ]] || return 1
  fragment="$(systemctl show --property=FragmentPath --value "$unit" 2>/dev/null || true)"
  [[ -z "$fragment" || "$fragment" == "$path" ]] || return 1
  valid_release_target "$REPO_DIR" "releases/$id" || return 1
  template="$REPO_DIR/releases/$id/deploy/sfci.service"
  [[ -f "$template" && ! -L "$template" &&
     "$(stat -c '%u:%g:%a:%h' "$template")" == "0:0:444:1" ]] || return 1
  isolated="$(mktemp /run/sfci/stale-candidate-isolated.XXXXXX)" || return 1
  legacy="$(mktemp /run/sfci/stale-candidate-legacy.XXXXXX)" || {
    rm -f "$isolated"; return 1
  }
  render_candidate_unit "$template" "$id" 1 "$isolated" || {
    rm -f "$isolated" "$legacy"; return 1
  }
  render_candidate_unit "$template" "$id" 0 "$legacy" || {
    rm -f "$isolated" "$legacy"; return 1
  }
  if cmp -s "$path" "$isolated" || cmp -s "$path" "$legacy"; then trusted=1; fi
  rm -f "$isolated" "$legacy"
  (( trusted == 1 ))
}

cleanup_stale_candidates() {
  local path unit id listeners names=""
  command -v ss >/dev/null 2>&1 || {
    echo "ss is required to prove the private candidate port is free" >&2; return 1
  }
  for path in /run/systemd/system/sfci-candidate-*.service; do
    [[ -e "$path" || -L "$path" ]] || continue
    names+="${path##*/}"$'\n'
  done
  while read -r unit _; do
    [[ -n "$unit" ]] || continue
    names+="$unit"$'\n'
  done < <(systemctl list-units --all --type=service --plain --no-legend \
    'sfci-candidate-*.service' 2>/dev/null || true)
  names="$(printf '%s' "$names" | sed '/^$/d' | sort -u)"
  while IFS= read -r unit; do
    [[ -n "$unit" ]] || continue
    candidate_unit_is_trusted "$unit" || {
      echo "untrusted stale candidate unit requires manual inspection: $unit" >&2; return 1
    }
  done <<<"$names"
  while IFS= read -r unit; do
    [[ -n "$unit" ]] || continue
    id="${unit#sfci-candidate-}"; id="${id%.service}"
    systemctl stop "$unit" >/dev/null 2>&1 || true
    systemctl reset-failed "$unit" >/dev/null 2>&1 || true
    rm -f "/run/systemd/system/$unit"
    rm -rf "/var/cache/sfci-candidate-$id"
  done <<<"$names"
  systemctl daemon-reload
  listeners="$(ss -H -ltn "sport = :$CANDIDATE_PORT")" || return 1
  [[ -z "$listeners" ]] || {
    echo "candidate port $CANDIDATE_PORT remains occupied by a non-candidate process" >&2; return 1
  }
}

stop_candidate() {
  systemctl stop "$CANDIDATE_UNIT" >/dev/null 2>&1 || true
  systemctl reset-failed "$CANDIDATE_UNIT" >/dev/null 2>&1 || true
  rm -f "$CANDIDATE_UNIT_FILE"
  systemctl daemon-reload >/dev/null 2>&1 || true
  rm -f "$CANDIDATE_MARKER"
  rm -rf "/var/cache/sfci-candidate-$RELEASE_ID"
}

cleanup_temps() {
  [[ -z "$CADDY_CANDIDATE" ]] || rm -f "$CADDY_CANDIDATE"
  [[ -z "$BUS_VHOST_TMP" ]] || rm -f "$BUS_VHOST_TMP"
  [[ -z "$HTTP_BODY_TMP" ]] || rm -f "$HTTP_BODY_TMP"
}

rollback_cutover() {
  (( ROLLBACK_RUNNING == 0 )) || return 1
  ROLLBACK_RUNNING=1
  local rollback_ok=1 firewall_closed=1
  log "rolling back failed release $NEW_TARGET"
  fail_closed_firewall || { rollback_ok=0; firewall_closed=0; }
  validate_rollback_artifacts "$DEPLOY_PATH_KIND" "$CADDY_HAD_FILE" || {
    stop_origin
    return 1
  }
  restore_deploy_path "$DEPLOY_PATH_KIND" "$DEPLOY_PATH_TARGET" "$DEPLOY_PATH_META" || rollback_ok=0
  remove_created_env "$ENV_CREATED" "$NEW_TARGET" || rollback_ok=0
  restore_origin_permissions "$ORIGIN_CERT_META" "$ORIGIN_KEY_META" || rollback_ok=0
  restore_runtime "$OLD_TARGET" "$OLD_PREVIOUS_TARGET" "$NEW_TARGET" "$LEGACY_PRESENT" || rollback_ok=0
  restore_caddy "$CADDY_HAD_FILE" "$CADDY_WAS_ACTIVE" "$CADDY_WAS_ENABLED" \
    "$CADDY_FILE_META" "$firewall_closed" || rollback_ok=0
  if (( firewall_closed == 1 )); then
    verify_no_non_cloudflare_web_ingress permanent || rollback_ok=0
    verify_no_non_cloudflare_web_ingress runtime || rollback_ok=0
  else
    stop_origin
  fi
  if (( rollback_ok == 1 )); then
    cleanup_committed_rollback_artifacts "$DEPLOY_PATH_KIND" || rollback_ok=0
    (( rollback_ok == 0 )) || remove_deploy_transaction || rollback_ok=0
  fi
  if (( rollback_ok == 0 )); then
    echo "rollback failed closed but is incomplete; transaction retained for recovery" >&2
  fi
  return $(( 1 - rollback_ok ))
}

finish_install() {
  local status=$?
  trap - EXIT INT TERM
  stop_candidate
  if (( CUTOVER_PENDING == 1 )); then
    if [[ -f "$TRANSACTION_FILE" && ! -L "$TRANSACTION_FILE" &&
          "$(stat -c '%u:%g:%a:%h' "$TRANSACTION_FILE")" == "0:0:600:1" ]] &&
       validate_transaction_schema && [[ "$(read_transaction_value phase)" == "committed" ]]; then
      CUTOVER_PENDING=0
    else
      rollback_cutover || status=1
    fi
  elif (( status != 0 && ENV_CREATED == 1 )); then
    remove_created_env "$ENV_CREATED" "$NEW_TARGET" || status=1
  fi
  cleanup_temps
  exit "$status"
}
trap finish_install EXIT
trap 'exit 130' INT TERM

cleanup_stale_candidates

# ---- Stable runtime environment ------------------------------------------------------------
if [[ ! -e "$SFCI_ENV" && ! -L "$SFCI_ENV" ]]; then
  log "writing $SFCI_ENV; add provider credentials before public use"
  install -m 600 -o root -g root /dev/null "$ENV_MARKER"
  install -m 600 -o root -g root /dev/null "$SFCI_ENV"
  ENV_CREATED=1
  printf '%s\n' 'RAPTOR_MC=1' 'RAPTOR_SEMANTIC=departafter' 'NUMBA_NUM_THREADS=2' \
    'GEOCODER=photon' 'API511_TOKEN=' 'GEOAPIFY_KEY=' > "$SFCI_ENV"
fi
[[ -f "$SFCI_ENV" && ! -L "$SFCI_ENV" &&
   "$(stat -c '%u:%g:%a:%h' "$SFCI_ENV")" == "0:0:600:1" ]] || {
  echo "$SFCI_ENV must be a root:root mode-600 single-link regular file" >&2; exit 1
}
if grep -Eq '^[[:space:]]*PORT[[:space:]]*=' "$SFCI_ENV"; then
  echo "$SFCI_ENV must not set PORT; sfci.service fixes it at $PRODUCTION_PORT" >&2; exit 1
fi

if [[ -e "$RELEASE_DIR/data" || -L "$RELEASE_DIR/data" ]]; then
  release_data_link="$(readlink "$RELEASE_DIR/data" 2>/dev/null || true)"
  [[ -L "$RELEASE_DIR/data" && "$release_data_link" =~ ^\.\./\.\./data-releases/[0-9]{8}([0-9]{6})?$ ]] || {
    echo "$RELEASE_DIR/data must pin a concrete data-releases/<id> directory" >&2; exit 1
  }
  PINNED_DATA_TARGET="${release_data_link#../../}"
  valid_data_target "$REPO_DIR" "$PINNED_DATA_TARGET" || {
    echo "pinned data release is missing or unsafe: $PINNED_DATA_TARGET" >&2; exit 1
  }
  SHARED_DATA="$REPO_DIR/$PINNED_DATA_TARGET"
  rm -f "$RELEASE_DIR/data"
fi
# Only the installer-created virtualenv is writable during dependency installation. Application
# source remains root-owned/read-only throughout, so neither pip nor the service account can swap
# requirements, units, Caddy configuration, or Python modules after validation.
if [[ -e "$RELEASE_DIR/.venv" || -L "$RELEASE_DIR/.venv" ]]; then
  [[ -d "$RELEASE_DIR/.venv" && ! -L "$RELEASE_DIR/.venv" ]] || {
    echo "release .venv must be an installer-created real directory" >&2; exit 1
  }
  chown -R -h "$USER_NAME:$USER_NAME" "$RELEASE_DIR/.venv"
  chmod -R u+rwX "$RELEASE_DIR/.venv"
else
  install -d -m 750 -o "$USER_NAME" -g "$USER_NAME" "$RELEASE_DIR/.venv"
fi
if [[ ! -x "$RELEASE_DIR/.venv/bin/pip" ]]; then
  sudo -u "$USER_NAME" "$PY_BIN" -m venv "$RELEASE_DIR/.venv"
fi
sudo -u "$USER_NAME" "$RELEASE_DIR/.venv/bin/pip" install --quiet --upgrade pip wheel setuptools
[[ -f "$RELEASE_DIR/requirements.txt" ]] || {
  echo "requirements.txt is required for a supported deployment" >&2; exit 1
}
sudo -u "$USER_NAME" "$RELEASE_DIR/.venv/bin/pip" install --quiet -r "$RELEASE_DIR/requirements.txt"

# Lock the completed virtualenv before candidate execution. Runtime caches and logs are external.
# Root owns the frozen environment, while the service group retains read/execute traversal.
# The top-level virtualenv starts as mode 750, so changing it to root:root before removing write
# access would leave the unprivileged service unable to execute its interpreter (mode 550).
chown -R -h "root:$USER_NAME" "$RELEASE_DIR/.venv"
chmod -R a-w "$RELEASE_DIR/.venv"
ln -s "../../$PINNED_DATA_TARGET" "$RELEASE_DIR/data"

candidate_curl() { curl --noproxy '*' --fail --silent --show-error "$@"; }

install_candidate_unit() {
  local template="$RELEASE_DIR/deploy/sfci.service" temp
  [[ -f "$template" && ! -L "$template" && "$(stat -c '%u:%g:%a:%h' "$template")" == "0:0:444:1" ]] || {
    echo "candidate unit template is not frozen root-owned source" >&2; return 1
  }
  grep -Fqx 'ExecStart=/opt/sfci/current/.venv/bin/python scripts/bootstrap_server.py' "$template" || {
    echo "production unit must run scripts/bootstrap_server.py" >&2; return 1
  }
  grep -q '^ExecStartPre=' "$template" && {
    echo "production unit cannot run heavy work before bootstrap binds" >&2; return 1
  }
  temp="$(mktemp /run/sfci/candidate-unit.XXXXXX)"
  render_candidate_unit "$template" "$RELEASE_ID" 1 "$temp"
  grep -qx "Environment=PORT=$CANDIDATE_PORT" "$temp" || {
    rm -f "$temp"; echo "candidate unit did not receive its private port" >&2; return 1
  }
  grep -q "/opt/sfci/current" "$temp" && {
    rm -f "$temp"; echo "candidate unit retained a live-release path" >&2; return 1
  }
  if ! grep -Fqx "Environment=NUMBA_CACHE_DIR=/var/cache/sfci-candidate-$RELEASE_ID/numba" "$temp" ||
     ! grep -Fqx "Environment=DEST_CACHE_FILE=/var/cache/sfci-candidate-$RELEASE_ID/dest_cache.json" "$temp" ||
     ! grep -Fqx "CacheDirectory=sfci-candidate-$RELEASE_ID" "$temp"; then
    rm -f "$temp"; echo "candidate unit did not receive isolated cache paths" >&2; return 1
  fi
  install -m 600 -o root -g root "$temp" "$CANDIDATE_UNIT_FILE"
  rm -f "$temp"
  systemd-analyze verify "$CANDIDATE_UNIT_FILE"
  systemctl daemon-reload
}

candidate_smoke() {
  local base="http://127.0.0.1:$CANDIDATE_PORT"
  local compute_json itinerary_json variance_json cell_id
  direct_release_checks
  [[ -z "$(ss -H -ltn "sport = :$CANDIDATE_PORT")" ]] || {
    echo "candidate port $CANDIDATE_PORT is already in use" >&2; return 1
  }

  touch "$CANDIDATE_MARKER"
  install_candidate_unit
  systemctl start "$CANDIDATE_UNIT"

  live_bound=0
  for _ in {1..50}; do
    if candidate_curl --max-time 1 "$base/livez" | jq -e '.ok == true' >/dev/null 2>&1; then
      live_bound=1; break
    fi
    sleep 0.2
  done
  (( live_bound == 1 )) || {
    journalctl -u "$CANDIDATE_UNIT" -n 80 --no-pager >&2 || true
    echo "production unit template did not bind liveness promptly" >&2; return 1
  }

  for _ in {1..90}; do
    if systemctl is-failed --quiet "$CANDIDATE_UNIT"; then
      journalctl -u "$CANDIDATE_UNIT" -n 80 --no-pager >&2 || true
      echo "candidate exited before readiness" >&2; return 1
    fi
    if candidate_curl --max-time 2 "$base/readyz" | jq -e '.ok == true' >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
  candidate_curl --max-time 5 "$base/livez" | jq -e '.ok == true' >/dev/null
  candidate_curl --max-time 5 "$base/readyz" | jq -e '.ok == true' >/dev/null
  candidate_curl --max-time 10 "$base/" >/dev/null
  compute_json="$(candidate_curl --max-time 90 "$base/compute?lat=37.7600&lon=-122.4200")"
  jq -e '(.dest | type == "array" and length == 2) and
         (.cells | type == "object" and length > 0)' <<<"$compute_json" >/dev/null
  cell_id="$(jq -r 'first(.cells | to_entries[] | select(.value[1] != null) | .key) // empty' \
    <<<"$compute_json")"
  [[ -n "$cell_id" ]] || { echo "candidate compute returned no cells" >&2; return 1; }
  itinerary_json="$(candidate_curl --max-time 45 \
    "$base/itinerary?id=$cell_id&dlat=37.7600&dlon=-122.4200")"
  jq -e '(.total | type == "number") and (.legs | type == "array" and length > 0)' \
    <<<"$itinerary_json" >/dev/null
  variance_json="$(candidate_curl --max-time 150 \
    "$base/variance?dlat=37.7600&dlon=-122.4200")"
  jq -e '(.dest | type == "array" and length == 2) and (.variance | type == "object")' \
    <<<"$variance_json" >/dev/null
  if find "$RELEASE_DIR" -xdev -type f -newer "$CANDIDATE_MARKER" -print -quit | grep -q .; then
    echo "candidate modified its immutable release source" >&2; return 1
  fi
}

log "validating sandboxed candidate on private port $CANDIDATE_PORT"
candidate_smoke
stop_candidate

# ---- Validate temporary Caddy configuration without changing the live file ----------------
[[ -f "$ORIGIN_CERT" && -f "$ORIGIN_KEY" ]] || {
  echo "install the Cloudflare Origin CA certificate and key before promoting a release" >&2; exit 1
}
ORIGIN_CERT_META="$(origin_file_meta "$ORIGIN_CERT")" || {
  echo "$ORIGIN_CERT must be a root-owned, single-link regular file" >&2; exit 1
}
ORIGIN_KEY_META="$(origin_file_meta "$ORIGIN_KEY")" || {
  echo "$ORIGIN_KEY must be a root-owned, single-link regular file" >&2; exit 1
}
mkdir -p /etc/caddy
CADDY_CANDIDATE="$(mktemp)"; BUS_VHOST_TMP="$(mktemp)"
CADDY_WAS_ENABLED="$(unit_enabled_state caddy)"
systemctl is-active --quiet caddy && CADDY_WAS_ACTIVE=1
if [[ -e "$CADDYFILE" || -L "$CADDYFILE" ]]; then
  [[ -f "$CADDYFILE" && ! -L "$CADDYFILE" &&
     "$(stat -c '%u:%h' "$CADDYFILE")" == "0:1" ]] || {
    echo "$CADDYFILE must be a root-owned single-link regular file" >&2; exit 1
  }
  CADDY_HAD_FILE=1
  CADDY_FILE_META="$(stat -c '%u:%g:%a' "$CADDYFILE")"
  starts="$(grep -c 'BUS-MARKER START (sf-muni-commute)' "$CADDYFILE" || true)"
  ends="$(grep -c 'BUS-MARKER END (sf-muni-commute)' "$CADDYFILE" || true)"
  [[ "$starts" == "$ends" && "$starts" -le 1 ]] || {
    echo "malformed or duplicate sf-muni-commute block in $CADDYFILE" >&2; exit 1
  }
  sed -n '/BUS-MARKER START (sf-muni-commute)/,/BUS-MARKER END (sf-muni-commute)/p' \
    "$CADDYFILE" > "$BUS_VHOST_TMP"
fi
install -m 644 "$RELEASE_DIR/deploy/Caddyfile" "$CADDY_CANDIDATE"
if [[ -s "$BUS_VHOST_TMP" ]]; then
  { printf '\n'; sed -n 'p' "$BUS_VHOST_TMP"; printf '\n'; } >> "$CADDY_CANDIDATE"
fi
caddy validate --adapter caddyfile --config "$CADDY_CANDIDATE"

if [[ -L "$REPO_DIR/deploy" ]]; then
  DEPLOY_PATH_KIND="symlink"; DEPLOY_PATH_TARGET="$(readlink "$REPO_DIR/deploy")"
  [[ "$DEPLOY_PATH_TARGET" == "current/deploy" ]] || {
    echo "$REPO_DIR/deploy symlink must target current/deploy" >&2; exit 1
  }
elif [[ -d "$REPO_DIR/deploy" ]]; then
  DEPLOY_PATH_KIND="legacy_dir"
  DEPLOY_PATH_META="$(stat -c '%u:%g:%a' "$REPO_DIR/deploy")"
elif [[ -e "$REPO_DIR/deploy" ]]; then
  echo "$REPO_DIR/deploy must be a directory or symlink" >&2; exit 1
fi

# ---- Transactional cutover ---------------------------------------------------------------
snapshot_runtime_state
if (( CADDY_HAD_FILE == 1 )); then
  install -m 600 -o root -g root "$CADDYFILE" "$CADDY_ROLLBACK"
fi
validate_rollback_artifacts "$DEPLOY_PATH_KIND" "$CADDY_HAD_FILE"
write_deploy_transaction pending
CUTOVER_PENDING=1

# Candidate has passed. Unit, Caddy, pointer, and firewall mutations are now rollback-protected.
fail_closed_firewall || { echo "could not prove origin ingress closed before cutover" >&2; exit 1; }
enforce_origin_permissions || { echo "could not enforce protected origin certificate modes" >&2; exit 1; }
install_runtime_units "$RELEASE_DIR"
install -m 644 "$CADDY_CANDIDATE" "$CADDYFILE"
switch_current "$REPO_DIR" "$NEW_TARGET"
activate_deploy_path "$DEPLOY_PATH_KIND"
systemctl daemon-reload
systemctl restart sfci

live_ok=0
for _ in {1..45}; do
  if candidate_curl --max-time 3 "http://127.0.0.1:$PRODUCTION_PORT/livez" 2>/dev/null |
       jq -e '.ok == true' >/dev/null &&
     candidate_curl --max-time 3 "http://127.0.0.1:$PRODUCTION_PORT/readyz" 2>/dev/null |
       jq -e '.ok == true' >/dev/null &&
     candidate_curl --max-time 10 "http://127.0.0.1:$PRODUCTION_PORT/" >/dev/null 2>&1; then
    live_ok=1; break
  fi
  sleep 2
done
(( live_ok == 1 )) || { echo "post-cutover application smoke failed" >&2; exit 1; }

# Unconditionally refresh and prove Cloudflare-only ingress; never open public HTTP/S.
ensure_cloudflare_lockdown || { echo "Cloudflare-only firewall verification failed" >&2; exit 1; }

rm -f /etc/logrotate.d/sfci
install -m 644 /dev/null /etc/logrotate.d/sfci
printf '%s\n' '/var/log/sfci/*.log {' '  weekly' '  rotate 4' '  compress' '  delaycompress' \
  '  missingok' '  notifempty' '  create 640 sfci sfci' '}' > /etc/logrotate.d/sfci

systemctl enable sfci sfci-keepalive.timer cloudflare-ufw.timer caddy
systemctl start sfci-keepalive.timer
systemctl restart caddy

# Exercise the real local proxy without ambient proxy variables, then public Cloudflare/TLS.
proxy_args=(--noproxy '*' --proto '=https' --tlsv1.2 --max-redirs 0 \
  --insecure --fail --silent --show-error --max-time 10 \
  --resolve sfcommutemap.com:443:127.0.0.1)
HTTP_BODY_TMP="$(mktemp)"
curl_200() {
  local output="$1"; shift
  [[ "$(curl "$@" --output "$output" --write-out '%{http_code}')" == "200" ]]
}
curl_200 "$HTTP_BODY_TMP" "${proxy_args[@]}" https://sfcommutemap.com/livez
jq -e '.ok == true' "$HTTP_BODY_TMP" >/dev/null
curl_200 "$HTTP_BODY_TMP" "${proxy_args[@]}" https://sfcommutemap.com/readyz
jq -e '.ok == true' "$HTTP_BODY_TMP" >/dev/null
curl_200 "$HTTP_BODY_TMP" "${proxy_args[@]}" https://sfcommutemap.com/
grep -qi '<!doctype html' "$HTTP_BODY_TMP"
public_args=(--noproxy '*' --proto '=https' --tlsv1.2 --max-redirs 0 \
  --fail --silent --show-error --max-time 20)
curl_200 "$HTTP_BODY_TMP" "${public_args[@]}" "$PUBLIC_SMOKE_URL/livez"
jq -e '.ok == true' "$HTTP_BODY_TMP" >/dev/null
curl_200 "$HTTP_BODY_TMP" "${public_args[@]}" "$PUBLIC_SMOKE_URL/readyz"
jq -e '.ok == true' "$HTTP_BODY_TMP" >/dev/null
curl_200 "$HTTP_BODY_TMP" "${public_args[@]}" "$PUBLIC_SMOKE_URL/"
grep -qi '<!doctype html' "$HTTP_BODY_TMP"
verify_cloudflare_lockdown || { echo "firewall changed during public smoke" >&2; exit 1; }

# Mark committed before non-critical retention cleanup.
write_deploy_transaction committed
remove_env_marker "$NEW_TARGET"
cleanup_committed_rollback_artifacts "$DEPLOY_PATH_KIND"
CUTOVER_PENDING=0
remove_deploy_transaction
prune_releases "$REPO_DIR" || log "warning: release pruning failed"
prune_data_releases "$REPO_DIR" || log "warning: data-release pruning failed"
cleanup_abandoned_stages "$REPO_DIR" || log "warning: abandoned-stage cleanup failed"
cleanup_abandoned_data_stages "$REPO_DIR" || log "warning: abandoned data-stage cleanup failed"

trap - EXIT INT TERM
cleanup_temps
log "release $RELEASE_ID is active"
systemctl --no-pager --plain status sfci sfci-keepalive.timer cloudflare-ufw.timer 2>&1 | head -25 || true
