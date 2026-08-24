#!/usr/bin/env bash
# Promote a separately built transit-data stage without changing any active data release.

set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "Run with sudo" >&2; exit 1; }

ROOT="/opt/sfci"
DATA_RELEASES="$ROOT/data-releases"
DATA_INCOMING="$ROOT/data-incoming"
DATA_TRANSACTION="$ROOT/.data-deploy-transaction"

usage() {
  echo "Usage: data-refresh.sh (--promote-id|--adopt-legacy) YYYYMMDD[HHMMSS] [--bootstrap-code-id YYYYMMDDHHMMSS]" >&2
  exit 2
}
BOOTSTRAP_CODE_ID=""
if [[ "$#" == 2 && ( "$1" == "--promote-id" || "$1" == "--adopt-legacy" ) &&
      "$2" =~ ^[0-9]{8}([0-9]{6})?$ ]]; then
  MODE="$1"; DATA_ID="$2"
elif [[ "$#" == 4 && "$1" == "--promote-id" &&
        "$2" =~ ^[0-9]{8}([0-9]{6})?$ && "$3" == "--bootstrap-code-id" &&
        "$4" =~ ^[0-9]{14}$ ]]; then
  MODE="$1"; DATA_ID="$2"; BOOTSTRAP_CODE_ID="$4"
else
  usage
fi

CONTROL_DIR="$(cd "$(dirname "$0")" && pwd -P)"
[[ "$CONTROL_DIR" =~ ^/opt/sfci/\.deploy-control/[0-9]{14}$ && ! -L "$CONTROL_DIR" &&
   "$(stat -c '%u:%g:%a' "$CONTROL_DIR")" == "0:0:700" ]] || {
  echo "run the root-owned data-refresh helper installed by deploy/push.sh" >&2; exit 1
}
[[ ! -L "$ROOT/.deploy-control" &&
   "$(stat -c '%u:%g:%a' "$ROOT/.deploy-control")" == "0:0:700" ]] || {
  echo "$ROOT/.deploy-control must be a root-owned mode-700 real directory" >&2; exit 1
}
for control_file in "$CONTROL_DIR/data-refresh.sh" "$CONTROL_DIR/release_ops.sh"; do
  expected_mode=500; [[ "$control_file" == */release_ops.sh ]] && expected_mode=400
  [[ -f "$control_file" && ! -L "$control_file" &&
     "$(stat -c '%u:%g:%a:%h' "$control_file")" == "0:0:$expected_mode:1" ]] || {
    echo "unsafe deployment control file: $control_file" >&2; exit 1
  }
done

[[ ! -L "$ROOT" ]] || { echo "$ROOT must not be a symlink" >&2; exit 1; }
chown root:root "$ROOT"; chmod 755 "$ROOT"
if [[ ! -e "$ROOT/.deploy.lock" && ! -L "$ROOT/.deploy.lock" ]]; then
  (umask 077; set -o noclobber; : > "$ROOT/.deploy.lock") 2>/dev/null || true
fi
[[ -f "$ROOT/.deploy.lock" && ! -L "$ROOT/.deploy.lock" &&
   "$(stat -c '%h' "$ROOT/.deploy.lock")" == "1" ]] || {
  echo "unsafe deployment lock" >&2; exit 1
}
chown root:root "$ROOT/.deploy.lock"
chmod 600 "$ROOT/.deploy.lock"
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 127; }
for prerequisite in jq unzip; do
  command -v "$prerequisite" >/dev/null 2>&1 || {
    echo "$prerequisite is required for immutable data validation" >&2; exit 127
  }
done
exec 9>"$ROOT/.deploy.lock"
flock -n 9 || { echo "another code or data deployment is running" >&2; exit 75; }

# shellcheck source=release_ops.sh disable=SC1091
source "$CONTROL_DIR/release_ops.sh"
for directory in "$DATA_RELEASES" "$DATA_INCOMING"; do
  [[ ! -L "$directory" ]] || { echo "$directory must not be a symlink" >&2; exit 1; }
  install -d -m 755 -o root -g root "$directory"
  [[ "$(cd "$directory" && pwd -P)" == "$directory" ]] || {
    echo "$directory escapes its canonical path" >&2; exit 1
  }
done

read_data_transaction_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$DATA_TRANSACTION"
}

validate_data_transaction_schema() {
  local key
  [[ "$(wc -l < "$DATA_TRANSACTION")" == "4" ]] || return 1
  for key in phase old previous new; do
    [[ "$(grep -c "^$key=" "$DATA_TRANSACTION")" == "1" ]] || return 1
  done
  awk -F= '$1 !~ /^(phase|old|previous|new)$/ { bad=1 } END { exit bad }' "$DATA_TRANSACTION"
}

write_data_transaction() {
  local phase="$1" old="$2" previous="$3" new="$4" temp="${DATA_TRANSACTION}.tmp.$$"
  [[ ! -e "$temp" && ! -L "$temp" ]] || return 1
  (umask 077
    set -o noclobber
    printf 'phase=%s\nold=%s\nprevious=%s\nnew=%s\n' "$phase" "$old" "$previous" "$new" \
      > "$temp")
  chown root:root "$temp"
  chmod 600 "$temp"
  mv -f "$temp" "$DATA_TRANSACTION"
  durable_sync_path "$DATA_TRANSACTION"
  durable_sync_path "$ROOT"
}

recover_data_transaction() {
  [[ -e "$DATA_TRANSACTION" || -L "$DATA_TRANSACTION" ]] || return 0
  [[ -f "$DATA_TRANSACTION" && ! -L "$DATA_TRANSACTION" &&
     "$(stat -c '%u:%g:%a:%h' "$DATA_TRANSACTION")" == "0:0:600:1" ]] || {
    echo "unsafe pending data transaction; refusing recovery" >&2; return 1
  }
  validate_data_transaction_schema || {
    echo "malformed pending data transaction; refusing recovery" >&2; return 1
  }
  local phase old previous new target
  phase="$(read_data_transaction_value phase)"
  old="$(read_data_transaction_value old)"
  previous="$(read_data_transaction_value previous)"
  new="$(read_data_transaction_value new)"
  [[ "$new" =~ ^data-releases/[0-9]{8}([0-9]{6})?$ ]] || return 1
  [[ -z "$old" ]] || valid_data_target "$ROOT" "$old" || return 1
  [[ -z "$previous" ]] || valid_data_target "$ROOT" "$previous" || return 1
  [[ -n "$old" || -z "$previous" ]] || return 1
  target="$ROOT/$new"
  if [[ "$phase" == "committed" ]]; then
    valid_data_target "$ROOT" "$new" || return 1
    [[ "$(data_current_target "$ROOT")" == "$new" ]] || return 1
    [[ "$(data_previous_target "$ROOT")" == "$old" ]] || return 1
    rm -f "$DATA_TRANSACTION"
    return 0
  fi
  [[ "$phase" == "pending" ]] || return 1
  restore_data_pointers_exact "$ROOT" "$old" "$previous"
  if [[ -d "$target" && ! -L "$target" && "$(data_current_target "$ROOT")" != "$new" ]]; then
    rm -rf "${target:?}"
  elif [[ -e "$target" || -L "$target" ]]; then
    return 1
  fi
  rm -f "$DATA_TRANSACTION"
}

recover_data_transaction
cleanup_abandoned_data_stages "$ROOT" "$DATA_ID"

validate_tree_links() {
  reject_data_payload_links "$1"
}

validate_data_contents() {
  local tree="$1" archive feed
  [[ -f "$tree/server_static.json" && -f "$tree/walk_graph.npz" &&
     -d "$tree/raptor_cache" ]] || {
    echo "data stage requires server_static.json, walk_graph.npz, and raptor_cache/" >&2; return 1
  }
  jq -e 'type == "object" and length > 0' "$tree/server_static.json" >/dev/null
  unzip -tqq "$tree/walk_graph.npz"
  archive="$(find "$tree/raptor_cache" -maxdepth 1 -type f \( -name '*.npz' -o -name '*.pkl' \) -print -quit)"
  [[ -n "$archive" ]] || { echo "raptor_cache contains no built artifacts" >&2; return 1; }
  for feed in muni_current.zip bart_gtfs.zip caltrain.zip; do
    [[ -f "$tree/$feed" ]] || { echo "data stage is missing required feed $feed" >&2; return 1; }
    unzip -tqq "$tree/$feed"
  done
}

freeze_and_validate() {
  local tree="$1" bad
  # Revoke the uploader's directory access before root traverses or reads any staged content.
  # The root-owned data-incoming parent prevents the timestamped entry itself being replaced.
  chown root:root "$tree"
  chmod 700 "$tree"
  validate_tree_links "$tree"
  chown -R -h root:root "$tree"
  # Builder umasks vary. Normalize the immutable release to service-readable permissions instead
  # of merely removing write bits (which would turn a mode-600 upload into unreadable mode 400).
  find "$tree" -xdev -type d -exec chmod 555 {} +
  find "$tree" -xdev -type f -exec chmod 444 {} +
  validate_tree_links "$tree"
  bad="$(find "$tree" -xdev \( ! -user root -o ! -group root -o -perm /222 \) -print -quit)"
  [[ -z "$bad" ]] || { echo "data release did not freeze root-owned/read-only: $bad" >&2; return 1; }
  validate_data_contents "$tree"
}

validate_full_readiness() {
  local tree="$1" code_target code_root python
  if [[ -n "$BOOTSTRAP_CODE_ID" ]]; then
    [[ "${CONTROL_DIR##*/}" == "$BOOTSTRAP_CODE_ID" ]] || {
      echo "bootstrap validation must use the control files from the same code release" >&2
      return 1
    }
    code_target="releases/$BOOTSTRAP_CODE_ID"
    valid_release_target "$ROOT" "$code_target" || {
      echo "bootstrap code release is missing or unsafe" >&2; return 1
    }
  else
    code_target="$(current_target "$ROOT")"
    valid_release_target "$ROOT" "$code_target" || {
      echo "a trusted active code release is required for full staged-data readiness" >&2
      return 1
    }
  fi
  code_root="$ROOT/$code_target"
  python="$code_root/.venv/bin/python"
  [[ -f "$code_root/scripts/core/readiness.py" &&
     ! -L "$code_root/scripts/core/readiness.py" &&
     "$(stat -c '%u:%g:%a:%h' "$code_root/scripts/core/readiness.py")" == "0:0:444:1" ]] || {
    echo "readiness validator is not frozen root-owned release source" >&2; return 1
  }
  [[ -x "$python" ]] || { echo "readiness release virtualenv is incomplete" >&2; return 1; }
  runuser -u sfci -- env PYTHONPATH="$code_root/scripts" "$python" - "$tree" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

from core import readiness

tree = Path(sys.argv[1])
bundle = json.loads((tree / "server_static.json").read_text())
service_date = str(bundle.get("svc_date", bundle.get("service_date", "")))
if len(service_date) != 8 or not service_date.isdigit():
    raise SystemExit("invalid staged service date")
stage_date = dt.datetime.strptime(service_date, "%Y%m%d").date()
if stage_date.weekday() != 2:
    raise SystemExit("staged service date must be a Wednesday")
stage_now = dt.datetime.combine(stage_date, dt.time(0, 0), tzinfo=readiness.LA)
grid_m = int(bundle.get("grid_m", 0))
grid_source_name = bundle.get("grid_source_name")
if not isinstance(grid_source_name, str) or Path(grid_source_name).name != grid_source_name:
    raise SystemExit("invalid grid source name in server_static.json")
feeds = {
    "muni": tree / "muni_current.zip",
    "bart": tree / "bart_gtfs.zip",
    "caltrain": tree / "caltrain.zip",
}
feed_sources = tuple(
    (path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in feeds.values()
)
bundle_sources = tuple(tuple(value) for value in bundle.get("source_mtimes", ()))
if bundle_sources != feed_sources:
    raise SystemExit("server_static.json was not built from the staged feed files")
access = tree / "raptor_cache" / f"access_walk_{grid_m}m_{service_date}.npz"
raptor_candidates = sorted((tree / "raptor_cache").glob(
    f"raptor_{service_date}_*_footpath*m.pkl"
))
if not raptor_candidates:
    raise SystemExit("no canonical RAPTOR cache matches the staged modeled Wednesday")
failures = []
for raptor in raptor_candidates:
    cache_check, cache_data = readiness.load_raptor_state(raptor, service_date)
    if not cache_check.ready or tuple(cache_data.get("source_mtimes", ())) != feed_sources:
        failures.append(f"{raptor.name}:stale_feed_sources")
        continue
    result = readiness.check_readiness(
        feeds,
        raptor,
        tree / "walk_graph.npz",
        access,
        tree / "server_static.json",
        now=stage_now,
        grid_m=grid_m,
        grid_source=tree / grid_source_name,
    )
    if result.ready:
        if result.service_date != service_date:
            raise SystemExit("readiness selected a different staged service date")
        print(json.dumps(result.as_dict(), sort_keys=True))
        break
    failures.append(f"{raptor.name}:{result.reason_code}:{result.detail or ''}")
else:
    raise SystemExit("staged data failed readiness: " + ", ".join(failures))
PY
}

TARGET="$DATA_RELEASES/$DATA_ID"

if [[ "$MODE" == "--promote-id" ]]; then
  validate_data_pointers "$ROOT" || { echo "unsafe data release pointers" >&2; exit 1; }
  if [[ -n "$BOOTSTRAP_CODE_ID" ]]; then
    [[ ! -e "$ROOT/current" && ! -L "$ROOT/current" &&
       ! -e "$ROOT/previous" && ! -L "$ROOT/previous" ]] || {
      echo "bootstrap data promotion is allowed only before the first immutable code cutover" >&2
      exit 1
    }
    [[ -L "$ROOT/data" && "$(readlink "$ROOT/data")" == "data-current" ]] || {
      echo "bootstrap data promotion requires an adopted legacy data pointer" >&2; exit 1
    }
  else
    validate_release_pointers "$ROOT" || { echo "unsafe code release pointers" >&2; exit 1; }
    [[ -L "$ROOT/current" ]] || {
      echo "promote code into the immutable release layout before advancing data-current" >&2
      exit 1
    }
  fi
  if [[ -e "$TARGET" || -L "$TARGET" ]]; then
    if valid_data_target "$ROOT" "data-releases/$DATA_ID" &&
       [[ "$(data_current_target "$ROOT")" == "data-releases/$DATA_ID" ]]; then
      reject_data_payload_links "$TARGET"
      validate_data_contents "$TARGET"
      validate_full_readiness "$TARGET"
      echo "[data-refresh] $DATA_ID is already the validated data-current release"
      exit 0
    fi
    echo "data release already exists but is not the active idempotent target: $TARGET" >&2
    exit 1
  fi
  STAGE="$DATA_INCOMING/$DATA_ID"
  [[ -d "$STAGE" && ! -L "$STAGE" && "$(cd "$STAGE" && pwd -P)" == "$STAGE" ]] || {
    echo "missing canonical data stage: $STAGE" >&2; exit 1
  }
  [[ "$(stat -c '%d' "$STAGE")" == "$(stat -c '%d' "$DATA_RELEASES")" ]] || {
    echo "data-incoming and data-releases must share one filesystem for atomic promotion" >&2
    exit 1
  }
  old_data_target="$(data_current_target "$ROOT")"
  old_data_previous="$(data_previous_target "$ROOT")"
  promotion_complete=0
  target_move_started=0
  pointer_change_started=0
  # Invoked indirectly by the EXIT trap below.
  # shellcheck disable=SC2329
  rollback_data_promotion() {
    local status=$? rollback_ok=1
    trap - EXIT INT TERM
    if (( promotion_complete == 0 )); then
      if (( pointer_change_started == 1 )); then
        restore_data_pointers_exact "$ROOT" "$old_data_target" "$old_data_previous" || rollback_ok=0
      fi
      if (( target_move_started == 1 )) && [[ -d "$TARGET" && ! -L "$TARGET" &&
           "$(data_current_target "$ROOT")" != "data-releases/$DATA_ID" ]]; then
        rm -rf "${TARGET:?}"
      fi
      if (( rollback_ok == 1 )); then
        rm -f "$DATA_TRANSACTION"
      else
        status=1
      fi
    fi
    exit "$status"
  }
  trap rollback_data_promotion EXIT
  trap 'exit 130' INT TERM
  freeze_and_validate "$STAGE"
  # Full modeled-Wednesday and artifact validation happens while the release is still an inactive
  # incoming sibling. A failure leaves it there for diagnosis and never consumes the stage.
  validate_full_readiness "$STAGE"
  write_data_transaction pending "$old_data_target" "$old_data_previous" \
    "data-releases/$DATA_ID"
  target_move_started=1
  mv "$STAGE" "$TARGET"
  durable_sync_path "$DATA_INCOMING"
  durable_sync_path "$DATA_RELEASES"
  pointer_change_started=1
  switch_data_current "$ROOT" "data-releases/$DATA_ID"
  validate_data_pointers "$ROOT"
  write_data_transaction committed "$old_data_target" "$old_data_previous" \
    "data-releases/$DATA_ID"
  promotion_complete=1
  trap - EXIT INT TERM
  rm -f "$DATA_TRANSACTION"
  prune_data_releases "$ROOT" || echo "[data-refresh] warning: data-release pruning failed" >&2
  echo "[data-refresh] promoted $DATA_ID; active code remains pinned to its existing data release"
  exit 0
fi

# One-time migration. Stop every possible reader before renaming/freezing the old mutable tree.
if [[ -e "$ROOT/data-current" || -L "$ROOT/data-current" ]]; then
  if validate_data_pointers "$ROOT" && valid_data_target "$ROOT" "data-releases/$DATA_ID" &&
     [[ "$(data_current_target "$ROOT")" == "data-releases/$DATA_ID" &&
        -L "$ROOT/data" && "$(readlink "$ROOT/data")" == "data-current" ]]; then
    reject_data_payload_links "$TARGET"
    validate_data_contents "$TARGET"
    echo "[data-refresh] legacy data is already adopted as $DATA_ID"
    exit 0
  fi
  echo "data-current already exists with a different or unsafe legacy-adoption state" >&2
  exit 1
fi
[[ ! -e "$ROOT/data-previous" && ! -L "$ROOT/data-previous" ]] || {
  echo "data-previous cannot exist before initial legacy adoption" >&2; exit 1
}
[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || { echo "data release already exists: $TARGET" >&2; exit 1; }
[[ -d "$ROOT/data" && ! -L "$ROOT/data" ]] || {
  echo "legacy adoption requires a real /opt/sfci/data directory" >&2; exit 1
}
sfci_was_active=0; caddy_was_active=0; migration_complete=0
systemctl is-active --quiet sfci && sfci_was_active=1
systemctl is-active --quiet caddy && caddy_was_active=1
restore_legacy_readers() {
  local status=$?
  trap - EXIT INT TERM
  if (( migration_complete == 0 )); then
    rm -f "$ROOT/data-current"
    [[ ! -L "$ROOT/data" ]] || rm -f "$ROOT/data"
    if [[ -d "$TARGET" && ! -L "$TARGET" && ! -e "$ROOT/data" ]]; then mv "$TARGET" "$ROOT/data"; fi
  fi
  (( sfci_was_active == 0 )) || systemctl start sfci >/dev/null 2>&1 || true
  (( caddy_was_active == 0 )) || systemctl start caddy >/dev/null 2>&1 || true
  exit "$status"
}
trap restore_legacy_readers EXIT
trap 'exit 130' INT TERM
systemctl stop caddy >/dev/null 2>&1 || true
systemctl stop sfci >/dev/null 2>&1 || true
mv "$ROOT/data" "$TARGET"
freeze_and_validate "$TARGET"
switch_data_current "$ROOT" "data-releases/$DATA_ID"
atomic_symlink "data-current" "$ROOT/data"
migration_complete=1
trap - EXIT INT TERM
(( sfci_was_active == 0 )) || systemctl start sfci
(( caddy_was_active == 0 )) || systemctl start caddy
echo "[data-refresh] adopted legacy data as immutable data release $DATA_ID"
