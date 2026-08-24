#!/usr/bin/env bash
# Small release and pointer helpers shared by install.sh and deterministic tests.

set -euo pipefail

release_id() {
  date -u +%Y%m%d%H%M%S
}

durable_sync_path() {
  local path="$1"
  # Deployment production is Linux/coreutils. Local macOS helper tests exercise pointer semantics
  # but cannot provide GNU sync -f's per-filesystem durability contract.
  [[ "$(uname -s)" == "Linux" ]] || return 0
  command -v sync >/dev/null 2>&1 || return 127
  sync -f "$path"
}

reject_code_payload_links() {
  local root="$1" bad
  bad="$(find "$root" -xdev \
    \( -path "$root/.venv" -o -path "$root/data" \) -prune -o \
    \( -type l -o \( -type f -links +1 \) \) -print -quit)"
  [[ -z "$bad" ]] || { echo "code payload contains a symlink or hardlinked file: $bad" >&2; return 1; }
}

reject_data_payload_links() {
  local root="$1" bad
  bad="$(find "$root" -xdev \( -type l -o \( -type f -links +1 \) \) -print -quit)"
  [[ -z "$bad" ]] || { echo "data payload contains a symlink or hardlinked file: $bad" >&2; return 1; }
}

acquire_deploy_lock() {
  local root="$1"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$root/.deploy.lock"
    flock -n 9
    return
  fi
  # macOS development/test hosts do not ship flock. Production must have a real descriptor lock;
  # this atomic-directory fallback exists only for local tests.
  [[ "$(uname -s)" == "Darwin" ]] || {
    echo "flock is required for production deployment" >&2
    return 127
  }
  SFCI_LOCAL_LOCK_DIR="$root/.deploy-lock-local"
  mkdir "$SFCI_LOCAL_LOCK_DIR"
  trap 'rmdir "$SFCI_LOCAL_LOCK_DIR" 2>/dev/null || true' EXIT
}

valid_release_target() {
  local root="$1"
  local target="$2"
  [[ "$target" =~ ^releases/[0-9]{14}$ ]] || return 1
  [[ -d "$root/$target" && ! -L "$root/$target" ]] || return 1
  local resolved releases_resolved
  resolved="$(cd "$root/$target" && pwd -P)"
  releases_resolved="$(cd "$root/releases" && pwd -P)"
  [[ "$resolved" == "$releases_resolved/"* ]]
}

valid_data_target() {
  local root="$1" target="$2" resolved data_releases_resolved
  [[ "$target" =~ ^data-releases/[0-9]{8}([0-9]{6})?$ ]] || return 1
  [[ -d "$root/$target" && ! -L "$root/$target" ]] || return 1
  resolved="$(cd "$root/$target" && pwd -P)"
  data_releases_resolved="$(cd "$root/data-releases" && pwd -P)"
  [[ "$resolved" == "$data_releases_resolved/"* ]]
}

data_current_target() {
  local root="$1"
  readlink "$root/data-current" 2>/dev/null || true
}

data_previous_target() {
  local root="$1"
  readlink "$root/data-previous" 2>/dev/null || true
}

validate_data_pointer() {
  local root="$1" name="$2" pointer target
  [[ "$name" == "data-current" || "$name" == "data-previous" ]] || return 1
  pointer="$root/$name"
  if [[ ! -e "$pointer" && ! -L "$pointer" ]]; then
    [[ "$name" == "data-previous" ]]
    return
  fi
  [[ -L "$pointer" ]] || return 1
  target="$(readlink "$pointer")"
  valid_data_target "$root" "$target"
}

validate_data_current() {
  validate_data_pointer "$1" data-current
}

validate_data_pointers() {
  local root="$1"
  validate_data_pointer "$root" data-current
  validate_data_pointer "$root" data-previous
}

switch_data_current() {
  local root="$1" next="$2" old
  if [[ -e "$root/data-current" || -L "$root/data-current" ]]; then
    validate_data_pointer "$root" data-current
  fi
  validate_data_pointer "$root" data-previous
  valid_data_target "$root" "$next"
  old="$(data_current_target "$root")"
  if [[ -n "$old" && "$old" != "$next" ]]; then
    atomic_symlink "$old" "$root/data-previous"
  fi
  atomic_symlink "$next" "$root/data-current"
}

restore_data_current_to() {
  local root="$1" old="$2" failed="$3"
  [[ -z "$old" ]] || valid_data_target "$root" "$old"
  [[ -z "$failed" ]] || valid_data_target "$root" "$failed"
  if [[ -n "$old" ]]; then
    atomic_symlink "$old" "$root/data-current"
    if [[ -n "$failed" && "$failed" != "$old" ]]; then
      atomic_symlink "$failed" "$root/data-previous"
    fi
  else
    rm -f "$root/data-current"
    if [[ "$(data_previous_target "$root")" == "$failed" ]]; then
      rm -f "$root/data-previous"
    fi
  fi
}

restore_data_pointers_exact() {
  local root="$1" current="$2" previous="$3"
  [[ -z "$current" ]] || valid_data_target "$root" "$current"
  [[ -z "$previous" ]] || valid_data_target "$root" "$previous"
  [[ -n "$current" || -z "$previous" ]] || return 1
  if [[ -n "$current" ]]; then
    atomic_symlink "$current" "$root/data-current"
  else
    rm -f "$root/data-current"
  fi
  if [[ -n "$previous" ]]; then
    atomic_symlink "$previous" "$root/data-previous"
  else
    rm -f "$root/data-previous"
  fi
}

validate_release_pointer() {
  local root="$1"
  local name="$2"
  local pointer="$root/$name"
  [[ "$name" == "current" || "$name" == "previous" ]] || return 1
  if [[ ! -e "$pointer" && ! -L "$pointer" ]]; then
    return 0
  fi
  [[ -L "$pointer" ]] || return 1
  valid_release_target "$root" "$(readlink "$pointer")"
}

validate_release_pointers() {
  local root="$1"
  validate_release_pointer "$root" current
  validate_release_pointer "$root" previous
  if [[ ! -L "$root/current" && -L "$root/previous" ]]; then
    return 1
  fi
}

atomic_symlink() {
  local target="$1"
  local link="$2"
  local tmp="${link}.tmp.$$"
  rm -f "$tmp"
  ln -s "$target" "$tmp"
  # Production is GNU/Linux, where -T atomically replaces the pointer itself. The fallback keeps
  # the helper testable with macOS's older mv; it is never used by the production host.
  if mv -fT "$tmp" "$link" 2>/dev/null; then
    durable_sync_path "$(dirname "$link")"
    return 0
  fi
  rm -f "$link"
  mv "$tmp" "$link"
  durable_sync_path "$(dirname "$link")"
}

current_target() {
  local root="$1"
  readlink "$root/current" 2>/dev/null || true
}

previous_target() {
  local root="$1"
  readlink "$root/previous" 2>/dev/null || true
}

switch_current() {
  local root="$1"
  local next="$2"
  local old
  validate_release_pointers "$root"
  valid_release_target "$root" "$next"
  old="$(current_target "$root")"
  if [[ -n "$old" ]]; then
    atomic_symlink "$old" "$root/previous"
  fi
  atomic_symlink "$next" "$root/current"
}

restore_current_to() {
  local root="$1"
  local old="$2"
  local failed="$3"
  [[ -z "$old" ]] || valid_release_target "$root" "$old"
  [[ -z "$failed" ]] || valid_release_target "$root" "$failed"

  if [[ -n "$old" ]]; then
    # Restore current first. If interrupted here, traffic is already back on the known-good code.
    atomic_symlink "$old" "$root/current"
    if [[ -n "$failed" && "$failed" != "$old" ]]; then
      atomic_symlink "$failed" "$root/previous"
    fi
  else
    # First install: there is no known-good target. Never leave the failed candidate active.
    rm -f "$root/current"
    if [[ "$(previous_target "$root")" == "$failed" ]]; then
      rm -f "$root/previous"
    fi
  fi
}

restore_release_pointers_exact() {
  local root="$1"
  local current="$2"
  local previous="$3"
  [[ -z "$current" ]] || valid_release_target "$root" "$current"
  [[ -z "$previous" ]] || valid_release_target "$root" "$previous"
  [[ -n "$current" || -z "$previous" ]] || return 1
  if [[ -n "$current" ]]; then
    atomic_symlink "$current" "$root/current"
  else
    rm -f "$root/current"
  fi
  if [[ -n "$previous" ]]; then
    atomic_symlink "$previous" "$root/previous"
  else
    rm -f "$root/previous"
  fi
}

prune_releases() {
  local root="$1"
  local releases="$root/releases"
  local current previous name path
  local kept_extra=0
  local release_names=""
  validate_release_pointers "$root"
  current="$(basename "$(current_target "$root")" 2>/dev/null || true)"
  previous="$(basename "$(previous_target "$root")" 2>/dev/null || true)"

  # Keep active, previous, and the newest additional release. Ignore non-timestamp directories.
  for path in "$releases"/*; do
    [[ -d "$path" && ! -L "$path" ]] || continue
    name="${path##*/}"
    [[ "$name" =~ ^[0-9]{14}$ ]] && release_names+="$name\n"
  done
  while IFS= read -r name; do
    [[ -n "$name" ]] || continue
    [[ "$name" == "$current" || "$name" == "$previous" ]] && continue
    if (( kept_extra == 0 )); then
      kept_extra=1
    else
      rm -rf "${releases:?}/$name"
    fi
  done < <(printf '%b' "$release_names" | sort -rn)
}

cleanup_abandoned_stages() {
  local root="$1"
  local incoming="$root/.incoming"
  local stage name
  [[ -d "$incoming" ]] || return 0
  while IFS= read -r stage; do
    name="${stage##*/}"
    [[ "$name" =~ ^[0-9]{14}$ ]] || continue
    rm -rf "${stage:?}"
  done < <(find "$incoming" -mindepth 1 -maxdepth 1 -type d -mtime +7 -print)
}

cleanup_abandoned_data_stages() {
  local root="$1" preserve_id="${2:-}"
  local incoming="$root/data-incoming"
  local stage name release data_link target
  local keep_names=$'\n'
  [[ -d "$incoming" && ! -L "$incoming" ]] || return 0
  [[ -z "$preserve_id" || "$preserve_id" =~ ^[0-9]{8}([0-9]{6})?$ ]] || return 1
  [[ -z "$preserve_id" ]] || keep_names+="$preserve_id"$'\n'
  for target in "$(data_current_target "$root")" "$(data_previous_target "$root")"; do
    [[ -z "$target" ]] || keep_names+="${target##*/}"$'\n'
  done
  for release in "$root/releases"/*; do
    [[ -d "$release" && ! -L "$release" ]] || continue
    data_link="$release/data"
    [[ -L "$data_link" ]] || continue
    target="$(readlink "$data_link")"
    [[ "$target" =~ ^\.\./\.\./data-releases/([0-9]{8}|[0-9]{14})$ ]] || return 1
    keep_names+="${target##*/}"$'\n'
  done
  while IFS= read -r stage; do
    name="${stage##*/}"
    [[ "$name" =~ ^[0-9]{8}([0-9]{6})?$ ]] || continue
    [[ "$keep_names" == *$'\n'"$name"$'\n'* ]] || rm -rf "${stage:?}"
  done < <(find "$incoming" -mindepth 1 -maxdepth 1 -type d -mtime +7 -print)
}

prune_data_releases() {
  local root="$1" release data_link target path name
  # Newline-delimited sentinels keep this helper compatible with macOS Bash 3.2 in local tests.
  # Production uses util-linux Bash/flock, but pointer behavior should be deterministic everywhere.
  local keep_names=$'\n'
  validate_data_pointers "$root"
  target="$(data_current_target "$root")"
  keep_names+="${target##*/}"$'\n'
  target="$(data_previous_target "$root")"
  if [[ -n "$target" ]]; then
    keep_names+="${target##*/}"$'\n'
  fi
  for release in "$root/releases"/*; do
    [[ -d "$release" && ! -L "$release" ]] || continue
    data_link="$release/data"
    [[ -L "$data_link" ]] || continue
    target="$(readlink "$data_link")"
    [[ "$target" =~ ^\.\./\.\./data-releases/([0-9]{8}|[0-9]{14})$ ]] || return 1
    valid_data_target "$root" "${target#../../}" || return 1
    [[ "$keep_names" == *$'\n'"${target##*/}"$'\n'* ]] || keep_names+="${target##*/}"$'\n'
  done
  for path in "$root/data-releases"/*; do
    [[ -d "$path" && ! -L "$path" ]] || continue
    name="${path##*/}"
    [[ "$name" =~ ^[0-9]{8}([0-9]{6})?$ ]] || continue
    [[ "$keep_names" == *$'\n'"$name"$'\n'* ]] || rm -rf "${path:?}"
  done
}
