"""Rootless, networkless tests for the deployment release contract."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "deploy" / "release_ops.sh"


def bash_script(script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"source {HELPER!s}\n{script}"],
        check=check,
        text=True,
        capture_output=True,
    )


def direct_rule_helper(
    script: str, rule: str, *, python: str | None = None
) -> subprocess.CompletedProcess[str]:
    install = (ROOT / "deploy" / "install.sh").read_text()
    start = install.index("direct_rule_fields() {")
    end = install.index("\nfirewall_call() {", start)
    functions = install[start:end]
    return subprocess.run(
        [
            "bash",
            "-c",
            f"PY_BIN={python or sys.executable}\n{functions}\n{script}",
            "sfci-test",
            rule,
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def firewall_primitive_helper(script: str, value: str) -> subprocess.CompletedProcess[str]:
    install = (ROOT / "deploy" / "install.sh").read_text()
    start = install.index("port_token_exposes_web() {")
    end = install.index("\ndirect_rule_fields() {", start)
    functions = install[start:end]
    return subprocess.run(
        ["bash", "-c", f"{functions}\n{script}", "sfci-test", value],
        check=False,
        text=True,
        capture_output=True,
    )


def direct_firewall_flow_helper(
    flow: str,
    rule: str,
    *,
    python: str | None = None,
    removal_status: int = 0,
) -> subprocess.CompletedProcess[str]:
    install = (ROOT / "deploy" / "install.sh").read_text()
    start = install.index("direct_rule_fields() {")
    end = install.index("\nverify_cloudflare_zone() {", start)
    functions = install[start:end]
    mock = r'''
TEST_RULE="$1"
REMOVE_STATUS="$2"
firewall_call() {
  local joined="$*"
  case "$joined" in
    *--get-zones*) printf '%s\n' public ;;
    *--get-target*) printf '%s\n' DROP ;;
    *--list-all*) printf '%s\n' '  target: DROP' ;;
    *--list-services*|*--list-ports*|*--list-protocols*|*--list-source-ports*|*--list-rich-rules*|*--list-forward-ports*) return 0 ;;
    *--get-all-rules*) printf '%s\n' "$TEST_RULE" ;;
    *--get-all-passthroughs*|*--get-policies*) return 0 ;;
    *--remove-rule*|*--remove-passthrough*) return "$REMOVE_STATUS" ;;
    *) return 1 ;;
  esac
}
service_exposes_web() { return 1; }
'''
    return subprocess.run(
        [
            "bash",
            "-c",
            f"PY_BIN={python or sys.executable}\n{functions}\n{mock}\n{flow}",
            "sfci-test",
            rule,
            str(removal_status),
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def make_release(root: Path, name: str) -> Path:
    path = root / "releases" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_release_id_is_only_a_utc_timestamp() -> None:
    assert re.fullmatch(r"\d{14}\n", bash_script("release_id").stdout)


def test_deploy_lock_rejects_a_concurrent_installer(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    root.mkdir()
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            f"source {HELPER!s}; acquire_deploy_lock {root!s}; echo locked; sleep 2",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        contender = bash_script(f"acquire_deploy_lock {root!s}", check=False)
        assert contender.returncode != 0
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_pointer_validation_rejects_files_and_escaping_targets(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    release = make_release(root, "20260823090000")
    (root / "current").write_text("not a pointer")
    assert bash_script(f"validate_release_pointers {root!s}", check=False).returncode != 0
    (root / "current").unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    escaping = root / "releases" / "20260823080000"
    escaping.symlink_to(outside, target_is_directory=True)
    (root / "current").symlink_to("releases/20260823080000")
    assert bash_script(f"validate_release_pointers {root!s}", check=False).returncode != 0
    (root / "current").unlink()
    aliased = root / "releases" / "20260823070000"
    aliased.symlink_to(release, target_is_directory=True)
    (root / "current").symlink_to("releases/20260823070000")
    assert bash_script(f"validate_release_pointers {root!s}", check=False).returncode != 0
    (root / "current").unlink()
    (root / "current").symlink_to(f"releases/{release.name}")
    bash_script(f"validate_release_pointers {root!s}")
    (root / "current").unlink()
    (root / "previous").symlink_to(f"releases/{release.name}")
    assert bash_script(f"validate_release_pointers {root!s}", check=False).returncode != 0


def test_payload_validation_rejects_symlinks_and_hardlinks(tmp_path: Path) -> None:
    code = tmp_path / "code"
    code.mkdir()
    source = code / "server.py"
    source.write_text("pass\n")
    bash_script(f"reject_code_payload_links {code!s}")

    link = code / "escape.py"
    link.symlink_to(source)
    assert bash_script(f"reject_code_payload_links {code!s}", check=False).returncode != 0
    link.unlink()

    hardlink = code / "same-inode.py"
    os.link(source, hardlink)
    assert bash_script(f"reject_code_payload_links {code!s}", check=False).returncode != 0
    hardlink.unlink()

    data = tmp_path / "data"
    data.mkdir()
    (data / "walk_graph.npz").symlink_to(source)
    assert bash_script(f"reject_data_payload_links {data!s}", check=False).returncode != 0


def test_cutover_failure_restores_explicit_old_target_idempotently(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    old = make_release(root, "20260823090000")
    new = make_release(root, "20260823100000")
    (root / "current").symlink_to(f"releases/{old.name}")
    bash_script(
        f"""
        switch_current {root!s} releases/{new.name}
        restore_current_to {root!s} releases/{old.name} releases/{new.name}
        restore_current_to {root!s} releases/{old.name} releases/{new.name}
        test "$(readlink {root!s}/current)" = releases/{old.name}
        test "$(readlink {root!s}/previous)" = releases/{new.name}
        """
    )


def test_first_install_failure_removes_failed_current(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    new = make_release(root, "20260823100000")
    bash_script(
        f"""
        switch_current {root!s} releases/{new.name}
        restore_current_to {root!s} '' releases/{new.name}
        test ! -e {root!s}/current
        test ! -L {root!s}/current
        """
    )


def test_exact_code_pointer_restore_preserves_original_previous(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    current = make_release(root, "20260823090000")
    previous = make_release(root, "20260823080000")
    failed = make_release(root, "20260823100000")
    (root / "current").symlink_to(f"releases/{current.name}")
    (root / "previous").symlink_to(f"releases/{previous.name}")
    bash_script(
        f"""
        switch_current {root!s} releases/{failed.name}
        restore_release_pointers_exact {root!s} releases/{current.name} releases/{previous.name}
        restore_release_pointers_exact {root!s} releases/{current.name} releases/{previous.name}
        test "$(readlink {root!s}/current)" = releases/{current.name}
        test "$(readlink {root!s}/previous)" = releases/{previous.name}
        """
    )


def test_prune_keeps_active_previous_and_newest_extra(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    names = [
        "20260823090000",
        "20260823100000",
        "20260823110000",
        "20260823120000",
        "20260823130000",
    ]
    for name in names:
        make_release(root, name)
    (root / "current").symlink_to("releases/20260823090000")
    (root / "previous").symlink_to("releases/20260823100000")
    bash_script(f"prune_releases {root!s}")
    assert sorted(path.name for path in (root / "releases").iterdir()) == [
        "20260823090000",
        "20260823100000",
        "20260823130000",
    ]


def test_abandoned_stage_cleanup_is_age_and_name_bounded(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    incoming = root / ".incoming"
    old = incoming / "20260801000000"
    fresh = incoming / "20260823120000"
    diagnostic = incoming / "keep-me"
    for path in (old, fresh, diagnostic):
        path.mkdir(parents=True)
    old_time = time.time() - 10 * 24 * 60 * 60
    os.utime(old, (old_time, old_time))
    bash_script(f"cleanup_abandoned_stages {root!s}")
    assert not old.exists()
    assert fresh.exists()
    assert diagnostic.exists()


def test_abandoned_data_cleanup_preserves_requested_and_referenced_ids(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    incoming = root / "data-incoming"
    releases = root / "data-releases"
    releases.mkdir(parents=True)
    names = ("20260818", "20260819", "20260820", "20260821", "20260822")
    for name in names:
        (incoming / name).mkdir(parents=True)
        (releases / name).mkdir()
        old_time = time.time() - 10 * 24 * 60 * 60
        os.utime(incoming / name, (old_time, old_time))
    (root / "data-current").symlink_to("data-releases/20260820")
    (root / "data-previous").symlink_to("data-releases/20260819")
    code = make_release(root, "20260823090000")
    (code / "data").symlink_to("../../data-releases/20260818")

    bash_script(f"cleanup_abandoned_data_stages {root!s} 20260821")

    assert sorted(path.name for path in incoming.iterdir()) == [
        "20260818",
        "20260819",
        "20260820",
        "20260821",
    ]


def test_data_prune_keeps_current_and_every_code_pinned_release(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    (root / "data-releases").mkdir(parents=True)
    for name in ("20260820", "20260821", "20260822", "20260823"):
        (root / "data-releases" / name).mkdir()
    (root / "data-current").symlink_to("data-releases/20260823")
    (root / "data-previous").symlink_to("data-releases/20260821")
    first = make_release(root, "20260823090000")
    second = make_release(root, "20260823100000")
    (first / "data").symlink_to("../../data-releases/20260820")
    (second / "data").symlink_to("../../data-releases/20260822")

    bash_script(f"prune_data_releases {root!s}")

    assert sorted(path.name for path in (root / "data-releases").iterdir()) == [
        "20260820",
        "20260821",
        "20260822",
        "20260823",
    ]


def test_data_pointer_cutover_and_rollback_are_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "sfci"
    (root / "data-releases").mkdir(parents=True)
    for name in ("20260820", "20260823"):
        (root / "data-releases" / name).mkdir()
    (root / "data-current").symlink_to("data-releases/20260820")
    bash_script(
        f"""
        switch_data_current {root!s} data-releases/20260823
        test "$(readlink {root!s}/data-current)" = data-releases/20260823
        test "$(readlink {root!s}/data-previous)" = data-releases/20260820
        restore_data_current_to {root!s} data-releases/20260820 data-releases/20260823
        restore_data_current_to {root!s} data-releases/20260820 data-releases/20260823
        test "$(readlink {root!s}/data-current)" = data-releases/20260820
        test "$(readlink {root!s}/data-previous)" = data-releases/20260823
        """
    )


def test_data_promotion_validates_before_move_and_has_preinstalled_rollback() -> None:
    refresh = (ROOT / "deploy" / "data-refresh.sh").read_text()
    branch = refresh.index('if [[ "$MODE" == "--promote-id" ]]')
    trap_install = refresh.index("trap rollback_data_promotion EXIT", branch)
    full_readiness = refresh.index('validate_full_readiness "$STAGE"', trap_install)
    transaction_pending = refresh.index("write_data_transaction pending", full_readiness)
    move = refresh.index('mv "$STAGE" "$TARGET"', full_readiness)
    sync_incoming = refresh.index('durable_sync_path "$DATA_INCOMING"', move)
    sync_releases = refresh.index('durable_sync_path "$DATA_RELEASES"', sync_incoming)
    switch = refresh.index('switch_data_current "$ROOT"', move)
    transaction_committed = refresh.index("write_data_transaction committed", switch)
    assert trap_install < full_readiness < transaction_pending < move
    assert move < sync_incoming < sync_releases < switch
    assert switch < transaction_committed
    assert refresh.index("recover_data_transaction\n") < branch
    assert 'restore_data_pointers_exact "$ROOT" "$old_data_target" "$old_data_previous"' in refresh
    assert '"$(data_current_target "$ROOT")" != "data-releases/$DATA_ID"' in refresh
    assert '[[ "$phase" == "pending" ]]' in refresh
    assert '[[ "$phase" == "committed" ]]' in refresh
    assert "validate_data_transaction_schema" in refresh
    assert '"$(data_previous_target "$ROOT")" == "$old"' in refresh
    assert 'now=stage_now' in refresh
    assert 'dt.time(0, 0)' in refresh
    assert 'stage_date.weekday() != 2' in refresh
    assert 'stat -c \'%d\' "$STAGE"' in refresh
    assert "must share one filesystem for atomic promotion" in refresh


def test_deploy_scripts_are_executable() -> None:
    for name in (
        "install.sh",
        "push.sh",
        "release_ops.sh",
        "data-refresh.sh",
        "cloudflare-ufw.sh",
    ):
        assert os.access(ROOT / "deploy" / name, os.X_OK), name


def test_scripts_encode_safe_staging_data_and_secret_contracts() -> None:
    push = (ROOT / "deploy" / "push.sh").read_text()
    install = (ROOT / "deploy" / "install.sh").read_text()
    docs = (ROOT / "deploy" / "DEPLOY.md").read_text()
    data_refresh = (ROOT / "deploy" / "data-refresh.sh").read_text()

    assert "--exclude '/data/***'" in push
    assert "--exclude '.claude/'" in push
    for development_path in (
        ".github/",
        ".ruff_cache/",
        "/prototypes/",
        "/tests/",
        "/requirements-dev.txt",
    ):
        assert f"--exclude '{development_path}'" in push
    assert "REMOTE_CONTROL=" in push
    assert "$REMOTE_CONTROL/install.sh --release-id" in push
    assert "'$REMOTE_STAGE/deploy/install.sh --release-id" not in push
    assert "stream_control_file deploy/install.sh install.sh 500" in push
    assert "stream_control_file deploy/release_ops.sh release_ops.sh 400" in push
    assert "stream_control_file deploy/data-refresh.sh data-refresh.sh 500" in push
    assert "--rsync-path='sudo rsync'" in push
    assert "--no-owner --no-group" in push
    assert "sudo install -d -m 700 -o root -g root '$REMOTE_STAGE'" in push
    assert "sudo chmod 700 '$REMOTE_STAGE'" in push
    assert "'$REMOTE_STAGE/deploy/install.sh' '$REMOTE_CONTROL/install.sh'" not in push
    for pattern in ("*.pem", "*.key", "*.csr", "*.p12", "*.pfx", ".env.*"):
        assert f"--exclude '{pattern}'" in push

    assert 'source "$CONTROL_DIR/release_ops.sh"' in install
    assert 'source "$SOURCE_DIR/' not in install
    assert 'chown -R -h root:root "$SOURCE_DIR"' in install
    assert 'chmod 700 "$SOURCE_DIR"' in install
    freeze = install.index('chmod -R a-w "$SOURCE_DIR"')
    requirements_read = install.index('"$RELEASE_DIR/requirements.txt"')
    assert freeze < requirements_read
    assert 'reject_code_payload_links "$SOURCE_DIR"' in install
    assert 'mkdir -p "$SHARED_DATA"' not in install
    assert 'chown -R "$USER_NAME:$USER_NAME" "$RELEASE_DIR"' not in install
    assert "merging staged data" not in install
    assert "data-releases" in docs
    assert "Legacy in-place trees are never copied into a release" in install
    assert 'SFCI_PUBLIC_SMOKE_URL=https://sfcommutemap.com' in push
    assert "SFCI_PUBLIC_SMOKE_URL" in docs
    assert 'switch_data_current "$ROOT" "data-releases/$DATA_ID"' in data_refresh
    assert 'restore_data_pointers_exact "$ROOT" "$old_data_target" "$old_data_previous"' in data_refresh
    assert '[[ -L "$ROOT/current" ]]' in data_refresh
    assert '--bootstrap-code-id' in data_refresh
    assert '"${CONTROL_DIR##*/}" == "$BOOTSTRAP_CODE_ID"' in data_refresh
    assert "bootstrap data promotion is allowed only before the first immutable code cutover" in data_refresh
    assert "bootstrap data promotion requires an adopted legacy data pointer" in data_refresh
    assert "matching frozen code" in docs
    assert 'systemctl stop sfci' in data_refresh
    assert 'freeze_and_validate "$STAGE"' in data_refresh
    assert 'find "$tree" -xdev -type d -exec chmod 555 {} +' in data_refresh
    assert 'find "$tree" -xdev -type f -exec chmod 444 {} +' in data_refresh
    assert 'prune_data_releases "$ROOT"' in data_refresh
    for required_feed in ("muni_current.zip", "bart_gtfs.zip", "caltrain.zip"):
        assert required_feed in data_refresh
    assert "readiness.check_readiness" in data_refresh
    assert "access_walk_{grid_m}m_{service_date}.npz" in data_refresh
    assert 'bundle_sources != feed_sources' in data_refresh
    assert 'cache_data.get("source_mtimes", ())' in data_refresh

    # Root/control/lock checks happen before any payload traversal.
    root_secure = install.index('chown root:root "$REPO_DIR"')
    control = install.index('trusted_path_ok "$CONTROL_DIR/install.sh"')
    lock = install.index('exec 9>"$REPO_DIR/.deploy.lock"')
    payload = install.index('source_canonical="$(cd "$SOURCE_DIR" && pwd -P)"')
    assert root_secure < control < lock < payload
    assert 'set -o noclobber; : > "$REPO_DIR/.deploy.lock"' in install
    assert 'set -o noclobber; : > "$ROOT/.deploy.lock"' in data_refresh
    recovery_call = install.index("recover_pending_cutover\n")
    freeze_call = install.index("freeze_source_payload\n", recovery_call)
    active_check = install.index('[[ "$OLD_TARGET" != "$NEW_TARGET" ]]', freeze_call)
    assert recovery_call < freeze_call < active_check
    assert "flock (util-linux) is required" in install
    assert "incoming release root must be a real directory" in install
    assert "resumable release root must be a real directory" in install

    for text in (
        push,
        install,
        data_refresh,
        (ROOT / "deploy" / "release_ops.sh").read_text(),
        (ROOT / "deploy" / "cloudflare-ufw.sh").read_text(),
    ):
        lowered = text.lower()
        assert "sha256" not in lowered
        assert "fingerprint" not in lowered
        assert "manifest" not in lowered


def test_candidate_uses_the_exact_bootstrap_production_unit_before_live_mutation() -> None:
    install = (ROOT / "deploy" / "install.sh").read_text()
    service = (ROOT / "deploy" / "sfci.service").read_text()

    candidate_passed = install.index("# Candidate has passed.")
    candidate_call = install.index("candidate_smoke\nstop_candidate")
    unit_mutation = install.index('install_runtime_units "$RELEASE_DIR"', candidate_passed)
    caddy_mutation = install.index('install -m 644 "$CADDY_CANDIDATE" "$CADDYFILE"', candidate_passed)
    assert candidate_call < candidate_passed < unit_mutation < caddy_mutation
    assert "systemd-run" not in install
    assert 'local template="$RELEASE_DIR/deploy/sfci.service"' in install
    assert 'systemd-analyze verify "$CANDIDATE_UNIT_FILE"' in install
    assert 'systemctl start "$CANDIDATE_UNIT"' in install
    assert 'render_candidate_unit "$template" "$RELEASE_ID" 1 "$temp"' in install
    assert 'chown -R -h "root:$USER_NAME" "$RELEASE_DIR/.venv"' in install
    assert install.index(
        'chown -R -h "root:$USER_NAME" "$RELEASE_DIR/.venv"'
    ) < install.index('chmod -R a-w "$RELEASE_DIR/.venv"')
    assert 'Environment=PORT=$CANDIDATE_PORT' in install
    assert 'systemctl stop "$CANDIDATE_UNIT"' in install
    assert 'USE_RAPTOR' not in install
    assert 'USE_WALK_GRAPH' not in install
    assert "candidate modified its immutable release source" in install
    assert "(.cells | type == \"object\" and length > 0)" in install
    assert "(.legs | type == \"array\" and length > 0)" in install
    assert "(.variance | type == \"object\")" in install
    assert "--noproxy '*'" in install
    assert 'Environment=PORT=8000' in service
    assert 'ExecStart=/opt/sfci/current/.venv/bin/python scripts/bootstrap_server.py' in service
    assert (
        "grep -Fqx 'ExecStart=/opt/sfci/current/.venv/bin/python "
        "scripts/bootstrap_server.py' \"$template\""
    ) in install
    assert "ExecStartPre=" not in service
    assert "ReadOnlyPaths=/opt/sfci/data-releases" in service
    assert "PORT=8000" not in install
    assert 'production unit template did not bind liveness promptly' in install
    assert "cleanup_stale_candidates\n" in install
    assert install.index("cleanup_stale_candidates\n") < install.index("candidate_smoke\nstop_candidate")
    assert "candidate_unit_is_trusted" in install
    assert 'cmp -s "$path" "$isolated" || cmp -s "$path" "$legacy"' in install
    assert 'valid_release_target "$REPO_DIR" "releases/$id"' in install
    assert "untrusted stale candidate unit requires manual inspection" in install
    assert "systemctl list-units --all --type=service" in install
    assert 'ss -H -ltn "sport = :$CANDIDATE_PORT"' in install
    assert "CacheDirectory=sfci-candidate-$RELEASE_ID" in install
    assert "NUMBA_CACHE_DIR=/var/cache/sfci-candidate-$RELEASE_ID/numba" in install
    assert 'rm -rf "/var/cache/sfci-candidate-$RELEASE_ID"' in install
    assert ".incoming and releases must share one filesystem" in install


def test_transaction_covers_legacy_firewall_proxy_and_first_install_rollback() -> None:
    install = (ROOT / "deploy" / "install.sh").read_text()
    docs = (ROOT / "deploy" / "DEPLOY.md").read_text()

    pending = install.index("CUTOVER_PENDING=1")
    switch = install.index('switch_current "$REPO_DIR" "$NEW_TARGET"', pending)
    pre_cutover_close = install.index(
        'fail_closed_firewall || { echo "could not prove origin ingress closed before cutover"',
        pending,
    )
    proxy_smoke = install.index("https://sfcommutemap.com/readyz", switch)
    committed = install.index("CUTOVER_PENDING=0", proxy_smoke)
    assert pending < pre_cutover_close < switch < proxy_smoke < committed
    assert 'rollback_cutover || status=1' in install
    assert "write_deploy_transaction pending" in install
    assert "write_deploy_transaction committed" in install
    assert "old_previous=%s" in install
    assert 'restore_release_pointers_exact "$REPO_DIR" "$old" "$old_previous"' in install
    assert "legacy=%s" in install
    assert '"$(read_transaction_value phase)" == "committed"' in install
    assert 'snapshot_runtime_state' in install
    assert 'restore_runtime_state' in install
    assert 'enabled-runtime) systemctl enable --runtime "$unit"' in install
    runtime_restore = install[install.index("restore_runtime_state()") : install.index("restore_caddy()")]
    caddy_restore = install[install.index("restore_caddy()") : install.index("origin_file_meta()")]
    assert "masked)" in runtime_restore and 'systemctl mask "$unit"' in runtime_restore
    assert "masked-runtime)" in runtime_restore and 'systemctl mask --runtime "$unit"' in runtime_restore
    assert "masked)" in caddy_restore and "systemctl mask caddy" in caddy_restore
    assert "masked-runtime)" in caddy_restore and "systemctl mask --runtime caddy" in caddy_restore
    assert '[[ "$actual_raw" == "$enabled_raw" ]]' in install
    assert 'LEGACY_PRESENT=1' in install
    assert 'logrotate-present' in install
    assert 'rm -f "$root/current"' in (ROOT / "deploy" / "release_ops.sh").read_text()
    assert "one-time migration" in docs.lower()
    assert "Do not copy the legacy working tree" in docs

    lockdown = install.index("ensure_cloudflare_lockdown", switch)
    public_smoke = install.index('"$PUBLIC_SMOKE_URL/readyz"', lockdown)
    committed_phase = install.index("write_deploy_transaction committed", public_smoke)
    assert switch < lockdown < public_smoke < committed_phase
    assert "systemctl start cloudflare-ufw.service" in install
    assert "verify_cloudflare_lockdown" in install
    assert "verify_no_non_cloudflare_web_ingress" in install
    assert "--add-service=http" not in install
    assert "--add-service=https" not in install
    for exposure in (
        "--list-services",
        "--list-ports",
        "--list-protocols",
        "--list-source-ports",
        "--list-rich-rules",
        "--list-forward-ports",
        "--get-all-rules",
        "--get-all-passthroughs",
        "--get-target",
        "--get-policies",
    ):
        assert exposure in install
    assert "stop_origin" in install
    assert "close_zone_web_ingress permanent cloudflare 0" in install
    assert "verify_no_non_cloudflare_web_ingress permanent 0" in install
    close_zone = install[install.index("close_zone_web_ingress()") : install.index("close_direct_web_ingress()")]
    assert 'if [[ "$mode" == "permanent" ]]' in close_zone
    assert 'firewall_call "$mode" --zone="$zone" --set-target=DROP' in close_zone
    assert 'firewall_zone_target "$mode" "$zone"' in close_zone
    assert install.count("--set-target=DROP") == 1
    target_helper = install[install.index("firewall_zone_target()") : install.index("service_is_required_non_web()")]
    assert 'firewall_call permanent --zone="$zone" --get-target' in target_helper
    assert 'firewall_call runtime --zone="$zone" --list-all' in target_helper
    assert 'restore_caddy "$CADDY_HAD_FILE" "$CADDY_WAS_ACTIVE" "$CADDY_WAS_ENABLED"' in install
    assert "caddy_enabled=%s" in install
    assert 'PUBLIC_SMOKE_URL must be exactly https://$PUBLIC_HOST' in install
    assert "--max-redirs 0" in install
    assert "--proto '=https'" in install
    assert "--write-out '%{http_code}'" in install
    assert '== "200"' in install
    assert 'jq -e \'.ok == true\'' in install
    assert '$SFCI_ENV must be a root:root mode-600 single-link regular file' in install
    assert 'chmod 640 "$path"' in install


def test_direct_firewall_classifier_ignores_only_provably_non_public_web_rules() -> None:
    safe_rules = (
        ("passthroughs", "ipv4 -N BareMetalInstanceServices"),
        ("passthroughs", "ipv4 -A OUTPUT -d 169.254.0.0/16 -j BareMetalInstanceServices"),
        ("passthroughs", "ipv4 -A BareMetalInstanceServices -d 169.254.0.2/32 -p tcp --dport 80 -j ACCEPT"),
        ("passthroughs", "ipv6 -A BareMetalInstanceServices -d fd00:c1::a9fe:0002/128 -p tcp --dport 80 -j ACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -p tcp --dport 22 -j ACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -p tcp --dport 80 -j DROP"),
        ("passthroughs", "ipv4 -A POSTROUTING -p tcp --dport 80 -j MARK"),
    )
    public_rules = (
        ("rules", "ipv4 filter INPUT 0 -p tcp --dport 80 -j ACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -ptcp --dport=80 -jACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -p all -j ACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -p 6 --dport 443 -j ACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -p udp --dport 443 -j ACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -p udp --dport 80 -j ACCEPT"),
        ("rules", "ipv4 filter INPUT 0 -p tcp ! --dport 22 -j ACCEPT"),
        ("passthroughs", "ipv4 -A INPUT -p tcp --dport 443 -j ACCEPT"),
        ("passthroughs", "ipv4 -A INPUT -p tcp -j ACCEPT"),
        ("passthroughs", "ipv4 -A INPUT -d 10.0.0.8/32 -p tcp --dport 80 -j ACCEPT"),
        ("passthroughs", "ipv4 -A INPUT ! -d 169.254.0.2/32 -p tcp --dport 80 -j ACCEPT"),
        ("passthroughs", "ipv4 ['-A', 'PREROUTING', '-p', 'tcp', '--dport', '80', '-j', 'DNAT']"),
        ("passthroughs", "ipv4 -P INPUT ACCEPT"),
    )

    for kind, rule in safe_rules:
        result = direct_rule_helper(f'direct_rule_classification {kind} "$1"', rule)
        assert result.returncode == 10, (rule, result.returncode, result.stderr)
    for kind, rule in public_rules:
        result = direct_rule_helper(f'direct_rule_classification {kind} "$1"', rule)
        assert result.returncode == 0, (rule, result.returncode, result.stderr)

    malformed = direct_rule_helper(
        'direct_rule_classification rules "$1"', "ipv4 filter INPUT not-a-priority -j ACCEPT"
    )
    assert malformed.returncode == 20
    missing_python = direct_rule_helper(
        'direct_rule_classification rules "$1"',
        "ipv4 filter INPUT 0 -p tcp --dport 80 -j ACCEPT",
        python="/definitely/missing/python",
    )
    assert missing_python.returncode not in {0, 10}


def test_firewall_port_classifier_covers_tcp_and_http3_udp() -> None:
    for token in ("80/tcp", "443/tcp", "443/udp", "70-90/udp"):
        assert firewall_primitive_helper('port_token_exposes_web "$1"', token).returncode == 0
    for token in ("22/tcp", "53/udp", "443/sctp"):
        assert firewall_primitive_helper('port_token_exposes_web "$1"', token).returncode != 0


def test_direct_firewall_parser_preserves_quoted_and_list_arguments() -> None:
    whitespace = (
        "ipv4 -A INPUT -p tcp --dport 80 -m comment "
        "--comment 'space preserving comment' -j ACCEPT"
    )
    parsed = direct_rule_helper(
        'direct_rule_to_array "$1" && printf "<%s>\\n" "${DIRECT_FIELDS[@]}"',
        whitespace,
    )
    assert parsed.returncode == 0, parsed.stderr
    assert "<space preserving comment>" in parsed.stdout

    listed = "ipv4 ['-A', 'INPUT', '-p', 'tcp', '--dport', '443', '-j', 'ACCEPT']"
    parsed_list = direct_rule_helper(
        'direct_rule_to_array "$1" && printf "<%s>\\n" "${DIRECT_FIELDS[@]}"',
        listed,
    )
    assert parsed_list.returncode == 0, parsed_list.stderr
    assert parsed_list.stdout.splitlines() == [
        "<ipv4>",
        "<-A>",
        "<INPUT>",
        "<-p>",
        "<tcp>",
        "<--dport>",
        "<443>",
        "<-j>",
        "<ACCEPT>",
    ]


def test_direct_firewall_flows_fail_on_classifier_or_removal_errors() -> None:
    public_rule = "ipv4 filter INPUT 0 -p tcp --dport 80 -j ACCEPT"
    safe_rule = "ipv4 filter INPUT 0 -p tcp --dport 22 -j ACCEPT"

    assert direct_firewall_flow_helper(
        "close_direct_web_ingress runtime", safe_rule
    ).returncode == 0
    assert direct_firewall_flow_helper(
        "verify_no_non_cloudflare_web_ingress runtime", safe_rule
    ).returncode == 0
    assert direct_firewall_flow_helper(
        "close_direct_web_ingress runtime", public_rule, removal_status=9
    ).returncode != 0
    assert direct_firewall_flow_helper(
        "verify_no_non_cloudflare_web_ingress runtime", public_rule
    ).returncode != 0
    assert direct_firewall_flow_helper(
        "close_direct_web_ingress runtime",
        public_rule,
        python="/definitely/missing/python",
    ).returncode != 0
    assert direct_firewall_flow_helper(
        "verify_no_non_cloudflare_web_ingress runtime",
        public_rule,
        python="/definitely/missing/python",
    ).returncode != 0


def test_transaction_metadata_env_and_rollback_paths_are_hardened() -> None:
    install = (ROOT / "deploy" / "install.sh").read_text()
    helper = (ROOT / "deploy" / "release_ops.sh").read_text()

    assert '"$(stat -c \'%u:%g:%a:%h\' "$TRANSACTION_FILE")" == "0:0:600:1"' in install
    assert '"$deploy_target" == "current/deploy"' in install
    assert 'validate_rollback_artifacts "$deploy_kind" "$caddy_had"' in install
    assert "validate_transaction_schema" in install
    assert '"$(stat -c \'%u:%g:%a\' "$UNIT_ROLLBACK_DIR")" == "0:0:700"' in install
    assert 'CADDY_ROLLBACK="$UNIT_ROLLBACK_DIR/Caddyfile"' in install
    assert 'install -m 600 -o root -g root "$CADDYFILE" "$CADDY_ROLLBACK"' in install
    assert "origin_cert_meta=%s" in install and "origin_key_meta=%s" in install
    assert 'restore_origin_permissions "$ORIGIN_CERT_META" "$ORIGIN_KEY_META"' in install
    assert 'remove_created_env "$ENV_CREATED" "$NEW_TARGET"' in install
    assert "ENV_CREATED=1" in install
    assert 'install -m 600 -o root -g root /dev/null "$ENV_MARKER"' in install
    assert "cleanup_orphan_env_marker" in install
    assert '"$(current_target "$REPO_DIR")" == "$new"' in install
    assert '"$(previous_target "$REPO_DIR")" == "$old"' in install
    assert 'cleanup_committed_rollback_artifacts "$deploy_kind"' in install
    committed = install.index("write_deploy_transaction committed")
    cleanup = install.index('cleanup_committed_rollback_artifacts "$DEPLOY_PATH_KIND"', committed)
    assert committed < cleanup < install.index("remove_deploy_transaction", cleanup)
    assert "cleanup_stale_committed_rollback\n" in install
    assert 'rm -rf --one-file-system "$DEPLOY_ROLLBACK_DIR"' in install
    assert install.count('rm -f "$TRANSACTION_FILE"') == 1
    recovery = install[install.index("recover_pending_cutover()") : install.index("cleanup_stale_committed_rollback()")]
    assert recovery.index('cleanup_committed_rollback_artifacts "$deploy_kind"') < recovery.index(
        "remove_deploy_transaction"
    )
    rollback = install[install.index("rollback_cutover()") : install.index("finish_install()")]
    assert rollback.index('cleanup_committed_rollback_artifacts "$DEPLOY_PATH_KIND"') < rollback.index(
        "remove_deploy_transaction"
    )
    assert '[[ -f "$SFCI_ENV" && ! -L "$SFCI_ENV"' in install
    assert "restore_release_pointers_exact()" in helper
    assert 'cleanup_abandoned_data_stages "$REPO_DIR"' in install


def test_transactions_and_pointer_swaps_are_durably_synced() -> None:
    install = (ROOT / "deploy" / "install.sh").read_text()
    refresh = (ROOT / "deploy" / "data-refresh.sh").read_text()
    helper = (ROOT / "deploy" / "release_ops.sh").read_text()

    install_replace = install.index('mv -f "$temp" "$TRANSACTION_FILE"')
    assert install_replace < install.index('durable_sync_path "$TRANSACTION_FILE"', install_replace)
    refresh_replace = refresh.index('mv -f "$temp" "$DATA_TRANSACTION"')
    refresh_file_sync = refresh.index('durable_sync_path "$DATA_TRANSACTION"', refresh_replace)
    refresh_dir_sync = refresh.index('durable_sync_path "$ROOT"', refresh_file_sync)
    assert refresh_replace < refresh_file_sync < refresh_dir_sync
    pointer_replace = helper.index('mv -fT "$tmp" "$link"')
    assert pointer_replace < helper.index('durable_sync_path "$(dirname "$link")"', pointer_replace)
    assert 'sync -f "$path"' in helper


def test_cloudflare_source_parser_rejects_broad_and_malformed_cidrs(tmp_path: Path) -> None:
    firewall = (ROOT / "deploy" / "cloudflare-ufw.sh").read_text()
    parser = firewall.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    v4 = tmp_path / "v4"
    v6 = tmp_path / "v6"
    v4.write_text("\n".join(f"192.0.{index}.0/24" for index in range(5)) + "\n")
    v6.write_text("\n".join(f"2001:db8:{index}::/48" for index in range(5)) + "\n")
    valid = subprocess.run(
        [sys.executable, "-c", parser, str(v4), str(v6)],
        text=True,
        capture_output=True,
    )
    assert valid.returncode == 0, valid.stderr

    v4.write_text("0.0.0.0/0\n" + v4.read_text())
    broad = subprocess.run(
        [sys.executable, "-c", parser, str(v4), str(v6)],
        text=True,
        capture_output=True,
    )
    assert broad.returncode != 0

    v4.write_text("\n".join(f"192.0.{index}.0/24" for index in range(5)) + "\n")
    v6.write_text("::/0\n" + v6.read_text())
    broad_v6 = subprocess.run(
        [sys.executable, "-c", parser, str(v4), str(v6)],
        text=True,
        capture_output=True,
    )
    assert broad_v6.returncode != 0

    v4.write_text("999.1.1.0/24\n")
    v6.write_text("\n".join(f"2001:db8:{index}::/48" for index in range(5)) + "\n")
    malformed = subprocess.run(
        [sys.executable, "-c", parser, str(v4), str(v6)],
        text=True,
        capture_output=True,
    )
    assert malformed.returncode != 0
    assert "ipaddress.ip_network" in firewall
    install = (ROOT / "deploy" / "install.sh").read_text()
    assert "minimum = 12 if n.version == 4 else 32" in install
