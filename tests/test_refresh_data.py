"""Offline contracts for the inactive data-generation workflow."""

from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import refresh_data


def _feed_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for table in refresh_data.REQUIRED_GTFS + ("calendar.txt",):
            z.writestr(table, "x\n")
    return buf.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_gtfs_validation_requires_core_tables_and_calendar(tmp_path):
    good = tmp_path / "good.zip"
    good.write_bytes(_feed_bytes())
    assert refresh_data.valid_gtfs_zip(good)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("routes.txt", "x\n")
    assert not refresh_data.valid_gtfs_zip(bad)


def test_fetch_is_atomic_and_does_not_replace_on_invalid_download(tmp_path):
    stage = tmp_path / "stage"
    old = stage / "muni_current.zip"
    stage.mkdir()
    old.write_bytes(b"prior-generation")

    def invalid(*_args, **_kwargs):
        return _Response(b"not a zip")

    with pytest.raises(ValueError, match="complete GTFS"):
        refresh_data._download("https://example.invalid", old, opener=invalid)
    assert old.read_bytes() == b"prior-generation"
    assert not list(stage.glob("*.part"))


def test_fetch_downloads_all_three_to_stage_and_never_uses_active_data(tmp_path):
    stage = tmp_path / "inactive"
    active = tmp_path / "active"
    active.mkdir()
    (active / "muni_current.zip").write_bytes(b"active")
    calls = []

    def opener(request, **_kwargs):
        calls.append(request)
        return _Response(_feed_bytes())

    result = refresh_data.fetch_feeds(stage, token="token", opener=opener)
    assert set(result) == {"muni", "bart", "caltrain"}
    assert {p.name for p in result.values()} == {
        "muni_current.zip", "bart_gtfs.zip", "caltrain.zip"
    }
    assert all(refresh_data.valid_gtfs_zip(p) for p in result.values())
    assert (active / "muni_current.zip").read_bytes() == b"active"
    assert any("api_key=token" in request.full_url for request in calls)
    assert all(request.get_header("User-agent") == refresh_data.DOWNLOAD_USER_AGENT
               for request in calls)


def test_prepare_stage_carries_stable_inputs_without_mutating_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("sf_neighborhoods.geojson", "walk_graph.npz", "osm_sf.pbf"):
        (source / name).write_bytes(name.encode())
    stage = refresh_data.prepare_stage(tmp_path / "stage", source)
    assert {p.name for p in stage.iterdir()} == {
        "sf_neighborhoods.geojson", "walk_graph.npz", "osm_sf.pbf"
    }
    assert (source / "walk_graph.npz").read_bytes() == b"walk_graph.npz"
    with pytest.raises(ValueError, match="empty"):
        refresh_data.prepare_stage(stage, source)


def test_config_data_override_is_resolved_relative_to_repo(monkeypatch):
    import importlib
    import core.config as config

    monkeypatch.setenv("SFCI_DATA_DIR", "tmp/stage")
    loaded = importlib.reload(config)
    assert loaded.DATA == (loaded.ROOT / "tmp/stage").resolve()
    monkeypatch.delenv("SFCI_DATA_DIR")
    loaded = importlib.reload(config)
    assert loaded.DATA == (loaded.ROOT / "data").resolve()


def test_stage_and_source_symlinks_and_active_root_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "walk_graph.npz").write_bytes(b"graph")
    linked_source = tmp_path / "linked-source"
    linked_source.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="source data root"):
        refresh_data.prepare_stage(tmp_path / "stage", linked_source)
    linked_stage = tmp_path / "linked-stage"
    linked_stage.symlink_to(tmp_path / "real-stage", target_is_directory=True)
    with pytest.raises(ValueError, match="stage"):
        refresh_data.prepare_stage(linked_stage, source)
    with pytest.raises(ValueError, match="inactive"):
        refresh_data.prepare_stage(ROOT / "data", source)


def test_active_root_descendants_and_linked_payloads_are_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    stable = source / "walk_graph.npz"
    stable.write_bytes(b"graph")
    active = tmp_path / "active"
    active.mkdir()
    monkeypatch.setenv("SFCI_DATA_DIR", str(active))
    with pytest.raises(ValueError, match="inactive"):
        refresh_data.prepare_stage(active / "nested", source)

    monkeypatch.delenv("SFCI_DATA_DIR")
    stage = refresh_data.prepare_stage(tmp_path / "stage", source)
    linked = stage / "linked.bin"
    linked.symlink_to(stable)
    with pytest.raises(ValueError, match="symlink"):
        refresh_data.build_stage(stage, service_date="20260819")

    stage = refresh_data.prepare_stage(tmp_path / "stage-hardlink", source)
    hardlinked = stage / "linked.bin"
    os.link(stable, hardlinked)
    with pytest.raises(ValueError, match="hard-linked"):
        refresh_data.build_stage(stage, service_date="20260819")


def test_deployment_data_paths_are_never_accepted_but_external_scratch_is(tmp_path):
    protected = (
        refresh_data.ROOT / "data-releases" / "20260823",
        refresh_data.ROOT / "data-current",
        refresh_data.ROOT / "data-current" / "nested",
        refresh_data.ROOT / "data-incoming" / "stage",
        Path("/opt/sfci/data-releases/20260823"),
        Path("/opt/sfci/data-current"),
        Path("/opt/sfci/data-current") / "nested",
        Path("/opt/sfci/data-incoming") / "stage",
    )
    for path in protected:
        with pytest.raises(ValueError, match="protected deployment"):
            refresh_data._inactive_directory(path, "stage")

    # A similarly named directory outside the deployment layouts is a valid
    # caller-owned inactive stage (including when it is created on demand).
    scratch = tmp_path / "data-releases" / "scratch"
    assert refresh_data._inactive_directory(scratch, "stage", create=True) == scratch
    assert scratch.is_dir()


def test_build_subprocess_gets_scripts_on_pythonpath(monkeypatch, tmp_path):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["env"] = kwargs["env"]

    monkeypatch.setattr(refresh_data.subprocess, "run", fake_run)
    refresh_data._run_build_command(tmp_path / "stage", ["python", "-c", "pass"], "20260819")
    assert seen["env"]["SFCI_DATA_DIR"] == str((tmp_path / "stage").resolve())
    assert str(refresh_data.SCRIPTS) in seen["env"]["PYTHONPATH"].split(os.pathsep)


def test_download_errors_do_not_leak_token_or_leave_partial_file(tmp_path):
    destination = tmp_path / "muni_current.zip"

    def raises(*_args, **_kwargs):
        raise OSError("https://api.511.org/?api_key=SECRET-TOKEN")

    with pytest.raises(RuntimeError) as exc:
        refresh_data._download("https://example.invalid", destination, opener=raises)
    assert "SECRET-TOKEN" not in str(exc.value)
    assert not list(tmp_path.glob("*.part"))


def test_cli_loads_repo_dotenv_without_overriding_environment(monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(refresh_data, "_load_repo_env", lambda: loaded.append(True))
    monkeypatch.setattr(refresh_data, "fetch_feeds", lambda *_args, **_kwargs: loaded.append("fetch"))
    assert refresh_data.main(["fetch", "--stage", str(tmp_path / "stage")]) == 0
    assert loaded == [True, "fetch"]


def test_repo_dotenv_token_is_loaded_without_printing_or_overriding(monkeypatch, tmp_path, capsys):
    from core import config
    (tmp_path / ".env").write_text("API511_TOKEN=mock-token\n")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("API511_TOKEN", raising=False)
    refresh_data._load_repo_env()
    assert os.environ["API511_TOKEN"] == "mock-token"
    assert "mock-token" not in capsys.readouterr().out
    monkeypatch.setenv("API511_TOKEN", "explicit-token")
    refresh_data._load_repo_env()
    assert os.environ["API511_TOKEN"] == "explicit-token"


def test_download_tempfiles_are_unique_for_repeated_attempts(tmp_path, monkeypatch):
    destination = tmp_path / "feed.zip"
    names = []
    real_mkstemp = refresh_data.tempfile.mkstemp

    def capture(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        names.append(name)
        return fd, name

    monkeypatch.setattr(refresh_data.tempfile, "mkstemp", capture)
    payload = _feed_bytes()
    refresh_data._download("https://example.invalid", destination,
                           opener=lambda *_args, **_kwargs: _Response(payload))
    refresh_data._download("https://example.invalid", destination,
                           opener=lambda *_args, **_kwargs: _Response(payload))
    assert len(names) == 2 and names[0] != names[1]
    assert not list(tmp_path.glob("*.part"))
