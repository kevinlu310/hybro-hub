"""Tests for hybro-hub update, restart, and installer detection."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from hub.cli import (
    _detect_installer_command,
    _installer_display_name,
    _installed_version,
    _parse_version,
    _query_pypi_versions,
    _read_post_upgrade_version,
    _spawn_start,
    _stop_daemon,
    main,
)


# ──── _parse_version ────


class TestParseVersion:
    def test_simple(self):
        assert _parse_version("0.1.16") == (0, 1, 16)

    def test_two_part(self):
        assert _parse_version("1.0") == (1, 0)

    def test_comparison(self):
        assert _parse_version("0.2.0") > _parse_version("0.1.99")

    def test_pre_release_raises(self):
        with pytest.raises(ValueError):
            _parse_version("0.2.0a1")


# ──── _installed_version ────


class TestInstalledVersion:
    def test_known_package(self):
        ver = _installed_version("click")
        assert ver is not None

    def test_missing_package(self):
        assert _installed_version("nonexistent-pkg-xyz-999") is None


# ──── _read_post_upgrade_version ────


class TestReadPostUpgradeVersion:
    def test_spawns_subprocess(self):
        with patch("hub.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="1.2.3\n")
            result = _read_post_upgrade_version("hybro-hub")
            assert result == "1.2.3"
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == sys.executable
            assert "-c" in args

    def test_returns_none_on_failure(self):
        with patch("hub.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert _read_post_upgrade_version("hybro-hub") is None


# ──── _detect_installer_command ────


class TestDetectInstallerCommand:
    def test_pipx_path(self, tmp_path):
        fake_prefix = tmp_path / "pipx" / "venvs" / "hybro-hub"
        fake_prefix.mkdir(parents=True)
        with (
            patch("sys.prefix", str(fake_prefix)),
            patch("shutil.which", return_value="/usr/local/bin/pipx"),
            patch("importlib.metadata.distribution") as mock_dist,
        ):
            mock_dist.return_value.read_text.return_value = None
            cmd = _detect_installer_command()
        assert cmd == ["/usr/local/bin/pipx", "upgrade", "hybro-hub"]

    def test_uv_tool_path(self, tmp_path):
        fake_prefix = tmp_path / "uv" / "tools" / "hybro-hub"
        fake_prefix.mkdir(parents=True)
        with (
            patch("sys.prefix", str(fake_prefix)),
            patch("shutil.which", return_value="/usr/local/bin/uv"),
            patch("importlib.metadata.distribution") as mock_dist,
        ):
            mock_dist.return_value.read_text.return_value = None
            cmd = _detect_installer_command()
        assert cmd == ["/usr/local/bin/uv", "tool", "upgrade", "hybro-hub"]

    def test_pip_fallback(self, tmp_path):
        fake_prefix = tmp_path / "some_venv"
        fake_prefix.mkdir(parents=True)
        with (
            patch("sys.prefix", str(fake_prefix)),
            patch("importlib.metadata.distribution") as mock_dist,
            patch("importlib.util.find_spec", return_value=True),
        ):
            mock_dist.return_value.read_text.return_value = None
            cmd = _detect_installer_command()
        assert cmd[0] == sys.executable
        assert "-m" in cmd and "pip" in cmd
        assert "hybro-hub" in cmd
        assert "a2a-adapter" in cmd
        assert "a2a-sdk" in cmd

    def test_uv_pip_fallback(self, tmp_path):
        fake_prefix = tmp_path / "some_venv"
        fake_prefix.mkdir(parents=True)
        with (
            patch("sys.prefix", str(fake_prefix)),
            patch("importlib.metadata.distribution") as mock_dist,
            patch("importlib.util.find_spec", return_value=None),
            patch("shutil.which", return_value="/usr/local/bin/uv"),
        ):
            mock_dist.return_value.read_text.return_value = None
            cmd = _detect_installer_command()
        assert cmd[0] == "/usr/local/bin/uv"
        assert "pip" in cmd
        assert "hybro-hub" in cmd

    def test_nothing_available(self, tmp_path):
        fake_prefix = tmp_path / "some_venv"
        fake_prefix.mkdir(parents=True)
        with (
            patch("sys.prefix", str(fake_prefix)),
            patch("importlib.metadata.distribution") as mock_dist,
            patch("importlib.util.find_spec", return_value=None),
            patch("shutil.which", return_value=None),
        ):
            mock_dist.return_value.read_text.return_value = None
            with pytest.raises(click.ClickException, match="Could not find"):
                _detect_installer_command()

    def test_conda_env(self, tmp_path):
        fake_prefix = tmp_path / "conda" / "envs" / "myenv"
        fake_prefix.mkdir(parents=True)
        with (
            patch("sys.prefix", str(fake_prefix)),
            patch.dict("os.environ", {"CONDA_PREFIX": str(fake_prefix)}),
            patch("importlib.metadata.distribution") as mock_dist,
        ):
            mock_dist.return_value.read_text.return_value = None
            with pytest.raises(click.ClickException, match="conda"):
                _detect_installer_command()

    def test_editable_install(self):
        editable_json = '{"url": "file:///home/user/hybro-hub", "dir_info": {"editable": true}}'
        with patch("importlib.metadata.distribution") as mock_dist:
            mock_dist.return_value.read_text.return_value = editable_json
            with pytest.raises(click.ClickException, match="editable"):
                _detect_installer_command()


# ──── _installer_display_name ────


class TestInstallerDisplayName:
    def test_pip(self):
        assert _installer_display_name([sys.executable, "-m", "pip", "install"]) == "pip"

    def test_pipx(self):
        assert _installer_display_name(["/usr/bin/pipx", "upgrade", "hybro-hub"]) == "pipx"

    def test_uv_tool(self):
        assert _installer_display_name(["/usr/bin/uv", "tool", "upgrade"]) == "uv tool"

    def test_uv_pip(self):
        assert _installer_display_name(["/usr/bin/uv", "pip", "install"]) == "uv"


# ──── _stop_daemon ────


class TestStopDaemon:
    def test_not_running(self):
        with patch("hub.cli.read_lock_pid", return_value=None):
            assert _stop_daemon() is False

    def test_stale_pid_no_process(self):
        import psutil

        with (
            patch("hub.cli.read_lock_pid", return_value=99999),
            patch("psutil.Process", side_effect=psutil.NoSuchProcess(99999)),
            patch("hub.cli._remove_lock_file") as mock_rm,
        ):
            assert _stop_daemon() is False
            mock_rm.assert_called_once_with(99999)

    def test_pid_different_process(self):
        mock_proc = MagicMock()
        mock_proc.cmdline.return_value = ["/usr/bin/nginx", "-g", "daemon off"]
        with (
            patch("hub.cli.read_lock_pid", return_value=12345),
            patch("psutil.Process", return_value=mock_proc),
        ):
            with pytest.raises(click.ClickException, match="different process"):
                _stop_daemon()

    def test_sigterm_succeeds(self):
        mock_proc = MagicMock()
        mock_proc.cmdline.return_value = ["python", "-m", "hub", "start"]
        with (
            patch("hub.cli.read_lock_pid", return_value=12345),
            patch("psutil.Process", return_value=mock_proc),
            patch("hub.cli._spinning_wait", return_value=True),
            patch("hub.cli._remove_lock_file") as mock_rm,
            patch("psutil.pid_exists", return_value=False),
        ):
            assert _stop_daemon() is True
            mock_proc.send_signal.assert_called_once()
            mock_rm.assert_called_once_with(12345)

    def test_sigterm_timeout_sigkill(self):
        mock_proc = MagicMock()
        mock_proc.cmdline.return_value = ["python", "-m", "hub", "start"]
        with (
            patch("hub.cli.read_lock_pid", return_value=12345),
            patch("psutil.Process", return_value=mock_proc),
            patch("hub.cli._spinning_wait", return_value=False),
            patch("hub.cli._remove_lock_file") as mock_rm,
        ):
            assert _stop_daemon() is True
            mock_proc.kill.assert_called_once()
            mock_rm.assert_called_once_with(12345)


# ──── _spawn_start ────


class TestSpawnStart:
    def test_background_success(self):
        with (
            patch("hub.cli.subprocess.run", return_value=MagicMock(returncode=0)),
            patch("hub.cli.read_lock_pid", return_value=12345),
            patch("time.sleep"),
        ):
            assert _spawn_start() is True

    def test_background_failure(self):
        with (
            patch("hub.cli.subprocess.run", return_value=MagicMock(returncode=1)),
        ):
            assert _spawn_start() is False

    def test_background_no_lock_after_start(self):
        with (
            patch("hub.cli.subprocess.run", return_value=MagicMock(returncode=0)),
            patch("hub.cli.read_lock_pid", return_value=None),
            patch("time.sleep"),
        ):
            assert _spawn_start() is False

    def test_foreground_flag(self):
        with patch("hub.cli.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = _spawn_start(foreground=True)
            assert result is True
            args = mock_run.call_args[0][0]
            assert "--foreground" in args


# ──── CLI integration: hybro-hub update ────


class TestUpdateCommand:
    def _run(self, args=None, **kwargs):
        runner = CliRunner()
        return runner.invoke(main, ["update"] + (args or []), catch_exceptions=False, **kwargs)

    def test_dry_run_shows_upgrades(self):
        with (
            patch("hub.cli._installed_version", side_effect=lambda p: {"hybro-hub": "0.1.16", "a2a-adapter": "0.2.5", "a2a-sdk": "0.3.25"}.get(p)),
            patch("hub.cli._query_pypi_versions", return_value={"hybro-hub": "0.1.18", "a2a-adapter": "0.2.7", "a2a-sdk": "0.3.25"}),
        ):
            result = self._run(["--dry-run"])
        assert result.exit_code == 0
        assert "\u2192" in result.output
        assert "0.1.18" in result.output

    def test_already_up_to_date(self):
        with (
            patch("hub.cli._installed_version", side_effect=lambda p: {"hybro-hub": "0.1.18", "a2a-adapter": "0.2.7", "a2a-sdk": "0.3.25"}.get(p)),
            patch("hub.cli._query_pypi_versions", return_value={"hybro-hub": "0.1.18", "a2a-adapter": "0.2.7", "a2a-sdk": "0.3.25"}),
        ):
            result = self._run()
        assert result.exit_code == 0
        assert "up to date" in result.output

    def test_pypi_unreachable_dry_run(self):
        with (
            patch("hub.cli._installed_version", return_value="0.1.16"),
            patch("hub.cli._query_pypi_versions", return_value={"hybro-hub": None, "a2a-adapter": None, "a2a-sdk": None}),
        ):
            result = self._run(["--dry-run"])
        assert result.exit_code == 0
        assert "offline" in result.output.lower() or "PyPI" in result.output

    def test_upgrade_subprocess_fails(self):
        with (
            patch("hub.cli._installed_version", side_effect=lambda p: "0.1.16"),
            patch("hub.cli._query_pypi_versions", return_value={"hybro-hub": "0.2.0", "a2a-adapter": "0.3.0", "a2a-sdk": "0.4.0"}),
            patch("hub.cli._detect_installer_command", return_value=[sys.executable, "-m", "pip", "install", "--upgrade", "hybro-hub"]),
            patch("hub.cli.subprocess.run", return_value=MagicMock(returncode=1, stderr="some error", stdout="")),
        ):
            result = self._run()
        assert result.exit_code == 1

    def test_permission_denied_suggests_sudo(self):
        with (
            patch("hub.cli._installed_version", side_effect=lambda p: "0.1.16"),
            patch("hub.cli._query_pypi_versions", return_value={"hybro-hub": "0.2.0", "a2a-adapter": "0.3.0", "a2a-sdk": "0.4.0"}),
            patch("hub.cli._detect_installer_command", return_value=[sys.executable, "-m", "pip", "install", "--upgrade", "hybro-hub"]),
            patch("hub.cli.subprocess.run", return_value=MagicMock(returncode=1, stderr="Permission denied", stdout="")),
        ):
            result = self._run()
        assert result.exit_code == 1
        assert "sudo" in result.output

    def test_restart_flag(self):
        with (
            patch("hub.cli._installed_version", side_effect=lambda p: "0.1.16"),
            patch("hub.cli._query_pypi_versions", return_value={"hybro-hub": "0.2.0", "a2a-adapter": "0.3.0", "a2a-sdk": "0.4.0"}),
            patch("hub.cli._detect_installer_command", return_value=[sys.executable, "-m", "pip", "install", "--upgrade", "hybro-hub"]),
            patch("hub.cli.subprocess.run", return_value=MagicMock(returncode=0, stderr="", stdout="")),
            patch("hub.cli._read_post_upgrade_version", return_value="0.2.0"),
            patch("hub.cli._stop_daemon", return_value=True) as mock_stop,
            patch("hub.cli._spawn_start", return_value=True) as mock_start,
        ):
            result = self._run(["--restart"])
        assert result.exit_code == 0
        mock_stop.assert_called_once()
        mock_start.assert_called_once()

    def test_shows_restart_hint_when_daemon_running(self):
        with (
            patch("hub.cli._installed_version", side_effect=lambda p: "0.1.16"),
            patch("hub.cli._query_pypi_versions", return_value={"hybro-hub": "0.2.0", "a2a-adapter": "0.3.0", "a2a-sdk": "0.4.0"}),
            patch("hub.cli._detect_installer_command", return_value=[sys.executable, "-m", "pip", "install", "--upgrade", "hybro-hub"]),
            patch("hub.cli.subprocess.run", return_value=MagicMock(returncode=0, stderr="", stdout="")),
            patch("hub.cli._read_post_upgrade_version", return_value="0.2.0"),
            patch("hub.cli.read_lock_pid", return_value=12345),
        ):
            result = self._run()
        assert result.exit_code == 0
        assert "hybro-hub restart" in result.output


# ──── CLI integration: hybro-hub restart ────


class TestRestartCommand:
    def _run(self, args=None):
        runner = CliRunner()
        return runner.invoke(main, ["restart"] + (args or []), catch_exceptions=False)

    def test_restart_running_daemon(self):
        with (
            patch("hub.cli._stop_daemon", return_value=True),
            patch("hub.cli._spawn_start", return_value=True) as mock_start,
        ):
            result = self._run()
        assert result.exit_code == 0
        mock_start.assert_called_once_with(foreground=False)

    def test_restart_when_not_running(self):
        with (
            patch("hub.cli._stop_daemon", return_value=False),
            patch("hub.cli._spawn_start", return_value=True),
        ):
            result = self._run()
        assert result.exit_code == 0
        assert "starting fresh" in result.output

    def test_restart_foreground(self):
        with (
            patch("hub.cli._stop_daemon", return_value=True),
            patch("hub.cli._spawn_start", return_value=True) as mock_start,
        ):
            result = self._run(["--foreground"])
        assert result.exit_code == 0
        mock_start.assert_called_once_with(foreground=True)

    def test_restart_pid_different_process(self):
        with patch("hub.cli._stop_daemon", side_effect=click.ClickException("different process")):
            result = self._run()
        assert result.exit_code != 0
        assert "different process" in result.output

    def test_restart_warns_if_start_fails(self):
        with (
            patch("hub.cli._stop_daemon", return_value=True),
            patch("hub.cli._spawn_start", return_value=False),
        ):
            result = self._run()
        assert result.exit_code == 0
        assert "may not have started" in result.output


# ──── CLI integration: hybro-hub --version ────


class TestVersionFlag:
    def test_version_output(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "hybro-hub" in result.output
