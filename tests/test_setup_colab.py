import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from setup_colab import setup


class SetupTests(unittest.TestCase):
    def folder(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="se")

    def test_new_runtime_reuses_drive_wheels_without_online_commands(self):
        with self.folder() as folder:
            project = Path(folder) / "drive"
            project.mkdir()
            (project / "pyproject.toml").write_text("dependency specification", encoding="utf-8")
            legacy = Path(folder) / "r1/.cache/pip"
            legacy.mkdir(parents=True)
            with patch.dict(os.environ, {"OPT_HMM_RUNTIME": str(Path(folder) / "r1")}), \
                    patch("setup_colab.venv.EnvBuilder"), patch("setup_colab.run") as run, \
                    redirect_stdout(io.StringIO()):
                setup(project)
            commands = [call.args[0] for call in run.call_args_list]
            fetch = next(call for call in run.call_args_list if call.args[0][2:4] == ["pip", "wheel"])
            self.assertEqual(Path(fetch.kwargs["env"]["PIP_CACHE_DIR"]), legacy.resolve())
            self.assertEqual(sum(command[2:4] == ["pip", "wheel"] for command in commands), 1)
            self.assertTrue(all("--no-index" in command for command in commands
                                if command[2:4] == ["pip", "install"]))
            self.assertEqual(len(list((project / ".cache/wheels").glob("*/project.txt"))), 1)
            with patch.dict(os.environ, {"OPT_HMM_RUNTIME": str(Path(folder) / "r2")}), \
                    patch("setup_colab.venv.EnvBuilder"), patch("setup_colab.run") as run, \
                    redirect_stdout(io.StringIO()):
                setup(project, offline=True)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertFalse(any(command[2:4] == ["pip", "wheel"] for command in commands))
            self.assertTrue(all("--no-index" in command for command in commands
                                if command[2:4] == ["pip", "install"]))

    def test_failed_install_does_not_mark_cache_complete(self):
        with self.folder() as folder:
            project = Path(folder)
            (project / "pyproject.toml").write_text("spec", encoding="utf-8")

            def fail(command, **kwargs):
                if "check" in command:
                    raise subprocess.CalledProcessError(1, command)

            with patch.dict(os.environ, {"OPT_HMM_RUNTIME": str(project / "rt")}), \
                    patch("setup_colab.venv.EnvBuilder"), patch("setup_colab.run", side_effect=fail), \
                    redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                setup(project)
            self.assertEqual(list((project / ".cache/wheels").glob("*/project.txt")), [])

    def test_offline_missing_or_changed_spec_never_downloads(self):
        with self.folder() as folder:
            project = Path(folder)
            spec = project / "pyproject.toml"
            spec.write_text("old", encoding="utf-8")
            with patch("setup_colab.run") as run, redirect_stdout(io.StringIO()), \
                    self.assertRaises(RuntimeError):
                setup(project, offline=True)
            run.assert_not_called()
            wheels = next((project / ".cache/wheels").iterdir())
            (wheels / "project.txt").write_text("old", encoding="utf-8")
            spec.write_text("new", encoding="utf-8")
            with patch("setup_colab.run") as run, redirect_stdout(io.StringIO()), \
                    self.assertRaises(RuntimeError):
                setup(project, offline=True)
            run.assert_not_called()

    def test_missing_ensurepip_bootstraps_virtualenv_offline(self):
        with self.folder() as folder:
            project = Path(folder)
            (project / "pyproject.toml").write_text("spec", encoding="utf-8")
            with patch.dict(os.environ, {"OPT_HMM_RUNTIME": str(project / "rt")}), \
                    patch("setup_colab.venv.EnvBuilder") as builder, \
                    patch("setup_colab.run") as run, redirect_stdout(io.StringIO()):
                builder.return_value.create.side_effect = subprocess.CalledProcessError(1, "ensurepip")
                setup(project)
            commands = [call.args[0] for call in run.call_args_list]
            bootstrap = next(command for command in commands if "--target" in command)
            self.assertIn("--no-index", bootstrap)
            seed = next(command for command in commands if command[2:3] == ["virtualenv"])
            self.assertIn("--no-download", seed)
            self.assertIn("--no-periodic-update", seed)


if __name__ == "__main__":
    unittest.main()
