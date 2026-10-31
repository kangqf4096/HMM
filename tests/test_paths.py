import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opt_hmm.cli import main
from opt_hmm.paths import COLAB_PROJECT, COLAB_RUNTIME, project_path, project_root, runtime_root
from colab_tools import project_environment
from download_model import download_model
from _compat import chdir


class PathTests(unittest.TestCase):
    def folder(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="pt")

    def test_colab_defaults_and_stale_project_variables_use_drive(self):
        with patch("opt_hmm.paths.on_colab", return_value=True), \
                patch("opt_hmm.paths.Path.resolve", lambda path: path), \
                patch("opt_hmm.paths.Path.is_dir", return_value=True):
            for stale in ("", "/content", COLAB_RUNTIME):
                with patch.dict(os.environ, {"OPT_HMM_PROJECT": stale, "OPT_HMM_RUNTIME": COLAB_RUNTIME}):
                    self.assertEqual(project_root().as_posix(), COLAB_PROJECT)
                    self.assertEqual(project_root(stale).as_posix(), COLAB_PROJECT)
                    self.assertEqual(project_path("models/opt-1.3b").as_posix(), COLAB_PROJECT + "/models/opt-1.3b")
                    self.assertEqual(runtime_root().as_posix(), COLAB_RUNTIME)

    def test_mount_is_required_before_persistent_directory_is_used(self):
        with patch("opt_hmm.paths.on_colab", return_value=True), \
                patch("opt_hmm.paths.Path.resolve", lambda path: path), \
                patch("opt_hmm.paths.Path.is_dir", return_value=False), \
                patch.dict(os.environ, {"OPT_HMM_PROJECT": ""}):
            with self.assertRaisesRegex(RuntimeError, "drive.mount"):
                project_root()

    def test_outside_colab_paths_use_checkout_or_selected_root_never_cwd(self):
        with self.folder() as folder:
            root = Path(folder)
            with chdir(root), patch("opt_hmm.paths.on_colab", return_value=False), \
                    patch.dict(os.environ, {"OPT_HMM_PROJECT": ""}):
                self.assertEqual(project_root(), ROOT)
            with chdir(root), patch.dict(os.environ, {"OPT_HMM_PROJECT": str(ROOT)}):
                self.assertEqual(project_path("configs/colab_arxiv.json"), ROOT / "configs/colab_arxiv.json")
                absolute = root / "external.json"
                self.assertEqual(project_path(absolute), absolute)

    def test_cli_prepare_resolves_config_model_and_output_from_selected_project(self):
        with self.folder() as folder:
            project = Path(folder) / "drive"
            project.mkdir()
            (project / "configs").mkdir()
            config = json.loads((ROOT / "configs/colab_arxiv.json").read_text(encoding="utf-8"))
            (project / "configs/colab_arxiv.json").write_text(json.dumps(config))
            with chdir(ROOT.parent), \
                    patch.dict(os.environ, {"OPT_HMM_PROJECT": str(project)}), \
                    patch("opt_hmm.cli.on_colab", return_value=True), \
                    patch("opt_hmm.data.prepare", return_value={}) as prepare, \
                    contextlib.redirect_stdout(io.StringIO()):
                main(["prepare", "--output", "data/ax0"])
            self.assertEqual(prepare.call_args.args[1:], (project / "models/opt-1.3b", project / "data/ax0"))
            self.assertEqual(prepare.call_args.args[0], config)

    def test_model_download_uses_project_even_when_called_from_content_equivalent(self):
        import types
        from unittest.mock import Mock

        with self.folder() as folder:
            project = Path(folder) / "drive"
            (project / "configs").mkdir(parents=True)
            config = json.loads((ROOT / "configs/colab_arxiv.json").read_text(encoding="utf-8"))
            (project / "configs/colab_arxiv.json").write_text(json.dumps(config))
            hub = types.ModuleType("huggingface_hub")

            def snapshot(**kwargs):
                destination = Path(kwargs["local_dir"])
                for name in ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "pytorch_model.bin"):
                    (destination / name).touch()

            hub.snapshot_download = Mock(side_effect=snapshot)
            with chdir(ROOT.parent), patch.dict(sys.modules, {"huggingface_hub": hub}), \
                    patch.dict(os.environ, {"OPT_HMM_PROJECT": str(project), "OPT_HMM_RUNTIME": str(Path(folder) / "run"),
                                            "HF_HOME": "stale-cache"}):
                result = download_model("configs/colab_arxiv.json")
                self.assertEqual(os.environ["HF_HOME"], str(project / ".cache/hf"))
            self.assertEqual(Path(result["model"]), project / "models/opt-1.3b")
            self.assertFalse((ROOT.parent / "models/opt-1.3b").exists())
