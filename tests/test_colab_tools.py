import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from colab_tools import OutputState, Panel, github_sync, run_live


class ColabTests(unittest.TestCase):
    def folder(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="cb")

    def test_carriage_returns_events_and_stage_changes(self):
        state = OutputState()
        state.feed("\rweights: 25%|##      | 1/4\rweights: 50%|####    | 2/4\r")
        self.assertEqual((state.stage, state.percent), ("weights", 50))
        payload = state.prefix + json.dumps({"stage": "loading", "current": None, "total": None})
        state.feed(payload[:20])
        state.feed(payload[20:] + "\n")
        self.assertEqual(state.stage, "loading")
        self.assertIsNone(state.percent)
        state.feed("last partial line")
        state.finish()
        self.assertEqual(state.lines[-1], "last partial line")

    def test_pipe_updates_before_child_exits_and_preserves_log(self):
        with self.folder() as folder:
            root = Path(folder)
            (root / "child.py").write_text(
                "import sys, time\nfrom pathlib import Path\n"
                "sys.stdout.write('weights: 50%|####| 1/2\\r')\n"
                "sys.stdout.flush()\nend = time.monotonic() + 5\n"
                "while not Path('ack').exists():\n"
                "    if time.monotonic() > end: raise RuntimeError('reader did not update live')\n"
                "    time.sleep(.02)\n"
                "sys.stderr.write('finished\\n')\n", encoding="utf-8")
            snapshots = []

            def draw(panel, state, status="运行中"):
                snapshots.append((state.percent, status))
                if state.percent == 50:
                    (root / "ack").touch()

            with patch.object(Panel, "draw", autospec=True, side_effect=draw):
                log = run_live([sys.executable, "-u", "child.py"], title="stream", log_name="s",
                               root=root, render=False)
            self.assertIn((50, "运行中"), snapshots)
            self.assertEqual(snapshots[-1], (100, "已完成"))
            self.assertIn("finished", log.read_text(encoding="utf-8"))
            self.assertTrue((root / "ack").is_file())

    def test_failed_child_keeps_output_and_raises(self):
        with self.folder() as folder:
            root = Path(folder)
            (root / "child.py").write_text("print('failure reason', flush=True)\nraise SystemExit(7)\n")
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                run_live([sys.executable, "-u", "child.py"], title="failure", log_name="f",
                         root=root, render=False)
            self.assertEqual(caught.exception.returncode, 7)
            self.assertIn("failure reason", (root / "tmp/f.log").read_text(encoding="utf-8"))

    def test_clone_and_pull_prompt_each_time_and_never_store_token(self):
        with self.folder() as folder:
            root = Path(folder)
            calls, environments = [], []

            def run(command, *, env, check):
                calls.append(command)
                self.assertNotIn("test-token", " ".join(command))
                self.assertTrue(env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic "))
                self.assertEqual(env["GIT_CONFIG_KEY_1"], "credential.helper")
                self.assertEqual(env["GIT_CONFIG_VALUE_1"], "")
                environments.append(env)

            before = dict(os.environ)
            with patch("colab_tools.getpass", return_value="test-token") as prompt, \
                    patch("colab_tools.subprocess.run", side_effect=run), \
                    patch("colab_tools.subprocess.check_output", return_value="experiment\n"), \
                    redirect_stdout(io.StringIO()):
                github_sync(root)
                (root / ".git").mkdir()
                github_sync(root)
            self.assertEqual(prompt.call_count, 2)
            self.assertEqual(calls[0][:2], ["git", "clone"])
            self.assertEqual(calls[1][-4:], ["pull", "--ff-only", "https://github.com/kangqf4096/HMM.git", "experiment"])
            self.assertTrue(all("GIT_CONFIG_VALUE_0" not in env for env in environments))
            self.assertEqual(dict(os.environ), before)
            self.assertEqual(list(root.iterdir()), [root / ".git"])


if __name__ == "__main__":
    unittest.main()
