"""Notebook progress display and private Git access, without saved credentials."""
import base64
import codecs
import html
import json
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from getpass import getpass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opt_hmm.paths import project_root, runtime_root


def github_sync(root=None):
    """Clone or fast-forward the current branch, prompting for a token each time."""
    project = project_root(root)
    repo = "https://github.com/kangqf4096/HMM.git"
    if (project / ".git").is_dir():
        branch = subprocess.check_output(
            ["git", "-C", str(project), "branch", "--show-current"], text=True).strip()
        if not branch:
            raise RuntimeError("Checkout is detached. Switch to the intended branch before pulling.")
        command = ["git", "-C", str(project), "pull", "--ff-only", repo, branch]
    else:
        command = ["git", "clone", repo, str(project)]
    token = getpass("GitHub token（每次输入，不保存）: ").strip()
    if not token:
        raise ValueError("Token is empty.")
    auth = base64.b64encode(f"kangqf4096:{token}".encode()).decode()
    git_env = os.environ.copy()
    git_env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth}",
        "GIT_CONFIG_KEY_1": "credential.helper",
        "GIT_CONFIG_VALUE_1": "",
    })
    try:
        subprocess.run(command, env=git_env, check=True)
    finally:
        git_env.pop("GIT_CONFIG_VALUE_0", None)
        token = auth = None
    print("代码已同步：", project)
    return project


def project_environment(root):
    root = project_root(root)
    runtime = runtime_root(root)
    paths = {"TMPDIR": runtime / "tmp", "HF_HOME": root / ".cache/hf",
             "VLLM_CACHE_ROOT": runtime / ".cache/vllm", "PIP_CACHE_DIR": runtime / ".cache/pip"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in paths.items()})
    env.update({"TMP": env["TMPDIR"], "TEMP": env["TMPDIR"],
                "OPT_HMM_PROJECT": str(root), "OPT_HMM_RUNTIME": str(runtime),
                "HF_HUB_CACHE": str(root / ".cache/hf/hub"),
                "HF_DATASETS_CACHE": str(root / ".cache/hf/datasets"),
                "PYTHONPATH": str(root / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
                "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "OPT_HMM_PROGRESS": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "0", "HF_DATASETS_DISABLE_PROGRESS_BARS": "0"})
    return env


class OutputState:
    """Parse flushed events and terminal bars, including carriage-return updates."""
    ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    native_bar = re.compile(r"([^\r\n]*?)\s*(\d+(?:\.\d+)?)%\|[^|]*\|([^\r\n]*)")
    prefix = "@@OPT_HMM_PROGRESS "

    def __init__(self):
        self.stage = "启动子进程"
        self.percent = None
        self.detail = ""
        self.pending = ""
        self.lines = deque(maxlen=12)
        self.errors = deque(maxlen=6)

    def _line(self, line):
        line = self.ansi.sub("", line).strip()
        if not line:
            return
        if re.match(r"^[\w.]*?(?:Error|Exception|ArrowInvalid):", line):
            self.errors.append(line)
        start = line.find(self.prefix)
        if start >= 0:
            try:
                event = json.loads(line[start + len(self.prefix):])
            except (ValueError, TypeError):
                self.lines.append(line)
                return
            self.stage = event["stage"]
            current, total = event.get("current"), event.get("total")
            self.percent = 100 * current / total if total and current is not None else None
            self.detail = event.get("detail", "")
            if total and current is not None:
                self.detail = f"{current}/{total}  {self.detail}".strip()
            return
        match = self.native_bar.search(line)
        if match:
            self.stage = match[1].strip(" :") or self.stage
            self.percent = float(match[2])
            self.detail = match[3].strip()
        self.lines.append(line)

    def feed(self, text):
        self.pending += text
        parts = re.split(r"[\r\n]", self.pending)
        self.pending = parts.pop()
        for line in parts:
            self._line(line)

    def finish(self):
        self._line(self.pending)
        self.pending = ""


class Panel:
    def __init__(self, title, log_path, enabled):
        self.title, self.log_path, self.enabled = title, log_path, enabled
        self.handle = None
        self.started = time.monotonic()

    def draw(self, state, status="运行中"):
        if not self.enabled:
            return
        from IPython.display import HTML, display
        elapsed = int(time.monotonic() - self.started)
        percent = state.percent
        value = f' value="{min(100, max(0, percent)):.2f}"' if percent is not None else ""
        caption = f"{percent:.1f}%" if percent is not None else "正在运行（当前阶段无总量）"
        tail = "\n".join(state.lines)
        errors = "\n".join(state.errors)
        error_markup = f'<pre style="color:#b42318;white-space:pre-wrap">{html.escape(errors)}</pre>' if errors else ""
        if state.pending:
            tail += "\n" + state.ansi.sub("", state.pending[-2000:])
        markup = (
            f'<div style="border:1px solid #bbb;padding:12px;border-radius:8px">'
            f'<b>{html.escape(self.title)}</b> · {html.escape(status)} · {elapsed // 60:02d}:{elapsed % 60:02d}'
            f'<div>{html.escape(state.stage)} · {caption}</div>'
            f'<progress max="100"{value} style="width:100%;height:20px"></progress>'
            f'<div>{html.escape(state.detail)}</div>'
            f'{error_markup}'
            f'<pre style="max-height:250px;overflow:auto;white-space:pre-wrap">{html.escape(tail)}</pre>'
            f'<small>完整日志：{html.escape(str(self.log_path))}</small></div>')
        if self.handle is None:
            self.handle = display(HTML(markup), display_id=True)
        else:
            self.handle.update(HTML(markup))


def _stop(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def run_live(command, *, title, log_name, root=None, render=True):
    """Drain a byte pipe continuously; refresh one notebook panel twice a second."""
    root = project_root(root)
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    env = project_environment(root)
    log_path = root / "tmp" / f"{Path(log_name).name}.log"
    state, chunks = OutputState(), queue.Queue()
    state.detail = f"项目目录：{root}"
    panel = Panel(title, log_path, render)
    panel.draw(state)
    process = subprocess.Popen(
        [str(arg) for arg in command], cwd=root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        start_new_session=(os.name == "posix"))

    def read_pipe():
        try:
            while True:
                data = os.read(process.stdout.fileno(), 4096)
                if not data:
                    break
                chunks.put(data)
        finally:
            chunks.put(None)

    reader = threading.Thread(target=read_pipe, daemon=True)
    reader.start()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    last_draw = 0
    try:
        with log_path.open("a", encoding="utf-8", newline="") as log:
            log.write(f"\n--- {title} ---\n")
            log.write(f"Project: {root}\nPython: {command[0]}\n")
            while True:
                try:
                    chunk = chunks.get(timeout=.2)
                except queue.Empty:
                    chunk = b""
                if chunk is None:
                    text = decoder.decode(b"", final=True)
                    log.write(text)
                    state.feed(text)
                    break
                if chunk:
                    text = decoder.decode(chunk)
                    log.write(text)
                    log.flush()
                    state.feed(text)
                if time.monotonic() - last_draw >= .5:
                    panel.draw(state)
                    last_draw = time.monotonic()
            process.wait()
            state.finish()
        if process.returncode:
            panel.draw(state, f"失败，退出码 {process.returncode}")
            raise subprocess.CalledProcessError(process.returncode, command, output="\n".join(state.errors))
        state.percent = 100
        panel.draw(state, "已完成")
    except KeyboardInterrupt:
        _stop(process)
        panel.draw(state, "已中断")
        raise
    except BaseException:
        _stop(process)
        raise
    finally:
        reader.join(timeout=1)
        process.stdout.close()
    return log_path


def run_step(*arguments, title, log_name, root=None):
    root = project_root(root)
    runtime = runtime_root(root)
    python = runtime / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"先运行安装环境单元格。缺少 Python：{python}")
    return run_live([python, "-u", *arguments], title=title, log_name=log_name, root=root)


def copy_runtime_assets(source="/content/opt-hmm", *, root=None):
    """Copy existing assets into Drive; preserve source and existing target files."""
    source, target = Path(source).resolve(), project_root(root)
    if source == target:
        raise ValueError("Source and persistent project directory must differ.")
    paths = []
    for name in ("models", "data", "runs", ".cache/hf"):
        folder = source / name
        if folder.is_dir():
            paths.extend(path for path in folder.rglob("*") if path.is_file())
    if (source / "tmp").is_dir():
        paths.extend((source / "tmp").glob("*.log"))
    state = OutputState()
    state.stage = "复制已有资源（按文件计数）"
    state.percent = 0 if paths else 100
    state.detail = f"0/{len(paths)}"
    panel = Panel("复制已有资源到 Drive", target, True)
    panel.draw(state)
    copied = skipped = 0
    for index, path in enumerate(paths, 1):
        relative = path.relative_to(source)
        destination = target / relative
        if destination.exists():
            skipped += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Leave an interrupted copy identifiable; never replace an existing asset.
            partial = destination.with_name(destination.name + ".part")
            shutil.copyfile(path, partial)
            partial.replace(destination)
            copied += 1
        state.stage = "复制已有资源（按文件计数）"
        state.percent = 100 * index / len(paths)
        state.detail = f"{index}/{len(paths)} · {relative}"
        panel.draw(state)
    panel.draw(state, "复制完成")
    print(f"已复制 {copied} 个文件，目标已存在而跳过 {skipped} 个文件。源文件保留：{source}")
    return {"copied": copied, "skipped": skipped}
