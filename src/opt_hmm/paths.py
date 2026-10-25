"""Resolve experiment resources from the persistent Colab project, never cwd."""
import os
from pathlib import Path

COLAB_PROJECT = "/content/drive/MyDrive/opt-hmm"
COLAB_RUNTIME = "/content/opt-hmm"
CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def on_colab():
    return os.name == "posix" and Path("/content").is_dir()


def project_root(root=None):
    selected = root or os.environ.get("OPT_HMM_PROJECT")
    if on_colab():
        if not selected or str(selected).rstrip("/") in ("/content", COLAB_RUNTIME):
            selected = COLAB_PROJECT
    project = Path(selected or CHECKOUT_ROOT).expanduser().resolve()
    if project.as_posix().startswith("/content/drive/") and not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("先运行 drive.mount('/content/drive')，再使用持久项目目录。")
    return project


def project_path(path, root=None):
    path = Path(path).expanduser()
    return path if path.is_absolute() else project_root(root) / path


def runtime_root(project=None):
    default = COLAB_RUNTIME if on_colab() else project_root(project)
    return Path(os.environ.get("OPT_HMM_RUNTIME") or default).expanduser().resolve()
