"""Small JSON artifacts; no system-temp directory and no mandatory checksums."""
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               default=_default, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_default, allow_nan=False) + "\n")


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output is not empty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def provenance():
    packages = {}
    for name in ("numpy", "vllm", "torch", "datasets", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "git_commit": commit}
