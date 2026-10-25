"""Persist dependency wheels on Drive; install the runtime without an index."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import venv

from colab_tools import project_environment


def run(command, *, root, env):
    subprocess.run([str(arg) for arg in command], cwd=root, env=env, check=True)


def setup(root=None, *, offline=False):
    env = project_environment(root)
    root = Path(env["OPT_HMM_PROJECT"])
    runtime = Path(env["OPT_HMM_RUNTIME"])
    # Wheels depend on Python ABI and platform, rather than the selected GPU.
    tag = f"{sys.implementation.cache_tag}-{sysconfig.get_platform()}"
    wheels = root / ".cache/wheels" / tag
    wheels.mkdir(parents=True, exist_ok=True)
    marker = wheels / "project.txt"
    spec = (root / "pyproject.toml").read_text(encoding="utf-8")
    ready = marker.is_file() and marker.read_text(encoding="utf-8") == spec
    print(f"持久依赖安装包：{wheels}", flush=True)
    print(f"本次 Python 环境：{runtime / '.venv'}", flush=True)
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if not ready:
        if offline:
            raise RuntimeError("当前 Python/平台和依赖声明尚无完整缓存；先不带 --offline 运行一次。")
        print("首次准备或依赖声明已变化：联网保存完整 wheel 安装包到 Drive。", flush=True)
        fetch_env = dict(env)
        legacy_cache = runtime / ".cache/pip"
        if legacy_cache.is_dir() and legacy_cache != Path(env["PIP_CACHE_DIR"]):
            # The current session may still hold large downloads from the old
            # setup. Reuse those responses while exporting wheels to Drive.
            fetch_env["PIP_CACHE_DIR"] = str(legacy_cache)
            print("利用本次会话遗留的 pip 缓存，将安装包保存到 Drive。", flush=True)
        run([sys.executable, "-m", "pip", "wheel", "--wheel-dir", wheels,
             ".[inference]", "setuptools>=68", "wheel", "virtualenv"], root=root, env=fetch_env)
    else:
        print("复用 Drive 安装包；本次依赖安装不访问软件源。", flush=True)

    environment = runtime / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    pip_ok = python.is_file() and subprocess.run(
        [str(python), "-m", "pip", "--version"], cwd=root, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not pip_ok:
        try:
            venv.EnvBuilder(with_pip=True).create(environment)
        except (OSError, subprocess.CalledProcessError):
            # Some Colab images lack ensurepip. Bootstrap from the same wheels,
            # without installing anything into the notebook's global Python.
            bootstrap = runtime / ".bootstrap"
            run([sys.executable, "-m", "pip", "install", "--no-index",
                 "--find-links", wheels, "--target", bootstrap, "virtualenv"], root=root, env=env)
            seed_env = dict(env)
            seed_env["PYTHONPATH"] = str(bootstrap) + os.pathsep + env["PYTHONPATH"]
            run([sys.executable, "-m", "virtualenv", "--no-download",
                 "--no-periodic-update", "--app-data", runtime / ".virtualenv",
                 environment], root=root, env=seed_env)
    run([python, "-m", "pip", "--version"], root=root, env=env)
    local = [python, "-m", "pip", "install", "--no-index", "--find-links", wheels]
    run([*local, "setuptools>=68", "wheel"], root=root, env=env)
    run([*local, "--no-build-isolation", "-e", ".[inference]"], root=root, env=env)
    run([python, "-m", "pip", "check"], root=root, env=env)
    # Only mark the cache reusable once an offline installation has succeeded.
    marker.write_text(spec, encoding="utf-8")
    print("环境安装完成；安装包已保存在 Drive，后续相同环境可离线复用。", flush=True)
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Require an existing complete wheel cache")
    args = parser.parse_args()
    setup(offline=args.offline)
