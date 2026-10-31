#!/usr/bin/env bash
# Install opt-hmm inside the vLLM 0.8.4 / Python 3.10 / CUDA 12.4 image.
#
# This script deliberately reuses the image's vLLM, PyTorch and CUDA runtime.
# It does not install a different vLLM, a new CUDA toolkit, or a new Python.
# By default it creates .venv with --system-site-packages so vLLM stays visible
# while the project and its light dependencies are installed separately.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"
USE_VENV=1
WITH_VLLM=0
SKIP_TESTS=0
SKIP_RUNTIME=0

usage() {
    cat <<'EOF'
Usage: bash scripts/install.sh [options]

Options:
  --python PATH     Python interpreter (default: python3)
  --venv PATH       Virtual environment path (default: .venv)
  --no-venv         Use the current Python with --user instead of a venv
  --with-vllm       Also install the inference extra (fresh non-image hosts only)
  --skip-tests      Do not run scripts/test.py
  --skip-runtime    Do not run scripts/run.py check-runtime
  -h, --help        Show this help

Environment:
  PIP_INDEX_URL     Package index (default: Tsinghua mirror)
  PIP_EXTRA_INDEX_URL, PIP_TRUSTED_HOST are passed through unchanged.

The default mode is for registry-cn-xibei2-internal.ebcloud.com/ebsys/vllm:
0.8.4-cuda12.4-python3.10-ubuntu22.04-v01. It never replaces the image's vLLM.
EOF
}

while (($#)); do
    case "$1" in
        --python) PYTHON="${2:?missing value for --python}"; shift 2 ;;
        --venv) VENV="${2:?missing value for --venv}"; shift 2 ;;
        --no-venv) USE_VENV=0; shift ;;
        --with-vllm) WITH_VLLM=1; shift ;;
        --skip-tests) SKIP_TESTS=1; shift ;;
        --skip-runtime) SKIP_RUNTIME=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi

"$PYTHON" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python 3.10-3.12 is required for vLLM 0.8.4; found {sys.version.split()[0]}. "
        "Use the server image or pass --python with a compatible interpreter."
    )
print(f"Python: {sys.version.split()[0]} ({sys.executable})")
PY

# Resource paths, caches, and VLLM_USE_V1=0 must be active before importing vLLM.
source "$ROOT/scripts/env.sh"

# In the image mode the selected interpreter must already see the image's vLLM.
# Conda base is a common trap: vLLM may live in a different env or in /usr/bin.
if (( ! WITH_VLLM )); then
    if ! "$PYTHON" - <<'PY'
import importlib.metadata as metadata
import sys
try:
    import vllm  # noqa: F401
except Exception as error:
    print(f"vLLM import failed: {error}", file=sys.stderr)
    raise SystemExit(1)
version = metadata.version("vllm")
print(f"vLLM: {version}")
if version.split("+", 1)[0] != "0.8.4":
    print(f"Expected vLLM 0.8.4, found {version}", file=sys.stderr)
    raise SystemExit(1)
try:
    import torch
except Exception as error:
    print(f"torch import failed: {error}", file=sys.stderr)
    raise SystemExit(1)
print(f"torch: {torch.__version__} (CUDA {torch.version.cuda})")
if torch.__version__.split("+", 1)[0] != "2.6.0":
    print("Warning: vLLM 0.8.4 wheels target torch 2.6.0; found a different torch.", file=sys.stderr)
PY
    then
        echo "Selected Python ($PYTHON) cannot use the image's vLLM 0.8.4." >&2
        echo "Activate the conda env that contains vLLM, or pass --python /path/to/that/python." >&2
        echo "If this is a fresh host, use --with-vllm to install the inference extra." >&2
        exit 1
    fi
fi

export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_DISABLE_PIP_VERSION_CHECK=1

PIP_PY="$PYTHON"
PIP_USER=()
IN_VENV=0

if (( USE_VENV )); then
    if [[ -x "$VENV/bin/python" ]]; then
        if "$VENV/bin/python" -c 'import vllm' >/dev/null 2>&1; then
            PIP_PY="$VENV/bin/python"
            IN_VENV=1
            echo "Reusing virtual environment: $VENV"
        else
            echo "Existing $VENV cannot import the image's vLLM; using --user instead." >&2
            PIP_USER=(--user)
        fi
    else
        if "$PYTHON" -m venv --system-site-packages "$VENV"; then
            PIP_PY="$VENV/bin/python"
            IN_VENV=1
            echo "Created virtual environment: $VENV"
        else
            echo "python -m venv is unavailable; falling back to --user." >&2
            PIP_USER=(--user)
        fi
    fi
else
    PIP_USER=(--user)
fi

if (( IN_VENV )); then
    "$PIP_PY" -m pip install "${PIP_USER[@]}" --upgrade "pip>=23" "setuptools>=68" "wheel"
fi

if (( WITH_VLLM )); then
    echo "Installing the inference extra, including vLLM 0.8.4 (not the default image mode)."
    "$PIP_PY" -m pip install "${PIP_USER[@]}" -e ".[inference]"
else
    # The image already provides vLLM/torch; only project, data, and tokenizer
    # dependencies are installed. The <4.56 cap keeps the image's 4.55.x
    # transformers instead of downgrading it for vLLM 0.8.4.
    "$PIP_PY" -m pip install "${PIP_USER[@]}" -e ".[data]"
fi

if ! "$PIP_PY" -m pip check; then
    echo "Warning: pip check reported a dependency problem. Inspect the output above." >&2
fi

if (( ! SKIP_TESTS )); then
    "$PIP_PY" "$ROOT/scripts/test.py"
fi

if (( ! SKIP_RUNTIME )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi not found; skipping GPU check-runtime." >&2
    elif nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
        "$PIP_PY" "$ROOT/scripts/run.py" check-runtime
    else
        echo "No GPU visible to this container; skipping check-runtime." >&2
        echo "After allocating a GPU, run:" >&2
        echo "  $PIP_PY $ROOT/scripts/run.py check-runtime" >&2
    fi
fi

if (( IN_VENV )); then
    ACTIVATE="source $VENV/bin/activate"
else
    ACTIVATE="# using --user install"
fi

cat <<EOF

Environment ready.
  project : $PWD
  python  : $PIP_PY
  vLLM    : 0.8.4 (V0 engine)

Activate and run:
  $ACTIVATE
  source scripts/env.sh
  python scripts/run.py simulate --config configs/sim_smoke.json --output runs/sim0
  python scripts/run.py generate --config configs/text_smoke.json \\
      --model /path/to/local/model --prompts data/ax0 --output runs/ax0-gen
EOF
