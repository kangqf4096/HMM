#!/usr/bin/env bash
# Source from any directory; resource paths always use the project root.
if [[ -d /content ]]; then
    if [[ ! -d /content/drive/MyDrive ]]; then
        printf '%s\n' "Mount Google Drive before sourcing scripts/env.sh." >&2
        return 1
    fi
    case "${OPT_HMM_PROJECT:-}" in
        ""|/content|/content/opt-hmm) OPT_HMM_PROJECT=/content/drive/MyDrive/opt-hmm ;;
    esac
    OPT_HMM_RUNTIME="${OPT_HMM_RUNTIME:-/content/opt-hmm}"
else
    OPT_HMM_PROJECT="${OPT_HMM_PROJECT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
    OPT_HMM_RUNTIME="${OPT_HMM_RUNTIME:-$OPT_HMM_PROJECT}"
fi
cd -- "$OPT_HMM_PROJECT" || return 1
mkdir -p tmp .cache/hf .cache/pip .cache/wheels "$OPT_HMM_RUNTIME/tmp" "$OPT_HMM_RUNTIME/.cache/vllm"
export OPT_HMM_PROJECT="$PWD"
export OPT_HMM_RUNTIME
export TMPDIR="$OPT_HMM_RUNTIME/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME="$PWD/.cache/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export VLLM_CACHE_ROOT="$OPT_HMM_RUNTIME/.cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# vLLM 0.8.4 V1 does not apply SamplingParams.logits_processors; the watermark
# sampler is a V0-only callable, so keep the project on the V0 engine.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
