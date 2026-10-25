#!/usr/bin/env bash
# Source this while your working directory is the repository root.
OPT_HMM_RUNTIME="${OPT_HMM_RUNTIME:-$PWD}"
mkdir -p tmp .cache/hf "$OPT_HMM_RUNTIME/tmp" "$OPT_HMM_RUNTIME/.cache/vllm" "$OPT_HMM_RUNTIME/.cache/pip"
export OPT_HMM_PROJECT="$PWD"
export OPT_HMM_RUNTIME
export TMPDIR="$OPT_HMM_RUNTIME/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME="$PWD/.cache/hf"
export PIP_CACHE_DIR="$OPT_HMM_RUNTIME/.cache/pip"
export VLLM_CACHE_ROOT="$OPT_HMM_RUNTIME/.cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
