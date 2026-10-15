#!/usr/bin/env bash
# Source this while your working directory is the repository root.
mkdir -p tmp .cache/hf .cache/vllm
export TMPDIR="$PWD/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HOME="$PWD/.cache/hf"
export VLLM_CACHE_ROOT="$PWD/.cache/vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
