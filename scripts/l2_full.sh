#!/usr/bin/env bash
# L2: run all GPU stages, then all CPU stages (see docs/l2_split.md).
# usage: bash scripts/l2_full.sh /path/to/local/model [run-name]
# Every command adds to runs/l2_progress.json:  watch -n 5 cat runs/l2_progress.json
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL_PATH="${1:?usage: bash scripts/l2_full.sh /path/to/local/model [run-name]}"
RUN="${2:-l2a}"
bash scripts/l2_gpu.sh "$MODEL_PATH" "$RUN"
bash scripts/l2_cpu.sh "$RUN"
