#!/usr/bin/env bash
# L2 CPU stages: mechanism, fit/freeze, test, cap sensitivity and progress weights.
# usage: WORKERS=10 bash scripts/l2_cpu.sh [run-name]
# Requires the two *-pos directories from l2_gpu.sh; no model or CUDA needed.
# Reuse complete stages and persistent SeedBS caches; archive incomplete outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=""
RUN="${1:-l2a}"
WORKERS="${WORKERS:-16}"
PROGRESS="${PROGRESS:-runs/l2_progress.json}"
WEIGHTS="${WEIGHTS:-runs/progress_weights.json}"

python scripts/l2_cpu.py --run "$RUN" --workers "$WORKERS" --progress "$PROGRESS" --weights "$WEIGHTS"
