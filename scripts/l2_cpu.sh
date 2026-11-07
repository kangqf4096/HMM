#!/usr/bin/env bash
# L2 CPU stages: mechanism, fit/freeze, test, cap sensitivity and progress weights.
# usage: WORKERS=10 bash scripts/l2_cpu.sh [run-name]
# Requires the two *-pos directories from l2_gpu.sh; no model or CUDA needed.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=""
RUN="${1:-l2a}"
WORKERS="${WORKERS:-16}"
PROGRESS="${PROGRESS:-runs/l2_progress.json}"
WEIGHTS="${WEIGHTS:-runs/progress_weights.json}"

# Check that both builds were transferred before starting the expensive fit.
for side in train test; do
    for name in manifest.json positions.jsonl articles.jsonl; do
        if [[ ! -f "runs/$RUN-$side-pos/$name" ]]; then
            printf 'Missing runs/%s-%s-pos/%s; complete l2_gpu.sh and copy both build directories first.\n' "$RUN" "$side" "$name" >&2
            exit 1
        fi
    done
done

python scripts/run.py l2-mechanism --input "runs/$RUN-train-pos" --output "runs/$RUN-mechanism" --progress "$PROGRESS"
python scripts/run.py l2-fit --input "runs/$RUN-train-pos" --output "runs/$RUN-fit" --workers "$WORKERS" --progress "$PROGRESS"
# Only after l2-fit writes frozen.json does evaluation load the test side.
python scripts/run.py l2-test --frozen "runs/$RUN-fit" --input "runs/$RUN-test-pos" --output "runs/$RUN-test" --workers "$WORKERS" --mechanism "runs/$RUN-mechanism/mechanism.json" --progress "$PROGRESS"
python scripts/run.py l2-cap --frozen "runs/$RUN-fit" --train "runs/$RUN-train-pos" --input "runs/$RUN-test-pos" --output "runs/$RUN-cap" --progress "$PROGRESS"
python scripts/run.py l2-weights --progress "$PROGRESS" --output "$WEIGHTS"
