#!/usr/bin/env bash
# L2 GPU stages: prepare prompts, generate both sides, replay and build positions.
# usage: bash scripts/l2_gpu.sh /path/to/local/model [run-name]
# Reuse complete stages; preserve incomplete outputs as *.oldN and rerun that stage.
# Test data may be prepared before freezing, but must not be used for selection.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL_PATH="${1:?usage: bash scripts/l2_gpu.sh /path/to/local/model [run-name]}"
RUN="${2:-l2a}"
CONFIG="${CONFIG:-configs/text_l2.json}"
PROGRESS="${PROGRESS:-runs/l2_progress.json}"
DTYPE_ARGS=()
if [[ -n "${DTYPE:-}" ]]; then
    DTYPE_ARGS=(--dtype "$DTYPE")
fi

python scripts/run.py check-runtime
python scripts/l2_gpu.py --model "$MODEL_PATH" --run "$RUN" --config "$CONFIG" --progress "$PROGRESS" "${DTYPE_ARGS[@]}"

printf 'GPU stages complete for %s. Next: WORKERS=10 bash scripts/l2_cpu.sh %q\n' "$RUN" "$RUN"
