#!/usr/bin/env bash
# L2 GPU stages: prepare prompts, generate both sides, replay and build positions.
# usage: bash scripts/l2_gpu.sh /path/to/local/model [run-name]
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
[ -f "data/$RUN/prompts.jsonl" ] || python scripts/run.py prepare --config "$CONFIG" --model "$MODEL_PATH" --output "data/$RUN"

# Training build fixes L_top, which the test build also uses.
python scripts/run.py l2-generate --config "$CONFIG" --model "$MODEL_PATH" --prompts "data/$RUN" --side train --output "runs/$RUN-train-gen" "${DTYPE_ARGS[@]}" --progress "$PROGRESS"
python scripts/run.py l2-build --input "runs/$RUN-train-gen" --output "runs/$RUN-train-pos" --model "$MODEL_PATH" --progress "$PROGRESS"
python scripts/run.py l2-generate --config "$CONFIG" --model "$MODEL_PATH" --prompts "data/$RUN" --side test --output "runs/$RUN-test-gen" "${DTYPE_ARGS[@]}" --progress "$PROGRESS"
python scripts/run.py l2-build --input "runs/$RUN-test-gen" --output "runs/$RUN-test-pos" --model "$MODEL_PATH" --train "runs/$RUN-train-pos" --progress "$PROGRESS"

printf 'GPU stages complete for %s. Next: WORKERS=10 bash scripts/l2_cpu.sh %q\n' "$RUN" "$RUN"
