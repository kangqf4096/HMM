#!/usr/bin/env bash
# L2 (docs/plan_l2_paper_setting_20261106.md). Server (GPU) steps 1-3 and 5, CPU steps 4 and 6-9.
# usage: bash scripts/l2_full.sh /path/to/local/model [run-name]
# Every command adds to runs/l2_progress.json:  watch -n 5 cat runs/l2_progress.json
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL_PATH="${1:?usage: bash scripts/l2_full.sh /path/to/local/model [run-name]}"
RUN="${2:-l2a}"
CONFIG="${CONFIG:-configs/text_l2.json}"
WORKERS="${WORKERS:-16}"
DTYPE_ARGS=()
if [[ -n "${DTYPE:-}" ]]; then
    DTYPE_ARGS=(--dtype "$DTYPE")
fi
PROGRESS="${PROGRESS:-runs/l2_progress.json}"
WEIGHTS="${WEIGHTS:-runs/progress_weights.json}"
python scripts/run.py check-runtime
# 1. prompts: rows disjoint from the earlier runs listed in l2.exclude_prompts
[ -f "data/$RUN/prompts.jsonl" ] || python scripts/run.py prepare --config "$CONFIG" --model "$MODEL_PATH" --output "data/$RUN"
# 2. training side (key K1): generate, build (replay needs the GPU), mechanism check
python scripts/run.py l2-generate --config "$CONFIG" --model "$MODEL_PATH" --prompts "data/$RUN" --side train --output "runs/$RUN-train-gen" "${DTYPE_ARGS[@]}" --progress "$PROGRESS"
python scripts/run.py l2-build --input "runs/$RUN-train-gen" --output "runs/$RUN-train-pos" --model "$MODEL_PATH" --progress "$PROGRESS"
python scripts/run.py l2-mechanism --input "runs/$RUN-train-pos" --output "runs/$RUN-mechanism" --progress "$PROGRESS"
# 3. estimation, dev selection, freeze (CPU). Stop here and read runs/$RUN-fit/fit_report.md and the mechanism report first.
python scripts/run.py l2-fit --input "runs/$RUN-train-pos" --output "runs/$RUN-fit" --workers "$WORKERS" --progress "$PROGRESS"
# 4. test side (key K2): do not read it before frozen.json exists
python scripts/run.py l2-generate --config "$CONFIG" --model "$MODEL_PATH" --prompts "data/$RUN" --side test --output "runs/$RUN-test-gen" "${DTYPE_ARGS[@]}" --progress "$PROGRESS"
python scripts/run.py l2-build --input "runs/$RUN-test-gen" --output "runs/$RUN-test-pos" --model "$MODEL_PATH" --train "runs/$RUN-train-pos" --progress "$PROGRESS"
# 5. one test run, then the cap sensitivity
python scripts/run.py l2-test --frozen "runs/$RUN-fit" --input "runs/$RUN-test-pos" --output "runs/$RUN-test" --workers "$WORKERS" --mechanism "runs/$RUN-mechanism/mechanism.json" --progress "$PROGRESS"
python scripts/run.py l2-cap --frozen "runs/$RUN-fit" --train "runs/$RUN-train-pos" --input "runs/$RUN-test-pos" --output "runs/$RUN-cap" --progress "$PROGRESS"
python scripts/run.py l2-weights --progress "$PROGRESS" --output "$WEIGHTS"
