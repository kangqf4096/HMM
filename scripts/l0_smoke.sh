#!/usr/bin/env bash
# S1a: 8 articles (one per role, test = one per scenario). Pipeline check only; no conclusions.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL_PATH="${1:?usage: bash scripts/l0_smoke.sh /path/to/local/model [run-name]}"
RUN_NAME="${2:-l0smoke0}"
CONFIG=configs/text_l0_smoke.json
python scripts/test.py
python scripts/run.py check-runtime
[ -f "data/$RUN_NAME/prompts.jsonl" ] || python scripts/run.py prepare --config "$CONFIG" --model "$MODEL_PATH" --output "data/$RUN_NAME"
python scripts/run.py l0-generate --config "$CONFIG" --model "$MODEL_PATH" --prompts "data/$RUN_NAME" --output "runs/$RUN_NAME-gen"
python scripts/run.py l0-build --input "runs/$RUN_NAME-gen" --output "runs/$RUN_NAME-pos" --model "$MODEL_PATH"
python scripts/run.py l0-evaluate --input "runs/$RUN_NAME-pos" --output "runs/$RUN_NAME-eval"
