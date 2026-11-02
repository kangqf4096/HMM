#!/usr/bin/env bash
# S1b: 1,050 articles. Run only after the smoke run passed; l0-evaluate freezes parameters before touching test.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL_PATH="${1:?usage: bash scripts/l0_full.sh /path/to/local/model [run-name]}"
RUN_NAME="${2:-l0full0}"
CONFIG=configs/text_l0_full.json
python scripts/run.py check-runtime
[ -f "data/$RUN_NAME/prompts.jsonl" ] || python scripts/run.py prepare --config "$CONFIG" --model "$MODEL_PATH" --output "data/$RUN_NAME"
python scripts/run.py l0-generate --config "$CONFIG" --model "$MODEL_PATH" --prompts "data/$RUN_NAME" --output "runs/$RUN_NAME-gen"
python scripts/run.py l0-build --input "runs/$RUN_NAME-gen" --output "runs/$RUN_NAME-pos" --model "$MODEL_PATH"
python scripts/run.py l0-evaluate --input "runs/$RUN_NAME-pos" --output "runs/$RUN_NAME-eval"
# Appendix: beta without the density cap, one run.
python scripts/run.py l0-evaluate --input "runs/$RUN_NAME-pos" --output "runs/$RUN_NAME-eval-nocap" --config configs/text_l0_nocap_appendix.json
