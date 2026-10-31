#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL_PATH="${1:?usage: bash scripts/pivot_smoke.sh /path/to/local/model [run-name] [prompts-dir]}"
RUN_NAME="${2:-piv0}"
PROMPTS_DIR="${3:-data/$RUN_NAME}"
python scripts/test.py
python scripts/run.py check-runtime
if [[ -e "$PROMPTS_DIR/manifest.json" ]]; then
    printf 'Reusing prepared prompts: %s\n' "$PROMPTS_DIR"
else
    python scripts/run.py prepare --config configs/text_smoke.json --model "$MODEL_PATH" --output "$PROMPTS_DIR"
fi
python scripts/run.py generate --config configs/text_smoke.json --model "$MODEL_PATH" --prompts "$PROMPTS_DIR" --output "runs/$RUN_NAME-gen" --pivot-diagnostics
# Separate process: vLLM exits before HF replay loads the same model on the GPU.
# diagnose-pivots runs the original five localization methods and all three extras.
python scripts/run.py diagnose-pivots --input "runs/$RUN_NAME-gen" --model "$MODEL_PATH" --output "runs/$RUN_NAME-diag"
