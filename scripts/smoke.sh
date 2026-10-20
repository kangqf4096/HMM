#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODEL_PATH="${1:-/home/MCEC_ADAS/kdz/models/Qwen2.5-Coder-3B-Instruct}"
RUN_NAME="${2:-arxiv0}"
python scripts/clean_wiki.py
python scripts/test.py
python scripts/run.py check-runtime
python scripts/run.py prepare --config configs/text_smoke.json --model "$MODEL_PATH" --output "data/$RUN_NAME"
python scripts/run.py generate --config configs/text_smoke.json --model "$MODEL_PATH" --prompts "data/$RUN_NAME" --output "runs/$RUN_NAME-gen"
python scripts/run.py evaluate --input "runs/$RUN_NAME-gen" --output "runs/$RUN_NAME-eval"
