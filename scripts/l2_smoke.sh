#!/usr/bin/env bash
# CPU-only pipeline check with the toy language model (never a result about OPT).
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/run.py l2-stages --config configs/text_l2.json
python scripts/run.py l2-synth --config configs/text_l2_smoke.json --output "${1:-runs/l2-synth}"
