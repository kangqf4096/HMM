#!/usr/bin/env bash
# L2 length study: how much does the frozen f1 depend on the text length? (docs/l2_length.md)
#   bash scripts/l2_len.sh gpu [/path/to/local/model]   # prompts, 800-token generation, 300-token derivation, builds
#   WORKERS=4 bash scripts/l2_len.sh cpu                # degenerate groups, filter, group A x3, group B x2, report
# Complete steps are skipped; an incomplete output is moved aside as <name>.old1, .old2, ... and the step restarts.
# BASE is the existing 500-token run (data/$BASE, runs/$BASE-*).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
MODE="${1:?usage: bash scripts/l2_len.sh gpu|cpu [model-path]}"
MODEL="${2:-/root/blockdata/models/qwen3-14b-fp8}"
BASE="${BASE:-l2q1}"
WORKERS="${WORKERS:-4}"
R=runs

archive() {
    local d="$1" i=1
    [[ -e "$d" ]] || return 0
    while [[ -e "$d.old$i" ]]; do i=$((i + 1)); done
    mv "$d" "$d.old$i"
    echo "[restart] kept previous output: $d.old$i"
}

# once MARKER OUTPUT command...   skip when MARKER is a non-empty file, else archive OUTPUT and run
once() {
    local marker="$1" out="$2"
    shift 2
    if [[ -s "$marker" ]]; then
        echo "[skip] $out"
        return 0
    fi
    archive "$out"
    echo "[run] $out"
    "$@"
}

py() { python scripts/run.py "$@"; }

gpu() {
    once data/l2L800/manifest.json data/l2L800 \
        py l2-derive prompts --input "data/$BASE" --output data/l2L800 --config configs/text_l2_L800.json
    CONFIG=configs/text_l2_L800.json PROGRESS="$R/l2L800_progress.json" bash scripts/l2_gpu.sh "$MODEL" l2L800
    for side in train test; do
        if [[ -s "$R/$BASE-$side-gen/sources.jsonl" ]]; then
            py l2-derive prefix --input "$R/l2L800-$side-gen" --short "$R/$BASE-$side-gen" \
                --output "$R/l2len-prefix-$side.json" | grep -E '"(all_identical|token_match_rate|articles_compared)"'
        else
            echo "[warn] $R/$BASE-$side-gen/sources.jsonl is missing: prefix check skipped for $side"
        fi
    done
    once data/l2L300/manifest.json data/l2L300 \
        py l2-derive prompts --input data/l2L800 --output data/l2L300 --config configs/text_l2_L300.json
    for side in train test; do
        once "$R/l2L300-$side-gen/metadata.json" "$R/l2L300-$side-gen" \
            py l2-derive sources --input "$R/l2L800-$side-gen" --output "$R/l2L300-$side-gen" --config configs/text_l2_L300.json
    done
    CONFIG=configs/text_l2_L300.json PROGRESS="$R/l2L300_progress.json" bash scripts/l2_gpu.sh "$MODEL" l2L300
    echo "GPU steps done. Next: WORKERS=$WORKERS bash scripts/l2_len.sh cpu"
}

cpu() {
    export CUDA_VISIBLE_DEVICES=""
    once "$R/l2len-degenerate.json" "$R/l2len-degenerate.json" \
        py l2-degenerate --output "$R/l2len-degenerate.json" --input \
        "$R/$BASE-train-pos" "$R/$BASE-test-pos" "$R/l2L800-train-pos" "$R/l2L800-test-pos" \
        "$R/l2L300-train-pos" "$R/l2L300-test-pos"
    for pair in "$BASE:f500" "l2L800:L800" "l2L300:L300"; do
        src="${pair%%:*}"
        cfg="configs/text_l2_${pair##*:}.json"
        for side in train test; do
            once "$R/${src}f-$side-pos/manifest.json" "$R/${src}f-$side-pos" \
                py l2-filter --input "$R/$src-$side-pos" --output "$R/${src}f-$side-pos" \
                --groups "$R/l2len-degenerate.json" --config "$cfg"
        done
    done
    if [[ -d "$R/$BASE-cache" && ! -e "$R/${BASE}f-cache" ]]; then
        cp -r "$R/$BASE-cache" "$R/${BASE}f-cache"   # SeedBS precomputes of unchanged articles are reused by hash
    fi
    for run in "${BASE}f" l2L800f l2L300f; do
        PROGRESS="$R/${run}_progress.json" WEIGHTS="$R/${run}_weights.json" WORKERS="$WORKERS" bash scripts/l2_cpu.sh "$run"
    done
    for run in l2L300f l2L800f; do
        once "$R/$run-reuse/frozen.json" "$R/$run-reuse" \
            py l2-reuse --frozen "$R/${BASE}f-fit" --train "$R/$run-train-pos" --base "$R/$run-fit" --output "$R/$run-reuse"
        once "$R/$run-reuse-test/report.json" "$R/$run-reuse-test" \
            py l2-test --frozen "$R/$run-reuse" --input "$R/$run-test-pos" --output "$R/$run-reuse-test" \
            --workers "$WORKERS" --cache "$R/$run-cache/test-seedbs"
    done
    archive "$R/l2len-report"
    py l2-lengths --runs l2L300f "${BASE}f" l2L800f --reference-tokens 500 \
        --degenerate "$R/l2len-degenerate.json" --output "$R/l2len-report"
    echo "Report: $R/l2len-report/report.md"
}

case "$MODE" in
    gpu) gpu ;;
    cpu) cpu ;;
    *) echo "usage: bash scripts/l2_len.sh gpu|cpu [model-path]" >&2; exit 2 ;;
esac
