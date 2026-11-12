#!/usr/bin/env bash
# Copy overlay weights to blockdata; keep the source until smoke passes.
set -euo pipefail
SRC="${1:-/root/models/qwen3-14b-fp8}"
DST="${2:-/root/blockdata/models/qwen3-14b-fp8}"
[[ -d "$SRC" && -f "$SRC/config.json" ]] || { echo "Invalid model source: $SRC" >&2; exit 1; }
[[ ! -e "$DST" && ! -e "$DST.part" ]] || { echo "Destination already exists: $DST or $DST.part" >&2; exit 1; }
[[ -d /root/blockdata ]] || { echo "blockdata is not mounted" >&2; exit 1; }
mkdir -p "$(dirname "$DST")"
need=$(du -sLb "$SRC" | cut -f1)
free=$(df -B1 --output=avail "$(dirname "$DST")" | tail -1 | tr -d ' ')
(( free > need + 2 * 1024 * 1024 * 1024 )) || { echo "Need model size plus 2 GiB free space" >&2; exit 1; }
# Dereference HF snapshot links so the blockdata copy is self contained.
cp -rL -- "$SRC" "$DST.part"
mv -- "$DST.part" "$DST"
echo "Blockdata model ready: $DST (overlay source retained until smoke passes)"
df -h "$DST" /dev/shm
