#!/usr/bin/env bash
# Copy weights to the local disk; keep the network copy as a backup.
set -euo pipefail
SRC="${1:-/root/blockdata/HMM/models/qwen3-14b-fp8}"
DST="${2:-/root/models/qwen3-14b-fp8}"
[[ -d "$SRC" && -f "$SRC/config.json" ]] || { echo "Invalid model source: $SRC" >&2; exit 1; }
[[ ! -e "$DST" && ! -e "$DST.part" ]] || { echo "Destination already exists: $DST or $DST.part" >&2; exit 1; }
mkdir -p "$(dirname "$DST")" /root/cache
need=$(du -sLb "$SRC" | cut -f1)
free=$(df -B1 --output=avail "$(dirname "$DST")" | tail -1 | tr -d ' ')
(( free > need + 2 * 1024 * 1024 * 1024 )) || { echo "Need model size plus 2 GiB free space" >&2; exit 1; }
# Dereference HF snapshot links: the local copy must not read remote weight blobs.
cp -rL -- "$SRC" "$DST.part"
mv -- "$DST.part" "$DST"
echo "Local model ready: $DST (network backup retained)"
df -h "$DST" /root/cache
