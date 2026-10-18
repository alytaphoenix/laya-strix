#!/usr/bin/env bash
# Tiny CPU smoke run: 8 rows, frozen encoder, one epoch.  Minutes, not hours.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/artifacts/training/example-cpu}"
exec "$REPO/deploy/train_laya.sh" \
  --device cpu \
  --data "$REPO/training/data/example.csv" \
  --out "$OUT" \
  --epochs 1 \
  --micro-batch 2 \
  --grad-accum 1 \
  --freeze-encoder \
  --loss soft-ce \
  --max-len 128 \
  --head-max-len 64
