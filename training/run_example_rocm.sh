#!/usr/bin/env bash
# ROCm iGPU run.  Stop the big model first (or use --device cpu); the wrapper
# refuses while llama-swap reports a loaded model unless you add
# --allow-gpu-contention.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/artifacts/training/example-rocm}"
exec "$REPO/deploy/train_laya.sh" \
  --device cuda \
  --data "$REPO/training/data/example.jsonl" \
  --out "$OUT" \
  --epochs 1 \
  --micro-batch 1 \
  --grad-accum 4 \
  --max-len 256
