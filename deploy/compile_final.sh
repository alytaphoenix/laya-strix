#!/usr/bin/env bash
# Compile the v0.1.0 deployed checkpoint (the fine-tuned 322 M multilingual
# encoder) to a single BF16 EP context model.
#
# Prerequisite: the exported artifacts at
# artifacts/onnx/multilingual_triage4_single/ (encoder_sim_nomask.onnx,
# head.onnx, tokenizer, rl_agent_config.json).  See docs/install-valhalla.md.
set -euo pipefail

REPO="${LAYA_STRIX_ROOT:-$HOME/laya-strix}"
VENV="${RYZEN_AI_VENV:-$HOME/ryzenai_1_8/venv}"
SEQ_LEN="${SEQ_LEN:-256}"
MODEL_NAME="${MODEL_NAME:-multilingual_triage4}"
MODEL_DIR="${MODEL_DIR:-$REPO/artifacts/onnx/${MODEL_NAME}_single}"
OUT="${OUT:-$REPO/artifacts/ep/${MODEL_NAME}_b1_l${SEQ_LEN}_m8.onnx}"
CACHE_KEY="${CACHE_KEY:-${MODEL_NAME}_b1_l${SEQ_LEN}_m8}"

source /opt/xilinx/xrt/setup.sh
source "$VENV/bin/activate"
source "$REPO/deploy/valhalla.env"
cd "$REPO"

python tools/compile_ep_context.py \
  --model "$MODEL_DIR/encoder_sim_nomask.onnx" \
  --out "$OUT" \
  --config "$REPO/deploy/vai_ep_config.json" \
  --cache-dir "$REPO/artifacts/ep-cache" \
  --cache-key "$CACHE_KEY" \
  --iterations 10

echo
echo "Compiled: $OUT"
echo "  LAYA_MODEL_DIR=$MODEL_DIR"
echo "  LAYA_ENCODER=$OUT"
echo "  LAYA_HEAD=$MODEL_DIR/head.onnx"
echo "  LAYA_SEQ_LEN=$SEQ_LEN"
