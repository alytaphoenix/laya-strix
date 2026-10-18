#!/usr/bin/env bash
# Compile the two-subgraph encoder (14 + 14 layers) for the NPU.
#
# The single 28-layer graph fails the AIE scheduler at fused layer 32 on this
# Ryzen AI 1.8 stack.  The split keeps the same weights and maths (max
# probability delta ~0.002 on the fixtures) and fits the NPU's buffers.
#
# Assumes the export exists at artifacts/onnx/english_split_256_layers
# (tools/export_split_layers.py, run on a machine with PyTorch/ONNX).
set -euo pipefail

REPO="${LAYA_STRIX_ROOT:-$HOME/laya-strix}"
VENV="${RYZEN_AI_VENV:-$HOME/ryzenai_1_8/venv}"
SEQ_LEN="${SEQ_LEN:-256}"
MODEL_DIR="${MODEL_DIR:-$REPO/artifacts/onnx/english_split_256_layers}"
A="$MODEL_DIR/encoder_a_sim_nomask.onnx"
B="$MODEL_DIR/encoder_b_sim_nomask.onnx"
OUT_DIR="$REPO/artifacts/ep"
CACHE="$REPO/artifacts/ep-cache"

source /opt/xilinx/xrt/setup.sh
source "$VENV/bin/activate"
source "$REPO/deploy/valhalla.env"
cd "$REPO"

if [ ! -f "$A" ]; then
  echo "== simplifying/rewriting encoder A =="
  python tools/simplify_graph.py \
    --model "$MODEL_DIR/encoder_a.onnx" --out "$MODEL_DIR/encoder_a_sim.onnx" \
    --verify "$MODEL_DIR/fixtures.npz"
  python tools/rewrite_mask.py \
    --model "$MODEL_DIR/encoder_a_sim.onnx" --out "$A" \
    --verify "$MODEL_DIR/fixtures.npz"
fi
if [ ! -f "$B" ]; then
  echo "== simplifying/rewriting encoder B =="
  python tools/simplify_graph.py \
    --model "$MODEL_DIR/encoder_b.onnx" --out "$MODEL_DIR/encoder_b_sim.onnx"
  python tools/rewrite_mask.py \
    --model "$MODEL_DIR/encoder_b_sim.onnx" --out "$B"
fi

mkdir -p "$OUT_DIR" "$CACHE"

echo "== compiling encoder A =="
python tools/compile_ep_context.py \
  --model "$A" --out "$OUT_DIR/english_a_b1_l${SEQ_LEN}_m8.onnx" \
  --config "$REPO/deploy/vai_ep_config.json" \
  --cache-dir "$CACHE" --cache-key "english_a_b1_l${SEQ_LEN}_m8" \
  --iterations 20

echo "== compiling encoder B =="
python tools/compile_ep_context.py \
  --model "$B" --out "$OUT_DIR/english_b_b1_l${SEQ_LEN}_m8.onnx" \
  --config "$REPO/deploy/vai_ep_config.json" \
  --cache-dir "$CACHE" --cache-key "english_b_b1_l${SEQ_LEN}_m8" \
  --iterations 20

echo
echo "Done. Point the service at:"
echo "  LAYA_ENCODER=$OUT_DIR/english_a_b1_l${SEQ_LEN}_m8.onnx"
echo "  LAYA_ENCODER_B=$OUT_DIR/english_b_b1_l${SEQ_LEN}_m8.onnx"
echo "  LAYA_MODEL_DIR=$MODEL_DIR"
echo "  LAYA_HEAD=$MODEL_DIR/head.onnx"
echo "  LAYA_SEQ_LEN=$SEQ_LEN"
