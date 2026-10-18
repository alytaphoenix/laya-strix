#!/usr/bin/env bash
# Compile the simplified English encoder to a BF16 VitisAI EP context model.
# Run on valhalla after install_valhalla.sh.
set -euo pipefail

REPO="${LAYA_STRIX_ROOT:-$HOME/laya-strix}"
VENV="${RYZEN_AI_VENV:-$HOME/ryzenai_1_8/venv}"
SEQ_LEN="${SEQ_LEN:-256}"
MODEL_DIR="${MODEL_DIR:-$REPO/artifacts/onnx/english_split_256}"
SRC="$MODEL_DIR/encoder.onnx"
SIM="$MODEL_DIR/encoder_sim.onnx"
NOMASK="$MODEL_DIR/encoder_sim_nomask.onnx"
OUT="$REPO/artifacts/ep/english_b1_l${SEQ_LEN}_m8.onnx"
CACHE_KEY="english_b1_l${SEQ_LEN}_m8"

source /opt/xilinx/xrt/setup.sh
source "$VENV/bin/activate"
source "$REPO/deploy/valhalla.env"
cd "$REPO"

if [ ! -f "$SIM" ]; then
  echo "== simplifying encoder =="
  python tools/simplify_graph.py \
    --model "$SRC" --out "$SIM" \
    --verify "$MODEL_DIR/fixtures.npz"
fi
if [ ! -f "$NOMASK" ]; then
  echo "== replacing mask GatherND =="
  python tools/rewrite_mask.py \
    --model "$SIM" --out "$NOMASK" \
    --verify "$MODEL_DIR/fixtures.npz"
fi

echo "== compiling BF16 EP context ($CACHE_KEY) =="
python tools/compile_ep_context.py \
  --model "$NOMASK" \
  --out "$OUT" \
  --config "$REPO/deploy/vai_ep_config.json" \
  --cache-dir "$REPO/artifacts/ep-cache" \
  --cache-key "$CACHE_KEY" \
  --iterations 20
