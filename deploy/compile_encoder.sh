#!/usr/bin/env bash
# Compile the simplified English encoder to a BF16 VitisAI EP context model.
# Run on valhalla after install_valhalla.sh.
set -euo pipefail

REPO="${LAYA_STRIX_ROOT:-$HOME/laya-strix}"
VENV="${RYZEN_AI_VENV:-$HOME/ryzenai_1_8/venv}"
SRC="$REPO/artifacts/onnx/english_split/encoder.onnx"
SIM="$REPO/artifacts/onnx/english_split/encoder_sim.onnx"
OUT="$REPO/artifacts/ep/english_b1_l512_m8.onnx"

source /opt/xilinx/xrt/setup.sh
source "$VENV/bin/activate"
source "$REPO/deploy/valhalla.env"
cd "$REPO"

if [ ! -f "$SIM" ]; then
  echo "== simplifying encoder =="
  python tools/simplify_graph.py \
    --model "$SRC" --out "$SIM" \
    --verify "$REPO/artifacts/onnx/english_split/fixtures.npz"
fi

echo "== compiling BF16 EP context =="
python tools/compile_ep_context.py \
  --model "$SIM" \
  --out "$OUT" \
  --config "$REPO/deploy/vai_ep_config.json" \
  --cache-dir "$REPO/artifacts/ep-cache" \
  --cache-key english_b1_l512_m8 \
  --iterations 20
