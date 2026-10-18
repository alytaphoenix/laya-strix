#!/usr/bin/env bash
# Install the laya-strix Python package into the Ryzen AI venv without
# replacing AMD's onnxruntime build.
#
#   bash deploy/install_server_deps.sh [venv]
#
# The base `laya` wheel needs torch/transformers but not onnxruntime; the
# `laya[onnx]` extra is deliberately not used here.  A constraints file pins
# the already-installed onnxruntime and numpy so pip cannot quietly swap the
# VitisAI EP for PyPI's CPU wheel.
set -euo pipefail

VENV="${1:-${RYZEN_AI_VENV:-$HOME/ryzenai_1_8/venv}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/xilinx/xrt/setup.sh
source "$VENV/bin/activate"
source "$REPO/deploy/valhalla.env"

CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
python - "$CONSTRAINTS" <<'PY'
import importlib.metadata as m
import sys
with open(sys.argv[1], "w") as f:
    for name in ("onnxruntime", "numpy"):
        try:
            f.write("%s==%s\n" % (name, m.version(name)))
        except m.PackageNotFoundError:
            pass
PY
echo "constraints:"; cat "$CONSTRAINTS"

if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "installing CPU torch"
  pip install --index-url https://download.pytorch.org/whl/cpu torch
fi

echo "installing laya + server deps"
pip install -c "$CONSTRAINTS" \
  "laya==0.4.11" "fastapi>=0.110" "uvicorn>=0.27" "numpy>=1.26" \
  "python-multipart>=0.0.9"

echo "installing laya-strix (no dependency changes)"
pip install --no-deps -e "$REPO"

echo "verifying the AMD execution provider survived"
python - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime", ort.__version__, providers)
if "VitisAIExecutionProvider" not in providers:
    raise SystemExit(
        "VitisAIExecutionProvider disappeared.  Reinstall the AMD onnxruntime "
        "from the Ryzen AI package (or rerun install_ryzen_ai.sh), then re-run "
        "this script."
    )
PY

echo "done: $(command -v laya-npu)"
