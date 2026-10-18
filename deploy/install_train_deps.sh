#!/usr/bin/env bash
# Create a separate training venv for Laya fine-tuning.
#
#   bash deploy/install_train_deps.sh          # CPU (safe default)
#   bash deploy/install_train_deps.sh --rocm   # ROCm 7.0 wheel for gfx1151
#
# This venv is deliberately independent of the Ryzen AI NPU venv: training
# pulls a GPU torch build, while inference keeps AMD's onnxruntime wheel.
set -euo pipefail

ROCM=0
if [ "${1:-}" = "--rocm" ]; then
  ROCM=1
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${LAYA_TRAIN_VENV:-$HOME/laya-train-venv}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.12
uv venv --python 3.12 "$VENV"
source "$VENV/bin/activate"

if [ "$ROCM" = "1" ]; then
  echo "== installing torch ROCm 7.0 (gfx1151) =="
  uv pip install --index-url https://download.pytorch.org/whl/rocm7.0 torch
else
  echo "== installing CPU torch =="
  uv pip install --index-url https://download.pytorch.org/whl/cpu torch
fi

echo "== installing laya training deps =="
uv pip install "laya==0.4.11"
uv pip install --no-deps -e "$REPO"

echo "== verification =="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("cuda_available", torch.cuda.is_available())
PY

if [ "$ROCM" = "1" ]; then
  # The ROCm 7.0 wheel can report a device and then segfault on the first
  # kernel against a newer ROCm/gfx1151 stack.  Probe in a child process.
  python - <<'PY'
import subprocess, sys
code = ("import torch\n"
        "x = torch.randn(16, 16, device='cuda')\n"
        "y = x @ x\n"
        "torch.cuda.synchronize()\n"
        "print('ok', float(y[0, 0]))\n")
p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print("gpu probe rc", p.returncode)
print((p.stdout or p.stderr).strip()[-400:])
if p.returncode != 0:
    print("\nWARNING: this ROCm wheel cannot run a kernel on this box.")
    print("Use the CPU training venv (bash deploy/install_train_deps.sh) for now.")
PY
fi

echo
echo "Training venv ready: $VENV"
echo "  deploy/train_laya.sh --help"
echo "  deploy/train_laya.sh --device cpu --data training/data/example.csv --out artifacts/training/example"
if [ "$ROCM" = "1" ]; then
  echo
  echo "If the wheel does not recognise gfx1151, try:"
  echo "  LAYA_HSA_OVERRIDE=11.0.0 deploy/train_laya.sh --device cuda ..."
fi
