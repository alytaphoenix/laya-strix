#!/usr/bin/env bash
# Run the Laya training wrapper in the training venv.
#
#   bash deploy/train_laya.sh --device cpu --data training/data/example.csv \
#       --out artifacts/training/example --epochs 1 --freeze-encoder
#
# The NPU inference venv is not touched.  `--device cuda` means the ROCm iGPU;
# the wrapper refuses while llama-swap has a model resident.
set -euo pipefail

REPO="${LAYA_STRIX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${LAYA_TRAIN_VENV:-$HOME/laya-train-venv}"

if [ ! -x "$VENV/bin/python" ]; then
  echo "train_laya: no training venv at $VENV; run deploy/install_train_deps.sh" >&2
  exit 2
fi

source "$VENV/bin/activate"
export LAYA_STRIX_ROOT="$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m laya_strix.cli train "$@"
