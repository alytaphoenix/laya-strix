#!/usr/bin/env bash
# Launch laya-npu with the Ryzen AI + XRT library paths.
# Used by deploy/laya-npu.service; can also be run by hand.
set -euo pipefail

REPO="${LAYA_STRIX_ROOT:-$HOME/laya-strix}"
VENV="${RYZEN_AI_VENV:-$HOME/ryzenai_1_8/venv}"

set +u
source /opt/xilinx/xrt/setup.sh
source "$VENV/bin/activate"
source "$REPO/deploy/valhalla.env"
set -u

exec laya-npu serve \
  --host "${LAYA_HOST:-127.0.0.1}" \
  --port "${LAYA_PORT:-8751}"
