#!/usr/bin/env bash
# Install the Ryzen AI 1.8 runtime on valhalla (Ubuntu 26.04 / kernel 7.0).
#
# XRT + amdxdna are already installed by the time this runs (see README).
# The only manual prerequisite is AMD's EULA package:
#
#   ~/ryzenai_1_8/ryzen_ai-1.8.0.tgz
#
# Usage:
#   bash deploy/install_valhalla.sh
#
# Idempotent: an existing venv is reused unless FORCE=1 is set.
set -euo pipefail
export LC_ALL=C

ROOT="${RYZENAI_ROOT:-$HOME/ryzenai_1_8}"
TGZ="${RYZEN_AI_TGZ:-$ROOT/ryzen_ai-1.8.0.tgz}"
VENV="${RYZEN_AI_VENV:-$ROOT/venv}"
SRC="$ROOT/ryzen_ai-1.8.0"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== checks =="
if [ ! -x /opt/xilinx/xrt/bin/xrt-smi ]; then
  echo "ERROR: XRT is not installed. Run the XRT step first." >&2
  exit 2
fi
if [ "$(ulimit -H -l)" != "unlimited" ]; then
  echo "ERROR: memlock hard limit is $(ulimit -H -l); open a new session after deploy/fix_memlock." >&2
  exit 2
fi
if [ ! -f "$TGZ" ]; then
  cat >&2 <<EOF
ERROR: AMD Ryzen AI package not found at:
  $TGZ

Download ryzen_ai-1.8.0.tgz from AMD's EULA page (account required) and put it
there, or set RYZEN_AI_TGZ=/path/to/ryzen_ai-1.8.0.tgz.  Nothing else in this
script needs the file.
EOF
  exit 2
fi

echo "== Python 3.12 =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.12
PYBIN="$(uv python find 3.12)"
export PATH="$(dirname "$PYBIN"):$PATH"
python3.12 --version

if [ "${FORCE:-0}" != "1" ] && [ -x "$VENV/bin/python" ]; then
  echo "== venv already exists: $VENV (FORCE=1 to reinstall) =="
else
  echo "== unpacking $TGZ =="
  rm -rf "$SRC"
  mkdir -p "$SRC"
  tar -xzf "$TGZ" -C "$SRC" --strip-components=0
  if [ ! -x "$SRC/install_ryzen_ai.sh" ]; then
    # Some archives wrap everything in a directory.
    found="$(find "$SRC" -maxdepth 3 -name install_ryzen_ai.sh -print -quit)"
    [ -n "$found" ] || { echo "ERROR: install_ryzen_ai.sh not found in $SRC" >&2; exit 2; }
    SRC="$(dirname "$found")"
  fi
  echo "== installing Ryzen AI into $VENV (this takes a while, ~24 GB) =="
  ( cd "$SRC" && ./install_ryzen_ai.sh -a yes -p "$VENV" )
fi

echo "== clearing the executable-stack bit =="
python3 "$REPO/deploy/fix_execstack.py" "$VENV"

echo "== verification =="
set +u
source /opt/xilinx/xrt/setup.sh
source "$VENV/bin/activate"
source "$REPO/deploy/valhalla.env"
set -u

python - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
print("onnxruntime", ort.__version__)
print("providers", providers)
if "VitisAIExecutionProvider" not in providers:
    raise SystemExit("VitisAIExecutionProvider missing")
PY

echo "== laya-strix server package =="
bash "$REPO/deploy/install_server_deps.sh" "$VENV"

echo
echo "Done.  Next:"
echo "  source /opt/xilinx/xrt/setup.sh"
echo "  source $VENV/bin/activate"
echo "  source $REPO/deploy/valhalla.env"
echo "  python $VENV/quicktest/quicktest.py"
