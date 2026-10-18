#!/usr/bin/env bash
# Install XRT 2.25.37 + the amdxdna DKMS plugin on Ubuntu 26.04 / kernel 7.0.
# Run from a real terminal (sudo).  Safe to re-run.
set -euo pipefail
export LC_ALL=C

ROOT="${RYZENAI_ROOT:-$HOME/ryzenai_1_8}"
ZIP="${XRT_ZIP:-$ROOT/RAI_1.8_Linux_NPU_XRT.zip}"
DIR="$ROOT/RAI_1.8_Linux_NPU_XRT"
URL="https://download.amd.com/opendownload/RyzenAI/Driver/RAI_1.8_Linux_NPU_XRT.zip"

mkdir -p "$ROOT"
if [ ! -f "$ZIP" ]; then
  echo "downloading $URL"
  curl -L --fail --max-time 600 -o "$ZIP" "$URL"
fi
if [ ! -f "$DIR/xrt_202620.2.25.37_24.04-amd64-base.deb" ]; then
  rm -rf "$DIR"
  mkdir -p "$DIR"
  python3 -m zipfile -e "$ZIP" "$DIR"
  # The archive nests one extra directory.
  if [ -d "$DIR/RAI_1.8_Linux_NPU_XRT" ]; then
    mv "$DIR"/RAI_1.8_Linux_NPU_XRT/*.deb "$DIR"/
    rmdir "$DIR/RAI_1.8_Linux_NPU_XRT"
  fi
fi

cd "$DIR"
echo "== dependencies (26.04 names) =="
sudo apt update
sudo apt install -y libboost-filesystem1.83.0 libboost-program-options1.83.0 \
  ocl-icd-opencl-dev uuid-dev

echo "== XRT base / dev / npu =="
sudo apt install --fix-broken -y ./xrt_202620.2.25.37_24.04-amd64-base.deb
sudo apt install --fix-broken -y ./xrt_202620.2.25.37_24.04-amd64-base-dev.deb
sudo apt install --fix-broken -y ./xrt_202620.2.25.37_24.04-amd64-npu.deb

echo "== amdxdna DKMS =="
sudo apt install --fix-broken -y ./xrt_plugin.2.25.260102.56.release_24.04-amd64-amdxdna.deb

echo "== verify =="
dkms status | grep -i amdxdna || true
modinfo amdxdna | grep -E '^(filename|version)' || true
ls -l /dev/accel/ || true
set +u
source /opt/xilinx/xrt/setup.sh
xrt-smi examine 2>&1 | head -40
