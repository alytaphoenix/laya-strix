# Install on valhalla (Strix Halo, Ubuntu 26.04)

Target: AMD Ryzen AI MAX+ 395, Radeon 8060S, XDNA2 NPU (`aie2p`, 6x8),
128 GB unified memory, Ubuntu 26.04, kernel 7.0.  The big model (gufo) runs on
ROCm `gfx1151`; this stack owns the NPU.

## 1. XRT and the amdxdna driver

```bash
bash deploy/install_xrt.sh
bash deploy/fix_memlock.sh     # then open a new SSH session
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

Expected: `NPU Strix Halo`, `aie2p`, 6x8, and a firmware version.  The
installer uses XRT 2.25.37 from
`https://download.amd.com/opendownload/RyzenAI/Driver/RAI_1.8_Linux_NPU_XRT.zip`
and the matching `amdxdna` DKMS plugin.  `fix_memlock.sh` writes PAM and
systemd limits; XRT's 64 MB `MAP_LOCKED` heap otherwise fails with `EAGAIN`.

## 2. Ryzen AI 1.8 runtime

Download the Linux package (7.3 GB) from AMD and place it at
`~/ryzenai_1_8/ryzen_ai-1.8.0.tgz`.  The public CDN path at the time of
writing was:

```text
https://download.amd.com/opendownload/RyzenAI/1.8.0/ryzen_ai-1.8.0.tgz
sha256 7532d511532dd8842995bc8dc2edd8a45fbf261312ea9abe8dd01b4d502ac98d
```

The package is subject to AMD's EULA; verify the SHA before running it.

```bash
bash deploy/install_valhalla.sh
```

The script applies the Ubuntu 26.04 workarounds:

- Python 3.12 from `uv` on `PATH` (the installer hard-codes `python3.12`;
  a symlink is not enough because the venv needs the stdlib next to the
  interpreter).
- The `libboost-filesystem1.83.0` / `libboost-program-options1.83.0` names
  (the docs' 1.74 is stale).
- Peano's library path, needed by `libonnxruntime_vitisai_ep.so`.
- XRT before the venv's `voe/lib` so the stale `libxrt_coreutil` is not
  loaded.
- The executable-stack bit on three Ryzen AI ELFs.
- `zip`, which the installer's post-check expects.

Verify:

```bash
source /opt/xilinx/xrt/setup.sh
source ~/ryzenai_1_8/venv/bin/activate
source deploy/valhalla.env
python ~/ryzenai_1_8/venv/quicktest/quicktest.py 2>&1 | grep -E 'NPU|Actually|Finished'
```

Expected: `NPU 398`, `VITIS_EP_CPU 2`, `Actually running on NPU 1`.

## 3. Server dependencies

`install_valhalla.sh` calls this automatically; it can be rerun:

```bash
bash deploy/install_server_deps.sh
```

It installs CPU torch, `laya==0.4.11`, FastAPI/uvicorn, and `laya-strix` into
the Ryzen AI venv.  The script pins the installed `onnxruntime`/`numpy` with a
constraints file and asserts that `VitisAIExecutionProvider` survives — a
stray PyPI `onnxruntime` would replace AMD's build.

## 4. Build the deployed checkpoint

v0.1.0 serves a **fine-tuned 322 M multilingual Laya encoder** (`mmBERT-base`)
at a 256-token profile.  The 421 M English ModernBERT-large BF16 graph does
not fit the current XDNA2 AIE scheduler: it fails at fused layer 32
(`Auto-phase failed ... hardware resources constraint`) at L=512, 256 and 128,
with the 14+14 layer split, and with the L3/spilling/optimisation config
knobs.  Quark A8W8/A16W8 quantization was not accurate enough for Laya's
calibrated answers.  The 322 M encoder fits, and after the English triage
fine-tune it passes Forseti's 9/9 gate.

### Train (CPU)

```bash
source ~/laya-train-venv/bin/activate   # deploy/install_train_deps.sh

# 1) full fine-tune of the multilingual base on the triage decisions
bash deploy/train_laya.sh --device cpu \
  --data training/data/forseti_triage.jsonl \
  --out artifacts/training/multilingual-triage2 \
  --base convaiinnovations/laya-multilingual \
  --epochs 8 --micro-batch 1 --grad-accum 1 --loss soft-ce \
  --max-len 256 --head-max-len 256

# 2) targeted abstention/confidence adjustment from run 1
bash deploy/train_laya.sh --device cpu \
  --data training/data/forseti_confusable_tune.jsonl \
  --out artifacts/training/multilingual-triage4 \
  --base artifacts/training/multilingual-triage2 \
  --epochs 6 --micro-batch 1 --grad-accum 1 --loss soft-ce \
  --encoder-lr 5e-6 --head-lr 5e-5 \
  --max-len 256 --head-max-len 256
```

`multilingual-triage4` is the v0.1.0 checkpoint.  Run the CPU gate:

```bash
# start a CPU service from the exported ONNX (below), or use laya.load()
```

The ROCm training wheel is experimental on this box's ROCm 10 stack; CPU is
the supported path.  See `docs/training.md`.

### Export

Run on a machine with PyTorch/ONNX (the Mac dev venv is fine), then copy the
directory to `valhalla`:

```bash
python tools/export_split_static.py \
  --model artifacts/training/multilingual-triage4 \
  --out artifacts/onnx/multilingual_triage4_single \
  --seq-len 256 --markers 8 --opset 17 --drop-final-norm

python tools/simplify_graph.py \
  --model artifacts/onnx/multilingual_triage4_single/encoder.onnx \
  --out artifacts/onnx/multilingual_triage4_single/encoder_sim.onnx \
  --verify artifacts/onnx/multilingual_triage4_single/fixtures.npz

python tools/rewrite_mask.py \
  --model artifacts/onnx/multilingual_triage4_single/encoder_sim.onnx \
  --out artifacts/onnx/multilingual_triage4_single/encoder_sim_nomask.onnx \
  --verify artifacts/onnx/multilingual_triage4_single/fixtures.npz
```

### Compile

```bash
bash deploy/compile_final.sh
```

It compiles the single encoder to
`artifacts/ep/multilingual_triage4_b1_l256_m8.onnx` and writes
`artifacts/ep-cache/multilingual_triage4_b1_l256_m8_report.json`.  Expect
~99.7% operator coverage and 100% GOPs on the NPU.

The English split/quantization experiments remain in the repo for a future
compiler/driver: `deploy/compile_layers.sh` (14+14 layers; fails at layer 32),
`deploy/vai_ep_config_l1.json` / `_l3.json` / `_l3b.json` (optimisation,
1 GB L3 and spilling variants; all fail), and `tools/quantize_a8w8.py`
(accuracy not acceptable).  They are not the v0.1.0 deployment path.

## 5. Run the service

```bash
cp deploy/laya-npu.env.example deploy/laya-npu.env
# edit paths if the repo is not ~/laya-strix
sudo cp deploy/laya-npu.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now laya-npu
systemctl status laya-npu --no-pager
curl -s http://127.0.0.1:8751/health
curl -s http://127.0.0.1:8751/npu/health
```

Default bind is loopback.  For LAN access set `LAYA_HOST=0.0.0.0` and
`LAYA_API_KEY`, and open only the LAN interface in the firewall.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `xrt-smi`: `mmap ... EAGAIN` | `deploy/fix_memlock.sh`; new SSH session; `ulimit -H -l` = unlimited |
| `import onnxruntime`: executable stack | `python3 deploy/fix_execstack.py ~/ryzenai_1_8/venv`; rerun after reinstall |
| `VitisAIExecutionProvider` missing | source `/opt/xilinx/xrt/setup.sh`, `venv/bin/activate`, `deploy/valhalla.env`; check the peano and XRT paths |
| `quicktest` prints `Test Finished` but no NPU line | missing peano path or stale `libxrt_coreutil`; use `deploy/valhalla.env` |
| Compile says zero NPU nodes | check the report; the fallback is decomposing `LayerNormalization`/`Softmax` or precomputing the 4D mask |
| Service exits: EP unavailable | run `install_server_deps.sh` and verify providers; do not set `LAYA_ALLOW_CPU_FALLBACK` in production |
| Port 8751 already used | stop the Mac/local `laya-serve`, or change `LAYA_PORT` and the tunnel |
