# laya-strix

Laya decision inference on the AMD XDNA2 NPU (Strix Halo), served beside a
larger model running on the iGPU.

```
Jev client / Forseti / hs-jev
        │  POST /v1/systemone  (Jev wire protocol)
        ▼
  laya-npu :8751
        ├── tokenizer + sequence builder + answers/calibration   (CPU)
        ├── ModernBERT-large encoder ── VitisAI EP ── XDNA2 NPU   (BF16)
        └── decision head ── CPU (sub-ms to ~20 ms; see below)
```

The target host is `valhalla` (Ryzen AI MAX+ 395, 128 GB, Ubuntu 26.04,
kernel 7.0, `aie2p` NPU).  The larger model is served by gufo/llama-swap on
ROCm `gfx1151`; Laya keeps the NPU busy and leaves the iGPU to that engine.

## Status

- [x] XRT 2.25.37 + amdxdna 2.25 installed; `xrt-smi examine` sees the NPU
      (firmware 1.1.2.65, `aie2p`, 6x8).
- [x] Static **split** ONNX export: encoder + head, opset 17, batch 1,
      sequence 512, 8 marker slots.  Parity against PyTorch: max
      `|dlogits| = 3.3e-05`.
- [x] Encoder simplified with `onnxsim`: 3667 -> 1280 nodes, all
      `Range`/`CastLike`/`Cos`/`Sin` folded, hidden-state parity 0.  The only
      non-BF16-table ops left are 57 `LayerNormalization`, 28 `Softmax` and
      one `GatherND` (the mask expander).
- [x] CPU end-to-end Jev server (`/v1/systemone`, `/batch`, `/health`,
      `/metrics`, `/npu/health`).  Forseti's 9-probe gate passes 9/9.
- [ ] Ryzen AI 1.8 runtime.  Blocked on AMD's EULA package
      (`ryzen_ai-1.8.0.tgz`, 7.3 GB).
- [ ] Encoder compile to a BF16 EP context + operator-assignment report.
- [ ] NPU latency / coverage gate and coexistence bench with gufo.

The fused export (one graph) works on CPU but contains ONNX sequence ops from
`qkv.unbind` / `chunk`; the split export replaces those with `select` / slices
and keeps the unsupported head ops (`TopK`, sequences) off the NPU.

## Layout

```
src/laya_strix/        service package (config, SplitSession, agent, Jev app)
tools/                 export, graph audit, EP-context compile, parity, bench
deploy/                valhalla install, XRT/memlock helpers, systemd unit
artifacts/             exported ONNX / compiled EP contexts (gitignored)
```

## Export and run locally (CPU)

```bash
uv venv .venv --python 3.12
uv pip install -e '.[export,dev]'

python tools/export_split_static.py --model english \
  --out artifacts/onnx/english_split --seq-len 512 --markers 8 --opset 17

LAYA_MODEL_DIR=$PWD/artifacts/onnx/english_split \
LAYA_PROVIDER=CPUExecutionProvider \
LAYA_PORT=8753 \
laya-npu serve --host 127.0.0.1 --port 8753

curl -s http://127.0.0.1:8753/health
curl -s -X POST http://127.0.0.1:8753/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"Duplicate charge on invoice 4411","questions":{"dept":{"type":"choice","instructions":"Which department?","criteria":{"billing":"payments and refunds","technical":"bugs and outages","other":"everything else"}}}}'
```

Forseti's gate:

```bash
FORSETI_LAYA_URL=http://127.0.0.1:8753 scripts/laya-eval.sh
# 9/9 probes passed
```

## Install the NPU stack on valhalla

1. XRT + amdxdna (already installed here):

   ```bash
   bash deploy/install_xrt.sh
   bash deploy/fix_memlock.sh   # log out / back in
   ```

2. AMD Ryzen AI 1.8 runtime.  Download `ryzen_ai-1.8.0.tgz` from AMD's
   EULA page (AMD account required; the endpoint is unreachable from this
   network) and place it at:

   ```
   /home/alyta/ryzenai_1_8/ryzen_ai-1.8.0.tgz
   ```

   Then:

   ```bash
   bash deploy/install_valhalla.sh
   # or manually, from a new shell:
   source /opt/xilinx/xrt/setup.sh
   source ~/ryzenai_1_8/venv/bin/activate
   source deploy/valhalla.env
   python ~/ryzenai_1_8/venv/quicktest/quicktest.py
   ```

   The installer applies the 26.04 workarounds: uv-managed Python 3.12,
   the peano library path, XRT-before-voe library order, and the
   executable-stack patch.

3. Compile the encoder:

   ```bash
   source /opt/xilinx/xrt/setup.sh
   source ~/ryzenai_1_8/venv/bin/activate
   source deploy/valhalla.env
   python tools/simplify_graph.py \
     --model artifacts/onnx/english_split/encoder.onnx \
     --out artifacts/onnx/english_split/encoder_sim.onnx \
     --verify artifacts/onnx/english_split/fixtures.npz

   python tools/compile_ep_context.py \
     --model artifacts/onnx/english_split/encoder_sim.onnx \
     --out artifacts/ep/english_b1_l512_m8.onnx \
     --config deploy/vai_ep_config.json
   ```

   The script prints the operator-assignment report and refuses a context
   with zero NPU nodes.  If the report shows CPU islands, the next knobs are:
   `optimize_level`, graph surgery for `Cos`/`Sin` (rotary tables), and
   precomputing the 4D attention mask.

4. Serve:

   ```bash
   cp deploy/laya-npu.env.example deploy/laya-npu.env   # edit paths
   sudo cp deploy/laya-npu.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now laya-npu
   ```

## Reaching the Jev endpoint

- **SSH tunnel (default, recommended):**

  ```bash
  ssh -N -L 8751:127.0.0.1:8751 llm_server
  ```

  Point Forseti at `http://127.0.0.1:8751` (stop the Mac's local Laya or use a
  different tunnel port).

- **LAN:** set `LAYA_HOST=0.0.0.0`, set `LAYA_API_KEY`, open only the LAN
  interface in the firewall.  Clients send `Authorization: Bearer ...`.
  `/health` liveness stays open by design; the detail keys need the token.

## Known limits (v1)

- English checkpoint only.  The Router will still try to route non-English
  text to `multilingual`; attach a compiled multilingual agent before enabling
  that traffic.
- One NPU row at a time (`LAYA_BATCH=1`), padded to 512 tokens.  Add compiled
  `B=2/4` and shorter `L` profiles after the first latency measurement.
- The head runs on CPU.  On this M4 it costs ~20 ms per 512-token row; the
  NPU encoder is the remaining cost.  If the head dominates, the next step is
  to remove its `TopK`/sequence ops and compile it for the NPU too.
- `laya-npu` reuses `laya.serve`, so the Jev contract, auth, admission limits
  and answer decoding are the upstream ones; `laya==0.4.11` is pinned.

## Attribution

The static exporter is derived from `NandhaKishorM/laya`'s
`laya-rust/tools/export_onnx.py` (Apache-2.0).  The 26.04 workarounds were
validated against `kotetsuy/ryzenai_1_8`.
