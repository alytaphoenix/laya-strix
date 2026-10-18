# laya-strix 0.1.0

Laya decision inference on the AMD XDNA2 NPU (Strix Halo), served beside a
larger model on the iGPU.

## Highlights

- **NPU-resident Laya encoder.** ModernBERT-large runs as a BF16 VitisAI EP
  context on the XDNA2 NPU; the small decision head stays on the CPU.
- **Jev-compatible service.** `/v1/systemone`, `/v1/systemone/batch`,
  `/health`, `/npu/health`, `/metrics`; drop-in for `hs-jev` and Forseti.
- **Static graph build with the unsupported ops removed.** The sender exports
  a split encoder/head, folds shape/rotary arithmetic with `onnxsim`, and
  rewrites the mask `GatherND`.  The compiled graph has no ONNX sequence ops,
  `CastLike`, `Range`, `Cos`, `Sin`, or `GatherND`.
- **Training loops.** `laya-npu train` wraps the official `laya-train` loop,
  with a separate CPU/ROCm venv, timestamped run directories, and a
  llama-swap contention guard.
- **MIT-licensed code.** Laya and its checkpoints remain Apache-2.0; see
  `NOTICE`.

## Validation

| Check | Result |
|---|---|
| XRT / amdxdna on Ubuntu 26.04 | XRT 2.25.37, amdxdna 2.25, firmware 1.1.2.65 |
| Ryzen AI quicktest (CNN) | NPU 398 / CPU 2; actually running on NPU 1 |
| Encoder export parity | max `|dlogits| = 3.3e-05` |
| Simplified encoder parity | hidden-state diff 0 |
| VitisAI preliminary coverage | 1265 / 1268 ops, 99.999% GOPs, 2 subgraphs |
| CPU Jev gate (Forseti) | 9/9 |
| NPU latency / coexistence | see `docs/benchmarks.md` |

## Install

See [`docs/install-valhalla.md`](install-valhalla.md).  Short version:

```bash
bash deploy/install_xrt.sh
bash deploy/fix_memlock.sh
bash deploy/install_valhalla.sh        # needs ryzen_ai-1.8.0.tgz
bash deploy/compile_encoder.sh
cp deploy/laya-npu.env.example deploy/laya-npu.env
sudo cp deploy/laya-npu.service /etc/systemd/system/
sudo systemctl enable --now laya-npu
```

## Jev quick check

```bash
curl -s http://127.0.0.1:8751/health
curl -s http://127.0.0.1:8751/npu/health
curl -s -X POST http://127.0.0.1:8751/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"Duplicate charge on invoice 4411","questions":{"dept":{"type":"choice","instructions":"Which department?","criteria":{"billing":"payments","technical":"bugs","other":"everything else"}}}}'
```

## Training

```bash
bash deploy/install_train_deps.sh   # CPU
bash deploy/install_train_deps.sh --rocm
bash training/run_example_cpu.sh
```

The fine-tuned checkpoint is a normal Laya directory; export/compile it with
the same tools.  See [`docs/training.md`](training.md).

## Known limits

- English checkpoint only in the shipped service; the router refuses to build
  a torch multilingual checkpoint behind the NPU process.
- One NPU row per forward, padded to 256 tokens; no short-length/batched
  profiles yet.
- The decision head runs on the CPU.
- Training defaults to CPU; the ROCm 7.0 torch wheel is experimental on this
  box's ROCm 10/gfx1151 stack (the wrapper probes and refuses/falls back), and
  the CUDA path refuses while llama-swap has a model resident.
- The service has no TLS; use an SSH tunnel or a reverse proxy off-host.

## Artifacts

```bash
make dist
# dist/laya_strix-0.1.0-py3-none-any.whl
# dist/laya_strix-0.1.0.tar.gz
```

## License

MIT.  Laya runtime/checkpoints Apache-2.0; AMD Ryzen AI/XRT are installed
from AMD and not redistributed.  See `LICENSE` and `NOTICE`.
