# laya-strix 0.1.0

Laya decision inference on the AMD XDNA2 NPU (Strix Halo), served beside a
larger model on the iGPU.

## Highlights

- **NPU-resident Laya encoder.** A fine-tuned 322 M `mmBERT-base` runs as a
  BF16 VitisAI EP context on the XDNA2 NPU; the small decision head stays on
  the CPU.
- **Jev-compatible service.** `/v1/systemone`, `/v1/systemone/batch`,
  `/health`, `/npu/health`, `/metrics`; drop-in for `hs-jev` and Forseti.
- **9/9 Forseti gate on the NPU**, with parity against the same checkpoint on
  CPU (0 argmax changes, max dp 0.0065).
- **Coexistence with gufo.** Pinned to CPUs 0-3, `Nice=10`, 2-thread CPU
  head; gufo decode regression ~4% while Laya serves ~2.6 req/s.
- **Static graph build with unsupported ops removed.** `onnxsim`, the mask
  `GatherND` -> `Unsqueeze` rewrite, and the final LayerNorm moved into the
  CPU head.  The NPU graph has no ONNX sequence ops, `CastLike`, `Range`,
  `Cos`, `Sin`, or `GatherND`.
- **Training loops.** `laya-npu train` wraps the official `laya-train` loop,
  with a separate CPU/ROCm venv, run artifacts, and a llama-swap contention
  guard.  The v0.1.0 checkpoint is reproduced from
  `training/data/forseti_triage.jsonl` + `forseti_confusable_tune.jsonl`.
- **MIT-licensed code.** Laya and its checkpoints remain Apache-2.0; see
  `NOTICE`.

## Validation

| Check | Result |
|---|---|
| XRT / amdxdna on Ubuntu 26.04 | XRT 2.25.37, amdxdna 2.25, firmware 1.1.2.65 |
| Ryzen AI quicktest (CNN) | NPU 398 / CPU 2; actually running on NPU 1 |
| Triage export parity (PyTorch) | max `\|dlogits\| = 1.1e-05` |
| Simplified encoder parity | hidden-state diff 0 |
| VitisAI assignment (322 M, L=256) | 992/997 VAIML, 5 CPU residual; 100% GOPs |
| EP context | 919 MB; compile 165 s |
| CPU Jev gate (Forseti) | 9/9 |
| NPU Jev gate (Forseti) | **9/9** |
| Jev parity, CPU vs NPU | 0 argmax changes; max dp 0.0065 (gate 0.01) |
| One question | p50 131.8 ms; 7.6 req/s |
| Three questions | p50 389.4 ms; 2.6 req/s |
| gufo decode, idle -> with Laya | 33.0 -> 31.7 tok/s (~4% regression) |
| Laya RSS / host memory | 2.4 GB / 39 GB of 122 GB |

Full method and raw counters: `docs/benchmarks.md`.

## Install

See [`docs/install-valhalla.md`](install-valhalla.md).  Short version:

```bash
bash deploy/install_xrt.sh
bash deploy/fix_memlock.sh
bash deploy/install_valhalla.sh        # needs ryzen_ai-1.8.0.tgz
# train/export the triage checkpoint, then:
bash deploy/compile_final.sh
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
bash deploy/install_train_deps.sh   # CPU (supported)
bash training/run_example_cpu.sh
```

The fine-tuned checkpoint is a normal Laya directory; export it with
`--drop-final-norm`, simplify/rewrite the mask, and run `deploy/compile_final.sh`.
See [`docs/training.md`](training.md).

## Known limits

- The shipped encoder is the **322 M multilingual** fine-tune, attached under
  `english,multilingual`.  The 421 M English ModernBERT-large BF16 graph does
  not fit the current AIE scheduler (layer-32 resource failure at L=512/256/
  128 and with the 14+14 split); Quark A8W8/A16W8 quantization was not
  accurate enough.  The English experiments remain for a future stack.
- One NPU row per forward, padded to 256 tokens; no short-length/batched
  profiles yet.  Requests truncate at 256 and report `usage.truncated`.
- The decision head runs on the CPU (~20 ms/row).
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
