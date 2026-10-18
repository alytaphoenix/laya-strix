# Training Laya on valhalla

The operational guide is [`training/README.md`](../training/README.md); this
page records the design and the hand-off to the NPU.

## Design

- Training uses the **official** `laya.train.finetune` loop via
  `laya-train`.  `laya-npu train` is a wrapper, not a second trainer.
- Training has its **own venv** (`~/laya-train-venv`) so a PyTorch ROCm/CPU
  build can coexist with the Ryzen AI venv's AMD `onnxruntime`.
- The default device is **CPU** so a training run cannot evict gufo from the
  iGPU.  `--device cuda` is opt-in and refused while llama-swap reports a
  loaded model (override: `--allow-gpu-contention`).
- Every run gets a timestamped directory with `run.json` (command, device,
  torch/HIP, git revision, timings) and `train.log`.
- ROCm runs set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`;
  `LAYA_HSA_OVERRIDE` sets `HSA_OVERRIDE_GFX_VERSION` for wheels that do not
  recognise gfx1151.

## Quick start

```bash
bash deploy/install_train_deps.sh          # CPU
bash training/run_example_cpu.sh           # 8 rows, frozen encoder, one epoch
```

Output: `artifacts/training/example-cpu/` with a checkpoint that loads with
`laya.load(...)`.

## Data

- CSV: `text,label`; Laya synthesizes one choice question from the labels.
- JSONL: `{state, questions, expected}` (hard label) or
  `{state, questions, gold}` (probability distribution).
- `choice`, `score`, and `noul` are supported; multiple questions per state
  are supported.

See `training/data/` and Laya's `docs/finetune.md`.

## Device and coexistence

| Device | Use |
|---|---|
| `cpu` | Default; no iGPU contention; slow for full fine-tunes |
| `cuda` (ROCm) | gfx1151 iGPU; stop gufo first or pass `--allow-gpu-contention` after measuring |
| `auto` | Picks CUDA if the current venv's torch sees it, otherwise CPU |

The wrapper's contention check is `GET http://127.0.0.1:8732/v1/models`; any
model not `unloaded`/`idle` blocks a CUDA run unless overridden.

## Training to NPU

The fine-tuned output is a normal Laya checkpoint directory, so the same build
pipeline applies:

```bash
python tools/export_split_static.py --model artifacts/training/my-run \
  --out artifacts/onnx/my-run --seq-len 512 --markers 8
python tools/simplify_graph.py --model artifacts/onnx/my-run/encoder.onnx \
  --out artifacts/onnx/my-run/encoder_sim.onnx --verify artifacts/onnx/my-run/fixtures.npz
python tools/rewrite_mask.py --model artifacts/onnx/my-run/encoder_sim.onnx \
  --out artifacts/onnx/my-run/encoder_sim_nomask.onnx --verify artifacts/onnx/my-run/fixtures.npz
python tools/compile_ep_context.py --model artifacts/onnx/my-run/encoder_sim_nomask.onnx \
  --out artifacts/ep/my-run_b1_l512_m8.onnx
```

Then update `LAYA_MODEL_DIR` / `LAYA_ENCODER` / `LAYA_HEAD` in
`deploy/laya-npu.env` and restart the service.

## Limits

- The official loop saves at the end, not per epoch; copy the run directory
  when a long run reaches a useful point.
- Full fine-tuning is memory-hungry; the 421 M encoder plus activations need
  substantially more than inference.  Use `--freeze-encoder` for rubric/head
  adaptation or the 322 M multilingual checkpoint when memory is tight.
- The NPU is inference-only in this release; training runs on CPU or the
  ROCm iGPU.
