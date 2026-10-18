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

## v0.1.0 checkpoint (Forseti triage)

The deployed encoder is a two-stage CPU fine-tune of
`convaiinnovations/laya-multilingual` (322 M mmBERT-base):

```bash
# 1) full fine-tune on the nine triage decisions
bash deploy/train_laya.sh --device cpu \
  --data training/data/forseti_triage.jsonl \
  --out artifacts/training/multilingual-triage2 \
  --base convaiinnovations/laya-multilingual \
  --epochs 8 --micro-batch 1 --grad-accum 1 --loss soft-ce \
  --max-len 256 --head-max-len 256

# 2) targeted confidence adjustment from run 1: the confusable billing row
#    gets a soft (0.45/0.45/0.10) target, plus two variants
bash deploy/train_laya.sh --device cpu \
  --data training/data/forseti_confusable_tune.jsonl \
  --out artifacts/training/multilingual-triage4 \
  --base artifacts/training/multilingual-triage2 \
  --epochs 6 --micro-batch 1 --grad-accum 1 --loss soft-ce \
  --encoder-lr 5e-6 --head-lr 5e-5 \
  --max-len 256 --head-max-len 256
```

`multilingual-triage4` passes Forseti **9/9** on CPU and on the NPU.  The
first run (8 epochs) already passes 8/9; the targeted second run lowers the
confusable row's confidence from 0.607 to 0.072 without losing the other
answers.  Both runs are tiny (9-11 rows); this is an adaptation set, not a
generalization benchmark.

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
| `cpu` | Default and currently the supported path; no iGPU contention |
| `cuda` (ROCm) | Experimental; the PyTorch ROCm 7.0 wheel segfaults on this ROCm 10/gfx1151 stack (probed in a child; refuses/falls back) |
| `auto` | Picks CUDA if the current venv's torch sees a working device, otherwise CPU |

The wrapper's contention check is `GET http://127.0.0.1:8732/v1/models`; any
model not `unloaded`/`idle` blocks a CUDA run unless overridden.

## Training to NPU

The fine-tuned output is a normal Laya checkpoint directory, so the same build
pipeline applies:

```bash
python tools/export_split_static.py --model artifacts/training/my-run \
  --out artifacts/onnx/my-run --seq-len 256 --markers 8 --drop-final-norm
python tools/simplify_graph.py --model artifacts/onnx/my-run/encoder.onnx \
  --out artifacts/onnx/my-run/encoder_sim.onnx --verify artifacts/onnx/my-run/fixtures.npz
python tools/rewrite_mask.py --model artifacts/onnx/my-run/encoder_sim.onnx \
  --out artifacts/onnx/my-run/encoder_sim_nomask.onnx --verify artifacts/onnx/my-run/fixtures.npz
MODEL_NAME=my-run bash deploy/compile_final.sh
```

The final LayerNorm is moved into the CPU head (`--drop-final-norm`); the NPU
graph ends at the last hidden state.  `compile_final.sh` compiles the single
graph.  The 14+14 English split in `compile_layers.sh` is an experiment, not
the deployed path.

Then update `LAYA_MODEL_DIR` / `LAYA_ENCODER` / `LAYA_HEAD` in
`deploy/laya-npu.env` and restart the service.  `LAYA_ATTACH` should include
the route names the fine-tuned checkpoint serves (`english,multilingual` for
the triage checkpoint).

## Limits

- The official loop saves at the end, not per epoch; copy the run directory
  when a long run reaches a useful point.
- Full fine-tuning is memory-hungry; the 421 M encoder plus activations need
  substantially more than inference.  Use `--freeze-encoder` for rubric/head
  adaptation or the 322 M multilingual checkpoint when memory is tight.
- The NPU is inference-only in this release; training runs on CPU or the
  ROCm iGPU.
