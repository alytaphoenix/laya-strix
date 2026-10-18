# Training Laya on `valhalla`

Fine-tuning uses the official `laya-train` loop (`laya.train.finetune`) in a
**separate venv** from the NPU inference stack.  The inference venv keeps
AMD's VitisAI `onnxruntime`; the training venv gets a normal PyTorch build.

## Install

```bash
# CPU (supported; no iGPU contention)
bash deploy/install_train_deps.sh

# Experimental ROCm iGPU (gfx1151; PyTorch ROCm 7.0 wheel)
bash deploy/install_train_deps.sh --rocm
```

The venv is `~/laya-train-venv` by default (`LAYA_TRAIN_VENV` overrides it).

> **ROCm note (2026-10-18).** The `torch 2.10.0+rocm7.0` wheel reports
> `gfx1151` but segfaults on the first kernel against this box's ROCm 10
> stack. `install_train_deps.sh --rocm` probes it in a child process and warns;
> `laya-npu train` does the same and refuses `--device cuda` (or falls back to
> CPU under `--device auto`).  CPU training is the supported path until a
> torch build matches the installed ROCm.

## Quick smoke run

```bash
bash training/run_example_cpu.sh
```

This trains only the decision heads on eight rows for one epoch, with
`--max-len 128 --head-max-len 64`, so it finishes in minutes on CPU.  The
checkpoint lands in `artifacts/training/example-cpu/`.

## Data

CSV — one text column and one label column; Laya synthesizes a choice question
from the distinct labels:

```csv
text,label
"Duplicate charge on March invoice",billing
"The API returns 502 after the deploy",technical
```

JSONL — explicit `state`, `questions`, and `expected` (or `gold`) rows:

```json
{"state": "Duplicate charge on March invoice",
 "questions": {"department": {"type": "choice", "instructions": "Which department?",
   "criteria": {"billing": "refunds", "technical": "bugs", "other": "everything else"}}},
 "expected": {"department": "billing"}}
```

`expected` accepts a choice label, a boolean for `noul`, or a level index/name
for `score`.  `gold` accepts a full probability distribution:

```json
{"gold": {"department": {"probabilities": {"billing": 0.8, "technical": 0.15, "other": 0.05}}}}
```

Examples: [`training/data/example.csv`](data/example.csv) and
[`training/data/example.jsonl`](data/example.jsonl).  The full schema is in
Laya's `docs/finetune.md` and `laya/train.py`.

## Run

```bash
bash deploy/train_laya.sh \
  --device cpu \
  --data training/data/example.csv \
  --out artifacts/training/my-run \
  --epochs 4 \
  --micro-batch 8 \
  --grad-accum 8 \
  --loss rlcd \
  --eval-data training/data/example.csv
```

Every `laya-train` option passes through: `--freeze-encoder`, `--loss soft-ce`,
`--label-smoothing`, `--shuffle-options`, `--target-error`, `--min-abstain-n`,
`--max-len`, `--head-max-len`, `--seed`, `--dry-run`, etc.

A run directory contains:

- `model.safetensors`, `rl_agent_config.json`, `tokenizer/` — a checkpoint
  loadable with `laya.load("<run-dir>")`;
- `train.log` — console output;
- `run.json` — command, device, torch/HIP versions, git revision, timing.

## Device and coexistence

- **Default `--device cpu`.**  It cannot evict gufo or Flash-Next.
- **`--device cuda`** means the ROCm iGPU.  The wrapper checks
  `http://127.0.0.1:8732/v1/models` and refuses while llama-swap reports a
  loaded model.  Stop the model, or pass `--allow-gpu-contention` if you have
  measured the memory headroom.
- The ROCm 7.0 wheel can report a gfx1151 device and then segfault on the
  first kernel against this ROCm 10 stack.  `laya-npu train` probes in a
  child process and refuses `--device cuda` (or falls back under `--device
  auto`); CPU is the supported path.
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is set by the wrapper.

## v0.1.0 triage checkpoint

The deployed encoder is the 322 M multilingual base after two CPU runs:

```bash
bash deploy/train_laya.sh --device cpu \
  --data training/data/forseti_triage.jsonl \
  --out artifacts/training/multilingual-triage2 \
  --base convaiinnovations/laya-multilingual \
  --epochs 8 --micro-batch 1 --grad-accum 1 --loss soft-ce \
  --max-len 256 --head-max-len 256

bash deploy/train_laya.sh --device cpu \
  --data training/data/forseti_confusable_tune.jsonl \
  --out artifacts/training/multilingual-triage4 \
  --base artifacts/training/multilingual-triage2 \
  --epochs 6 --micro-batch 1 --grad-accum 1 --loss soft-ce \
  --encoder-lr 5e-6 --head-lr 5e-5 \
  --max-len 256 --head-max-len 256
```

`multilingual-triage4` passes Forseti's 9/9 gate on CPU and on the NPU.  The
first run passes 8/9; the second (soft-target confusable row plus two
variants) drops the confusable confidence from 0.607 to 0.072.  This is a
small adaptation set, not a generalization claim.  Full recipe:
[`../docs/training.md`](../docs/training.md).

## After training: put the fine-tune on the NPU

The fine-tuned output is a standard Laya checkpoint directory, so the same
export/compile path works:

```bash
python tools/export_split_static.py \
  --model artifacts/training/my-run \
  --out artifacts/onnx/my-run \
  --seq-len 256 --markers 8 --drop-final-norm

python tools/simplify_graph.py \
  --model artifacts/onnx/my-run/encoder.onnx \
  --out artifacts/onnx/my-run/encoder_sim.onnx \
  --verify artifacts/onnx/my-run/fixtures.npz

python tools/rewrite_mask.py \
  --model artifacts/onnx/my-run/encoder_sim.onnx \
  --out artifacts/onnx/my-run/encoder_sim_nomask.onnx \
  --verify artifacts/onnx/my-run/fixtures.npz

MODEL_NAME=my-run MODEL_DIR=$PWD/artifacts/onnx/my-run \
  bash deploy/compile_final.sh
```

Then point `LAYA_MODEL_DIR`/`LAYA_ENCODER`/`LAYA_HEAD` at the new artifacts
and set `LAYA_ATTACH` to the route names it serves.

## Notes

- The official loop saves the checkpoint at the end, not per epoch; keep runs
  short or copy the run directory when a long run reaches a useful point.
- `--freeze-encoder` is the fastest way to adapt the heads to a new rubric.
- Full fine-tuning on CPU is slow; the ROCm 7.0 wheel is experimental on this
  box's ROCm 10/gfx1151 stack (the wrapper probes/refuses).  The 322 M
  multilingual checkpoint is the practical target; `--freeze-encoder` is the
  fastest way to adapt the heads.
