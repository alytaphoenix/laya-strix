# Training Laya on `valhalla`

Fine-tuning uses the official `laya-train` loop (`laya.train.finetune`) in a
**separate venv** from the NPU inference stack.  The inference venv keeps
AMD's VitisAI `onnxruntime`; the training venv gets a normal PyTorch build.

## Install

```bash
# CPU (safe default; no iGPU contention)
bash deploy/install_train_deps.sh

# ROCm iGPU (gfx1151; PyTorch ROCm 7.0 wheel)
bash deploy/install_train_deps.sh --rocm
```

The venv is `~/laya-train-venv` by default (`LAYA_TRAIN_VENV` overrides it).

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
- For a ROCm wheel that does not recognise gfx1151, try
  `LAYA_HSA_OVERRIDE=11.0.0 ... --device cuda`; drop the override once a newer
  wheel supports the device natively.
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is set by the wrapper.

## After training: put the fine-tune on the NPU

The fine-tuned output is a standard Laya checkpoint directory, so the same
export/compile path works:

```bash
python tools/export_split_static.py \
  --model artifacts/training/my-run \
  --out artifacts/onnx/my-run \
  --seq-len 512 --markers 8

python tools/simplify_graph.py \
  --model artifacts/onnx/my-run/encoder.onnx \
  --out artifacts/onnx/my-run/encoder_sim.onnx \
  --verify artifacts/onnx/my-run/fixtures.npz

python tools/rewrite_mask.py \
  --model artifacts/onnx/my-run/encoder_sim.onnx \
  --out artifacts/onnx/my-run/encoder_sim_nomask.onnx \
  --verify artifacts/onnx/my-run/fixtures.npz

python tools/compile_ep_context.py \
  --model artifacts/onnx/my-run/encoder_sim_nomask.onnx \
  --out artifacts/ep/my-run_b1_l512_m8.onnx
```

Then point `LAYA_MODEL_DIR`/`LAYA_ENCODER`/`LAYA_HEAD` at the new artifacts.

## Notes

- The official loop saves the checkpoint at the end, not per epoch; keep runs
  short or copy the run directory when a long run reaches a useful point.
- `--freeze-encoder` is the fastest way to adapt the heads to a new rubric.
- Full fine-tuning on CPU is slow; use the ROCm venv for real runs, or a
  smaller checkpoint (`multilingual`, 322 M) when memory is tight.
