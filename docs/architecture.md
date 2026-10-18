# Architecture

`laya-strix` runs Laya's non-autoregressive decision encoder on the AMD XDNA2
NPU while a larger model keeps the iGPU.

```
Jev client / Forseti / hs-jev
        │  POST /v1/systemone
        ▼
  laya-npu :8751
        │
        ├── laya.serve: Jev contract, auth, admission, batch, health
        ├── tokenizer + sequence builder + typed-answer decoding/calibration (CPU)
        ├── fine-tuned 322 M mmBERT encoder ── VitisAI EP ── XDNA2 NPU (BF16)
        └── decision head ── CPU ONNX Runtime
```

The big model (gufo/llama.cpp under llama-swap) is a separate process on the
ROCm iGPU (`:8731` → `:8732`).  The two processes share unified memory but not
an accelerator: Laya's NPU work does not consume iGPU compute.  The service is
pinned to CPUs 0-3 with a 2-thread head and `Nice=10`, leaving the rest of the
box to gufo.

## Why encoder + head, and why not English ModernBERT-large

The fused Laya export contains `SplitToSequence`/`SequenceAt` from
`qkv.unbind` and `chunk`, plus a head `TopK`.  The VitisAI compiler does not
accept ONNX sequence types.  The build therefore traces two graphs:

- `encoder.onnx`: the transformer, inputs `input_ids`/`attention_mask`,
  output `hidden_states`.
- `head.onnx`: Laya's two-layer decision head and typed-answer heads.  It is
  ~60 MB and runs on the CPU; it is not the inference bottleneck.

The deployed v0.1.0 encoder is the **322 M multilingual `mmBERT-base`**
checkpoint, fine-tuned on the English triage decisions.  The 421 M English
ModernBERT-large BF16 graph does not fit the current XDNA2 AIE scheduler: it
fails at fused layer 32 at `L=512`, 256 and 128, with a 14+14 layer split,
and with the L3/spilling/optimisation knobs.  Quark A8W8/A16W8 quantization
was not accurate enough for Laya's calibrated answers.  The 322 M graph fits
at `L=256` and passes Forseti's 9/9 gate after the fine-tune; the English
experiments remain in `deploy/compile_layers.sh` and `tools/quantize_a8w8.py`
for a future compiler/driver.

## Graph build pipeline

```
train (CPU)             -> multilingual-triage4 checkpoint
export_split_static.py  -> encoder.onnx + head.onnx (opset 17, B=1, L=256, M=8)
simplify_graph.py       -> onnxsim folds shape/rotary arithmetic
rewrite_mask.py         -> mask GatherND becomes an Unsqueeze
compile_final.sh        -> BF16 VitisAI EP context + operator-assignment report
```

The encoder's final LayerNorm is moved into the CPU head (`--drop-final-norm`)
so the NPU graph ends at the last hidden state.  The clean encoder has only
standard ONNX ops from AMD's BF16 table plus `LayerNormalization` and
`Softmax` (both handled for BERT-family encoders; the compile report is the
authority).  The deployed context is 919 MB; the compile reports 992/997
nodes on VAIML and 5 CPU residual nodes, with 100% GOPs.

## Shapes and padding

The compiled encoder is static: one row, **256 tokens**, 8 marker slots.  The
session pads each request row to the profile with the tokenizer pad id, zeros
the attention mask for the tail, pads marker metadata with `marker_mask=false`,
and slices logits back to the request's marker count.  `LAYA_SEQ_LEN` /
`LAYA_MARKERS` must match the compiled artifact, and `agent.cfg["max_len"]`
is pinned to `LAYA_SEQ_LEN` so `usage.truncated` is honest.

A request with N questions becomes N rows; the session runs them one at a time
(`LAYA_BATCH=1`).  Short-length and batched profiles are the next performance
step; the three-question body uses ~82 tokens per row against 256, so padding
is the main latency lever.

## Service and data flow

1. `laya.serve.create_app(router)` handles the Jev request validation and the
   single-worker inference gate.
2. `_AttachedOnlyRouter` routes to the attached agent and refuses to build a
   torch checkpoint behind the NPU service's back.  `LAYA_ATTACH` lists the
   names the single compiled agent is attached under (`english,multilingual`
   for the deployed fine-tune).
3. `SplitSession.run` pads, runs the NPU encoder, runs the CPU head, and
   returns `logits`/`act_logits` exactly as the fused session would.
4. `laya.ONNXAgent` decodes typed answers and applies the checkpoint
   temperatures; the wire response is unchanged from `laya-serve`.

## Memory and coexistence

- Encoder context: 919 MB BF16; head ~60 MB; runtime (torch + ORT + Python)
  ~1.4 GB.  Service RSS measured at 2.4 GB.
- The systemd unit caps the service at 4 GB (`MemoryHigh`) / 5 GB
  (`MemoryMax`), pins `CPUAffinity=0-3`, sets `Nice=10`, and limits the head
  to two ORT threads.
- The NPU device is exclusive: one process/context at a time.  The health
  endpoint and logs make a driver/EP failure visible instead of silently
  falling back to CPU.
- The pinned profile leaves gufo's CPU sampling intact; the measured decode
  regression is ~4% with Laya at ~2.6 req/s.
