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
        ├── ModernBERT-large encoder ── VitisAI EP ── XDNA2 NPU (BF16)
        └── decision head ── CPU ONNX Runtime
```

The big model (gufo/llama.cpp under llama-swap) is a separate process on the
ROCm iGPU (`:8731` → `:8732`).  The two processes share unified memory but not
an accelerator: Laya's NPU work does not consume iGPU compute.

## Why split the graph

The fused Laya export contains `SplitToSequence`/`SequenceAt` from
`qkv.unbind` and `chunk`, plus a head `TopK`.  The VitisAI compiler does not
accept ONNX sequence types.  The build therefore traces two graphs:

- `encoder.onnx`: ModernBERT-large, inputs `input_ids`/`attention_mask`,
  output `hidden_states`.  `Tensor.select` replaces `unbind`; two static
  slices replace the GeGLU `chunk`.
- `head.onnx`: Laya's two-layer decision head and typed-answer heads.  It is
  ~100 MB and runs on the CPU; it is not the inference bottleneck.

The NPU gets the 421 M-parameter transformer (99.9% of the FLOPs); the CPU
keeps tokenization, the head, and the calibration/answer math.

## Graph build pipeline

```
export_split_static.py  -> encoder.onnx + head.onnx (opset 17, B=1, L=512, M=8)
simplify_graph.py       -> onnxsim folds shape/rotary arithmetic (3667 -> 1280 nodes)
rewrite_mask.py         -> mask GatherND becomes an Unsqueeze (1279 nodes)
compile_ep_context.py   -> BF16 VitisAI EP context + operator-assignment report
```

The clean encoder contains only standard ONNX ops from AMD's BF16 table plus
`LayerNormalization` and `Softmax` (both handled for BERT-family encoders;
the compile report is the authority).

## Shapes and padding

The compiled encoder is static: one row, 512 tokens, 8 marker slots.  The
session pads each request row to the profile with the tokenizer pad id, zeros
the attention mask for the tail, pads marker metadata with `marker_mask=false`,
and slices logits back to the request's marker count.  `LAYA_SEQ_LEN` /
`LAYA_MARKERS` must match the compiled artifact.

A request with N questions becomes N rows; the session runs them one at a time
(`LAYA_BATCH=1`).  Short-length and batched profiles are the next performance
step; they trade compile time and NPU memory for lower padding.

## Service and data flow

1. `laya.serve.create_app(router)` handles the Jev request validation and the
   single-worker inference gate.
2. `_AttachedOnlyRouter` routes to the attached English agent and refuses to
   build a torch checkpoint behind the NPU service's back.
3. `SplitSession.run` pads, runs the encoder, runs the head, and returns
   `logits`/`act_logits` exactly as the fused session would.
4. `laya.ONNXAgent` decodes typed answers and applies the checkpoint
   temperatures; the wire response is unchanged from `laya-serve`.

## Memory and coexistence

- Encoder: ~1.6 GB BF16 weights + small activations.
- Head: ~0.1 GB, CPU.
- Runtime (torch import + ORT + Python): a few hundred MB.
- The systemd unit caps the service at 4 GB and pins it to a small CPU set,
  leaving the rest of the box to gufo.
- The NPU device is exclusive: one process/context at a time.  The health
  endpoint and logs make a driver/EP failure visible instead of silently
  falling back to CPU.
