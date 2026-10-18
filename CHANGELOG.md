# Changelog

## 0.1.0 — unreleased

First release of the Laya-on-XDNA2 deployment.

### Added

- Static encoder/head ONNX export for the English Laya checkpoint
  (`B=1`, `L=512`, 8 marker slots, opset 17), with PyTorch fixture parity.
- Graph cleanup: `onnxsim` folds shape/rotary arithmetic; a targeted rewrite
  replaces the mask `GatherND` with `Unsqueeze`.  The compiled encoder has no
  ONNX sequence ops and no `CastLike`/`Range`/`Cos`/`Sin`/`GatherND`.
- `SplitSession`: static-shape padding, NPU encoder + CPU head, single-flight
  locking, latency/padding counters.
- Jev-compatible service: `/v1/systemone`, `/v1/systemone/batch`, `/health`,
  `/npu/health`, `/metrics`; English-only router that refuses to build a
  torch checkpoint behind the NPU service's back.
- `laya-npu` CLI: `serve` and `train`.
- `laya-npu train`: wrapper around the official `laya-train` loop with a
  timestamped run directory, `run.json`, `train.log`, CPU-default device
  policy, llama-swap contention guard, and ROCm allocator defaults.
- Deployment: XRT/amdxdna installer, memlock and execstack helpers, Ryzen AI
  runtime installer, systemd unit, `valhalla.env`, compile runner.
- Benchmarks and parity tools: `tools/bench.py`, `tools/parity.py`.

### Known limits

- English checkpoint only; multilingual/typed-decisions need a compiled,
  attached agent.
- One NPU row per forward (`B=1`), padded to 512 tokens; no short-length
  buckets yet.
- The decision head runs on the CPU (~ms to ~20 ms per 512-token row).
- Training and inference share the box; the default training device is CPU and
  the ROCm path refuses while llama-swap has a model resident.
