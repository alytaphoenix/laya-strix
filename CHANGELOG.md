# Changelog

## 0.1.0 — 2026-10-18

First release of the Laya-on-XDNA2 deployment.

### Added

- **NPU service.** A fine-tuned 322 M `mmBERT-base` Laya encoder runs as a
  BF16 VitisAI EP context on the XDNA2 NPU; the decision head runs on the CPU.
- **Jev-compatible API.** `/v1/systemone`, `/v1/systemone/batch`, `/health`,
  `/npu/health`, `/metrics`, strict mode, bearer auth and the upstream
  `laya.serve` contract.  The single compiled agent is attached under
  `english,multilingual` (`LAYA_ATTACH`).
- **Static graph build.** Export (`--drop-final-norm`), `onnxsim`, mask
  `GatherND` -> `Unsqueeze`, and a BF16 EP-context compile.  The deployed
  context is 919 MB; 992/997 nodes run on VAIML.
- **Training loops.** `laya-npu train` wraps the official `laya-train` loop,
  with a separate CPU/ROCm venv, `run.json`/`train.log`, a CPU-default device
  policy, a llama-swap contention guard, and a child-process ROCm probe.
- **v0.1.0 checkpoint recipe.** `training/data/forseti_triage.jsonl` +
  `forseti_confusable_tune.jsonl` fine-tune the multilingual base; the result
  passes Forseti 9/9 on CPU and NPU.
- **Deployment.** XRT/amdxdna installer, memlock/execstack helpers, Ryzen AI
  1.8 installer, systemd unit (`CPUAffinity=0-3`, `Nice=10`, 4/5 GB memory
  bounds, 2-thread CPU head), `valhalla.env`, `compile_final.sh`.
- **Benchmarks/parity.** `tools/bench.py`, `tools/parity.py`; one question
  p50 131.8 ms, three questions p50 389.4 ms, max dp 0.0065, gufo decode
  regression ~4%.

### Changed

- The service pins `agent.cfg["max_len"]` to the compiled `LAYA_SEQ_LEN` so
  `usage.truncated` is honest.
- ORT provider options are passed only to the VitisAI EP; the CPU EP is added
  by ORT, avoiding the equal-length options error.
- `valhalla.env` adds the uv Python libdir for `libflexmlrt`'s
  `libpython3.12.so.1.0`.
- `compile_ep_context.py` accepts the `VAIML` device name in the report and
  handles encoder graphs with one output.

### Known limits

- The 421 M English ModernBERT-large BF16 graph does not fit the current AIE
  scheduler (fused layer 32) at L=512/256/128, with a 14+14 layer split, or
  with the L3/spilling/optimisation knobs.  Quark A8W8/A16W8 quantization was
  not accurate enough (max dp 0.14-0.80).  The shipped encoder is the 322 M
  multilingual fine-tune.
- One NPU row per forward, 256-token profile; no short-length/batched
  profiles.  Padding is the main latency lever.
- The head is CPU-side (~20 ms/row).
- Training is CPU-first; the PyTorch ROCm 7.0 wheel segfaults on this box's
  ROCm 10/gfx1151 stack.  The wrapper probes in a child and fails over or
  refuses instead of crashing mid-run.
- No TLS in the service; use an SSH tunnel or reverse proxy off-host.
