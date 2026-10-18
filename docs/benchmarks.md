# Benchmarks

Measured on `valhalla` (Ryzen AI MAX+ 395, 128 GB unified memory, Ubuntu
26.04, XRT 2.25.37, Ryzen AI 1.8, NPU firmware 1.1.2.65) with the v0.1.0
fine-tuned 322 M encoder.

## Method

### NPU decision latency

On valhalla, with the service bound to loopback:

```bash
python3 tools/bench.py --url http://127.0.0.1:8751 --requests 20 --warmup 2
python3 tools/bench.py --url http://127.0.0.1:8751 --requests 20 --warmup 2 \
  --questions-file /tmp/one_q.json
```

The default body has three questions; `/tmp/one_q.json` is a single choice
question.  `tools/bench.py` reports min/p50/p95/max plus the service's
encoder/head/padding counters from `/npu/health`.

### Jev parity

The CPU reference and the NPU service must run the **same checkpoint**.  The
reference is a CPU service built from the same exported triage4 ONNX on the
Mac (`:8756`); the NPU is reached through an SSH tunnel (`:8754`):

```bash
ssh -N -L 8754:127.0.0.1:8751 llm_server
python tools/parity.py \
  --reference http://127.0.0.1:8756 \
  --candidate http://127.0.0.1:8754 \
  --probes ../forseti/crew/eval/laya-probe.jsonl
```

Gate: no argmax changes, max probability delta <= 0.01, and Forseti's
`scripts/laya-eval.sh` 9/9 against the NPU endpoint.

### Coexistence with gufo

`gufo-27b-q4` was loaded through llama-swap (`:8731`), warmed with a cached
prompt, and decoded for 89 tokens (`max_tokens=128`).  Baseline was measured
idle; then `tools/bench.py` drove Laya for 80 requests (three questions each)
while five gufo requests ran.  Laya is pinned to CPUs 0-3 with a 2-thread CPU
head and `Nice=10`; gufo keeps the rest.

## Results

| Measurement | Result |
|---|---|
| CPU reference (same checkpoint), Forseti 9-probe | 9/9 |
| NPU service, Forseti 9-probe | 9/9 |
| NPU encoder, one row, 256 tokens | p50 `109 ms` (service average; compile probe 110 ms) |
| NPU head, one row | p50 `~20 ms` (CPU, 2 threads) |
| NPU service, one question | p50 `131.8 ms`, p95 `132.7 ms`, `7.6 req/s` |
| NPU service, three questions | p50 `389.4 ms`, p95 `391.2 ms`, `2.6 req/s` |
| NPU encoder coverage | 992/997 nodes on VAIML, 5 CPU residual; 100% GOPs |
| EP context | 919 MB; compile 165 s |
| Jev parity vs CPU | 0 argmax changes; max dp `0.0065` (gate 0.01); PASS |
| gufo decode, idle | `33.0 tok/s` (89 tokens, cached prompt) |
| gufo decode, Laya @ ~2.6 req/s | `31.7 tok/s` (89 tokens); **~4% regression** |
| gufo TTFT, cached prompt | `3.3 ms` |
| Laya service RSS | `2.4 GB` |
| Host memory during test | `39 GB used / 122 GB` (`82 GB` available) |
| Laya p50 under gufo load | `385.6 ms` (three questions; within noise of idle) |

## Notes

- The deployed v0.1.0 encoder is the fine-tuned 322 M multilingual
  `mmBERT-base`.  The 421 M English ModernBERT-large BF16 graph does not fit
  the current AIE scheduler: it fails at fused layer 32 at L=512, 256 and
  128, with a 14+14 layer split, and with the L3/spilling/optimisation config
  knobs.  Quark A8W8/A16W8 quantization was not accurate enough for Laya's
  calibrated answers (max probability deltas 0.14-0.80).  See
  `docs/install-valhalla.md`.
- Padding is the biggest latency lever: the three-question body uses ~82
  tokens per row against a 256-token profile (`padding_ratio` ~3.0).  A
  validated 128-token or bucketed profile should roughly halve the encoder
  time.  The 322 M graph has not been compiled at 128 yet.
- The head is CPU-side and currently ~20 ms/row with two threads.  It is
  small; compiling it for the NPU would remove that cost but is not a v0.1.0
  blocker.
- The CPU reference is not a performance target; it is the correctness
  reference for probabilities and argmax.
- The earlier unpinned Laya service (all cores, no thread cap) slowed gufo
  decode by ~6x under load.  Pinning to 0-3, `Nice=10`, and a 2-thread head
  reduced the regression to ~4%; the systemd unit carries those settings.
