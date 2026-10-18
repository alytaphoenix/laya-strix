# Benchmarks

Numbers are filled from the 0.1.0 validation run on `valhalla`
(Ryzen AI MAX+ 395, 128 GB, Ubuntu 26.04, XRT 2.25.37, Ryzen AI 1.8,
firmware 1.1.2.65).  Until the NPU compile gate passes, the NPU rows are
marked TBD.

## Method

### NPU decision latency

On valhalla, with the service bound to loopback:

```bash
source /opt/xilinx/xrt/setup.sh
source ~/ryzenai_1_8/venv/bin/activate
source deploy/valhalla.env
python tools/bench.py --url http://127.0.0.1:8751 --requests 30 --warmup 3
```

`tools/bench.py` sends the built-in four-question triage request and reports
min/p50/p95/max plus the service's encoder/head/padding counters from
`/npu/health`.  A single-question gate is measured with `--questions-file`.

### Jev parity

From the Mac, with the CPU reference on `:8753` and an SSH tunnel to the NPU
on `:8751`:

```bash
ssh -N -L 8751:127.0.0.1:8751 llm_server
python tools/parity.py \
  --reference http://127.0.0.1:8753 \
  --candidate http://127.0.0.1:8751 \
  --probes ../forseti/crew/eval/laya-probe.jsonl
```

Gate: no argmax changes, max probability delta <= 0.01, and Forseti's own
`scripts/laya-eval.sh` 9/9 against the NPU endpoint.

### Coexistence with gufo

Load a gufo model through llama-swap (`:8731`), record decode tokens/s and
TTFT, then repeat while `tools/bench.py` drives Laya at ~20 qps.  Also record
`free -g`, `rocm-smi`, and NPU utilization (`xrt-smi`/`amdtop`) before and
after.  Target: <5% decode regression, <10% prefill regression, no OOM over a
one-hour soak.

## Results

| Measurement | Result |
|---|---|
| CPU reference, Forseti 9-probe | 9/9 (M4 CPU, 512-token profile) |
| NPU encoder, single row, 256 tokens | TBD |
| NPU service, four-question triage | TBD |
| NPU encoder coverage | preliminary: 1265/1268 ops, 99.999% GOPs, 2 subgraphs |
| Jev parity vs CPU | TBD (max |dp|, argmax changes) |
| gufo decode before / with Laya @20 qps | TBD |
| gufo prefill before / with Laya @20 qps | TBD |
| Laya RSS | TBD (target <= 2 GB) |

## Notes

- The 512-token profile reached the AIE backend but failed the scheduler with
  `Auto-phase failed ... hardware resources constraint at layer 32`.  The
  256-token profile is the v0.1.0 service profile: same 1265/1268 operator
  coverage, half the GOPs, and enough buffer headroom.  A 512 profile needs
  the encoder split into subgraphs or a compiler configuration that reduces
  layer-32 pressure.
- The head runs on CPU in v0.1.  Its latency is reported separately by
  `/npu/health` and `tools/bench.py`; if it dominates, the next step is
  removing the head's `TopK`/sequence ops and compiling it for the NPU too.
- The v0.1.0 encoder profile is `B=1, L=256, M=8`; short-length buckets and
  batched profiles are expected to reduce padding and improve throughput.
- The CPU reference is not a performance target; it is the correctness
  reference for probabilities and argmax.
