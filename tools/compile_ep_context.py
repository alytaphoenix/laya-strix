#!/usr/bin/env python3
"""Compile a static Laya ONNX graph to a VitisAI BF16 EP context model.

Run this on the target (valhalla) inside the Ryzen AI 1.8 venv, with XRT's
``setup.sh`` sourced and the two extra library paths from deploy/valhalla.env.
The first session creation does the compile and can take minutes; the EP
context file it writes is what the service loads at runtime.

    source /opt/xilinx/xrt/setup.sh
    source ~/ryzenai_1_8/venv/bin/activate
    source deploy/valhalla.env
    python tools/compile_ep_context.py \
        --model artifacts/onnx/english/model.onnx \
        --out artifacts/ep/english_b1_l512_m8.onnx

The operator assignment report is written next to the cache and printed on
success; use it to confirm the encoder actually landed on the NPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _dtype(dtype: str):
    import numpy as np

    return {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
        "tensor(float)": np.float32,
        "tensor(bfloat16)": np.float32,
    }[dtype]


def _feed(session):
    """Build a valid dummy feed for the graph's static input signature."""
    import numpy as np

    feeds = {}
    for inp in session.get_inputs():
        shape = [1 if isinstance(d, str) else d for d in inp.shape]
        if inp.name == "attention_mask":
            arr = np.ones(shape, dtype=_dtype(inp.type))
        elif inp.name == "marker_mask":
            arr = np.ones(shape, dtype=_dtype(inp.type))
        else:
            arr = np.zeros(shape, dtype=_dtype(inp.type))
        feeds[inp.name] = arr
    return feeds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Static ONNX graph.")
    ap.add_argument("--out", required=True, help="EP context output path.")
    ap.add_argument("--config", default="deploy/vai_ep_config.json")
    ap.add_argument("--cache-dir", default="artifacts/ep-cache")
    ap.add_argument("--cache-key", default=None,
                    help="Cache subdirectory (default: output basename).")
    ap.add_argument("--report-file", default=None,
                    help="Operator assignment JSON (default: cache dir/vitisai_ep_report.json).")
    ap.add_argument("--no-embed", action="store_true",
                    help="Do not embed the context in the ONNX file.")
    ap.add_argument("--iterations", type=int, default=10)
    args = ap.parse_args(argv)

    import onnxruntime as ort

    providers = ort.get_available_providers()
    print("onnxruntime:", ort.__version__)
    print("providers:", providers)
    if "VitisAIExecutionProvider" not in providers:
        raise SystemExit(
            "VitisAIExecutionProvider is not available.  This is usually one of:\n"
            "  - the AMD onnxruntime wheel was replaced by PyPI's (reinstall Ryzen AI)\n"
            "  - deploy/valhalla.env was not sourced (libpeano / XRT paths)\n"
        )

    model = os.path.abspath(args.model)
    out = os.path.abspath(args.out)
    cache_dir = os.path.abspath(args.cache_dir)
    cache_key = args.cache_key or os.path.splitext(os.path.basename(out))[0]
    report = os.path.abspath(args.report_file or os.path.join(cache_dir, "vitisai_ep_report.json"))
    config = os.path.abspath(args.config)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = report
    # The report is written into the cache directory; make sure it is not stale.
    if os.path.exists(report):
        os.remove(report)

    so = ort.SessionOptions()
    so.log_severity_level = 1
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", out)
    so.add_session_config_entry("ep.context_embed_mode", "0" if args.no_embed else "1")

    options = {
        "config_file": config,
        "cache_dir": cache_dir,
        "cache_key": cache_key,
        "enable_cache_file_io_in_mem": 0,
    }

    print("\ncompiling %s" % model)
    print("  out:       %s" % out)
    print("  config:    %s" % config)
    print("  cache:     %s/%s" % (cache_dir, cache_key))
    print("  report:    %s" % report)
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        model,
        sess_options=so,
        providers=["VitisAIExecutionProvider"],
        provider_options=[options],
    )
    elapsed = time.perf_counter() - t0
    print("\ncompiled in %.1f s" % elapsed)
    print("session providers:", session.get_providers())
    for i in session.get_inputs():
        print("  in  %-16s %s %s" % (i.name, i.type, i.shape))
    for o in session.get_outputs():
        print("  out %-16s %s %s" % (o.name, o.type, o.shape))

    feeds = _feed(session)
    logits, act = session.run(None, feeds)
    print("output shapes: logits=%s act=%s" % (logits.shape, act.shape))
    times = []
    for _ in range(args.iterations):
        t = time.perf_counter()
        session.run(None, feeds)
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    print("latency: p50 %.1f ms  p95 %.1f ms  min %.1f ms  (n=%d)" %
          (times[len(times) // 2], times[int(len(times) * 0.95)],
           times[0], len(times)))

    if not os.path.exists(out):
        raise SystemExit("session succeeded but no EP context was written to %s" % out)
    print("context: %.1f MB" % (os.path.getsize(out) / 1e6))

    if os.path.exists(report):
        with open(report) as f:
            data = json.load(f)
        print("\noperator assignment:")
        for row in data.get("deviceStat", []):
            print("  %-6s nodes=%s" % (row.get("name"), row.get("nodeNum")))
            print("         %s" % ", ".join(row.get("supportedOpType", [])[:12]))
        devices = {row.get("name"): row.get("nodeNum") for row in data.get("deviceStat", [])}
        if devices.get("NPU", 0) == 0:
            raise SystemExit("report says zero nodes on the NPU; do not deploy this context")
    else:
        print("WARNING: no operator assignment report at %s" % report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
