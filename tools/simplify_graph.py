#!/usr/bin/env python3
"""Constant-fold and simplify a static Laya encoder before NPU compilation.

The dynamo export leaves a lot of shape arithmetic, `Range`, `CastLike` and
rotary `Cos`/`Sin` nodes around.  With a static sequence length the position
table is constant, so `onnxsim` folds those away.  On the English encoder this
takes 3667 nodes to 1280 and removes every `CastLike`/`Range`/`Cos`/`Sin`, at
identical logits (max diff ~3e-5 against the PyTorch fixtures).

Usage:
    python tools/simplify_graph.py \
        --model artifacts/onnx/english_split/encoder.onnx \
        --out artifacts/onnx/english_split/encoder_sim.onnx \
        --verify artifacts/onnx/english_split/fixtures.npz
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time


RISKY = {"Cos", "Gelu", "LayerNormalization", "Softmax", "TopK", "Relu",
         "Clip", "CastLike", "GatherND", "GreaterOrEqual", "LessOrEqual",
         "SequenceAt", "SplitToSequence", "IsNaN", "Range"}


def _save(model, out: str):
    import onnx

    for p in (out, out + ".data"):
        if os.path.exists(p):
            os.remove(p)
    onnx.save_model(
        model,
        out,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(out) + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )


def _verify(original: str, simplified: str, fixtures: str):
    import numpy as np
    import onnxruntime as ort

    a = ort.InferenceSession(original, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(simplified, providers=["CPUExecutionProvider"])
    L = int(a.get_inputs()[0].shape[1])
    z = np.load(fixtures)
    worst = 0.0
    for g in sorted({k.split("/")[0] for k in z.files}):
        if f"{g}/input_ids" not in z.files:
            continue
        ids = z[f"{g}/input_ids"]
        att = z[f"{g}/attention_mask"]
        for i in range(ids.shape[0]):
            n = min(int(att[i].sum()), L)
            xid = np.zeros((1, L), dtype=np.int64)
            xa = np.zeros((1, L), dtype=np.int64)
            xid[0, :n] = ids[i, :n]
            xa[0, :n] = 1
            ha = a.run(None, {"input_ids": xid, "attention_mask": xa})[0]
            hb = b.run(None, {"input_ids": xid, "attention_mask": xa})[0]
            d = float(np.abs(ha - hb).max())
            worst = max(worst, d)
    print("  encoder parity: max |dhidden| %.3e" % worst)
    if worst > 1e-3:
        raise SystemExit("simplified graph diverges from the original")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default=None,
                    help="fixtures.npz for an original-vs-simplified check.")
    ap.add_argument("--check", type=int, default=0,
                    help="onnxsim check_n (0 disables the ORT check; faster).")
    args = ap.parse_args(argv)

    import onnx
    import onnxsim

    original = onnx.load(args.model, load_external_data=False)
    t0 = time.perf_counter()
    simplified, ok = onnxsim.simplify(
        args.model, check_n=args.check, perform_optimization=True)
    if not ok:
        raise SystemExit("onnxsim reported failure")
    print("simplified in %.1f s: %d -> %d nodes" %
          (time.perf_counter() - t0, len(original.graph.node),
           len(simplified.graph.node)))
    _save(simplified, args.out)
    print("wrote %s (%.1f MB) + %s (%.1f MB)" %
          (args.out, os.path.getsize(args.out) / 1e6,
           os.path.basename(args.out) + ".data",
           os.path.getsize(args.out + ".data") / 1e6))

    counts = collections.Counter(n.op_type for n in simplified.graph.node)
    flagged = {k: v for k, v in counts.items() if k in RISKY}
    if flagged:
        print("remaining risky ops: %s" % ", ".join("%s=%d" % kv for kv in sorted(flagged.items())))
    else:
        print("no risky ops left")

    if args.verify:
        _verify(args.model, args.out, args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
