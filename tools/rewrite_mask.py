#!/usr/bin/env python3
"""Replace the encoder's mask GatherND with a broadcast-friendly Unsqueeze.

The simplified encoder has one ``GatherND`` left.  It gathers the attention
mask with a constant index tensor of ``[0, i]`` pairs, i.e. it is just a
reshape of the [1, L] mask to [1, 1, L] so the following ``And`` broadcasts it
against the [1, 1, L, L] sliding-window mask.  ``GatherND`` is not in AMD's
BF16 operator table; ``Unsqueeze`` and ``And`` are.

    python tools/rewrite_mask.py \
        --model artifacts/onnx/english_split/encoder_sim.onnx \
        --out artifacts/onnx/english_split/encoder_sim_nomask.onnx \
        --verify artifacts/onnx/english_split/fixtures.npz
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


RISKY = {"Cos", "Gelu", "LayerNormalization", "Softmax", "TopK", "Relu",
         "Clip", "CastLike", "GatherND", "GreaterOrEqual", "LessOrEqual",
         "SequenceAt", "SplitToSequence", "IsNaN", "Range"}


def _save(model, out: str):
    for p in (out, out + ".data"):
        if os.path.exists(p):
            os.remove(p)
    onnx.save_model(
        model, out,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(out) + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )


def rewrite(model):
    graph = model.graph
    keep = []
    axes_name = "laya_mask_axes"
    replaced = 0
    for node in graph.node:
        if node.op_type != "GatherND":
            keep.append(node)
            continue
        # data -> [1, L] mask; indices -> constant [1,1,L,2] of [0, i] pairs.
        if len(node.input) != 2:
            raise SystemExit("unexpected GatherND signature: %s" % list(node.input))
        data, indices = node.input
        init = next((t for t in graph.initializer if t.name == indices), None)
        if init is not None:
            arr = numpy_helper.to_array(init)
        else:
            const = next((n for n in graph.node if n.op_type == "Constant"
                          and indices in n.output), None)
            if const is None:
                raise SystemExit("GatherND indices %r are not constant" % indices)
            arr = numpy_helper.to_array(
                next(a for a in const.attribute if a.name == "value").t)
        if arr.ndim < 3 or arr.shape[-1] != 2 or not np.all(arr[..., 0] == 0):
            raise SystemExit("GatherND is not the [0, i] mask gather: %s" % (arr.shape,))
        # output rank is arr.ndim - 1; data is [1, L].  Insert the singleton
        # axes between batch and sequence: [1, L] -> [1, 1, ..., 1, L].
        out_rank = arr.ndim - 1
        axes = list(range(1, out_rank - 1))
        axes_init = numpy_helper.from_array(np.array(axes, dtype=np.int64),
                                            name=axes_name)
        if not any(t.name == axes_name for t in graph.initializer):
            graph.initializer.append(axes_init)
        keep.append(helper.make_node(
            "Unsqueeze", [data, axes_name], list(node.output),
            name=node.name + "_unsqueeze"))
        replaced += 1

    if not replaced:
        print("no GatherND nodes; nothing to do")
        return model, 0

    del graph.node[:]
    graph.node.extend(keep)
    # Drop the now-unused index initializer and its Constant producer.
    used = {i for n in graph.node for i in n.input}
    live = [n for n in graph.node
            if n.op_type != "Constant" or any(o in used for o in n.output)]
    del graph.node[:]
    graph.node.extend(live)
    inits = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(inits)
    return model, replaced


def verify(a_path: str, b_path: str, fixtures: str):
    import onnxruntime as ort

    a = ort.InferenceSession(a_path, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(b_path, providers=["CPUExecutionProvider"])
    L = int(a.get_inputs()[0].shape[1])
    z = np.load(fixtures)
    worst = 0.0
    cases = []
    for g in sorted({k.split("/")[0] for k in z.files}):
        if f"{g}/input_ids" not in z.files:
            continue
        ids, att = z[f"{g}/input_ids"], z[f"{g}/attention_mask"]
        for i in range(ids.shape[0]):
            cases.append((ids[i], att[i]))
    # A trailing-padding case exercises the mask broadcast directly.
    pad_ids = np.zeros(L, dtype=np.int64)
    pad_att = np.zeros(L, dtype=np.int64)
    pad_att[: L // 2] = 1
    cases.append((pad_ids, pad_att))

    for ids, att in cases:
        n = min(int(att.sum()), L)
        xid = np.zeros((1, L), dtype=np.int64)
        xa = np.zeros((1, L), dtype=np.int64)
        xid[0, :n] = ids[:n]
        xa[0, :n] = 1
        ha = a.run(None, {"input_ids": xid, "attention_mask": xa})[0]
        hb = b.run(None, {"input_ids": xid, "attention_mask": xa})[0]
        worst = max(worst, float(np.abs(ha - hb).max()))
    print("hidden-state parity: max |d| %.3e" % worst)
    if worst > 1e-3:
        raise SystemExit("mask rewrite diverges")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default=None)
    args = ap.parse_args(argv)

    model = onnx.load(args.model, load_external_data=True)
    print("loaded %d nodes" % len(model.graph.node))
    model, n = rewrite(model)
    if n:
        _save(model, args.out)
        print("replaced %d GatherND node(s); wrote %s (%.1f MB) + data (%.1f MB)" %
              (n, args.out, os.path.getsize(args.out) / 1e6,
               os.path.getsize(args.out + ".data") / 1e6))
    counts = collections.Counter(x.op_type for x in model.graph.node)
    flagged = {k: v for k, v in counts.items() if k in RISKY}
    print("remaining risky ops: %s" %
          (", ".join("%s=%d" % kv for kv in sorted(flagged.items())) or "none"))
    if args.verify and n:
        verify(args.model, args.out, args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
