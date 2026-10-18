#!/usr/bin/env python3
"""Inventory an ONNX graph before handing it to the VitisAI compiler.

Prints the opset, input/output signatures, operator counts and the initializer
size, then flags operators that are not in AMD's Ryzen AI 1.8 BF16 support
table.  A flagged op is not automatically fatal: VitisAI partitions the graph
and runs unsupported nodes on the CPU, but a CPU island in the middle of the
encoder is slow, so this is the list to design around.

Usage:
    python tools/audit_graph.py artifacts/onnx/english/model.onnx
"""

from __future__ import annotations

import argparse
import collections
import os
import sys


# Ryzen AI 1.8 "Supported Operators" table, BF16 column.
# Source: https://ryzenai.docs.amd.com/en/latest/ops_support.html
BF16_OPS = {
    "Abs", "Add", "And", "ArgMax", "ArgMin", "AveragePool", "BatchNormalization",
    "BitShift", "BitwiseAnd", "BitwiseNot", "BitwiseOr", "BitwiseXor", "Cast",
    "Ceil", "Concat", "Constant", "ConstantOfShape", "Conv", "ConvTranspose",
    "CumSum", "DepthToSpace", "Div", "Einsum", "Equal", "Erf", "Exp", "Expand",
    "Flatten", "Floor", "Gather", "GatherElements", "Gemm", "GlobalAveragePool",
    "GridSample", "GroupConv", "Identity", "InstanceNormalization", "LSTM",
    "Less", "Log", "MatMul", "Max", "MaxPool", "Min", "Mod", "Mul", "Neg",
    "Not", "Or", "Pad", "Pow", "Range", "Reciprocal", "ReduceMax", "ReduceMean",
    "ReduceMin", "ReduceSum", "Reshape", "Resize", "Round", "ScatterND", "Shape",
    "Sign", "Sin", "Size", "Slice", "Split", "Sqrt", "Squeeze", "Sub", "Tanh",
    "Tile", "Transpose", "Unsqueeze", "Upsample", "Where", "Xor",
}
# Ops that show up in the table for other precisions but not BF16.  Keep them
# separate: they are the usual reasons a transformer graph needs a CPU tail.
NOT_BF16 = {"Cos", "Gelu", "LayerNormalization", "Softmax", "TopK", "Relu",
            "Sigmoid", "Clip", "LeakyRelu", "HardSigmoid", "HardSwish",
            "GroupNormalization", "QLinearConv", "QLinearMatMul"}


def _dim(d):
    if d.dim_param:
        return d.dim_param
    if d.HasField("dim_value"):
        return d.dim_value
    return "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--external-data", action="store_true",
                    help="Load external weight data (slow; only needed for size reporting).")
    args = ap.parse_args(argv)

    import onnx

    if not os.path.exists(args.model):
        raise SystemExit("no such model: %s" % args.model)

    model = onnx.load(args.model, load_external_data=args.external_data)
    graph = model.graph

    print("model: %s (%.1f MB)" % (args.model, os.path.getsize(args.model) / 1e6))
    print("opset: %s" % ", ".join("%s=%d" % (o.domain or "ai.onnx", o.version)
                                   for o in model.opset_import))
    print("producer: %s %s" % (model.producer_name, model.producer_version))

    print("\ninputs:")
    for i in graph.input:
        if i.name in {x.name for x in graph.initializer}:
            continue
        t = i.type.tensor_type
        print("  %-16s %-14s %s" % (i.name, onnx.TensorProto.DataType.Name(t.elem_type),
                                     [_dim(d) for d in t.shape.dim]))
    print("outputs:")
    for o in graph.output:
        t = o.type.tensor_type
        print("  %-16s %-14s %s" % (o.name, onnx.TensorProto.DataType.Name(t.elem_type),
                                     [_dim(d) for d in t.shape.dim]))

    counts = collections.Counter(n.op_type for n in graph.node)
    domains = collections.Counter((n.domain or "ai.onnx") for n in graph.node)
    print("\noperators (%d nodes, domains: %s):" % (len(graph.node), dict(domains)))
    for op, n in counts.most_common():
        mark = ""
        if op not in BF16_OPS:
            mark = "  <-- not in BF16 table" if op in NOT_BF16 else "  <-- unknown"
        print("  %-22s %4d%s" % (op, n, mark))

    init_bytes = 0
    for t in graph.initializer:
        init_bytes += t.ByteSize()
    print("\ninitializers: %d tensors, %.1f MB inline" %
          (len(graph.initializer), init_bytes / 1e6))
    data = args.model + ".data"
    if os.path.exists(data):
        print("external data: %s (%.1f MB)" % (data, os.path.getsize(data) / 1e6))

    unknown = sorted({n.op_type for n in graph.node
                      if n.op_type not in BF16_OPS and n.op_type not in NOT_BF16})
    risky = sorted({n.op_type for n in graph.node if n.op_type in NOT_BF16})
    if risky:
        print("\nBF16-risky ops: %s" % ", ".join(risky))
    if unknown:
        print("unclassified ops: %s" % ", ".join(unknown))
    if not risky and not unknown:
        print("\nall operators are in the BF16 table")
    return 0


if __name__ == "__main__":
    sys.exit(main())
