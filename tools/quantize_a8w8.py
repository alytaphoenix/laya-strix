#!/usr/bin/env python3
"""Quantize the English encoder to AMD Quark A8W8 (int8 activations/weights).

The 421 M ModernBERT encoder does not fit the XDNA2 BF16 scheduler at L=256
(`AIE auto-phase ... hardware resources constraint at layer 32`), even split
into 14-layer graphs.  The 322 M multilingual encoder fits.  A8W8 halves the
activation/weight buffers, so the English graph can fit at the cost of
quantization error; the Forseti/parity gates decide whether it is acceptable.

    source ~/quark-venv/bin/activate
    python tools/quantize_a8w8.py \
      --model artifacts/onnx/english_split_256_nonorm/encoder_sim_nomask.onnx \
      --out artifacts/onnx/english_split_256_nonorm/encoder_a8w8.onnx \
      --fixtures artifacts/onnx/english_split_256_nonorm/fixtures.npz

The calibration reader uses the PyTorch fixture sequences (real Laya rows),
padded to the compiled profile.  Add `--random N` for extra synthetic rows.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


class Reader:
    def __new__(cls, feeds):
        from onnxruntime.quantization import CalibrationDataReader

        class _Reader(CalibrationDataReader):
            def __init__(self, feeds):
                self.feeds = feeds
                self.i = 0

            def get_next(self):
                if self.i >= len(self.feeds):
                    return None
                f = self.feeds[self.i]
                self.i += 1
                return f

            def rewind(self):
                self.i = 0

        return _Reader(feeds)


def _fixture_feeds(path, seq_len, pad_id=0):
    z = np.load(path)
    feeds = []
    for g in sorted({k.split("/")[0] for k in z.files}):
        if f"{g}/input_ids" not in z.files:
            continue
        ids = z[f"{g}/input_ids"]
        att = z[f"{g}/attention_mask"]
        for i in range(ids.shape[0]):
            n = min(int(att[i].sum()), seq_len)
            xid = np.full((1, seq_len), pad_id, dtype=np.int64)
            xa = np.zeros((1, seq_len), dtype=np.int64)
            xid[0, :n] = ids[i, :n]
            xa[0, :n] = 1
            feeds.append({"input_ids": xid, "attention_mask": xa})
    return feeds


def _random_feeds(n, seq_len, rng):
    feeds = []
    for _ in range(n):
        L = int(rng.integers(max(8, seq_len // 4), seq_len + 1))
        ids = rng.integers(0, 30000, size=(1, seq_len), dtype=np.int64)
        att = np.zeros((1, seq_len), dtype=np.int64)
        att[0, :L] = 1
        ids[0, L:] = 0
        feeds.append({"input_ids": ids, "attention_mask": att})
    return feeds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--random", type=int, default=8)
    ap.add_argument("--pad-id", type=int, default=0)
    ap.add_argument("--config", choices=["a8w8", "a16w8"], default="a8w8")
    ap.add_argument("--matmul-only", action="store_true",
                    help="Quantize MatMul only; leave every other operator in float.")
    args = ap.parse_args(argv)

    from quark.onnx import A8W8_QCONFIG, A16W8_QCONFIG, ModelQuantizer
    base_config = A8W8_QCONFIG if args.config == "a8w8" else A16W8_QCONFIG

    feeds = []
    if args.fixtures:
        feeds += _fixture_feeds(args.fixtures, args.seq_len, args.pad_id)
    feeds += _random_feeds(args.random, args.seq_len, np.random.default_rng(0))
    if not feeds:
        raise SystemExit("no calibration data")
    print("calibration rows:", len(feeds))

    # Quark 0.13's LayerNormalization quantizer assumes an optional bias input
    # (input[2]); the Laya graph has 2-input LayerNorm nodes.  Leave the
    # normalization/activation islands in float (BF16 on the NPU); quantize the
    # MatMul/MLP path, which is where the buffer pressure is.
    import onnx
    graph = onnx.load(args.model, load_external_data=False).graph
    skip_types = {"LayerNormalization", "Softmax", "Erf", "Gelu", "Sigmoid", "Tanh"}
    excluded = [n.name for n in graph.node if n.op_type in skip_types and n.name]
    print("excluding %d float-only nodes from quantization" % len(excluded))
    config = base_config
    if args.matmul_only:
        non_matmul = sorted({n.op_type for n in graph.node if n.op_type != "MatMul"})
        config.layer_type_config = {None: non_matmul}
        config.exclude = [n.name for n in graph.node
                          if n.op_type != "MatMul" and n.name]
        print("MatMul-only quantization; excluding:", ", ".join(non_matmul))
    else:
        config.exclude = excluded
        # Matching by op type is more reliable than node names: the quantizer
        # may rename/fuse nodes before applying the exclusion list.
        config.layer_type_config = {
            None: ["LayerNormalization", "Softmax", "Erf", "Gelu", "Sigmoid", "Tanh"]
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    quantizer = ModelQuantizer(config)
    print("quantizing", args.model)
    quantizer.quantize_model(
        args.model,
        args.out,
        calibration_data_reader=Reader(feeds),
    )
    print("wrote", args.out, "%.1f MB" % (os.path.getsize(args.out) / 1e6))
    sidecar = args.out + ".data"
    if os.path.exists(sidecar):
        print("sidecar", "%.1f MB" % (os.path.getsize(sidecar) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
