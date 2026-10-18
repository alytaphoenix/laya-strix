#!/usr/bin/env python3
"""Export the Laya encoder as two consecutive NPU subgraphs.

The VitisAI scheduler cannot fit the full 28-layer ModernBERT encoder at
`L=256` or `L=512`: it fails at fused layer 32 with
`Auto-phase failed ... hardware resources constraint`.  Splitting the layer
stack into two graphs (default 14 + 14) halves the per-graph buffering while
keeping the same weights and maths.  The service runs graph A, then graph B,
then the CPU head.

    python tools/export_split_layers.py --model english \
        --out artifacts/onnx/english_split_256_layers --seq-len 256 --split-at 14 \
        --drop-final-norm

`--drop-final-norm` moves the encoder's final LayerNorm into the CPU head, as
in the single-graph exporter.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import export_onnx_static as base  # noqa: E402
import export_split_static as split  # noqa: E402


class PartA:
    def __new__(cls, encoder, split_at):
        import torch

        class _PartA(torch.nn.Module):
            def __init__(self, encoder, split_at):
                super().__init__()
                self.enc = encoder
                self.split_at = split_at

            def forward(self, input_ids, attention_mask):
                from transformers.models.modernbert import modeling_modernbert as mm
                import torch as _torch

                seq_len = input_ids.shape[1]
                position_ids = _torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                hidden = self.enc.embeddings(input_ids=input_ids)
                masks = {
                    "full_attention": mm.create_bidirectional_mask(
                        self.enc.config, hidden, attention_mask),
                    "sliding_attention": mm.create_bidirectional_sliding_window_mask(
                        self.enc.config, hidden, attention_mask),
                }
                pos = {lt: self.enc.rotary_emb(hidden, position_ids, lt)
                       for lt in set(self.enc.config.layer_types)}
                for layer in self.enc.layers[:self.split_at]:
                    hidden = layer(
                        hidden,
                        attention_mask=masks[layer.attention_type],
                        position_embeddings=pos[layer.attention_type],
                    )
                return hidden

        return _PartA(encoder, split_at)


class PartB:
    def __new__(cls, encoder, split_at, final_norm=None):
        import torch

        class _PartB(torch.nn.Module):
            def __init__(self, encoder, split_at, final_norm=None):
                super().__init__()
                self.enc = encoder
                self.split_at = split_at
                self.final_norm = final_norm

            def forward(self, hidden_states, attention_mask):
                from transformers.models.modernbert import modeling_modernbert as mm
                import torch as _torch

                hidden = hidden_states
                seq_len = hidden.shape[1]
                position_ids = _torch.arange(seq_len, device=hidden.device).unsqueeze(0)
                masks = {
                    "full_attention": mm.create_bidirectional_mask(
                        self.enc.config, hidden, attention_mask),
                    "sliding_attention": mm.create_bidirectional_sliding_window_mask(
                        self.enc.config, hidden, attention_mask),
                }
                pos = {lt: self.enc.rotary_emb(hidden, position_ids, lt)
                       for lt in set(self.enc.config.layer_types)}
                for layer in self.enc.layers[self.split_at:]:
                    hidden = layer(
                        hidden,
                        attention_mask=masks[layer.attention_type],
                        position_embeddings=pos[layer.attention_type],
                    )
                if self.final_norm is not None:
                    hidden = self.final_norm(hidden)
                return hidden

        return _PartB(encoder, split_at, final_norm)


def _parity(out_dir, fixtures, L, M, pad_id):
    import numpy as np
    import onnxruntime as ort

    a = ort.InferenceSession(os.path.join(out_dir, "encoder_a.onnx"),
                             providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(os.path.join(out_dir, "encoder_b.onnx"),
                             providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(os.path.join(out_dir, "head.onnx"),
                                providers=["CPUExecutionProvider"])
    z = np.load(fixtures)
    worst = 0.0
    for g in sorted({k.split("/")[0] for k in z.files}):
        need = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
        if not all("%s/%s" % (g, n) in z.files for n in need):
            continue
        ids = z["%s/input_ids" % g]
        att = z["%s/attention_mask" % g]
        mpos = z["%s/marker_pos" % g]
        mmask = z["%s/marker_mask" % g]
        qtype = z["%s/qtype" % g]
        ref = z["%s/ref_logits" % g]
        for i in range(ids.shape[0]):
            n = min(int(att[i].sum()), L)
            xid = np.full((1, L), pad_id, dtype=np.int64)
            xa = np.zeros((1, L), dtype=np.int64)
            xmp = np.zeros((1, M), dtype=np.int64)
            xmm = np.zeros((1, M), dtype=np.bool_)
            xq = np.zeros((1,), dtype=np.int64)
            xid[0, :n] = ids[i, :n]
            xa[0, :n] = 1
            m = min(int(mmask[i].sum()), M)
            xmp[0, :m] = mpos[i, :m]
            xmm[0, :m] = True
            xq[0] = qtype[i]
            h1 = a.run(None, {"input_ids": xid, "attention_mask": xa})[0]
            h2 = b.run(None, {"hidden_states_in": h1, "attention_mask": xa})[0]
            logits, act = head.run(None, {
                "hidden_states": h2,
                "attention_mask": xa,
                "marker_pos": xmp,
                "marker_mask": xmm,
                "qtype": xq,
            })
            d = float(np.abs(logits[0, :m] - ref[i, :m]).max())

            def softmax(x):
                e = np.exp(x - x.max())
                return e / e.sum()

            dp = float(np.abs(softmax(logits[0, :m]) - softmax(ref[i, :m])).max())
            worst = max(worst, dp)
            print("    %-14s row %d  logits %.3e  prob %.3e" % (g, i, d, dp))
            if dp > 1e-2:
                raise SystemExit("layer-split parity failed: %s row %d, dp %.3e" %
                                 (g, i, dp))
    print("  layer-split parity OK (max |dprob| %.3e)" % worst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="english")
    ap.add_argument("--out", default="artifacts/onnx/english_split_layers")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--markers", type=int, default=8)
    ap.add_argument("--split-at", type=int, default=14)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--drop-final-norm", action="store_true")
    ap.add_argument("--token", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    import torch

    model_dir, subfolder, label = base._resolve_spec(args.model, args.token)
    if subfolder:
        model_dir = os.path.join(model_dir, subfolder)
    base._fix_tokenizer_config(model_dir)
    cfg_path = os.path.join(model_dir, "rl_agent_config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    out_dir = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out_dir, "encoder_a.onnx")) and not args.force:
        raise SystemExit("%s already exists; pass --force" % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    enc_dir = os.path.join(model_dir, "encoder")
    model = base.build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    model.encoder.config._attn_implementation = "eager"
    model.eval()
    split._patch_attention(model)
    split._patch_mlp(model)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    shutil.copy2(cfg_path, os.path.join(out_dir, "rl_agent_config.json"))
    base.copy_tokenizer(model_dir, out_dir)

    print("== fixtures ==")
    with torch.no_grad():
        base.generate_fixtures(model, tok, cfg, out_dir)

    final_norm = None
    if args.drop_final_norm:
        import torch.nn as nn
        final_norm = getattr(model.encoder, "final_norm", None)
        if final_norm is None:
            raise SystemExit("encoder has no final_norm to move")
        model.encoder.final_norm = nn.Identity()
        print("  moved encoder.final_norm into the CPU head")

    B, L, M = 1, args.seq_len, args.markers
    ids = torch.zeros(B, L, dtype=torch.long)
    att = torch.ones(B, L, dtype=torch.long)
    hidden = torch.zeros(B, L, model.encoder.config.hidden_size)
    mpos = torch.zeros(B, M, dtype=torch.long)
    mmask = torch.ones(B, M, dtype=torch.bool)
    qtype = torch.zeros(B, dtype=torch.long)

    print("== part A (layers 0..%d) ==" % (args.split_at - 1))
    t0 = time.perf_counter()
    split._export(PartA(model.encoder, args.split_at), (ids, att),
                  os.path.join(out_dir, "encoder_a.onnx"),
                  ["input_ids", "attention_mask"], ["hidden_states"], opset=args.opset)
    print("  %.1f s" % (time.perf_counter() - t0))

    print("== part B (layers %d..) ==" % args.split_at)
    t0 = time.perf_counter()
    split._export(PartB(model.encoder, args.split_at, final_norm), (hidden, att),
                  os.path.join(out_dir, "encoder_b.onnx"),
                  ["hidden_states_in", "attention_mask"], ["hidden_states_out"],
                  opset=args.opset)
    print("  %.1f s" % (time.perf_counter() - t0))

    print("== head ==")
    t0 = time.perf_counter()
    split._export(split.HeadWrapper(model, final_norm), (hidden, att, mpos, mmask, qtype),
                  os.path.join(out_dir, "head.onnx"),
                  ["hidden_states", "attention_mask", "marker_pos", "marker_mask", "qtype"],
                  ["logits", "act_logits"], opset=args.opset)
    print("  %.1f s" % (time.perf_counter() - t0))

    print("== parity ==")
    _parity(out_dir, os.path.join(out_dir, "fixtures.npz"), L, M, tok.pad_token_id or 0)
    print("\ndone -> %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
