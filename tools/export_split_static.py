#!/usr/bin/env python3
"""Export Laya as a static encoder + head pair for the NPU.

Why split: the fused graph's attention uses ``qkv.unbind`` and the decision
head uses ``in_proj_weight.split``; the dynamo exporter turns both into ONNX
sequence ops (``SplitToSequence`` + ``SequenceAt``), which the VitisAI compiler
does not accept.  The encoder is 99.9% of the FLOPs, so we trace it separately
with ``Tensor.select`` instead of ``unbind``, compile that graph for the NPU,
and leave the tiny head on the CPU (regular ONNX Runtime).

The encoder graph is otherwise the same ModernBERT-large encoder used by the
fused export.  Inputs are ``input_ids`` and ``attention_mask``; output is
``hidden_states`` [B, L, hidden].  The head takes those hidden states plus the
marker metadata and returns the same ``logits`` / ``act_logits`` as upstream.

Usage (on a development machine, CPU is fine):

    python tools/export_split_static.py --model english \
        --out artifacts/onnx/english_split --seq-len 512 --markers 8

Then copy ``encoder.onnx`` + ``encoder.onnx.data`` + ``head.onnx`` + the
tokenizer/config to valhalla for the EP-context compile.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import types

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import export_onnx_static as base  # noqa: E402


def _save(prog, out_path: str):
    """Save a dynamo ONNXProgram with a sibling external-data file."""
    import onnx

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    data = tmp + ".data"
    try:
        prog.save(tmp, external_data=False)
        proto = onnx.load(tmp)
    finally:
        for p in (tmp, data):
            if os.path.exists(p):
                os.remove(p)

    final = out_path
    sidecar = final + ".data"
    for p in (final, sidecar):
        if os.path.exists(p):
            os.remove(p)
    onnx.save_model(
        proto,
        final,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(sidecar),
        size_threshold=1024,
        convert_attribute=False,
    )
    print("  wrote %s (%.1f MB) + %s (%.1f MB)" %
          (final, os.path.getsize(final) / 1e6,
           os.path.basename(sidecar), os.path.getsize(sidecar) / 1e6))


def _patch_attention(model):
    """Replace ``qkv.unbind`` with ``qkv.select`` in every encoder layer.

    This is the only change from the upstream eager forward.  It keeps the
    exact same maths (the three slices are the same tensors) while exporting
    ordinary Slice ops instead of ONNX sequences.
    """
    from transformers.models.modernbert import modeling_modernbert as mm

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(*input_shape, 3, -1, self.head_dim)
        query_states = qkv.select(-3, 0).transpose(1, 2)
        key_states = qkv.select(-3, 1).transpose(1, 2)
        value_states = qkv.select(-3, 2).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = mm.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1)

        attention_interface = mm.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, mm.eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.head_dim ** -0.5,
            sliding_window=self.sliding_window,
            deterministic=self.deterministic_flash_attn,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_drop(self.Wo(attn_output))
        return attn_output, attn_weights

    n = 0
    for module in model.encoder.modules():
        if isinstance(module, mm.ModernBertAttention):
            module.forward = types.MethodType(forward, module)
            n += 1
    if not n:
        raise SystemExit("no ModernBertAttention modules found; cannot patch")
    print("  patched %d encoder attention layers (unbind -> select)" % n)


def _patch_mlp(model):
    """Replace the GeGLU ``chunk(2)`` with two static slices.

    ``torch.chunk`` returns a tuple, so the exporter emits
    ``SplitToSequence`` + ``SequenceAt`` (one pair per encoder layer).  The
    intermediate size is static here, so slicing the last dimension is
    equivalent and exports to ordinary ``Slice`` ops.
    """
    from transformers.models.modernbert import modeling_modernbert as mm

    def forward(self, hidden_states):
        wi = self.Wi(hidden_states)
        half = wi.shape[-1] // 2
        input_states = wi[..., :half]
        gate = wi[..., half:]
        return self.Wo(self.drop(self.act(input_states) * gate))

    n = 0
    for module in model.encoder.modules():
        if isinstance(module, mm.ModernBertMLP):
            module.forward = types.MethodType(forward, module)
            n += 1
    if not n:
        raise SystemExit("no ModernBertMLP modules found; cannot patch")
    print("  patched %d encoder MLP layers (chunk -> slice)" % n)


class EncoderWrapper:
    """Callable module wrapper (avoids importing torch before the model is built)."""

    def __new__(cls, encoder):
        import torch

        class _Encoder(torch.nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder

            def forward(self, input_ids, attention_mask):
                return self.encoder(input_ids=input_ids,
                                    attention_mask=attention_mask).last_hidden_state

        return _Encoder(encoder)


class HeadWrapper:
    def __new__(cls, model, final_norm=None):
        import torch

        class _Head(torch.nn.Module):
            def __init__(self, model, final_norm=None):
                super().__init__()
                self.model = model
                self.final_norm = final_norm

            def forward(self, hidden_states, attention_mask, marker_pos,
                        marker_mask, qtype):
                m = self.model
                h = hidden_states
                if self.final_norm is not None:
                    h = self.final_norm(h)
                h = h + m.type_emb(qtype)[:, None, :]
                pad = ~attention_mask.bool()
                for layer in m.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
                idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
                gathered = torch.gather(h, 1, idx)
                logits = m.scorer(gathered).squeeze(-1).float()
                logits = logits.masked_fill(~marker_mask, -1e4)
                p = torch.softmax(logits.detach(), -1)
                k = marker_mask.sum(-1).clamp(min=2).float()
                ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
                top2 = p.topk(2, -1).values
                feats = torch.stack(
                    [top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
                pooled = h[:, 0].float()
                act_input = torch.cat([pooled, feats], -1)
                act_input = act_input.to(m.act_head[0].weight.dtype)
                return logits, m.act_head(act_input)

        return _Head(model, final_norm)


def _export(module, args, out_path, input_names, output_names, opset=17):
    import torch

    with torch.no_grad():
        prog = torch.onnx.export(
            module,
            args,
            f=None,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=None,
            opset_version=opset,
            dynamo=True,
            external_data=False,
            optimize=False,
            report=False,
        )
    if prog is None:
        raise RuntimeError("torch.onnx.export returned None")
    _save(prog, out_path)


def _parity(out_dir, fixtures, batch_size, seq_len, markers, pad_id):
    import numpy as np
    import onnxruntime as ort

    enc = ort.InferenceSession(os.path.join(out_dir, "encoder.onnx"),
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
            n = min(int(att[i].sum()), seq_len)
            x_ids = np.full((batch_size, seq_len), pad_id, dtype=np.int64)
            x_att = np.zeros((batch_size, seq_len), dtype=np.int64)
            x_mpos = np.zeros((batch_size, markers), dtype=np.int64)
            x_mmask = np.zeros((batch_size, markers), dtype=np.bool_)
            x_qtype = np.zeros((batch_size,), dtype=np.int64)
            x_ids[0, :n] = ids[i, :n]
            x_att[0, :n] = 1
            m = min(int(mmask[i].sum()), markers)
            x_mpos[0, :m] = mpos[i, :m]
            x_mmask[0, :m] = True
            x_qtype[0] = qtype[i]
            hidden = enc.run(None, {"input_ids": x_ids,
                                    "attention_mask": x_att})[0]
            logits, act = head.run(None, {
                "hidden_states": hidden,
                "attention_mask": x_att,
                "marker_pos": x_mpos,
                "marker_mask": x_mmask,
                "qtype": x_qtype,
            })
            d = float(np.abs(logits[0, :m] - ref[i, :m]).max())
            worst = max(worst, d)
            print("    %-14s row %d  logits %.3e" % (g, i, d))
            if d > 1e-3:
                raise SystemExit("split parity failed: %s row %d, diff %.3e" % (g, i, d))
    print("  split parity OK (max |dlogits| %.3e)" % worst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="english", help="Checkpoint name or local path.")
    ap.add_argument("--out", default="artifacts/onnx/english_split")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--markers", type=int, default=8)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--token", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--drop-final-norm", action="store_true",
                    help="Move the encoder's final LayerNorm into the CPU head. "
                         "The NPU scheduler cannot fit the final layer at L=256/512; "
                         "the head applies the identical normalization on CPU.")
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
    if os.path.exists(os.path.join(out_dir, "encoder.onnx")) and not args.force:
        raise SystemExit("%s already exists; pass --force" % out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("== building %s ==" % label)
    enc_dir = os.path.join(model_dir, "encoder")
    model = base.build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    try:
        model.encoder.config.reference_compile = False
    except Exception:
        pass
    model.encoder.config._attn_implementation = "eager"
    model.eval()
    _patch_attention(model)
    _patch_mlp(model)
    print("  %.1f M parameters" % (sum(p.numel() for p in model.parameters()) / 1e6))

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

    print("== encoder ==")
    t0 = time.perf_counter()
    _export(EncoderWrapper(model.encoder), (ids, att),
            os.path.join(out_dir, "encoder.onnx"),
            ["input_ids", "attention_mask"], ["hidden_states"], opset=args.opset)
    print("  %.1f s" % (time.perf_counter() - t0))

    print("== head ==")
    hidden = torch.zeros(B, L, model.encoder.config.hidden_size)
    mpos = torch.zeros(B, M, dtype=torch.long)
    mmask = torch.ones(B, M, dtype=torch.bool)
    qtype = torch.zeros(B, dtype=torch.long)
    t0 = time.perf_counter()
    _export(HeadWrapper(model, final_norm), (hidden, att, mpos, mmask, qtype),
            os.path.join(out_dir, "head.onnx"),
            ["hidden_states", "attention_mask", "marker_pos", "marker_mask", "qtype"],
            ["logits", "act_logits"], opset=args.opset)
    print("  %.1f s" % (time.perf_counter() - t0))

    print("== parity ==")
    _parity(out_dir, os.path.join(out_dir, "fixtures.npz"), B, L, M,
            tok.pad_token_id or 0)
    print("\ndone -> %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
