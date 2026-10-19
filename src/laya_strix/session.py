"""Split encoder/head ONNX session with static-shape padding.

The encoder graph is compiled for a fixed (batch, sequence, markers) profile
on the NPU.  Laya's HTTP layer hands us variable-length rows, so this adapter
pads each row to the compiled profile, runs the encoder and the CPU head, and
slices the outputs back to the caller's marker count.

The head is deliberately on the CPU: it is ~100 MB, contains the sequence ops
and TopK that the NPU compiler does not accept, and costs well under a
millisecond.  The encoder is the 421 M-parameter transformer and is the part
that must be on the NPU.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .config import Settings

log = logging.getLogger("laya_strix.session")


@dataclass
class Stats:
    runs: int = 0
    rows: int = 0
    tokens: int = 0
    padded_tokens: int = 0
    encoder_ms_total: float = 0.0
    head_ms_total: float = 0.0
    last_encoder_ms: float = 0.0
    last_head_ms: float = 0.0
    errors: int = 0

    def as_dict(self) -> dict:
        n = max(self.runs, 1)
        return {
            "runs": self.runs,
            "rows": self.rows,
            "tokens": self.tokens,
            "padded_tokens": self.padded_tokens,
            "padding_ratio": round(self.padded_tokens / max(self.tokens, 1), 3),
            "encoder_ms_avg": round(self.encoder_ms_total / n, 2),
            "head_ms_avg": round(self.head_ms_total / n, 3),
            "encoder_ms_last": round(self.last_encoder_ms, 2),
            "head_ms_last": round(self.last_head_ms, 3),
            "errors": self.errors,
        }


class SplitSession:
    """Drop-in replacement for the fused ONNXAgent session."""

    def __init__(self, settings: Settings):
        import onnxruntime as ort

        self.settings = settings
        self.stats = Stats()
        self._lock = threading.Lock()

        available = ort.get_available_providers()
        provider = settings.provider
        if provider not in available:
            if settings.allow_cpu_fallback:
                log.warning("%s unavailable; falling back to CPU (LAYA_ALLOW_CPU_FALLBACK=1)",
                            provider)
                provider = "CPUExecutionProvider"
            else:
                raise RuntimeError(
                    "%s is not available under this process. Providers: %s. "
                    "Source /opt/xilinx/xrt/setup.sh and deploy/valhalla.env, "
                    "or set LAYA_ALLOW_CPU_FALLBACK=1 for a CPU-only development run."
                    % (provider, available)
                )
        self.provider = provider

        so = ort.SessionOptions()
        so.log_severity_level = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The decision head is CPU-side.  Keep it from fanning out to every
        # core and starving gufo's sampling thread; the NPU encoder is the
        # real work.  The service is also pinned with systemd CPUAffinity.
        so.intra_op_num_threads = settings.head_threads
        so.inter_op_num_threads = 1

        # A loaded EP context model already contains the compiled graph; ORT
        # warns if a cache key is supplied with it.  Keep the BF16 config file
        # (required) and drop only the cache key.
        runtime_options = {k: v for k, v in settings.provider_options.items()
                           if k != "cache_key"}
        if provider == "CPUExecutionProvider":
            providers = ["CPUExecutionProvider"]
            provider_options = None
        else:
            # VitisAI partitions the graph itself and ORT adds the CPU EP for
            # the residual subgraphs; listing providers=[EP, CPU] with one
            # options dict trips ORT's equal-length check.
            providers = [provider]
            provider_options = [runtime_options] if runtime_options else None
        t0 = time.perf_counter()
        self.encoder = ort.InferenceSession(
            str(settings.encoder),
            sess_options=so,
            providers=providers,
            provider_options=provider_options,
        )
        self.encoder_b = None
        if settings.encoder_b is not None:
            self.encoder_b = ort.InferenceSession(
                str(settings.encoder_b),
                sess_options=so,
                providers=providers,
                provider_options=provider_options,
            )
        self.encoder_load_s = time.perf_counter() - t0

        self.head = ort.InferenceSession(
            str(settings.head),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        log.info("encoder A loaded in %.1f s on %s", self.encoder_load_s,
                 self.encoder.get_providers())
        if self.encoder_b is not None:
            log.info("encoder B loaded on %s", self.encoder_b.get_providers())
        log.info("head loaded on %s", self.head.get_providers())

        self._check_shapes()
        self.warmup()

    def _check_shapes(self):
        expected = {"input_ids": self.settings.seq_len,
                    "attention_mask": self.settings.seq_len}
        for i in self.encoder.get_inputs():
            if i.name in expected and len(i.shape) == 2 and isinstance(i.shape[1], int):
                if i.shape[1] != expected[i.name]:
                    raise RuntimeError(
                        "encoder input %s has shape %s; LAYA_SEQ_LEN is %d. "
                        "Re-export/compile the graph for this sequence length."
                        % (i.name, i.shape, expected[i.name]))
        if self.encoder_b is not None:
            for i in self.encoder_b.get_inputs():
                if i.name == "attention_mask" and isinstance(i.shape[1], int):
                    if i.shape[1] != self.settings.seq_len:
                        raise RuntimeError(
                            "encoder B attention_mask shape %s does not match "
                            "LAYA_SEQ_LEN=%d" % (i.shape, self.settings.seq_len))
        log.info("encoder A inputs: %s", [(i.name, i.shape) for i in self.encoder.get_inputs()])
        if self.encoder_b is not None:
            log.info("encoder B inputs: %s", [(i.name, i.shape) for i in self.encoder_b.get_inputs()])
        log.info("head inputs: %s", [(i.name, i.shape) for i in self.head.get_inputs()])

    def warmup(self, rows: int = 1):
        m = self.settings.markers
        feeds = {
            "input_ids": np.ones((rows, self.settings.seq_len), dtype=np.int64),
            "attention_mask": np.ones((rows, self.settings.seq_len), dtype=np.int64),
            "marker_pos": np.zeros((rows, m), dtype=np.int64),
            "marker_mask": np.ones((rows, m), dtype=np.bool_),
            "qtype": np.zeros((rows,), dtype=np.int64),
        }
        self.run(["logits", "act_logits"], feeds)
        self.stats = Stats()

    def _run_one(self, ids, att, mpos, mmask, qtype):
        L = self.settings.seq_len
        M = self.settings.markers
        n = min(int(att.sum()), L)
        if n <= 0:
            raise ValueError("empty row: attention_mask has no active tokens")

        x_ids = np.full((1, L), self.settings.pad_id, dtype=np.int64)
        x_att = np.zeros((1, L), dtype=np.int64)
        x_mpos = np.zeros((1, M), dtype=np.int64)
        x_mmask = np.zeros((1, M), dtype=np.bool_)
        x_qtype = np.zeros((1,), dtype=np.int64)

        x_ids[0, :n] = ids[:n]
        x_att[0, :n] = 1
        m = min(mpos.shape[0], M)
        active = np.asarray(mmask[:m], dtype=bool)
        if active.any():
            positions = np.asarray(mpos[:m])[active]
            if positions.min() < 0 or positions.max() >= L:
                raise ValueError(
                    "marker position outside the compiled sequence: %s" %
                    (positions.tolist(),))
        x_mpos[0, :m] = mpos[:m]
        x_mmask[0, :m] = active
        x_qtype[0] = int(qtype)

        t0 = time.perf_counter()
        hidden = self.encoder.run(
            None, {"input_ids": x_ids, "attention_mask": x_att})[0]
        if self.encoder_b is not None:
            hidden = self.encoder_b.run(
                None, {"hidden_states_in": hidden, "attention_mask": x_att})[0]
        enc_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        logits, act = self.head.run(None, {
            "hidden_states": hidden,
            "attention_mask": x_att,
            "marker_pos": x_mpos,
            "marker_mask": x_mmask,
            "qtype": x_qtype,
        })
        head_ms = (time.perf_counter() - t1) * 1000.0
        return logits[0, :mpos.shape[0]], act[0], enc_ms, head_ms, n

    def run(self, output_names, feeds):
        """Fused-session API: ``run(["logits","act_logits"], feeds)``."""
        ids = np.asarray(feeds["input_ids"])
        att = np.asarray(feeds["attention_mask"])
        mpos = np.asarray(feeds["marker_pos"])
        mmask = np.asarray(feeds["marker_mask"])
        qtype = np.asarray(feeds["qtype"])

        if ids.ndim != 2 or att.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 2-D")
        if ids.shape != att.shape:
            raise ValueError("input_ids and attention_mask shapes differ")
        rows = ids.shape[0]
        if rows == 0:
            raise ValueError("empty batch")

        if mpos.ndim == 1:
            mpos = mpos[None, :]
        if mmask.ndim == 1:
            mmask = mmask[None, :]
        if mpos.ndim != 2 or mmask.ndim != 2:
            raise ValueError("marker_pos and marker_mask must be 2-D")
        if mpos.shape != mmask.shape:
            raise ValueError("marker_pos and marker_mask shapes differ")
        if mpos.shape[0] != rows:
            raise ValueError("marker rows do not match input rows")
        if mpos.shape[1] > self.settings.markers:
            raise ValueError(
                "request has %d markers, compiled profile has %d; "
                "increase LAYA_MARKERS and recompile" %
                (mpos.shape[1], self.settings.markers))

        qtype = qtype.reshape(-1)
        if qtype.size == 1 and rows > 1:
            qtype = np.repeat(qtype, rows)
        if qtype.size != rows:
            raise ValueError("qtype rows do not match input rows")

        out_logits = []
        out_act = []
        with self._lock:
            try:
                for i in range(rows):
                    logits, act, enc_ms, head_ms, n = self._run_one(
                        ids[i], att[i], mpos[i], mmask[i], qtype[i])
                    out_logits.append(logits)
                    out_act.append(act)
                    self.stats.runs += 1
                    self.stats.rows += 1
                    self.stats.tokens += n
                    self.stats.padded_tokens += self.settings.seq_len
                    self.stats.encoder_ms_total += enc_ms
                    self.stats.head_ms_total += head_ms
                    self.stats.last_encoder_ms = enc_ms
                    self.stats.last_head_ms = head_ms
            except Exception:
                self.stats.errors += 1
                raise

        return [np.stack(out_logits, axis=0), np.stack(out_act, axis=0)]

    # Health/utility surface used by /npu/health.
    def get_providers(self):
        return self.encoder.get_providers()

    def get_inputs(self):
        return self.encoder.get_inputs()

    def get_outputs(self):
        return self.encoder.get_outputs()

    def describe(self) -> dict:
        return {
            "backend": "vitisai-npu" if "VitisAIExecutionProvider" in self.provider else "cpu",
            "provider": self.provider,
            "encoder_path": str(self.settings.encoder),
            "encoder_b_path": str(self.settings.encoder_b) if self.settings.encoder_b else None,
            "head_path": str(self.settings.head),
            "encoder_load_s": round(self.encoder_load_s, 2),
            "encoder_inputs": [(i.name, i.shape) for i in self.encoder.get_inputs()],
            "encoder_outputs": [(o.name, o.shape) for o in self.encoder.get_outputs()],
            "encoder_b_inputs": ([(i.name, i.shape) for i in self.encoder_b.get_inputs()]
                                 if self.encoder_b is not None else None),
            "encoder_b_outputs": ([(o.name, o.shape) for o in self.encoder_b.get_outputs()]
                                  if self.encoder_b is not None else None),
            "head_providers": self.head.get_providers(),
            "seq_len": self.settings.seq_len,
            "markers": self.settings.markers,
            "stats": self.stats.as_dict(),
        }
