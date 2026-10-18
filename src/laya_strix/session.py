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


def _np_dtype(ort_type: str):
    return {
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(bool)": np.bool_,
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(bfloat16)": np.float32,
    }[ort_type]


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

        provider_options = [settings.provider_options] if settings.provider_options else None
        t0 = time.perf_counter()
        self.encoder = ort.InferenceSession(
            str(settings.encoder),
            sess_options=so,
            providers=[provider, "CPUExecutionProvider"],
            provider_options=provider_options,
        )
        self.encoder_load_s = time.perf_counter() - t0

        self.head = ort.InferenceSession(
            str(settings.head),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )

        log.info("encoder loaded in %.1f s on %s", self.encoder_load_s,
                 self.encoder.get_providers())
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
        for i in self.head.get_inputs():
            if i.name == "input_ids":
                continue
        log.info("encoder inputs: %s", [(i.name, i.shape) for i in self.encoder.get_inputs()])
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
            n = 1

        x_ids = np.full((1, L), self.settings.pad_id, dtype=np.int64)
        x_att = np.zeros((1, L), dtype=np.int64)
        x_mpos = np.zeros((1, M), dtype=np.int64)
        x_mmask = np.zeros((1, M), dtype=np.bool_)
        x_qtype = np.zeros((1,), dtype=np.int64)

        x_ids[0, :n] = ids[:n]
        x_att[0, :n] = 1
        m = min(mpos.shape[0], M)
        x_mpos[0, :m] = mpos[:m]
        x_mmask[0, :m] = mmask[:m]
        x_qtype[0] = int(qtype)

        t0 = time.perf_counter()
        hidden = self.encoder.run(
            None, {"input_ids": x_ids, "attention_mask": x_att})[0]
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

        rows = ids.shape[0]
        if mpos.shape[1] > self.settings.markers:
            raise ValueError(
                "request has %d markers, compiled profile has %d; "
                "increase LAYA_MARKERS and recompile" %
                (mpos.shape[1], self.settings.markers))

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
            "head_path": str(self.settings.head),
            "encoder_load_s": round(self.encoder_load_s, 2),
            "encoder_inputs": [(i.name, i.shape) for i in self.encoder.get_inputs()],
            "encoder_outputs": [(o.name, o.shape) for o in self.encoder.get_outputs()],
            "head_providers": self.head.get_providers(),
            "seq_len": self.settings.seq_len,
            "markers": self.settings.markers,
            "stats": self.stats.as_dict(),
        }
