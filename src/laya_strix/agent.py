"""Build a Laya ONNXAgent whose session is the split encoder/head session.

Upstream `ONNXAgent` constructs a fused ORT session in its constructor.  We
want the rest of its pipeline (sequence building, collation, typed answer
decoding, calibration) unchanged, so we temporarily substitute the session
constructor while it initialises, then hand it our `SplitSession`.

This is the smallest reliable seam until upstream accepts a `session=` or
`providers=` argument.  The substitution is scoped to one constructor call
and restored in a `finally` block.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger("laya_strix.agent")
_PATCH_LOCK = threading.Lock()


def build_agent(model_dir: Path, session, *, calibration: str | None = None):
    import laya.onnx_agent as oa
    import onnxruntime as ort

    real_session = ort.InferenceSession
    real_available = ort.get_available_providers

    def _fake_session(*args, **kwargs):
        return session

    def _fake_available():
        # The fused constructor only uses this to choose CUDA vs CPU for the
        # session it would otherwise build; our fake ignores it either way.
        return ["CPUExecutionProvider"]

    with _PATCH_LOCK:
        ort.InferenceSession = _fake_session
        ort.get_available_providers = _fake_available
        try:
            agent = oa.ONNXAgent(
                str(model_dir),
                onnx_path=str(session.settings.encoder),
                calibration=calibration,
            )
        finally:
            ort.InferenceSession = real_session
            ort.get_available_providers = real_available

    agent.session = session
    agent.npu_session = session
    # The compiled profile is the real state budget.  Without this the tokenizer
    # would build 512-token rows and the session would silently slice them to
    # the compiled length, so `usage.truncated` would lie.
    agent.cfg["max_len"] = session.settings.seq_len
    agent.backend = "vitisai" if "VitisAIExecutionProvider" in session.get_providers() else "cpu"
    agent.device = "npu" if agent.backend == "vitisai" else "cpu"
    agent.cpu_fallback_count = 0
    agent.last_fallback_reason = None
    log.info("Laya agent ready: %s (%s)", agent.model_id, agent.device)
    return agent
