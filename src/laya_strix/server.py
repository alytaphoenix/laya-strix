"""Jev-compatible HTTP surface for the NPU-backed Laya agent."""

from __future__ import annotations

import logging
import os
import time

from .agent import build_agent
from .config import Settings
from .session import SplitSession

log = logging.getLogger("laya_strix.server")


def build_app(settings: Settings | None = None, *, calibration: str | None = None):
    """Return (FastAPI app, SplitSession).  Kept separate for tests."""
    settings = settings or Settings.from_env()

    # The Router may otherwise probe torch devices (MPS on a Mac) even though
    # the agent is ONNX-only.  Pin it before importing laya.
    os.environ.setdefault("LAYA_DEVICE", "cpu")

    from laya import Router
    from laya.serve import create_app

    session = SplitSession(settings)
    agent = build_agent(settings.model_dir, session, calibration=calibration)

    router = Router(default="english", max_loaded=1, preload=False)
    router.attach("english", agent)
    app = create_app(router)

    counts = {"requests": 0, "errors": 0, "started": time.time()}
    started = time.monotonic()

    @app.middleware("http")
    async def _count(request, call_next):
        path = request.url.path
        if path.startswith("/v1/systemone"):
            counts["requests"] += 1
        try:
            response = await call_next(request)
        except Exception:
            counts["errors"] += 1
            raise
        if response.status_code >= 500:
            counts["errors"] += 1
        return response

    @app.get("/npu/health")
    def npu_health():
        body = session.describe()
        body.update({
            "status": "ok",
            "uptime_s": round(time.monotonic() - started, 1),
            "requests": counts["requests"],
            "errors": counts["errors"],
        })
        return body

    @app.get("/metrics")
    def metrics():
        s = session.stats
        lines = [
            "# HELP laya_npu_requests_total Jev decision requests seen.",
            "# TYPE laya_npu_requests_total counter",
            "laya_npu_requests_total %d" % counts["requests"],
            "# HELP laya_npu_errors_total Requests that failed or returned 5xx.",
            "# TYPE laya_npu_errors_total counter",
            "laya_npu_errors_total %d" % counts["errors"],
            "# HELP laya_npu_rows_total Question rows evaluated.",
            "# TYPE laya_npu_rows_total counter",
            "laya_npu_rows_total %d" % s.rows,
            "# HELP laya_npu_encoder_ms Average encoder latency in milliseconds.",
            "# TYPE laya_npu_encoder_ms gauge",
            "laya_npu_encoder_ms %.3f" % s.as_dict()["encoder_ms_avg"],
            "# HELP laya_npu_head_ms Average head latency in milliseconds.",
            "# TYPE laya_npu_head_ms gauge",
            "laya_npu_head_ms %.3f" % s.as_dict()["head_ms_avg"],
            "# HELP laya_npu_padding_ratio Padded tokens per real token.",
            "# TYPE laya_npu_padding_ratio gauge",
            "laya_npu_padding_ratio %.3f" % s.as_dict()["padding_ratio"],
        ]
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse("\n".join(lines) + "\n")

    log.info("laya-npu configured: %s", settings.describe())
    return app, session


def run(host: str | None = None, port: int | None = None) -> None:
    import uvicorn

    settings = Settings.from_env()
    app, session = build_app(settings)
    host = host or settings.host
    port = int(port or settings.port)
    log.info("serving Jev /v1/systemone on %s:%d (%s)", host, port,
             session.describe()["backend"])
    uvicorn.run(app, host=host, port=port,
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"))
