"""Jev-compatible HTTP surface for the NPU-backed Laya agent."""

from __future__ import annotations

import logging
import os
import threading
import time

from .agent import build_agent
from .auth import make_api_key_dependency
from .config import Settings
from .session import SplitSession

log = logging.getLogger("laya_strix.server")


def build_app(settings: Settings | None = None, *, calibration: str | None = None):
    """Return (FastAPI app, SplitSession).  Kept separate for tests."""
    settings = settings or Settings.from_env()
    settings.validate()
    settings.validate_tls()

    # The Router may otherwise probe torch devices (MPS on a Mac) even though
    # the agent is ONNX-only.  Pin it before importing laya.
    os.environ.setdefault("LAYA_DEVICE", "cpu")

    from fastapi import Depends
    from laya import Router
    from laya.serve import create_app

    class _AttachedOnlyRouter(Router):
        """Refuse to build a torch checkpoint behind the NPU service's back."""

        def _build(self, key: str):
            raise ValueError(
                "laya-npu serves the attached checkpoint(s) only; %r is not attached. "
                "Compile and attach it before enabling that traffic." % key)

    # Resolve a file-backed bearer key before laya.serve.create_app reads the
    # environment, and reuse it for the extension endpoints below.
    api_key = settings.apply_api_key()

    session = SplitSession(settings)
    agent = build_agent(settings.model_dir, session, calibration=calibration)

    names = settings.attach or ("english",)
    router = _AttachedOnlyRouter(default=names[0], max_loaded=len(names), preload=False)
    for name in names:
        router.attach(name, agent)
    app = create_app(router)
    if not settings.enable_docs:
        # FastAPI installs /docs, /redoc and /openapi.json at construction.
        # They expose the schema but not data; keep the public surface minimal
        # unless the operator opts in with LAYA_ENABLE_DOCS=1.
        hidden = {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
        app.router.routes = [
            r for r in app.router.routes
            if getattr(r, "path", None) not in hidden
        ]
    auth_dependency = make_api_key_dependency(api_key)

    counts = {"requests": 0, "errors": 0, "started": time.time()}
    counts_lock = threading.Lock()
    started = time.monotonic()

    @app.middleware("http")
    async def _count(request, call_next):
        path = request.url.path
        if path.startswith("/v1/systemone"):
            with counts_lock:
                counts["requests"] += 1
        try:
            response = await call_next(request)
        except Exception:
            with counts_lock:
                counts["errors"] += 1
            raise
        if response.status_code >= 500:
            with counts_lock:
                counts["errors"] += 1
        return response

    @app.get("/npu/health", dependencies=[Depends(auth_dependency)])
    def npu_health():
        body = session.describe()
        with counts_lock:
            requests = counts["requests"]
            errors = counts["errors"]
        body.update({
            "status": "ok",
            "uptime_s": round(time.monotonic() - started, 1),
            "requests": requests,
            "errors": errors,
        })
        return body

    @app.get("/metrics", dependencies=[Depends(auth_dependency)])
    def metrics():
        s = session.stats
        with counts_lock:
            requests = counts["requests"]
            errors = counts["errors"]
        lines = [
            "# HELP laya_npu_requests_total Jev decision requests seen.",
            "# TYPE laya_npu_requests_total counter",
            "laya_npu_requests_total %d" % requests,
            "# HELP laya_npu_errors_total Requests that failed or returned 5xx.",
            "# TYPE laya_npu_errors_total counter",
            "laya_npu_errors_total %d" % errors,
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


def run(host: str | None = None, port: int | None = None, *,
        tls_cert: str | None = None, tls_key: str | None = None,
        tls_ca: str | None = None, tls_client_auth: bool | None = None,
        api_key_file: str | None = None) -> None:
    import dataclasses
    import ssl
    from pathlib import Path

    import uvicorn

    settings = Settings.from_env()
    overrides = {}
    if tls_cert:
        overrides["tls_cert"] = Path(tls_cert)
    if tls_key:
        overrides["tls_key"] = Path(tls_key)
    if tls_ca:
        overrides["tls_ca"] = Path(tls_ca)
    if tls_client_auth is not None:
        overrides["tls_client_auth"] = tls_client_auth
    if api_key_file:
        overrides["api_key_file"] = Path(api_key_file)
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    settings.validate_tls()

    app, session = build_app(settings)
    host = settings.host if host is None else host
    port = settings.port if port is None else int(port)

    ssl_kwargs = {}
    if settings.tls_enabled:
        ssl_kwargs["ssl_certfile"] = str(settings.tls_cert)
        ssl_kwargs["ssl_keyfile"] = str(settings.tls_key)
        ssl_kwargs["ssl_version"] = ssl.PROTOCOL_TLS_SERVER
        if settings.tls_client_auth:
            ssl_kwargs["ssl_ca_certs"] = str(settings.tls_ca)
            ssl_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED
    scheme = "https" if settings.tls_enabled else "http"
    log.info("serving Jev /v1/systemone on %s://%s:%d (%s)", scheme, host, port,
             session.describe()["backend"])
    uvicorn.run(app, host=host, port=port,
                log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
                **ssl_kwargs)
