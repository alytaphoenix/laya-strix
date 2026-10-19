"""`laya-npu verify`: preflight the environment without loading the NPU.

A running service owns the NPU context, so this intentionally checks config,
files, secret permissions and execution-provider availability rather than
constructing a second `SplitSession`.
"""

from __future__ import annotations

import sys

from .config import Settings


def verify_config(settings: Settings | None = None) -> int:
    settings = settings or Settings.from_env()
    failures = []

    try:
        settings.validate()
        settings.validate_tls()
        key = settings.resolve_api_key()
    except Exception as exc:
        print("config: FAIL: %s" % exc, file=sys.stderr)
        return 1

    print("model_dir: %s (%s)" % (settings.model_dir,
                                  "ok" if settings.model_dir.is_dir() else "missing"))
    print("encoder:   %s (%s)" % (settings.encoder,
                                  "ok" if settings.encoder.is_file() else "missing"))
    print("head:      %s (%s)" % (settings.head,
                                  "ok" if settings.head.is_file() else "missing"))
    print("seq_len:   %d  markers: %d" % (settings.seq_len, settings.markers))
    print("attach:    %s" % ",".join(settings.attach))
    print("api key:   %s" % ("configured" if key else "not configured"))
    print("tls:       %s%s" % ("on" if settings.tls_enabled else "off",
                               ", mTLS" if settings.tls_client_auth else ""))
    if settings.api_key_file is not None:
        print("key file:  %s" % settings.api_key_file)

    if not settings.model_dir.is_dir():
        failures.append("model directory missing")
    if not settings.encoder.is_file():
        failures.append("encoder missing")
    if not settings.head.is_file():
        failures.append("head missing")

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
    except Exception as exc:
        print("onnxruntime: FAIL: %s" % exc, file=sys.stderr)
        return 1
    print("providers: %s" % ", ".join(providers))
    if settings.provider not in providers and not settings.allow_cpu_fallback:
        failures.append("%s unavailable" % settings.provider)

    if failures:
        print("verify: FAIL: %s" % "; ".join(failures), file=sys.stderr)
        return 1
    print("verify: OK")
    return 0
