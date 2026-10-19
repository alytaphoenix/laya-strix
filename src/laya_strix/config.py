"""Environment configuration for laya-npu."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _check_secret_file(path: Path, label: str) -> None:
    """Reject a group/world-readable secret file.

    The service is Linux/macOS; this is a cheap guard against the common
    ``0644 api.key`` mistake.  Certificates are intentionally public and are
    not checked here.
    """
    try:
        info = path.stat()
    except OSError as exc:
        raise RuntimeError("%s does not exist: %s (%s)" % (label, path, exc)) from exc
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError("%s is not a regular file: %s" % (label, path))
    if info.st_size == 0:
        raise RuntimeError("%s is empty: %s" % (label, path))
    if os.name == "posix":
        mode = stat.S_IMODE(info.st_mode)
        if mode & 0o077:
            raise RuntimeError(
                "%s %s is %04o; chmod 600 (or 400) it so the service secret "
                "is not group/world readable" % (label, path, mode))


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    encoder: Path
    head: Path
    encoder_b: Path | None = None
    provider: str = "VitisAIExecutionProvider"
    provider_options: dict = field(default_factory=dict)
    seq_len: int = 256
    markers: int = 8
    batch_size: int = 1
    pad_id: int = 0
    allow_cpu_fallback: bool = False
    attach: tuple[str, ...] = ("english",)
    head_threads: int = 2
    api_key_file: Path | None = None
    tls_cert: Path | None = None
    tls_key: Path | None = None
    tls_ca: Path | None = None
    tls_client_auth: bool = False
    enable_docs: bool = False
    host: str = "127.0.0.1"
    port: int = 8751

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = Path(os.environ.get(
            "LAYA_MODEL_DIR",
            Path(__file__).resolve().parents[2] / "artifacts" / "onnx" / "english_split_256",
        ))
        encoder = Path(os.environ.get("LAYA_ENCODER", model_dir / "encoder.onnx"))
        head = Path(os.environ.get("LAYA_HEAD", model_dir / "head.onnx"))
        encoder_b_raw = os.environ.get("LAYA_ENCODER_B")
        encoder_b = Path(encoder_b_raw) if encoder_b_raw else None
        provider = os.environ.get("LAYA_PROVIDER", "VitisAIExecutionProvider")

        options: dict = {}
        config_file = os.environ.get("LAYA_EP_CONFIG")
        if config_file:
            options["config_file"] = str(Path(config_file).resolve())
        cache_dir = os.environ.get("LAYA_EP_CACHE")
        if cache_dir:
            options["cache_dir"] = str(Path(cache_dir).resolve())
            options["cache_key"] = os.environ.get(
                "LAYA_EP_CACHE_KEY", encoder.stem)
            options["enable_cache_file_io_in_mem"] = 0

        def _path(name: str) -> Path | None:
            raw = os.environ.get(name)
            return Path(raw) if raw else None

        return cls(
            model_dir=model_dir,
            encoder=encoder,
            head=head,
            encoder_b=encoder_b,
            provider=provider,
            provider_options=options,
            seq_len=int(os.environ.get("LAYA_SEQ_LEN", "256")),
            markers=int(os.environ.get("LAYA_MARKERS", "8")),
            batch_size=int(os.environ.get("LAYA_BATCH", "1")),
            pad_id=int(os.environ.get("LAYA_PAD_ID", "0")),
            allow_cpu_fallback=_bool("LAYA_ALLOW_CPU_FALLBACK", False),
            attach=tuple(x.strip() for x in
                         os.environ.get("LAYA_ATTACH", "english").split(",")
                         if x.strip()),
            head_threads=int(os.environ.get("LAYA_HEAD_THREADS", "2")),
            api_key_file=_path("LAYA_API_KEY_FILE"),
            tls_cert=_path("LAYA_TLS_CERT"),
            tls_key=_path("LAYA_TLS_KEY"),
            tls_ca=_path("LAYA_TLS_CA"),
            tls_client_auth=_bool("LAYA_TLS_CLIENT_AUTH", False),
            enable_docs=_bool("LAYA_ENABLE_DOCS", False),
            host=os.environ.get("LAYA_HOST", "127.0.0.1"),
            port=int(os.environ.get("LAYA_PORT", "8751")),
        )

    def validate(self) -> None:
        if self.seq_len <= 0:
            raise RuntimeError("LAYA_SEQ_LEN must be positive")
        if self.markers <= 0:
            raise RuntimeError("LAYA_MARKERS must be positive")
        if self.batch_size <= 0:
            raise RuntimeError("LAYA_BATCH must be positive")
        if self.head_threads <= 0:
            raise RuntimeError("LAYA_HEAD_THREADS must be positive")
        if self.pad_id < 0:
            raise RuntimeError("LAYA_PAD_ID must be non-negative")
        if not (0 < self.port < 65536):
            raise RuntimeError("LAYA_PORT must be between 1 and 65535")
        if not self.host:
            raise RuntimeError("LAYA_HOST must not be empty")

    def resolve_api_key(self) -> str | None:
        """Return the configured bearer key without logging it.

        ``LAYA_API_KEY`` wins over ``LAYA_API_KEY_FILE``.  The file form keeps
        the secret out of the systemd environment dump and process args.
        """
        env_key = os.environ.get("LAYA_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip()
        if self.api_key_file is not None:
            _check_secret_file(self.api_key_file, "LAYA_API_KEY_FILE")
            try:
                key = self.api_key_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise RuntimeError(
                    "cannot read LAYA_API_KEY_FILE %s: %s" %
                    (self.api_key_file, exc)) from exc
            if not key:
                raise RuntimeError("LAYA_API_KEY_FILE %s is empty" %
                                   self.api_key_file)
            return key
        return None

    def apply_api_key(self) -> str | None:
        """Make a file-backed key visible to upstream ``laya.serve``.

        ``laya.serve.create_app`` reads ``LAYA_API_KEY`` when it builds the
        app, so this must run first.  An explicit env key already wins.  The
        environment key is normalised so our extension-endpoint check and the
        upstream check compare the same bytes.
        """
        env_key = os.environ.get("LAYA_API_KEY")
        if env_key is not None:
            stripped = env_key.strip()
            if not stripped:
                os.environ.pop("LAYA_API_KEY", None)
            elif stripped != env_key:
                os.environ["LAYA_API_KEY"] = stripped

        key = self.resolve_api_key()
        if key and not os.environ.get("LAYA_API_KEY"):
            os.environ["LAYA_API_KEY"] = key
        return key

    def validate_tls(self) -> None:
        cert, key = self.tls_cert, self.tls_key
        if bool(cert) != bool(key):
            raise RuntimeError(
                "TLS needs both LAYA_TLS_CERT and LAYA_TLS_KEY (or neither)")
        if cert is not None and key is not None:
            if cert.resolve() == key.resolve():
                raise RuntimeError("LAYA_TLS_CERT and LAYA_TLS_KEY must differ")
            if not cert.is_file():
                raise RuntimeError("LAYA_TLS_CERT does not exist: %s" % cert)
            _check_secret_file(key, "LAYA_TLS_KEY")
        if self.tls_client_auth:
            if not self.tls_enabled:
                raise RuntimeError(
                    "LAYA_TLS_CLIENT_AUTH=1 requires LAYA_TLS_CERT/KEY")
            if self.tls_ca is None:
                raise RuntimeError(
                    "LAYA_TLS_CLIENT_AUTH=1 requires LAYA_TLS_CA")
            if not self.tls_ca.is_file():
                raise RuntimeError("LAYA_TLS_CA does not exist: %s" % self.tls_ca)
        elif self.tls_ca is not None and not self.tls_ca.is_file():
            raise RuntimeError("LAYA_TLS_CA does not exist: %s" % self.tls_ca)

    @property
    def tls_enabled(self) -> bool:
        return self.tls_cert is not None and self.tls_key is not None

    def describe(self) -> dict:
        return {
            "model_dir": str(self.model_dir),
            "encoder": str(self.encoder),
            "encoder_b": str(self.encoder_b) if self.encoder_b else None,
            "head": str(self.head),
            "provider": self.provider,
            "provider_options": self.provider_options,
            "seq_len": self.seq_len,
            "markers": self.markers,
            "batch_size": self.batch_size,
            "allow_cpu_fallback": self.allow_cpu_fallback,
            "attach": list(self.attach),
            "head_threads": self.head_threads,
            "api_key_configured": bool(os.environ.get("LAYA_API_KEY")) or
                                  self.api_key_file is not None,
            "api_key_file": str(self.api_key_file) if self.api_key_file else None,
            "tls": self.tls_enabled,
            "tls_client_auth": self.tls_client_auth,
            "docs_enabled": self.enable_docs,
        }
