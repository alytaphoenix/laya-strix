"""Environment configuration for laya-npu."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    encoder: Path
    head: Path
    encoder_b: Path | None = None
    provider: str = "VitisAIExecutionProvider"
    provider_options: dict = field(default_factory=dict)
    seq_len: int = 512
    markers: int = 8
    batch_size: int = 1
    pad_id: int = 0
    allow_cpu_fallback: bool = False
    attach: tuple[str, ...] = ("english",)
    head_threads: int = 2
    host: str = "127.0.0.1"
    port: int = 8751

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = Path(os.environ.get(
            "LAYA_MODEL_DIR",
            Path(__file__).resolve().parents[2] / "artifacts" / "onnx" / "english_split",
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

        return cls(
            model_dir=model_dir,
            encoder=encoder,
            head=head,
            encoder_b=encoder_b,
            provider=provider,
            provider_options=options,
            seq_len=int(os.environ.get("LAYA_SEQ_LEN", "512")),
            markers=int(os.environ.get("LAYA_MARKERS", "8")),
            batch_size=int(os.environ.get("LAYA_BATCH", "1")),
            pad_id=int(os.environ.get("LAYA_PAD_ID", "0")),
            allow_cpu_fallback=_bool("LAYA_ALLOW_CPU_FALLBACK", False),
            attach=tuple(x.strip() for x in
                         os.environ.get("LAYA_ATTACH", "english").split(",")
                         if x.strip()),
            head_threads=int(os.environ.get("LAYA_HEAD_THREADS", "2")),
            host=os.environ.get("LAYA_HOST", "127.0.0.1"),
            port=int(os.environ.get("LAYA_PORT", "8751")),
        )

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
        }
