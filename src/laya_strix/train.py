"""`laya-npu train`: a safe wrapper around the official Laya fine-tuning loop.

The model/loop lives in `laya.train` / `laya.train_cli`; this wrapper adds the
operational bits the box needs:

  * a default CPU device, so a training run cannot silently evict gufo from the
    iGPU; `--device cuda` is opt-in and refuses while llama-swap reports a
    loaded model unless `--allow-gpu-contention` is given;
  * a timestamped run directory, `run.json` (command, device, torch/HIP, git
    revision), and a `train.log` with the console output;
  * the environment knobs that make ROCm allocator behaviour stable.

The heavy lifting is still `laya-train`; every option it accepts can be passed
through after the wrapper's own flags.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _git_revision(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def _probe_cuda(env: dict) -> tuple[bool, str]:
    """Run a tiny GPU op in a child process.

    A ROCm wheel can report ``torch.cuda.is_available() == True`` and then
    segfault on the first real kernel (the ROCm 7.0 wheel against this box's
    ROCm 10/gfx1151 is one example).  A child process keeps that crash out of
    the training wrapper.
    """
    code = (
        "import torch\n"
        "x = torch.randn(16, 16, device='cuda')\n"
        "y = x @ x\n"
        "torch.cuda.synchronize()\n"
        "print('ok', float(y[0, 0]))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env,
                          capture_output=True, text=True)
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return False, "rc=%d %s" % (proc.returncode, " | ".join(tail))


def _loaded_models(url: str = "http://127.0.0.1:8732/v1/models") -> list[str]:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            body = json.load(r)
    except Exception:
        return []
    loaded = []
    for m in body.get("data", []):
        status = (m.get("status") or {})
        value = status.get("value") if isinstance(status, dict) else status
        if value not in (None, "unloaded", "idle"):
            loaded.append(str(m.get("id")))
    return loaded


def _parse(argv):
    ap = argparse.ArgumentParser(
        prog="laya-npu train",
        description="Fine-tune a Laya checkpoint (wraps laya-train).",
        epilog="Run `laya-train --help` for the full training option list.",
    )
    ap.add_argument("--data", help="JSONL or CSV training file.")
    ap.add_argument("--out", help="Checkpoint output directory.")
    ap.add_argument("--eval-data", dest="eval_data", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"],
                    help="Default cpu; cuda means the ROCm iGPU. (default: %(default)s)")
    ap.add_argument("--run-root", default=None,
                    help="Where to create a timestamped run dir if --out is omitted "
                         "(default: <repo>/artifacts/training).")
    ap.add_argument("--allow-gpu-contention", action="store_true",
                    help="Allow --device cuda while llama-swap has a model loaded.")
    ap.add_argument("--no-log", action="store_true",
                    help="Do not write train.log/run.json (still prints).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build items and print stats without loading weights.")
    args, rest = ap.parse_known_args(argv)
    args.rest = rest
    return args


def run_training(argv=None) -> int:
    args = _parse(argv)

    repo = Path(os.environ.get("LAYA_STRIX_ROOT", Path(__file__).resolve().parents[2]))
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else None
    if out is None and not args.dry_run:
        root = Path(args.run_root) if args.run_root else repo / "artifacts" / "training"
        stem = Path(args.data).stem if args.data else "run"
        out = root / f"{stamp}-{stem}"

    if not args.data:
        print("laya-npu train: --data is required", file=sys.stderr)
        return 2
    if out is None and args.dry_run:
        out = Path(args.run_root or (repo / "artifacts" / "training")) / f"{stamp}-dry-run"

    # Device guard: never take the iGPU while a big model is resident.
    was_auto = args.device == "auto"
    if args.device in ("cuda", "auto"):
        import torch

        if torch.cuda.is_available():
            args.device = "cuda"
        elif was_auto:
            args.device = "cpu"
        else:
            print(
                "laya-npu train: --device cuda requested, but torch reports no CUDA/ROCm "
                "device.  Install the ROCm training venv (deploy/install_train_deps.sh --rocm) "
                "or use --device cpu.", file=sys.stderr)
            return 2
    if args.device == "cuda":
        loaded = _loaded_models()
        if loaded and not args.allow_gpu_contention:
            print(
                "laya-npu train: llama-swap reports loaded model(s): %s.\n"
                "Training on the iGPU would contend with inference.  Stop the model, use "
                "--device cpu, or pass --allow-gpu-contention." % ", ".join(loaded),
                file=sys.stderr)
            return 2

    # ROCm allocator: fragmentation is the usual cause of a mid-run OOM on
    # unified-memory Strix Halo.  HSA_OVERRIDE_GFX_VERSION is opt-in for
    # wheels that do not recognise gfx1151 natively.
    env = dict(os.environ)
    env.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("LAYA_HSA_OVERRIDE"):
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", os.environ["LAYA_HSA_OVERRIDE"])

    if args.device == "cuda":
        ok, detail = _probe_cuda(env)
        if not ok:
            if was_auto:
                print("laya-npu train: GPU probe failed (%s); falling back to cpu" % detail,
                      file=sys.stderr)
                args.device = "cpu"
            else:
                print(
                    "laya-npu train: the ROCm GPU probe failed (%s).  The wheel reports a "
                    "device but cannot run a kernel; use --device cpu, or install a torch "
                    "build that matches this ROCm/gfx1151 stack." % detail, file=sys.stderr)
                return 2

    train_argv = ["--data", str(args.data), "--device", args.device]
    if out is not None:
        train_argv += ["--out", str(out)]
    if args.eval_data:
        train_argv += ["--eval-data", str(args.eval_data)]
    if args.dry_run:
        train_argv.append("--dry-run")
    train_argv += args.rest

    meta = {
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "data": str(args.data),
        "out": str(out),
        "eval_data": args.eval_data,
        "device": args.device,
        "command": [sys.executable, "-m", "laya.train_cli"] + train_argv,
        "git": _git_revision(repo),
    }
    try:
        import torch
        meta["torch"] = torch.__version__
        meta["torch_cuda"] = bool(torch.cuda.is_available())
        meta["torch_hip"] = getattr(torch.version, "hip", None)
        meta["torch_device_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception as exc:  # pragma: no cover - torch is a hard dependency of laya
        meta["torch_error"] = repr(exc)

    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    meta_path = out / "run.json"
    if not args.no_log:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        log = open(log_path, "w", encoding="utf-8")
    else:
        log = None

    print("laya-npu train: %s -> %s (device %s)" %
          (args.data, out, args.device), flush=True)
    if meta.get("torch_hip"):
        print("laya-npu train: torch %s, HIP %s, %s" %
              (meta["torch"], meta["torch_hip"], meta.get("torch_device_name")),
              flush=True)

    # The official trainer runs in this process and reads os.environ; apply
    # the allocator/HSA settings built above (a subprocess would inherit them).
    os.environ.update(env)

    from laya.train_cli import main as train_main

    t0 = time.perf_counter()
    rc = 1
    try:
        if log is not None:
            with contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
                 contextlib.redirect_stderr(_Tee(sys.stderr, log)):
                rc = train_main(train_argv)
        else:
            rc = train_main(train_argv)
    except KeyboardInterrupt:
        rc = 130
        print("laya-npu train: interrupted", file=sys.stderr)
    finally:
        meta["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        meta["elapsed_s"] = round(time.perf_counter() - t0, 1)
        meta["returncode"] = rc
        if log is not None:
            log.close()
        if not args.no_log:
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    if rc == 0:
        print("\nlaya-npu train: checkpoint -> %s" % out)
        print("  load:   laya.load(%r)" % str(out))
        print("  serve:  compile it for the NPU with tools/export_split_static.py,")
        print("          then tools/simplify_graph.py, tools/rewrite_mask.py and")
        print("          deploy/compile_encoder.sh.")
        print("  logs:   %s" % log_path)
    else:
        print("laya-npu train: failed (rc=%d); see %s" % (rc, log_path), file=sys.stderr)
    return rc
