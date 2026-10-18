"""Command line entry point (serve/build/verify/bench land here)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `train` has its own parser; intercept it before the serve flags would
    # reject --data/--out/--epochs.
    if argv and argv[0] == "train":
        from .train import run_training
        return run_training(argv[1:])

    ap = argparse.ArgumentParser(
        prog="laya-npu",
        epilog="Subcommands: serve, train (see `laya-npu train --help`).",
    )
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve", "verify", "bench"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8751)
    args = ap.parse_args(argv)

    if args.command == "serve":
        from .server import run
        run(host=args.host, port=args.port)
        return 0

    ap.error("command not implemented yet: %s" % args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
