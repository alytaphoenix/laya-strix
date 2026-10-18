"""Command line entry point (serve/build/verify/bench land here)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="laya-npu")
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
