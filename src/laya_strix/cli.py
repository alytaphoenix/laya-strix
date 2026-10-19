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
        epilog="Subcommands: serve, verify, train (see `laya-npu train --help`).",
    )
    ap.add_argument("command", nargs="?", default="serve",
                    choices=["serve", "verify"])
    ap.add_argument("--host", default=None,
                    help="Bind address (default: LAYA_HOST or 127.0.0.1).")
    ap.add_argument("--port", type=int, default=None,
                    help="Bind port (default: LAYA_PORT or 8751).")
    ap.add_argument("--tls-cert", default=None,
                    help="PEM server certificate; enables HTTPS.")
    ap.add_argument("--tls-key", default=None,
                    help="PEM private key for --tls-cert.")
    ap.add_argument("--tls-ca", default=None,
                    help="CA bundle for client-certificate verification.")
    ap.add_argument("--tls-client-auth", action="store_true", default=None,
                    help="Require a client certificate (needs --tls-ca).")
    ap.add_argument("--api-key-file", default=None,
                    help="Read the bearer key from a file instead of LAYA_API_KEY.")
    args = ap.parse_args(argv)

    if args.command == "serve":
        from .server import run
        run(host=args.host, port=args.port,
            tls_cert=args.tls_cert, tls_key=args.tls_key,
            tls_ca=args.tls_ca, tls_client_auth=args.tls_client_auth,
            api_key_file=args.api_key_file)
        return 0

    if args.command == "verify":
        from .verify import verify_config
        return verify_config()

    ap.error("command not implemented yet: %s" % args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
