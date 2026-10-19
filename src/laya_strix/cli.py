"""Command line entry point: serve, verify and train."""

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
        import dataclasses
        from pathlib import Path

        from .config import Settings
        from .verify import verify_config

        settings = Settings.from_env()
        overrides = {}
        if args.tls_cert:
            overrides["tls_cert"] = Path(args.tls_cert)
        if args.tls_key:
            overrides["tls_key"] = Path(args.tls_key)
        if args.tls_ca:
            overrides["tls_ca"] = Path(args.tls_ca)
        if args.tls_client_auth is not None:
            overrides["tls_client_auth"] = args.tls_client_auth
        if args.api_key_file:
            overrides["api_key_file"] = Path(args.api_key_file)
        if overrides:
            settings = dataclasses.replace(settings, **overrides)
        return verify_config(settings)

    ap.error("command not implemented yet: %s" % args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
