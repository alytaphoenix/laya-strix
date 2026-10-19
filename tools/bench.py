#!/usr/bin/env python3
"""Latency/throughput probe for a Jev /v1/systemone endpoint.

    python tools/bench.py --url http://127.0.0.1:8751 --requests 20
    python tools/bench.py --url http://127.0.0.1:8751 --requests 50 --concurrency 4

Also prints the service's /npu/health counters before and after, so the
encoder/head split and padding ratio are visible.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import ssl
import statistics
import sys
import time
import urllib.request


DEFAULT_STATE = (
    "Subject: Double charged on invoice 88213\n\n"
    "Hi, I was billed twice this month for the Pro plan and the second "
    "charge has not been refunded. This is the third time I have written in."
)
DEFAULT_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
            "billing": "invoices, payments, double charges, refunds",
            "technical": "bugs, outages, integration errors",
            "account": "logins, seats, plan changes",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "this week", "today", "drop everything"],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "The customer threatens to cancel the account.",
    },
}


def _headers(api_key: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _post(url: str, body: dict, timeout: float, api_key: str | None = None,
          context: ssl.SSLContext | None = None) -> float:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone", data=data,
        headers=_headers(api_key))
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        resp.read()
    return (time.perf_counter() - t0) * 1000.0


def _get(url: str, path: str, timeout: float = 3.0,
         api_key: str | None = None,
         context: ssl.SSLContext | None = None):
    try:
        req = urllib.request.Request(url.rstrip("/") + path,
                                     headers=_headers(api_key))
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=context) as r:
            return json.load(r)
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8751")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--state-file", default=None)
    ap.add_argument("--questions-file", default=None)
    ap.add_argument("--api-key-file", default=None,
                    help="Read a bearer key from this file (0600).")
    ap.add_argument("--ca-cert", default=None,
                    help="CA bundle for a self-signed HTTPS endpoint.")
    ap.add_argument("--insecure", action="store_true",
                    help="Disable TLS verification (testing only).")
    args = ap.parse_args(argv)

    api_key = None
    if args.api_key_file:
        api_key = open(args.api_key_file, encoding="utf-8").read().strip()
    context = None
    if args.insecure:
        context = ssl._create_unverified_context()
    elif args.ca_cert:
        context = ssl.create_default_context(cafile=args.ca_cert)

    state = open(args.state_file).read() if args.state_file else DEFAULT_STATE
    questions = (json.load(open(args.questions_file))
                 if args.questions_file else DEFAULT_QUESTIONS)
    body = {"state": state, "questions": questions}

    before = _get(args.url, "/npu/health", api_key=api_key, context=context)
    print("target: %s" % args.url)
    if before:
        print("backend: %s  provider: %s" % (before.get("backend"), before.get("provider")))
        print("shape: seq=%s markers=%s" % (before.get("seq_len"), before.get("markers")))

    for _ in range(args.warmup):
        _post(args.url, body, args.timeout, api_key, context)

    times = []
    errors = 0
    t0 = time.perf_counter()
    if args.concurrency <= 1:
        for _ in range(args.requests):
            try:
                times.append(_post(args.url, body, args.timeout,
                                    api_key, context))
            except Exception as exc:
                errors += 1
                print("error:", exc, file=sys.stderr)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(_post, args.url, body, args.timeout,
                                   api_key, context)
                       for _ in range(args.requests)]
            for f in futures:
                try:
                    times.append(f.result())
                except Exception as exc:
                    errors += 1
                    print("error:", exc, file=sys.stderr)
    wall = time.perf_counter() - t0

    times.sort()
    if times:
        p50 = statistics.median(times)
        p95 = times[min(len(times) - 1, int(len(times) * 0.95))]
        print("\nrequests: %d  errors: %d  wall: %.1f s" %
              (args.requests, errors, wall))
        print("latency ms: min %.1f  p50 %.1f  p95 %.1f  max %.1f" %
              (times[0], p50, p95, times[-1]))
        print("throughput: %.1f req/s (concurrency %d)" %
              (len(times) / wall if wall else 0.0, args.concurrency))

    after = _get(args.url, "/npu/health", api_key=api_key, context=context)
    if after:
        s = after.get("stats", {})
        print("\nservice stats: runs=%s rows=%s tokens=%s padding_ratio=%s" %
              (s.get("runs"), s.get("rows"), s.get("tokens"), s.get("padding_ratio")))
        print("encoder ms avg/last: %s / %s" %
              (s.get("encoder_ms_avg"), s.get("encoder_ms_last")))
        print("head ms avg/last: %s / %s" %
              (s.get("head_ms_avg"), s.get("head_ms_last")))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
