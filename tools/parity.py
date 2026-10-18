#!/usr/bin/env python3
"""Compare two Jev endpoints on the same probe set.

Use the NPU service as ``--candidate`` and the CPU/MPS implementation as
``--reference``.  For every probe it prints argmax agreement and the largest
probability difference; the acceptance gate is usually:

    no argmax changes, max |dp| <= 0.01, noul/score within 0.01.

Probe format is Forseti's ``crew/eval/laya-probe.jsonl``:

    {"kind": "choice"|"noul_high"|"score_high", "expected": ...,
     "state": "...", "criteria": {...}}

    python tools/parity.py \
        --reference http://127.0.0.1:8753 \
        --candidate http://127.0.0.1:8751 \
        --probes ../forseti/crew/eval/laya-probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def _request_for(row):
    kind = row["kind"]
    state = {"input": row["state"]}
    if kind == "choice":
        criteria = dict(row.get("criteria") or {})
        if not criteria:
            criteria = {"yes": "the output fits", "other": "none of the options fit"}
        q = {"type": "choice",
             "instructions": "Which option best fits this output?",
             "criteria": criteria}
        return state, {"decide": q}
    if kind == "noul_high":
        q = {"type": "noul",
             "instructions": "Does the state contain an explicit threat to cancel or block progress?"}
        return state, {"threat": q}
    if kind == "score_high":
        q = {"type": "score", "instructions": "How urgent is this?",
             "criteria": ["routine", "soon", "blocking"]}
        return state, {"urgency": q}
    raise ValueError("unknown probe kind %r" % kind)


def _post(url, state, questions):
    body = json.dumps({"state": state, "questions": questions}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _answer(out):
    answers = out.get("answers", {})
    return next(iter(answers.values())) if answers else {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--max-dp", type=float, default=0.01)
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(args.probes) if line.strip()]
    worst = 0.0
    argmax_changes = 0
    failures = []
    for i, row in enumerate(rows, 1):
        state, questions = _request_for(row)
        ref = _answer(_post(args.reference, state, questions))
        cand = _answer(_post(args.candidate, state, questions))

        if ref.get("type") == "noul":
            dp = abs(float(ref.get("noul", 0)) - float(cand.get("noul", 0)))
            print("  [%d] noul  dp=%.4f" % (i, dp))
        elif ref.get("type") == "score":
            rp = ref.get("probabilities", {})
            cp = cand.get("probabilities", {})
            dp = max((abs(rp[k] - cp.get(k, 0)) for k in rp), default=0.0)
            ra = max(rp, key=rp.get) if rp else None
            ca = max(cp, key=cp.get) if cp else None
            if ra != ca:
                argmax_changes += 1
            print("  [%d] score ref=%s cand=%s  dp=%.4f" % (i, ra, ca, dp))
        else:
            rp = ref.get("probabilities", {})
            cp = cand.get("probabilities", {})
            dp = max((abs(rp[k] - cp.get(k, 0)) for k in rp), default=0.0)
            ra = max(rp, key=rp.get) if rp else None
            ca = max(cp, key=cp.get) if cp else None
            if ra != ca:
                argmax_changes += 1
            print("  [%d] choice ref=%s cand=%s  dp=%.4f" % (i, ra, ca, dp))
        worst = max(worst, dp)
        if dp > args.max_dp:
            failures.append((i, dp))

    print("\nprobes: %d  argmax changes: %d  max |dp|: %.4f  gate: %.4f" %
          (len(rows), argmax_changes, worst, args.max_dp))
    if failures:
        print("FAIL: %s" % failures)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
