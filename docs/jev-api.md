# Jev API

`laya-npu` exposes the same TypeSafe Jev wire contract as upstream
`laya-serve`.  A client written for Jev (`hs-jev`, Forseti) can point its
`baseUrl` at this service without changes.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/systemone` | One state, one or more typed questions. |
| `POST` | `/v1/systemone/batch` | Several states sharing a question set. |
| `GET` | `/health` | Liveness; resident checkpoint and device. |
| `GET` | `/npu/health` | NPU/EP detail, shapes, encoder/head counters. |
| `GET` | `/metrics` | Prometheus text: requests, errors, latency, padding. |

## Request

```json
{
  "state": "Hi, we were billed twice for March. Please refund the duplicate today.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "other": "everything else"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this?",
      "criteria": ["not urgent", "soon", "blocking"]
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "Does the user threaten to cancel or leave?"
    }
  }
}
```

Optional controls: `model`, `max_len`, `head_max_len`, `task`, `lang`,
`lang_guess`, `min_confidence`.  The body accepts a JSON object, string, or
conversation turn list as `state`.

## Response

```json
{
  "model": "laya-rl-agent-onnx",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.9865, "technical": 0.008, "other": 0.0055},
      "confidence": 0.9267,
      "answer_confidence": 0.9865,
      "action": {"act_probability": 1.0},
      "x_jev_confidence": 0.9797
    },
    "urgency": {
      "type": "score",
      "score": 1.7722,
      "legend": {"0": "not urgent", "1": "soon", "2": "blocking"},
      "probabilities": {"0": 0.0389, "1": 0.15, "2": 0.8111}
    },
    "churn_risk": {"type": "noul", "noul": 0.879, "confidence": 0.879}
  },
  "usage": {
    "input_tokens": 164,
    "output_tokens": 0,
    "state_tokens": 21,
    "state_tokens_dropped": 0,
    "truncated": false,
    "truncated_questions": []
  },
  "routing": {"model": "english", "reason": "English Latin text", "detection": {"language": "en"}}
}
```

`confidence` is Laya's normalised entropy; `x_jev_confidence` is Jev's own
formula for the same probabilities.  `answer_confidence` is the top
probability (or the larger side of `noul`).  `action.act_probability` is the
auxiliary act-versus-escalate head.

## Batch

```json
{
  "states": ["first ticket", "second ticket"],
  "questions": {"department": {"type": "choice", "instructions": "...", "criteria": ["billing", "technical"]}},
  "batch_size": 8
}
```

The upstream `laya.serve` limits and chunking apply.  The NPU session still
runs one row per forward in v0.1.

## Auth and access

- Loopback default: `127.0.0.1:8751`; use an SSH tunnel from another host.
- LAN: set `LAYA_HOST=0.0.0.0` and configure `LAYA_API_KEY` or
  `LAYA_API_KEY_FILE`; clients send `Authorization: Bearer <key>`.
  `/v1/systemone*`, `/npu/health` and `/metrics` return `401` without it.
  `/health` remains a minimal open liveness response
  (`{"status":"ok"}`); the full health/checkpoint payload needs the key.
- TLS is terminated by uvicorn: set `LAYA_TLS_CERT`/`LAYA_TLS_KEY`, optionally
  `LAYA_TLS_CA` with `LAYA_TLS_CLIENT_AUTH=1` for mTLS.  Generate local
  certificates with `bash deploy/gen_tls.sh <host>`; for the public internet
  use a real CA or a TLS-terminating reverse proxy.
- API key rotation requires a service restart; the key is never logged.  See
  [`tls-and-auth.md`](tls-and-auth.md) for the full setup, curl/Python
  examples and key-file permissions.

## Strict Jev mode

Set `LAYA_JEV_STRICT=1` to drop Laya-specific additions (root `routing`,
per-answer `action`/`answer_confidence`, `noul.confidence`, extended `usage`)
and serve only Jev's documented fields.

## Errors

- Unknown checkpoint / non-attached model: the attached-only router refuses
  to build a torch checkpoint behind the NPU service.  The deployed
  fine-tuned 322 M agent is attached under `english,multilingual`
  (`LAYA_ATTACH`); attach a compiled typed-decisions agent before enabling
  that traffic.
- NPU/EP unavailable: startup fails closed (no silent CPU fallback) unless
  `LAYA_ALLOW_CPU_FALLBACK=1` is set for a development run.
- Malformed bodies, over-budget requests and admission overflow: the upstream
  `laya-serve` status codes and messages apply.
