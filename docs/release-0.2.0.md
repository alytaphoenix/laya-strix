# laya-strix 0.2.0

Access and hardening release: HTTPS, mTLS, bearer API keys, secret-file
checks, input validation and a defensive default HTTP surface.

## Highlights

- **TLS terminated by uvicorn.** Set `LAYA_TLS_CERT` + `LAYA_TLS_KEY` for
  HTTPS; add `LAYA_TLS_CA` and `LAYA_TLS_CLIENT_AUTH=1` for mutual TLS.  The
  CLI also accepts `--tls-cert`, `--tls-key`, `--tls-ca`,
  `--tls-client-auth`.
- **API keys everywhere.** `LAYA_API_KEY_FILE` (0600) or `LAYA_API_KEY`
  protects `/v1/systemone*`, and this release adds the same constant-time
  bearer check to `/npu/health` and `/metrics`.  `/health` remains a minimal
  open liveness response (`{"status":"ok"}`); the full payload needs the key.
- **Setup scripts.** `deploy/gen_api_key.sh` writes a 256-bit key with 0600
  permissions; `deploy/gen_tls.sh <host>` creates a local CA, a server
  certificate with SANs and a client certificate for mTLS.
- **Docs.** `docs/tls-and-auth.md` covers LAN, mTLS, reverse proxies,
  rotation, systemd permissions, curl and Python clients.
- **Defensive defaults.** API/TLS secret files must be 0600 (or 0400);
  empty/mismatched rows and out-of-range marker positions are rejected;
  `/docs`, `/redoc` and `/openapi.json` are off unless `LAYA_ENABLE_DOCS=1`.
- **Operational tools.** `laya-npu verify` preflights config/files/secrets/
  providers without loading the NPU; `tools/bench.py` and `tools/parity.py`
  understand bearer keys, self-signed CAs and per-endpoint parity keys.

## Validation

| Check | Result |
|---|---|
| Unit tests | 14 passed (`pytest`) |
| `/health` without key | `200 {"status":"ok"}` |
| `/health` with key | full checkpoint/device payload |
| `/npu/health`, `/metrics` without key | `401`, `WWW-Authenticate: Bearer` |
| `/v1/systemone` without/with key | `401` / `200` |
| TLS with local CA | `curl --cacert ca.crt` works; wrong CA fails |
| mTLS | no client cert fails; `client.crt` + `client.key` works |
| CLI flags | `--tls-cert/--tls-key/--api-key-file` verified on CPU |
| Secret-file permissions | insecure API/TLS key modes fail closed |
| `/docs` / `/openapi.json` | `404` by default |
| Bench/parity over HTTPS + key | `tools/bench.py` and `tools/parity.py` pass |

The NPU runtime/checkpoint and its 9/9 Forseti gate are unchanged from 0.1.0;
the TLS/auth path was exercised with the same exported encoder on CPU so the
exclusive NPU context was not disturbed.

## Upgrade

```bash
git pull
uv build   # optional; the release also attaches wheel/sdist
sudo cp deploy/laya-npu.service /etc/systemd/system/
sudo systemctl daemon-reload
# add TLS/API key settings to deploy/laya-npu.env, then:
sudo systemctl restart laya-npu
```

Loopback + no key still works as before; `/docs` is now disabled by default.

## License

MIT.  Laya runtime/checkpoints Apache-2.0; AMD Ryzen AI/XRT are installed
from AMD and not redistributed.  See `LICENSE` and `NOTICE`.
