# TLS and API keys

`laya-npu` uses the upstream `laya.serve` bearer check and adds the same check
to its own `/npu/health` and `/metrics` endpoints.  TLS is terminated by the
uvicorn process; no reverse proxy is required for a small deployment.

## Quick start (LAN)

```bash
cd ~/laya-strix

# 1) bearer key, 0600 permissions
bash deploy/gen_api_key.sh /home/alyta/laya-strix/deploy/api.key

# 2) self-signed CA + server cert (+ client cert for mTLS)
bash deploy/gen_tls.sh valhalla
```

Edit `deploy/laya-npu.env`:

```bash
LAYA_HOST=0.0.0.0
LAYA_API_KEY_FILE=/home/alyta/laya-strix/deploy/api.key
LAYA_TLS_CERT=/home/alyta/laya-strix/deploy/tls/server.crt
LAYA_TLS_KEY=/home/alyta/laya-strix/deploy/tls/server.key
LAYA_TLS_CA=/home/alyta/laya-strix/deploy/tls/ca.crt
LAYA_TLS_CLIENT_AUTH=0
```

```bash
chmod 600 deploy/laya-npu.env deploy/api.key deploy/tls/*.key
sudo systemctl restart laya-npu
systemctl is-active laya-npu
```

Verify:

```bash
KEY=$(cat deploy/api.key)
curl --cacert deploy/tls/ca.crt https://valhalla:8751/health
curl --cacert deploy/tls/ca.crt \
  -H "Authorization: Bearer $KEY" https://valhalla:8751/health
curl --cacert deploy/tls/ca.crt \
  -H "Authorization: Bearer $KEY" https://valhalla:8751/npu/health
```

Without the bearer:

- `/v1/systemone`, `/v1/systemone/batch`, `/npu/health`, `/metrics` → `401`
  with `WWW-Authenticate: Bearer`.
- `/health` → `200 {"status":"ok"}` (minimal liveness only; no checkpoint,
  device or revision data).

## API key details

- `LAYA_API_KEY` is the upstream `laya.serve` variable.
- `LAYA_API_KEY_FILE` is read at startup and exported to the upstream app.
  It keeps the secret out of `/proc/$pid/environ`, `systemctl show` and the
  shell history.  An explicit `LAYA_API_KEY` wins if both are set.
- The key is compared with `hmac.compare_digest`; the service never logs it.
- If `LAYA_API_KEY_FILE` (or the TLS private key) is group/world readable,
  startup fails with a `chmod 600` message instead of silently using it.
- Rotate by writing a new key, then `systemctl restart laya-npu`.  The old key
  stops working immediately.  One key is shared by all clients; use a reverse
  proxy if you need per-client keys or revocation without a restart.
- Do not put `LAYA_API_KEY` on the `laya-npu` command line; process arguments
  are visible to other local users.

## TLS details

- `deploy/gen_tls.sh HOST` creates a local CA, a server certificate with
  SANs for `HOST`, `localhost` and `127.0.0.1`, and a client certificate for
  mTLS.  Output is `deploy/tls/` (gitignored).
- For a public hostname, use a real certificate (for example Let's Encrypt)
  or terminate TLS at a reverse proxy.  Do not expose the generated CA
  private key; the CA certificate itself is meant to be distributed to
  clients.
- `LAYA_TLS_CERT` and `LAYA_TLS_KEY` must be set together.
- `LAYA_TLS_CLIENT_AUTH=1` requires `LAYA_TLS_CA` and a client certificate:
  `curl --cacert ca.crt --cert client.crt --key client.key ...`.
- uvicorn loads the certificate at startup.  Replace the files and restart
  the unit to renew; there is no hot reload.
- FastAPI `/docs`, `/redoc` and `/openapi.json` are removed by default.  Set
  `LAYA_ENABLE_DOCS=1` only for a trusted/authenticated deployment.
- If a proxy terminates TLS, keep `laya-npu` on `127.0.0.1` and have the proxy
  forward `Authorization: Bearer ...`; do not expose the backend directly.

## Systemd

The unit reads `/home/alyta/laya-strix/deploy/laya-npu.env` as root, then runs
as `alyta`.  Keep the env file and key files `0600`; keep `deploy/tls/` `0700`.
The unit already pins CPUs 0-3, `Nice=10` and a two-thread head.

```bash
sudo systemctl restart laya-npu
journalctl -u laya-npu -n 50 --no-pager
```

## Benchmark/parity tools

`tools/bench.py` and `tools/parity.py` accept the same deployment:

```bash
python tools/bench.py --url https://valhalla:8751 \
  --api-key-file deploy/api.key --ca-cert deploy/tls/ca.crt

python tools/parity.py --reference https://cpu-ref:8751 \
  --candidate https://valhalla:8751 \
  --api-key-file deploy/api.key --ca-cert deploy/tls/ca.crt \
  --probes ../forseti/crew/eval/laya-probe.jsonl
```

For endpoints with different keys, parity accepts
`--reference-api-key-file` and `--candidate-api-key-file`.  `--insecure`
disables certificate verification for a one-off test only; do not use it in a
production check.

## Clients

Forseti / hs-jev: set the base URL to `https://valhalla:8751` and configure
the bearer token in the client's headers.  For the self-signed CA, point the
client's trust store at `deploy/tls/ca.crt`.

Python:

```python
import httpx

key = open("deploy/api.key").read().strip()
with httpx.Client(verify="deploy/tls/ca.crt", timeout=30) as client:
    r = client.post(
        "https://valhalla:8751/v1/systemone",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "state": "Duplicate charge on invoice 4411",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which department?",
                    "criteria": {"billing": "payments", "technical": "bugs",
                                 "other": "everything else"},
                }
            },
        },
    )
    r.raise_for_status()
    print(r.json()["answers"])
```

## Threat model

- A bearer key over plain HTTP is visible to anyone on the path.  Use TLS for
  any non-loopback traffic.
- The self-signed CA is suitable for a trusted LAN or an SSH tunnel.  For the
  public internet use a real CA and a firewall/reverse proxy.
- `/health` is intentionally usable as a liveness probe without a key.  It
  does not expose paths, revisions, device state or counters.
- The upstream request size/admission limits still apply after auth.
