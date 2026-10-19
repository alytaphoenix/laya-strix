#!/usr/bin/env bash
# Generate a local CA, a server certificate and a client certificate for mTLS.
#
#   bash deploy/gen_tls.sh valhalla
#   bash deploy/gen_tls.sh 192.168.0.142
#
# Output defaults to deploy/tls (gitignored).  For a public hostname use a
# real CA (Let's Encrypt) or terminate TLS at a reverse proxy instead.
set -euo pipefail

HOST="${1:-localhost}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${LAYA_TLS_DIR:-$REPO/deploy/tls}"

if ! [[ "$HOST" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "gen_tls: HOST must be a hostname or IP address, got: $HOST" >&2
  exit 2
fi

if ! command -v openssl >/dev/null 2>&1; then
  echo "gen_tls: openssl is required" >&2
  exit 2
fi

umask 077
mkdir -p "$OUT"
chmod 700 "$OUT"

case "$HOST" in
  *:*)       SAN="IP:$HOST,DNS:localhost,IP:127.0.0.1" ;;
  *[!0-9.]*) SAN="DNS:$HOST,DNS:localhost,IP:127.0.0.1" ;;
  *)         SAN="IP:$HOST,DNS:localhost,IP:127.0.0.1" ;;
esac

cat > "$OUT/ca.cnf" <<'EOF'
[req]
distinguished_name = dn
x509_extensions = v3_ca
prompt = no
[dn]
CN = Laya Strix Local CA
[v3_ca]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
EOF

cat > "$OUT/server.cnf" <<EOF
[req]
distinguished_name = dn
req_extensions = v3_req
prompt = no
[dn]
CN = $HOST
[v3_req]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = $SAN
EOF

cat > "$OUT/client.cnf" <<'EOF'
[req]
distinguished_name = dn
req_extensions = v3_req
prompt = no
[dn]
CN = laya-client
[v3_req]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = clientAuth
EOF

echo "== CA =="
openssl req -x509 -newkey rsa:4096 -sha256 -days 825 -nodes \
  -keyout "$OUT/ca.key" -out "$OUT/ca.crt" -config "$OUT/ca.cnf" 2>/dev/null

echo "== server ($HOST) =="
openssl req -newkey rsa:2048 -nodes \
  -keyout "$OUT/server.key" -out "$OUT/server.csr" -config "$OUT/server.cnf" 2>/dev/null
openssl x509 -req -in "$OUT/server.csr" -CA "$OUT/ca.crt" -CAkey "$OUT/ca.key" \
  -CAcreateserial -out "$OUT/server.crt" -days 825 -sha256 \
  -extfile "$OUT/server.cnf" -extensions v3_req 2>/dev/null

echo "== client (mTLS) =="
openssl req -newkey rsa:2048 -nodes \
  -keyout "$OUT/client.key" -out "$OUT/client.csr" -config "$OUT/client.cnf" 2>/dev/null
openssl x509 -req -in "$OUT/client.csr" -CA "$OUT/ca.crt" -CAkey "$OUT/ca.key" \
  -CAcreateserial -out "$OUT/client.crt" -days 825 -sha256 \
  -extfile "$OUT/client.cnf" -extensions v3_req 2>/dev/null

chmod 600 "$OUT"/*.key
chmod 644 "$OUT"/*.crt
rm -f "$OUT"/*.csr "$OUT"/*.srl

cat <<EOF

Generated:
  CA:      $OUT/ca.crt
  server:  $OUT/server.crt + server.key
  client:  $OUT/client.crt + client.key (only for mTLS)

Server env:
  LAYA_TLS_CERT=$OUT/server.crt
  LAYA_TLS_KEY=$OUT/server.key
  LAYA_TLS_CA=$OUT/ca.crt
  LAYA_TLS_CLIENT_AUTH=0        # set 1 to require client.crt

Client:
  curl --cacert $OUT/ca.crt https://$HOST:8751/health
  # mTLS:
  curl --cacert $OUT/ca.crt --cert $OUT/client.crt --key $OUT/client.key \\
       https://$HOST:8751/health
EOF
