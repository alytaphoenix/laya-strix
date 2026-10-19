#!/usr/bin/env bash
# Generate a 256-bit bearer key.  With a path, write it with 0600 permissions:
#   bash deploy/gen_api_key.sh /home/alyta/laya-strix/deploy/api.key
set -euo pipefail

if ! command -v openssl >/dev/null 2>&1; then
  echo "gen_api_key: openssl is required" >&2
  exit 2
fi

KEY="$(openssl rand -hex 32)"
if [ "${1:-}" ]; then
  umask 077
  printf '%s\n' "$KEY" > "$1"
  chmod 600 "$1"
  echo "wrote $1" >&2
else
  printf '%s\n' "$KEY"
fi
