#!/usr/bin/env bash
# Raise RLIMIT_MEMLOCK for XRT.  XRT mmaps its 64 MB device heap with
# MAP_LOCKED; Ubuntu's 8 MB default makes xrt-smi and inference fail with
# EAGAIN.  Log out and back in (or start a new SSH session) afterwards.
set -euo pipefail

sudo tee /etc/security/limits.d/99-xrt-memlock.conf >/dev/null <<'EOF'
*       soft    memlock     unlimited
*       hard    memlock     unlimited
root    soft    memlock     unlimited
root    hard    memlock     unlimited
EOF

sudo mkdir -p /etc/systemd/system.conf.d /etc/systemd/user.conf.d
printf '[Manager]\nDefaultLimitMEMLOCK=infinity\n' \
  | sudo tee /etc/systemd/system.conf.d/99-xrt-memlock.conf >/dev/null
printf '[Manager]\nDefaultLimitMEMLOCK=infinity\n' \
  | sudo tee /etc/systemd/user.conf.d/99-xrt-memlock.conf >/dev/null
sudo systemctl daemon-reexec

echo "configured; open a new session and check: ulimit -H -l  # unlimited"
