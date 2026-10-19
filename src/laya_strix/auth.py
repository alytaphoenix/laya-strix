"""Bearer auth for the extension endpoints that upstream laya.serve does not
cover.

``laya.serve`` already enforces ``LAYA_API_KEY`` on ``/v1/systemone*`` and on
the detailed ``/health`` payload.  ``/npu/health`` and ``/metrics`` are added
by this package, so they get the same constant-time check explicitly.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException


def _expected(api_key: str) -> bytes:
    # Match laya.serve's wire encoding, including non-ASCII keys.
    return ("Bearer " + api_key).encode("utf-8", "surrogateescape")


def make_api_key_dependency(api_key: str | None):
    """Return a FastAPI dependency, or a no-op when no key is configured."""

    def dependency(authorization: str | None = Header(default=None)) -> None:
        if not api_key:
            return
        supplied = (authorization or "").encode("utf-8", "surrogateescape")
        if not hmac.compare_digest(supplied, _expected(api_key)):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return dependency
