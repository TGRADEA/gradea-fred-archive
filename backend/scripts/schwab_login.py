#!/usr/bin/env python
"""schwab_login.py — mint this project's Schwab token without a web server.

Usage:
    python scripts/schwab_login.py

Prints the Schwab authorization URL, waits for you to paste the URL Schwab
redirected you to (the address bar of the page that fails to load — that is
expected, nothing listens on the callback), exchanges the code, and writes the
token file to SCHWAB_TOKEN_PATH (default: <backend dir>/tokens.json).

This is the same OAuth flow gradea-trading-platform serves at /api/auth/login,
driven from a terminal. It needs SCHWAB_APP_KEY, SCHWAB_APP_SECRET and
SCHWAB_CALLBACK_URL in backend/.env; the callback must match the one
registered on the Schwab developer app byte for byte.

Register a separate Schwab developer app for this project. A refresh token is
single-use, so two programs sharing one registration revoke each other's
session (see schwab_client.py). Never share tokens.json between projects.

Exit code 0 on success, 1 on any auth or config error.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import schwab_client


def _print_status() -> None:
    st = schwab_client.auth_status()
    print(f"state: {st.get('state')}  token_path: {st.get('token_path')}")
    ttl = st.get("refresh_expires_in_seconds")
    if ttl:
        print(f"refresh token expires in ~{ttl // 86400} days (Schwab caps this at 7)")


def main() -> int:
    try:
        ctx = schwab_client.login_context(force=True)
    except schwab_client.AuthError as exc:
        print(f"cannot start login: {exc}", file=sys.stderr)
        missing = schwab_client.missing_env(require_callback=True)
        if missing:
            print(f"missing env: {', '.join(missing)} (see env.example)", file=sys.stderr)
        return 1

    print("1. Open this URL in a browser and sign in to Schwab:\n")
    print(f"   {ctx['authorize_url']}\n")
    print(f"2. Schwab will redirect to {ctx['callback_url']}. The page will not load —")
    print("   that is expected. Copy the FULL URL from the address bar and paste it here.\n")
    try:
        raw = input("redirect URL: ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted", file=sys.stderr)
        return 1
    try:
        schwab_client.complete_from_url(raw)
    except schwab_client.AuthError as exc:
        print(f"login failed: {exc}", file=sys.stderr)
        return 1
    print("\nSchwab token written.")
    _print_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
