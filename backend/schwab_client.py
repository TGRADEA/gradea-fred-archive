"""schwab_client.py — Schwab token custody + client cache.

Since FOUNDATION-AUTH-INTERNAL-1, GradeA owns its Schwab credential end to end:
its own developer app registration, its own OAuth flow served from its own
origin (8002), and its own token file that nothing else writes. Auth is
internal to one project.

That rule is not tidiness. A Schwab refresh token is single-use — refreshing
rotates it and revokes the previous one — so two processes sharing one
registration take turns invalidating each other's credential, and each side
sees an intermittent `invalid_grant` it cannot attribute to anything it did.
One credential per project is the only configuration in which that cannot
happen.

Token custody rules:
- One path: `SCHWAB_TOKEN_PATH`, default `<data_dir>/tokens.json`. No fallback
  chain, no candidate search — a missing token names the exact path checked.
- GradeA is the sole writer. The OAuth routes below mint the file; schwab-py
  refreshes it in place at the same path.
- A token file in another application's shape is refused, not adapted. Reading
  a sibling project's token is the failure mode this slice exists to remove.
- The built client is cached against the *identity* of that file, so replacing
  the token takes effect on the next call rather than on restart (INV-H).
- No token value is ever logged or returned in a response (INV-E).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

from settings import (
    describe_config_problem,
    invalid_schwab_keys,
    missing_schwab_keys,
)
from settings import settings as _settings

APP_KEY = _settings.schwab_app_key
APP_SECRET = _settings.schwab_app_secret
CALLBACK_URL = _settings.schwab_callback_url

# The one token file GradeA reads and writes (FOUNDATION-AUTH-INTERNAL-1).
# GradeA holds its own Schwab app registration and runs its own OAuth flow, so
# it is the sole refresher of this credential. No other process writes here.
TOKEN_PATH = _settings.schwab_token_path

# Schwab refresh tokens are hard-capped at 7 days. Surfaced so the UI can tell
# the user that re-login is a recurring, expected action rather than a fault.
REFRESH_TOKEN_TTL_DAYS = 7

_RECONNECT_HINT = "Connect Schwab from the dashboard to sign in again."


class AuthError(Exception):
    """Raised when no usable Schwab credentials/token are available.

    `missing` / `invalid` are populated for configuration failures so routes can
    name the offending env keys. Neither ever carries a credential *value*.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "auth_required",
        missing: list[str] | None = None,
        invalid: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.missing = list(missing or [])
        self.invalid = dict(invalid or {})


class ConfigError(AuthError):
    """Raised when .env is missing or still holds env.example placeholders."""

    def __init__(self, missing: list[str], invalid: dict[str, str]):
        super().__init__(
            describe_config_problem(missing, invalid),
            code="not_configured",
            missing=missing,
            invalid=invalid,
        )


_client = None  # module-level cache
# Identity of the token file `_client` was built from; see `_source_fingerprint`.
_client_source: tuple | None = None


def missing_env(*, require_callback: bool = False) -> list[str]:
    """Schwab env keys that are absent or blank.

    By default the callback URL is not among them: reading the stored token and
    refreshing it need only the app key and secret. Pass
    `require_callback=True` from the OAuth routes, which do send a redirect URI
    to Schwab and cannot work without one.
    """
    return missing_schwab_keys(
        APP_KEY, APP_SECRET, CALLBACK_URL, require_callback=require_callback
    )


def invalid_env() -> dict[str, str]:
    """Schwab env keys that are set but unusable → secret-free reason."""
    return invalid_schwab_keys(APP_KEY, APP_SECRET, CALLBACK_URL)


def is_configured() -> bool:
    return not missing_env() and not invalid_env()


def active_token_path() -> Path:
    """The token file GradeA will actually read. One path, no fallback.

    Returned whether or not it exists, so a missing token fails with the exact
    path that was checked. A fallback chain here is what once let GradeA read a
    dead file while reporting a healthy connection.
    """
    return TOKEN_PATH


def _source_fingerprint() -> tuple | None:
    """Identity of the token file GradeA would read right now, or None if absent.

    Path, mtime and size together rather than mtime alone: two writes inside one
    filesystem mtime tick still change the file's length. GradeA is now the only
    writer, so this mostly fires on GradeA's own refreshes -- a rebuild that
    re-reads the file authlib just wrote, which is correct if redundant. It stays
    because it is also what makes a manual token replacement take effect without
    a restart.
    """
    path = active_token_path()
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def token_source() -> str:
    """Which custody the active token came from: "gradea" or "none".

    Narrowed from "vault"|"legacy"|"none" by FOUNDATION-AUTH-INTERNAL-1. There
    is now exactly one custodian, so the only question this can answer is
    whether a token exists. Kept rather than replaced by `token_present`
    because the status panel and its two frontend consumers key off this field.
    """
    return "gradea" if active_token_path().exists() else "none"


def has_token() -> bool:
    return TOKEN_PATH.exists()


def _require_config(*, require_callback: bool = False) -> None:
    """Raise ConfigError unless the Schwab app config is present and usable.

    `require_callback` is for the OAuth routes only. The data path (stored
    token -> schwab-py client) never sends a redirect URI, so demanding one
    there would block a perfectly working setup.
    """
    missing, invalid = missing_env(require_callback=require_callback), invalid_env()
    if missing or invalid:
        raise ConfigError(missing, invalid)


# ── Token document reading ───────────────────────────────────────────────
# GradeA writes exactly one shape: schwab-py's envelope
# `{"creation_timestamp": <epoch>, "token": {...}}`. Schwab's own flat payload
# (`access_token`/`refresh_token`/`expires_at`/…) is still *recognized* so that
# a path aimed at another application's token file fails with that diagnosis
# instead of a generic parse error — recognized, never loaded. Everything below
# reads *timing*; token values never leave this module except to schwab-py.


def _as_epoch(value: object) -> float | None:
    """Epoch seconds from a number or an ISO-8601 string, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    return None


def _read_token_document(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _token_facts(doc: dict | None) -> dict:
    """Shape + expiry timing for a token document. Carries no token material.

    Returns `{shape, access_expires_at, refresh_expires_at, has_refresh,
    has_access}` where `shape` is "schwab_py" (GradeA's own), "flat" (Schwab's
    raw payload, as some other applications persist it), or "" when the document
    is unrecognisable.
    A `None` expiry means "not stated by the file", which is treated as
    non-expiring rather than expired — schwab-py is the authority on a token it
    can still refresh.
    """
    unknown = {
        "shape": "",
        "access_expires_at": None,
        "refresh_expires_at": None,
        "has_refresh": False,
        "has_access": False,
    }
    if not doc:
        return unknown

    inner = doc.get("token")
    if isinstance(inner, dict):
        created = _as_epoch(doc.get("creation_timestamp"))
        access_at = _as_epoch(inner.get("expires_at"))
        if access_at is None and created is not None:
            expires_in = _as_epoch(inner.get("expires_in"))
            access_at = created + expires_in if expires_in is not None else None
        return {
            "shape": "schwab_py",
            "access_expires_at": access_at,
            # schwab-py records only when the token was minted; the 7-day cap is
            # Schwab's, not the file's.
            "refresh_expires_at": (
                created + REFRESH_TOKEN_TTL_DAYS * 86400 if created is not None else None
            ),
            "has_refresh": bool(inner.get("refresh_token")),
            "has_access": bool(inner.get("access_token")),
        }

    if doc.get("access_token") or doc.get("refresh_token"):
        return {
            "shape": "flat",
            "access_expires_at": _as_epoch(doc.get("expires_at")),
            "refresh_expires_at": _as_epoch(doc.get("refresh_expires_at")),
            "has_refresh": bool(doc.get("refresh_token")),
            "has_access": bool(doc.get("access_token")),
        }

    return unknown


def _remaining(expires_at: float | None, now: float) -> int | None:
    return None if expires_at is None else int(expires_at - now)


def _modified_at(path: Path) -> str | None:
    """UTC ISO-8601 mtime of a token file, or None when it is not there."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, UTC).replace(microsecond=0).isoformat()


# ── Auth diagnostics (SCHWAB-AUTH-BRIDGE-DIAGNOSTICS-1) ──────────────────
# A Schwab consent screen that never returns is indistinguishable, from the
# GradeA side, from a callback GradeA dropped on the floor. Recording when a
# callback last arrived makes the difference observable: no `last_callback_at`
# after a consent click means Schwab never redirected, so the fault is upstream
# of this process. Process-local and in-memory, like the pending-state store.

_AUTH_EVENT_MAX_LEN = 200
_SECRETLIKE = re.compile(
    r"(?i)\b(code|state|token|secret|access_token|refresh_token)\b\s*[=:]\s*\S+"
)
_auth_diagnostic: dict[str, str | None] = {
    "last_auth_event": None,
    "last_auth_event_at": None,
    "last_callback_at": None,
    "auth_error": None,
}


def _redact(text: str) -> str:
    """Strip anything shaped like a credential, then bound the length."""
    return _SECRETLIKE.sub(r"\1=[redacted]", text).strip()[:_AUTH_EVENT_MAX_LEN]


def record_auth_event(event: str, error: str | None = None) -> None:
    """Note that an auth milestone happened. Never records code/state/token.

    `event` is one of a small vocabulary chosen by the caller (`login_started`,
    `callback_received`, `callback_error`, `callback_exchange_failed`,
    `token_stored`). Events named `callback*` also stamp `last_callback_at`,
    which is the field that answers "did Schwab ever come back?".
    """
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    _auth_diagnostic["last_auth_event"] = _redact(event)
    _auth_diagnostic["last_auth_event_at"] = now
    if event.startswith("callback"):
        _auth_diagnostic["last_callback_at"] = now
    _auth_diagnostic["auth_error"] = _redact(error) if error else None
    detail = f" — {_auth_diagnostic['auth_error']}" if error else ""
    print(f"[gradea][auth] {_auth_diagnostic['last_auth_event']}{detail}")


def auth_diagnostic() -> dict[str, str | None]:
    """Snapshot of the diagnostic fields. Safe to serialize."""
    return dict(_auth_diagnostic)


def reset_auth_diagnostic() -> None:
    for key in _auth_diagnostic:
        _auth_diagnostic[key] = None


def auth_status() -> dict:
    """Non-throwing status snapshot. Contains no credential or token material.

    Legacy keys (`configured`, `token_present`, `callback_url`, `token_path`)
    are preserved verbatim; everything else is additive (INV-B).
    """
    missing, invalid = missing_env(), invalid_env()
    configured = not missing and not invalid
    source = token_source()
    path = active_token_path()

    # The callback URL is withheld while it is a placeholder: echoing it back
    # invites the user to register the template string at Schwab for real.
    callback = "" if "SCHWAB_CALLBACK_URL" in invalid else CALLBACK_URL

    status = {
        # ── legacy keys — shape frozen by INV-B ──
        "configured": configured,
        "token_present": source != "none",
        "callback_url": callback,
        "token_path": str(path),
        # ── additive ──
        "authenticated": False,
        "state": "not_configured",
        "message": describe_config_problem(missing, invalid),
        "missing_env": missing,
        "invalid_env": sorted(invalid),
        "invalid_env_detail": invalid,
        "refresh_token_ttl_days": REFRESH_TOKEN_TTL_DAYS,
        "refresh_expires_in_seconds": None,
        # ── token custody (FOUNDATION-AUTH-INTERNAL-1) ──
        "token_source": source,
        "access_expires_in_seconds": None,
        # Which file GradeA actually read, and whether the live client was built
        # from that same version of it. Without this pair, "I reconnected but
        # GradeA disagrees" is unfalsifiable from the browser.
        "token_modified_at": _modified_at(path),
        "client_fresh": _client is None or _client_source == _source_fingerprint(),
        "access_stale": False,
        **auth_diagnostic(),
    }
    if not configured:
        return status

    if source == "none":
        status.update(
            state="auth_required",
            message=(
                f"No Schwab token at {path}. {_RECONNECT_HINT}"
            ),
        )
        return status

    facts = _token_facts(_read_token_document(path))
    if not facts["shape"]:
        status.update(
            state="token_unreadable",
            message=(
                f"{path} is not a Schwab token file GradeA can read. "
                f"{_RECONNECT_HINT}"
            ),
        )
        return status
    if facts["shape"] == "flat":
        # Recognizable, but another application's shape. `_loadable_token_path()`
        # refuses it, so the banner has to say so too: reporting "connected" here
        # while every tile 401s is exactly the silent lie this codebase bans.
        status.update(
            state="token_unreadable",
            message=(
                f"{path} holds another application's Schwab token, not GradeA's. "
                f"Point SCHWAB_TOKEN_PATH at GradeA's own token file. "
                f"{_RECONNECT_HINT}"
            ),
        )
        return status

    now = time.time()
    access_left = _remaining(facts["access_expires_at"], now)
    refresh_left = _remaining(facts["refresh_expires_at"], now)
    status.update(
        access_expires_in_seconds=access_left,
        refresh_expires_in_seconds=refresh_left,
    )

    # A live access token is enough on its own; otherwise a refresh token that
    # has not passed Schwab's 7-day cap is, because schwab-py renews on demand.
    access_live = access_left is not None and access_left > 0
    refresh_live = facts["has_refresh"] and (refresh_left is None or refresh_left > 0)
    if access_live or refresh_live:
        if not facts["has_access"]:
            # Within its window but unsignable: authlib reads the access token
            # straight out of the file and reports its absence as a bare
            # `unsupported_token_type`, which looks like a Schwab outage. Say so
            # here instead of letting every data call 502. Reported as
            # `token_unreadable` rather than as a new state: `state` is a pinned
            # contract enum and the frontend renders no banner at all for a value
            # it has no copy for. The specific reason travels in `message`.
            status.update(
                state="token_unreadable",
                message=(
                    f"The Schwab token at {path} has no access token. "
                    f"{_RECONNECT_HINT}"
                ),
            )
            return status
        # Past expiry but renewable is still "authenticated" -- schwab-py
        # refreshes on demand. Flagged rather than hidden: when every tile is
        # erroring, a stale access token that never renews is the difference
        # between "signed out" and "the refresh token Schwab holds no longer
        # matches this file", and a flat "Connected to Schwab." hid that.
        stale = not access_live
        message = "Connected to Schwab."
        if stale:
            message = (
                "Connected to Schwab. Access token is past expiry; schwab-py "
                "will renew it on the next call."
            )
        status.update(
            authenticated=True,
            state="authenticated",
            access_stale=stale,
            message=message,
        )
        return status

    status.update(
        state="refresh_expired",
        message=(
            f"Schwab session has expired — refresh tokens last "
            f"{REFRESH_TOKEN_TTL_DAYS} days. {_RECONNECT_HINT}"
        ),
    )
    return status


def _write_private(path: Path, payload: dict) -> None:
    """Serialize a token-bearing document, owner-readable only where supported."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows ACLs; the file still sits inside the app directory.


def _loadable_token_path() -> Path:
    """The token file, verified to be one schwab-py can load.

    Returns the active path itself — never a derived copy. GradeA mints this
    file through its own OAuth flow and schwab-py refreshes it in place, so the
    file loaded and the file written are the same file. The previous
    adapt-to-a-side-copy step existed only to avoid writing another project's
    token file, and it was the source of the worst class of bug here: a derived
    copy that could outrank the real token and pin a refresh token that had
    already been rotated away.
    """
    path = active_token_path()
    if not path.exists():
        raise AuthError(
            f"No Schwab token at {path}. {_RECONNECT_HINT}", code="auth_required"
        )
    facts = _token_facts(_read_token_document(path))
    shape = facts["shape"]
    if shape == "schwab_py":
        if not facts["has_access"]:
            # Parses, but authlib signs requests with the access token it reads
            # straight out of this file. Absent, it raises a bare
            # `unsupported_token_type` that reads as a Schwab outage. Refuse
            # here so the failure names the token instead. `auth_status()`
            # reports the same file as `token_unreadable` for the same reason —
            # the banner and the loader must never disagree.
            raise AuthError(
                f"The Schwab token at {path} has no access token. "
                f"{_RECONNECT_HINT}",
                code="auth_required",
            )
        return path
    if shape == "flat":
        # Schwab's raw payload rather than schwab-py's envelope: SCHWAB_TOKEN_PATH
        # is aimed at some other application's token file. GradeA will not read
        # or adapt it. Sharing a credential is what makes two processes rotate
        # one refresh chain out from under each other.
        raise AuthError(
            f"{path} holds another application's Schwab token, not GradeA's. "
            f"Point SCHWAB_TOKEN_PATH at GradeA's own token file. "
            f"{_RECONNECT_HINT}",
            code="token_unreadable",
        )
    raise AuthError(
        f"{path} is not a Schwab token file GradeA can read. {_RECONNECT_HINT}",
        code="token_unreadable",
    )


def _build() -> object:
    """Build (or rebuild) the schwab-py client from the stored token.

    `client_from_token_file` gives the returned client authlib's automatic
    refresh, which is safe here precisely because GradeA is the sole refresher
    of this credential: nothing else can rotate the chain underneath it. That is
    also what lets the long-lived streamer client keep working — it captures
    this object once and refreshes itself for the life of the stream.

    No OAuth flow is started here; a terminal-driven manual flow would block an
    HTTP request. The browser flow lives in the routes below. Anything short of
    a loadable token is an AuthError telling the operator to connect.
    """
    _require_config()

    try:
        import schwab  # type: ignore
    except ImportError as exc:
        raise AuthError(
            "schwab-py not installed. Run: pip install -r requirements.txt"
        ) from exc

    load_path = _loadable_token_path()
    try:
        return schwab.auth.client_from_token_file(
            token_path=str(load_path),
            api_key=APP_KEY,
            app_secret=APP_SECRET,
        )
    except Exception as exc:  # noqa: BLE001
        raise AuthError(
            f"Stored Schwab token could not be loaded. {_RECONNECT_HINT}",
            code="token_unreadable",
        ) from exc


def invalidate_client() -> None:
    """Drop the cached client so the next call re-reads the stored token."""
    global _client, _client_source
    _client = None
    _client_source = None


def get_client():
    """Return the cached client, rebuilding it when the token file has changed.

    Reconnecting mints a new refresh token and revokes the previous one, so a
    client cached across that event holds credentials Schwab now rejects
    (`invalid_grant`) — and because schwab-py loads the token into memory once,
    nothing about the new file on disk would ever reach it. Keying the cache on
    the source file's identity is what makes a reconnect take effect on the next
    call instead of requiring a GradeA restart.

    Now that GradeA is the only writer, the usual trigger is GradeA's own
    authlib refresh rewriting the file, and the rebuild simply re-reads what was
    just written. Harmless, and still the mechanism that makes a fresh login or
    a hand-replaced token take effect immediately.

    The fingerprint is taken before the build, so a write that races the build is
    picked up by the following call rather than being cached over.
    """
    global _client, _client_source
    fingerprint = _source_fingerprint()
    if _client is None or fingerprint != _client_source:
        _client = _build()
        _client_source = fingerprint
    return _client


def clear_token() -> bool:
    """Delete GradeA's token. Returns True if a file was removed.

    Only GradeA's own file is touched, because only GradeA's own file exists in
    this project's data path. Sibling projects hold separate Schwab app
    registrations and separate tokens, so signing out here signs out of nothing
    else -- which is the whole point of one credential per project.
    """
    invalidate_client()
    path = active_token_path()
    if path.exists():
        path.unlink()
        return True
    return False


# ── Browser OAuth (AUTH-SCHWAB-OAUTH-1) ──────────────────────────────────
# Primary and only auth path since FOUNDATION-AUTH-INTERNAL-1. GradeA holds its
# own Schwab app registration, so the callback below is registered to GradeA's
# own origin and the token it mints lands at `active_token_path()` -- the same
# file schwab-py then refreshes in place.
#
# Route-friendly three-legged flow built on schwab-py's serializable
# `get_auth_context` / `client_from_received_url` primitives, so the OAuth
# boundary (INV-C) stays inside this module. The HTTP routes in app.py only
# call build_authorization_url() and exchange_code().
#
# State store is process-local and in-memory: a single-user local desktop app
# does not warrant a DB. Limitation: pending states do not survive a server
# restart, and a horizontally-scaled deployment would need shared storage.

_AUTH_STATE_TTL_SEC = 600
_pending_states: dict[str, float] = {}  # state -> created epoch seconds


def _prune_states(now: float | None = None) -> None:
    now = time.time() if now is None else now
    expired = [s for s, ts in _pending_states.items() if now - ts > _AUTH_STATE_TTL_SEC]
    for s in expired:
        _pending_states.pop(s, None)


def _require_schwab_auth():
    """Import schwab.auth or raise AuthError with install guidance.

    Only the OAuth routes reach this, and they do send a redirect URI to
    Schwab, so the callback URL is required here even though the data path does
    not need it.
    """
    _require_config(require_callback=True)
    try:
        import schwab.auth as schwab_auth  # type: ignore
    except ImportError as exc:
        raise AuthError(
            "schwab-py not installed. Run: pip install -r requirements.txt"
        ) from exc
    return schwab_auth


def login_context(force: bool = False) -> dict:
    """Begin a browser OAuth flow and describe it to the caller.

    Generates a CSRF state token, asks schwab-py for the Schwab authorization
    URL bound to our registered callback (CALLBACK_URL), records the state, and
    returns `{authorize_url, callback_url, state, forced}`. Raises ConfigError
    when the Schwab app config is missing or still holds placeholders — the
    backend refuses to hand out a login URL it knows Schwab will reject.

    `force` is the contract behind the "Connect Schwab" button: a login must be
    startable even while a valid token is already stored, and it must not be
    completable by a state left over from an earlier, abandoned attempt. Holding
    a token is therefore never a reason to refuse, and forcing drops every
    pending state so the returned one is the only one that can be redeemed. The
    stored token is left alone — an abandoned re-login must not disconnect a
    working session.

    No secrets are logged: the returned `state` is a single-use CSRF nonce, and
    `authorize_url` carries only the app key Schwab already knows.
    """
    schwab_auth = _require_schwab_auth()
    state = secrets.token_urlsafe(24)
    ctx = schwab_auth.get_auth_context(APP_KEY, CALLBACK_URL, state=state)
    if force:
        _pending_states.clear()
    else:
        _prune_states()
    _pending_states[ctx.state] = time.time()
    record_auth_event("login_started")
    return {
        "authorize_url": ctx.authorization_url,
        "callback_url": CALLBACK_URL,
        "state": ctx.state,
        "forced": force,
    }


def build_authorization_url(force: bool = False) -> str:
    """Authorization URL only — the redirect form of `login_context()`."""
    return login_context(force=force)["authorize_url"]


def _latest_pending_state() -> str | None:
    """Most recently issued un-consumed state, or None."""
    _prune_states()
    if not _pending_states:
        return None
    return max(_pending_states, key=_pending_states.__getitem__)


def parse_completion(raw: str) -> tuple[str, str]:
    """Split a pasted redirect URL (or a bare code) into `(code, state)`.

    Schwab requires an https callback, which a plain-http local server cannot
    receive; pasting the address bar is the documented fallback. The pasted
    value never reaches a log — only the extracted `state` does, and only as a
    lookup key.

    Raises AuthError(code="missing_code") with actionable guidance when the
    input carries no authorization code.
    """
    raw = (raw or "").strip()
    if not raw:
        raise AuthError(
            "No authorization code supplied. Paste the full URL Schwab "
            "redirected you to (including ?code=...), or the code by itself.",
            code="missing_code",
        )

    if not raw.lower().startswith(("http://", "https://")):
        return raw, ""

    query = parse_qs(urlparse(raw).query)
    provider_error = (query.get("error") or [""])[0]
    if provider_error:
        raise AuthError(
            f"Schwab returned an OAuth error: {provider_error}",
            code="oauth_denied",
        )
    code = (query.get("code") or [""])[0]
    if not code:
        raise AuthError(
            "That URL has no ?code= parameter. Copy the FULL address-bar URL "
            "from the page Schwab redirected you to.",
            code="missing_code",
        )
    return code, (query.get("state") or [""])[0]


def complete_from_url(raw: str) -> dict:
    """Finish a browser OAuth flow from a pasted redirect URL or bare code.

    Accepts either form, then hands off to `exchange_code()` — one exchange
    path, one state check. When the paste carries no `state` (a bare code), the
    most recent pending state is used: this is a single-user local app, so at
    most one login is ever in flight.
    """
    code, state = parse_completion(raw)
    if not state:
        state = _latest_pending_state() or ""
        if not state:
            raise AuthError(
                "No Schwab login is in progress. Start one with "
                "POST /api/auth/login/start, then paste the redirect URL.",
                code="no_pending_login",
            )
    return exchange_code(code, state)


def exchange_code(code: str, state: str) -> dict:
    """Complete a browser OAuth flow.

    Validates the returned `state` against the pending store, reconstructs the
    received-URL from the registered CALLBACK_URL (so the redirect_uri matches
    exactly regardless of how the browser reached us), exchanges the `code` for
    a token, and persists it to TOKEN_PATH via schwab-py's own writer.

    Returns the auth status dict on success. Raises AuthError on any failure.
    Never returns or logs token material.
    """
    if not code:
        raise AuthError("missing authorization code", code="missing_code")
    _prune_states()
    if state not in _pending_states:
        raise AuthError("invalid or expired OAuth state", code="invalid_state")
    _pending_states.pop(state, None)

    schwab_auth = _require_schwab_auth()
    ctx = schwab_auth.get_auth_context(APP_KEY, CALLBACK_URL, state=state)

    received_url = f"{CALLBACK_URL}?{urlencode({'code': code, 'state': state})}"

    def _write_token(token, *args, **kwargs):
        # schwab-py hands us the metadata-wrapped envelope
        # ({"creation_timestamp": ..., "token": {...}}), which is exactly what
        # client_from_token_file expects to read back. Serialize it verbatim —
        # unwrapping or re-shaping here makes the token unloadable. It goes to
        # the one path GradeA reads, so the client built on the next call loads
        # exactly this file.
        _write_private(active_token_path(), token)

    global _client, _client_source
    _client = schwab_auth.client_from_received_url(
        APP_KEY, APP_SECRET, ctx, received_url, _write_token,
    )
    _client_source = _source_fingerprint()
    return auth_status()
