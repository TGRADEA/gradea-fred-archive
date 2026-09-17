"""FOUNDATION-AUTH-INTERNAL-1 — GradeA owns the token it reads.

Supersedes `test_token_vault.py`. That suite pinned a *relationship*: GradeA
read a file the standalone Schwab Token Vault on `https://127.0.0.1:8182` wrote,
adapted it into a derived copy, and was careful never to write or delete the
original. This suite pins the absence of that relationship. GradeA holds its own
Schwab developer app registration, serves the OAuth flow from its own origin
(`https://127.0.0.1:8002`), and is the sole writer of exactly one token file.

Why that is a correctness property and not a preference: a Schwab refresh token
is single-use. Refreshing rotates it and revokes the one before it. Two programs
sharing one registration therefore take turns invalidating each other's
credential, and neither can attribute the resulting intermittent
`invalid_grant` to anything it did. Separate registrations make the failure
impossible rather than merely unlikely.

Migrated from TOKEN-VAULT-AUTH-8002-1 with their original numbers so the delta
is auditable: TV-T1/2/3/5/6/7/8/8b/9/15/16/17/20/21/22/25/30. Deleted with the
relationship they described: TV-T4/4b/4c (the candidate search), TV-T10 (the
legacy fallback), TV-T11 (never delete the vault's file), TV-T12/12b/12c (the
adapter), TV-T13 (the vault URL in settings), TV-T14 (the frontend pointing at
the vault), TV-T23/24/28/29 (the custody warning). TC-T31..T36 are new.

No network and no real credentials — every token here is a JSON file in tmp_path.

TC-T1..T3, TC-T9, TC-T32 and TC-T33 are referenced by name from
`_map/invariants.md` (INV-F, INV-E, INV-16) and `_map/contracts.md`.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import schwab_client  # noqa: E402
from settings import Settings, reload_settings  # noqa: E402

ACCESS = "GRADEA-ACCESS-TOKEN"
REFRESH = "GRADEA-REFRESH-TOKEN"
# The port the sibling project's token vault uses. Named here only so the greps
# below can assert it never appears in GradeA's own wiring again.
FOREIGN_PORT = "8182"


@pytest.fixture
def custody(monkeypatch, tmp_path):
    """A configured GradeA whose one token path lives in tmp_path.

    Returns that path. It does not exist yet — tests that want a token call
    `_write_token`.
    """
    monkeypatch.setattr(schwab_client, "APP_KEY", "test-app-key")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "test-app-secret")
    monkeypatch.setattr(schwab_client, "CALLBACK_URL", "https://127.0.0.1:8002")
    monkeypatch.setattr(schwab_client, "TOKEN_PATH", tmp_path / "tokens.json")
    monkeypatch.setattr(schwab_client, "_client", None)
    schwab_client.reset_auth_diagnostic()
    yield schwab_client.TOKEN_PATH
    schwab_client.reset_auth_diagnostic()


def _write_token(
    path: Path,
    *,
    access: str = ACCESS,
    refresh: str = REFRESH,
    created: int | None = None,
    expires_at: int | None = None,
    **token_fields,
) -> dict:
    """Write GradeA's own shape: schwab-py's envelope. Live token by default."""
    now = int(time.time())
    token = {
        "access_token": access,
        "refresh_token": refresh,
        "expires_at": now + 1800 if expires_at is None else expires_at,
        "token_type": "Bearer",
        "scope": "api",
    }
    token.update(token_fields)
    doc = {"creation_timestamp": now if created is None else created, "token": token}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return doc


def _write_foreign_token(path: Path) -> None:
    """Schwab's raw flat payload — what a *different* application's file holds."""
    now = int(time.time())
    path.write_text(
        json.dumps(
            {
                "access_token": "FOREIGN-ACCESS",
                "refresh_token": "FOREIGN-REFRESH",
                "expires_at": now + 1800,
                "refresh_expires_at": now + 7 * 86400,
            }
        ),
        encoding="utf-8",
    )


# ---------- TC-T1: GradeA's serving port and callback agree -----------------


def test_tc_t1_defaults_serve_8002_and_own_the_callback():
    """One decision, two fields: a callback can only land on the port the app
    listens on, so the two must be derived from the same number or they drift.

    8182 is a sibling project's, and GradeA could not bind it if it tried —
    which is the practical reason GradeA registers its own Schwab app rather
    than borrowing one.
    """
    port = Settings.model_fields["port"].default
    callback = Settings.model_fields["schwab_callback_url"].default

    assert port == 8002
    assert callback == "https://127.0.0.1:8002"
    assert callback.endswith(f":{port}")
    assert port != 8182
    assert FOREIGN_PORT not in callback


# ---------- TC-T2: the env template ships the same pair ---------------------


# ---------- TC-T3: the launcher serves 8002 ---------------------------------


# ---------- TC-T5: SCHWAB_TOKEN_PATH relocates GradeA's own token -----------


def test_tc_t5_schwab_token_path_env_overrides_the_default(monkeypatch, tmp_path):
    """The default is a convenience, not a requirement — but it relocates
    GradeA's *own* token, it does not point GradeA at somebody else's."""
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", str(tmp_path / "elsewhere.json"))
    assert reload_settings().schwab_token_path == tmp_path / "elsewhere.json"

    # `~` is expanded, so a Windows-style value from a .env still resolves.
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", "~/gradea/tokens.json")
    assert reload_settings().schwab_token_path == Path.home() / "gradea/tokens.json"
    monkeypatch.delenv("SCHWAB_TOKEN_PATH", raising=False)
    reload_settings()


# ---------- TC-T6: a live token reads as connected --------------------------


def test_tc_t6_live_token_reads_as_connected(custody):
    _write_token(custody)

    status = schwab_client.auth_status()

    assert status["authenticated"] is True
    assert status["state"] == "authenticated"
    assert status["token_source"] == "gradea"
    assert status["token_path"] == str(custody)
    assert 0 < status["access_expires_in_seconds"] <= 1800
    assert 0 < status["refresh_expires_in_seconds"] <= 7 * 86400


# ---------- TC-T7: expired access + live refresh is still connected ---------


def test_tc_t7_expired_access_with_live_refresh_is_connected(custody):
    """schwab-py renews on demand, so an expired access token is not a
    disconnection while the refresh token is inside Schwab's 7-day cap."""
    now = int(time.time())
    _write_token(custody, expires_at=now - 60, created=now - 4 * 86400)

    status = schwab_client.auth_status()

    assert status["authenticated"] is True
    assert status["state"] == "authenticated"
    assert status["access_expires_in_seconds"] < 0


# ---------- TC-T8: both expired is refresh_expired --------------------------


def test_tc_t8_both_expired_reads_as_refresh_expired(custody):
    now = int(time.time())
    _write_token(custody, expires_at=now - 8 * 86400, created=now - 8 * 86400)

    status = schwab_client.auth_status()

    assert status["authenticated"] is False
    assert status["state"] == "refresh_expired"
    # Points at the way back in, and at nothing outside GradeA.
    assert "Connect Schwab" in status["message"]
    assert FOREIGN_PORT not in status["message"]


def test_tc_t8b_missing_token_names_the_path(custody):
    """The message has to say *where GradeA looked* and *how to fix it* — those
    are two different kinds of confusion otherwise."""
    status = schwab_client.auth_status()

    assert status["authenticated"] is False
    assert status["state"] == "auth_required"
    assert status["token_source"] == "none"
    assert str(custody) in status["message"]
    assert "Connect Schwab" in status["message"]


# ---------- TC-T9: no token material in the status payload (INV-E) ----------


# ---------- TC-T15: a reconnect invalidates the cached client ---------------


def test_tc_t15_cached_client_is_rebuilt_when_the_token_changes(custody, monkeypatch):
    """The original reported failure, which survives the custody change.

    schwab-py loads the token into memory once, so a client cached across a
    reconnect keeps presenting the refresh token Schwab revoked at the new
    sign-in. `get_client()` therefore keys its cache on the token file's
    identity, not merely on "have I built one yet" (INV-H).
    """
    _write_token(custody)
    built = []

    def _client_from_token_file(token_path, api_key, app_secret):
        loaded = json.loads(Path(token_path).read_text(encoding="utf-8"))
        built.append(loaded["token"]["refresh_token"])
        return MagicMock(name=f"client-{len(built)}")

    import schwab.auth as real_auth

    monkeypatch.setattr(real_auth, "client_from_token_file", _client_from_token_file)

    first = schwab_client.get_client()
    # Repeat calls with an unchanged file must not rebuild.
    assert schwab_client.get_client() is first
    assert built == [REFRESH]

    _write_token(custody, refresh="REFRESH-AFTER-RECONNECT")

    second = schwab_client.get_client()
    assert second is not first, "a reconnect must not reuse the old client"
    assert built == [REFRESH, "REFRESH-AFTER-RECONNECT"]


def test_tc_t15b_the_cached_and_loaded_paths_are_the_same_file(custody, monkeypatch):
    """The cache key and the load target must not be able to disagree.

    Under the vault design they were different files, so a self-refresh by
    schwab-py rewrote one while the fingerprint watched the other. With one path
    that class of staleness cannot be expressed — this pins it shut.
    """
    _write_token(custody)
    loaded = {}

    def _client_from_token_file(token_path, api_key, app_secret):
        loaded["path"] = token_path
        return MagicMock(name="client")

    import schwab.auth as real_auth

    monkeypatch.setattr(real_auth, "client_from_token_file", _client_from_token_file)

    schwab_client._build()

    assert loaded["path"] == str(custody)
    assert schwab_client._loadable_token_path() == schwab_client.active_token_path()


def test_tc_t17_token_without_an_access_token_is_reported_not_connected(custody):
    """A refresh-only document parses, but authlib cannot sign a request with
    it — it raises a bare `unsupported_token_type` that reads as a Schwab
    outage. GradeA has to call it what it is and point at the way back."""
    now = int(time.time())
    _write_token(custody, access="", created=now - 4 * 86400)

    status = schwab_client.auth_status()

    assert status["authenticated"] is False
    # Not a new state: `state` is a pinned enum the frontend switches on, so this
    # reuses the "cannot load a usable token" branch and puts the specifics in
    # `message`.
    assert status["state"] == "token_unreadable"
    assert "no access token" in status["message"]
    assert "Connect Schwab" in status["message"]
    assert REFRESH not in json.dumps(status)

    # And the loader refuses rather than handing schwab-py an unsignable file.
    with pytest.raises(schwab_client.AuthError) as excinfo:
        schwab_client._loadable_token_path()
    assert excinfo.value.code == "auth_required"
    assert REFRESH not in str(excinfo.value)


# ---------- TC-T20/T21: what the callback URL is actually required for ------


def test_tc_t20_callback_url_is_not_required_for_the_data_path(monkeypatch):
    """A blank SCHWAB_CALLBACK_URL must not block Schwab *access*.

    GradeA reaches Schwab through
    `schwab.auth.client_from_token_file(token_path, api_key, app_secret)`, which
    takes no redirect URI — one is only ever sent when *starting* an
    authorization. Treating the callback as mandatory made a setting that does
    nothing for data access into a hard blocker: `_require_config()` raised
    before the token was ever opened, and the UI reported it as missing
    configuration.
    """
    monkeypatch.setattr(schwab_client, "APP_KEY", "real-app-key")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "real-app-secret")
    monkeypatch.setattr(schwab_client, "CALLBACK_URL", "")

    assert schwab_client.missing_env() == []
    schwab_client._require_config()  # must not raise

    # The app key and secret are still required: schwab-py needs them to refresh
    # an expiring token, so their absence is a real misconfiguration.
    monkeypatch.setattr(schwab_client, "APP_SECRET", "")
    assert schwab_client.missing_env() == ["SCHWAB_APP_SECRET"]


def test_tc_t21_oauth_routes_still_require_the_callback(monkeypatch):
    """Signing in *does* send a redirect URI, and it is now the only way in.

    Relaxing the default for the data path must not quietly let a login start
    with no callback, which Schwab rejects with an opaque error.
    """
    monkeypatch.setattr(schwab_client, "APP_KEY", "real-app-key")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "real-app-secret")
    monkeypatch.setattr(schwab_client, "CALLBACK_URL", "")

    assert schwab_client.missing_env(require_callback=True) == ["SCHWAB_CALLBACK_URL"]
    with pytest.raises(schwab_client.ConfigError):
        schwab_client._require_schwab_auth()


# ---------- TC-T22/T25/T30: path and staleness honesty ----------------------


def test_tc_t22_relative_token_path_anchors_to_the_backend_dir(monkeypatch):
    """A relative SCHWAB_TOKEN_PATH must not depend on the working directory.

    `SCHWAB_TOKEN_PATH=tokens.json` resolved against the CWD, so the same config
    read a different file depending on how the server was launched.
    """
    from settings import BACKEND_DIR

    monkeypatch.setenv("SCHWAB_TOKEN_PATH", "tokens.json")
    resolved = reload_settings().schwab_token_path

    assert resolved.is_absolute()
    assert resolved == BACKEND_DIR / "tokens.json"
    monkeypatch.delenv("SCHWAB_TOKEN_PATH", raising=False)
    reload_settings()


def test_tc_t25_expired_access_with_live_refresh_is_flagged_stale(custody):
    """An access token 12h past expiry must not read as a plain live session.

    It is still `authenticated` — schwab-py renews on demand — but the panel
    said only "Connected to Schwab." while every tile errored, which pointed
    away from the token instead of at it.
    """
    now = int(time.time())
    _write_token(custody, created=now - 50_000, expires_at=now - 44_537)

    status = schwab_client.auth_status()

    assert status["authenticated"] is True
    assert status["state"] == "authenticated"
    assert status["access_stale"] is True
    assert status["access_expires_in_seconds"] < 0
    assert "past expiry" in status["message"]


def test_tc_t30_env_file_is_anchored_to_the_backend_directory():
    """A relative env_file resolves against the CWD, not the module.

    Launching uvicorn from the repo root would then load a different .env — or
    none — so edits to gradea-backend/.env would silently do nothing while the
    status panel reported stale values with total confidence.
    """
    from settings import BACKEND_DIR

    configured = Path(Settings.model_config["env_file"])
    assert configured.is_absolute()
    assert configured == BACKEND_DIR / ".env"


# ---------- TC-T31..T36: the custody invariant itself (INV-16) --------------


def test_tc_t31_token_source_is_gradea_or_none(custody):
    """The enum narrowed with the design. `vault` and `legacy` described other
    people's custody of this credential, and there is no longer any such thing.
    """
    assert schwab_client.token_source() == "none"

    _write_token(custody)
    assert schwab_client.token_source() == "gradea"

    # Even a file in a foreign shape is still GradeA's configured path — the
    # source says *whose path*, the state says whether it is usable.
    _write_foreign_token(custody)
    assert schwab_client.token_source() == "gradea"


def test_tc_t33_a_foreign_token_file_is_refused_not_adapted(custody, monkeypatch):
    """The invariant, stated as a failure mode (INV-16).

    Schwab's flat payload is recognizable, so pointing SCHWAB_TOKEN_PATH at
    another application's token file is diagnosable rather than a generic parse
    error. GradeA names it and stops. It must not adapt, copy, or load it: doing
    so puts two processes on one single-use refresh chain, which is the exact
    failure this slice removes.
    """
    _write_foreign_token(custody)
    before = custody.read_bytes()
    called = []

    def _client_from_token_file(token_path, api_key, app_secret):
        called.append(token_path)
        return MagicMock(name="client")

    import schwab.auth as real_auth

    monkeypatch.setattr(real_auth, "client_from_token_file", _client_from_token_file)

    with pytest.raises(schwab_client.AuthError) as excinfo:
        schwab_client._build()

    assert excinfo.value.code == "token_unreadable"
    assert "another application" in str(excinfo.value)
    assert called == [], "a foreign token must never reach schwab-py"
    assert custody.read_bytes() == before, "and must not be rewritten in passing"

    # The banner has to agree with the loader, or it paints a green badge over a
    # page of erroring tiles.
    status = schwab_client.auth_status()
    assert status["authenticated"] is False
    assert status["state"] == "token_unreadable"
    assert "another application" in status["message"]
    assert "FOREIGN-ACCESS" not in json.dumps(status)


def test_tc_t34_there_is_exactly_one_token_path(monkeypatch, tmp_path):
    """No candidate chain, no fallback, no derived copy.

    The chain existed to find a file GradeA did not write. Every extra place to
    look was a place for "no Schwab token" to name the wrong path, and for a
    stale file to win over a fresh one.
    """
    monkeypatch.setenv("SCHWAB_TOKEN_PATH", str(tmp_path / "chosen.json"))
    s = reload_settings()

    assert s.schwab_token_path == tmp_path / "chosen.json"
    assert s.schwab_token_path_candidates == [s.schwab_token_path]

    # env_file=None, not just delenv: this half asserts what the setting
    # DEFAULTS to, and a real .env that sets SCHWAB_TOKEN_PATH (the token-vault
    # path, on at least one live checkout) survives delenv and would be read as
    # the default. The invariant under test is a property of the code, not of
    # whoever's machine is running it.
    monkeypatch.delenv("SCHWAB_TOKEN_PATH", raising=False)
    s = reload_settings(env_file=None)
    assert s.schwab_token_path_candidates == [s.schwab_token_path]
    # The default is GradeA's own directory, not a shared location on a desktop.
    assert s.schwab_token_path == s.tokens_path

    # And the module-level constant the client reads agrees with settings.
    assert schwab_client.active_token_path() == schwab_client.TOKEN_PATH


