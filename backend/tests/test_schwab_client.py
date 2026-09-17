"""tests-schwab — coverage for schwab_client.py.

The module reads APP_KEY / APP_SECRET / TOKEN_PATH at import time, so all
tests monkeypatch module attributes directly (not env vars). schwab-py is
NOT installed in this sandbox, so _build()'s `import schwab` is exercised
via sys.modules injection.

Test naming: CLIENT-T1..T13.
"""
from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import schwab_client

# ---------- helpers --------------------------------------------------------

def _make_fake_schwab_module(token_file_client=None, manual_flow_client=None,
                              token_file_raises: Exception | None = None,
                              manual_flow_raises: Exception | None = None):
    """Build a SimpleNamespace mimicking schwab.auth.* for sys.modules injection."""
    tf = MagicMock(name="client_from_token_file")
    if token_file_raises is not None:
        tf.side_effect = token_file_raises
    else:
        tf.return_value = token_file_client or MagicMock(name="token_file_client")

    mf = MagicMock(name="client_from_manual_flow")
    if manual_flow_raises is not None:
        mf.side_effect = manual_flow_raises
    else:
        mf.return_value = manual_flow_client or MagicMock(name="manual_flow_client")

    auth_ns = SimpleNamespace(
        client_from_token_file=tf,
        client_from_manual_flow=mf,
    )
    return SimpleNamespace(auth=auth_ns), tf, mf


@pytest.fixture
def configured(monkeypatch):
    """Module is fully configured with both key + secret."""
    monkeypatch.setattr(schwab_client, "APP_KEY", "test-app-key")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "test-app-secret")
    monkeypatch.setattr(schwab_client, "_client", None)
    return None


def _envelope(access: str = "X", refresh: str = "R") -> str:
    """A schwab-py token envelope, minted now — the shape `_build()` can load."""
    return json.dumps({
        "creation_timestamp": int(time.time()),
        "token": {"access_token": access, "refresh_token": refresh},
    })


@pytest.fixture
def isolated_token_path(monkeypatch, tmp_path):
    """Sandbox GradeA's token path; returns it.

    One path, because FOUNDATION-AUTH-INTERNAL-1 left GradeA with exactly one:
    the file it mints, refreshes and deletes. A real token on the developer's
    machine would otherwise make these look connected.
    """
    p = tmp_path / "tokens.json"
    monkeypatch.setattr(schwab_client, "TOKEN_PATH", p)
    return p


# ---------- CLIENT-T1 ------------------------------------------------------

def test_t1_auth_error_is_exception_subclass():
    assert issubclass(schwab_client.AuthError, Exception)
    # Instantiable with a message.
    err = schwab_client.AuthError("boom")
    assert str(err) == "boom"


# ---------- CLIENT-T2 ------------------------------------------------------

def test_t2_is_configured_false_when_app_key_missing(monkeypatch):
    monkeypatch.setattr(schwab_client, "APP_KEY", "")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "secret")
    assert schwab_client.is_configured() is False


# ---------- CLIENT-T3 ------------------------------------------------------

def test_t3_is_configured_false_when_app_secret_missing(monkeypatch):
    monkeypatch.setattr(schwab_client, "APP_KEY", "key")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "")
    assert schwab_client.is_configured() is False


# ---------- CLIENT-T4 ------------------------------------------------------

def test_t4_is_configured_true_when_both_set(monkeypatch):
    monkeypatch.setattr(schwab_client, "APP_KEY", "key")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "secret")
    assert schwab_client.is_configured() is True


# ---------- CLIENT-T5 ------------------------------------------------------

def test_t5_has_token_reflects_token_path_existence(isolated_token_path):
    # Initially absent.
    assert schwab_client.has_token() is False
    # Create the file.
    isolated_token_path.write_text("{}")
    assert schwab_client.has_token() is True
    # Remove again.
    isolated_token_path.unlink()
    assert schwab_client.has_token() is False


# ---------- CLIENT-T6 ------------------------------------------------------

def test_t6_auth_status_shape(configured, isolated_token_path):
    status = schwab_client.auth_status()
    # Legacy keys are frozen by INV-B; FOUNDATION-AUTH-CONFIG-1 only adds to them.
    assert {"configured", "token_present", "callback_url", "token_path"} <= set(status)
    assert status["configured"] is True
    assert status["token_present"] is False
    assert isinstance(status["callback_url"], str)
    # token_path should reflect our patched TOKEN_PATH.
    assert status["token_path"] == str(isolated_token_path)


# ---------- CLIENT-T7 ------------------------------------------------------

def test_t7_build_raises_auth_error_when_not_configured(monkeypatch):
    monkeypatch.setattr(schwab_client, "APP_KEY", "")
    monkeypatch.setattr(schwab_client, "APP_SECRET", "")
    with pytest.raises(schwab_client.AuthError) as excinfo:
        schwab_client._build()
    assert ".env" in str(excinfo.value)
    assert "SCHWAB_APP_KEY" in str(excinfo.value)


# ---------- CLIENT-T8 ------------------------------------------------------

def test_t8_build_raises_auth_error_when_schwab_import_fails(
    configured, isolated_token_path, monkeypatch
):
    # Force `import schwab` to raise ImportError by setting sys.modules entry
    # to None (Python treats this as "module is known to be unimportable").
    monkeypatch.setitem(sys.modules, "schwab", None)
    with pytest.raises(schwab_client.AuthError) as excinfo:
        schwab_client._build()
    assert "schwab-py" in str(excinfo.value)


# ---------- CLIENT-T9 ------------------------------------------------------

def test_t9_build_uses_token_file_when_token_present(
    configured, isolated_token_path, monkeypatch
):
    isolated_token_path.write_text(_envelope())
    fake_client = MagicMock(name="token_client")
    fake_mod, tf, mf = _make_fake_schwab_module(token_file_client=fake_client)
    monkeypatch.setitem(sys.modules, "schwab", fake_mod)
    monkeypatch.setitem(sys.modules, "schwab.auth", fake_mod.auth)

    result = schwab_client._build()

    assert result is fake_client
    tf.assert_called_once_with(
        token_path=str(isolated_token_path),
        api_key="test-app-key",
        app_secret="test-app-secret",
    )
    mf.assert_not_called()


# ---------- CLIENT-T10 -----------------------------------------------------

def test_t10_build_rejects_an_unusable_token_file(
    configured, isolated_token_path, monkeypatch
):
    """schwab-py's manual flow reads a pasted URL from stdin, which inside an
    HTTP worker means a hung request, so it is never used. An unusable file is
    therefore an error that names the file and points at the way back in."""
    isolated_token_path.write_text("garbage")
    fake_mod, tf, mf = _make_fake_schwab_module()
    monkeypatch.setitem(sys.modules, "schwab", fake_mod)
    monkeypatch.setitem(sys.modules, "schwab.auth", fake_mod.auth)

    with pytest.raises(schwab_client.AuthError) as excinfo:
        schwab_client._build()

    assert excinfo.value.code == "token_unreadable"
    assert str(isolated_token_path) in str(excinfo.value)
    assert "Connect Schwab" in str(excinfo.value)
    assert "8182" not in str(excinfo.value), "no pointer at another project's server"
    tf.assert_not_called()
    mf.assert_not_called()


def test_t10b_build_reports_a_token_schwab_py_refuses(
    configured, isolated_token_path, monkeypatch
):
    """A file of the right shape that schwab-py still rejects is the same
    outcome, without the exception escaping raw."""
    isolated_token_path.write_text(_envelope())
    fake_mod, tf, mf = _make_fake_schwab_module(
        token_file_raises=RuntimeError("corrupt token")
    )
    monkeypatch.setitem(sys.modules, "schwab", fake_mod)
    monkeypatch.setitem(sys.modules, "schwab.auth", fake_mod.auth)

    with pytest.raises(schwab_client.AuthError) as excinfo:
        schwab_client._build()

    assert excinfo.value.code == "token_unreadable"
    tf.assert_called_once()
    mf.assert_not_called()


# ---------- CLIENT-T11 -----------------------------------------------------

def test_t11_build_reports_auth_required_when_no_token_file(
    configured, isolated_token_path, monkeypatch
):
    assert not isolated_token_path.exists()

    fake_mod, tf, mf = _make_fake_schwab_module()
    monkeypatch.setitem(sys.modules, "schwab", fake_mod)
    monkeypatch.setitem(sys.modules, "schwab.auth", fake_mod.auth)

    with pytest.raises(schwab_client.AuthError) as excinfo:
        schwab_client._build()

    assert excinfo.value.code == "auth_required"
    # Names the path that was checked and where to go to fill it.
    assert str(isolated_token_path) in str(excinfo.value)
    assert "Connect Schwab" in str(excinfo.value)
    assert "8182" not in str(excinfo.value), "no pointer at another project's server"
    tf.assert_not_called()
    mf.assert_not_called()


# ---------- CLIENT-T12 -----------------------------------------------------

def test_t12_get_client_caches_result(configured, isolated_token_path, monkeypatch):
    """First call builds; second call returns cached client without re-building."""
    build_calls = {"count": 0}
    sentinel = MagicMock(name="sentinel_client")

    def fake_build():
        build_calls["count"] += 1
        return sentinel

    monkeypatch.setattr(schwab_client, "_build", fake_build)

    first = schwab_client.get_client()
    second = schwab_client.get_client()
    third = schwab_client.get_client()

    assert first is sentinel
    assert second is sentinel
    assert third is sentinel
    assert build_calls["count"] == 1, "get_client should cache _build's result"


# ---------- CLIENT-T13 -----------------------------------------------------

def test_t13_clear_token_deletes_gradeas_token_and_clears_cache(
    configured, isolated_token_path, monkeypatch
):
    """`clear_token()` deletes the token and drops the cached client.

    A real sign-out since FOUNDATION-AUTH-INTERNAL-1 (INV-16): GradeA owns this
    credential outright, so deleting the file ends the session rather than
    dropping one copy of a file another program keeps alive.
    """
    sentinel = MagicMock(name="cached")
    monkeypatch.setattr(schwab_client, "_client", sentinel)
    isolated_token_path.write_text(_envelope())
    assert schwab_client._client is sentinel

    # First call: token present → returns True, file gone, cache cleared.
    assert schwab_client.clear_token() is True
    assert not isolated_token_path.exists()
    assert schwab_client._client is None

    # Second call: nothing left → returns False, no error.
    assert schwab_client.clear_token() is False
    assert schwab_client._client is None
