"""Centralized configuration via Pydantic Settings.

All env vars the app reads live here. Modules import `settings` from this file
and access typed attributes — no scattered `os.getenv` calls.

Loading order:
1. Defaults from this file
2. .env file (if present, gitignored)
3. Process environment (highest priority)

Run-time validation: pydantic raises on bad values at startup, not at first use.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The backend package directory. Relative token paths anchor here rather than
# to the working directory, which varies by launcher.
BACKEND_DIR = Path(__file__).resolve().parent

# Universe defaults live in universe.py (INV-UNI-1, FOUNDATION-UNIVERSE-1).
# These names are kept so env overrides and existing imports keep working;
# they are views into the registry seeds, not declarations.
from universe import seed_csv as _seed_csv

DEFAULT_EARNINGS_UNIVERSE = _seed_csv("earnings_default")
DEFAULT_RECORDER_SYMBOLS = _seed_csv("recorder_default")  # UNIVERSE-CONFIG-1
DEFAULT_TRACE_SYMBOLS = _seed_csv("trace_default")
DEFAULT_STREAM_SYMBOLS = _seed_csv("stream_default")


def _csv(v: str | list[str]) -> list[str]:
    if isinstance(v, list):
        return v
    return [s.strip() for s in v.split(",") if s.strip()]


# ── Schwab config validation (FOUNDATION-AUTH-CONFIG-1) ──────────────────────
# A copied-but-unedited env.example is indistinguishable from a real config
# unless the template values are rejected outright — otherwise GradeA builds a
# Schwab authorize URL with client_id=your_schwab_app_key_here and Schwab
# rejects the login with an opaque error.

SCHWAB_ENV_KEYS = ("SCHWAB_APP_KEY", "SCHWAB_APP_SECRET", "SCHWAB_CALLBACK_URL")

# Only these are needed to *use* Schwab. The data path reaches Schwab through
# `schwab.auth.client_from_token_file(token_path, api_key, app_secret)`, which
# takes no redirect URI: a redirect URI is only ever sent when *starting* an
# authorization. The app key and secret are always required, because schwab-py
# needs them to refresh an expiring token.
#
# SCHWAB_CALLBACK_URL is therefore optional at import time. It is validated when
# set and required by the OAuth routes, but demanding it up front turned a
# setting that does nothing for data access into a hard blocker for reading an
# already-stored token.
SCHWAB_REQUIRED_ENV_KEYS = ("SCHWAB_APP_KEY", "SCHWAB_APP_SECRET")

PLACEHOLDER_REASON = "still set to an env.example placeholder"
NOT_HTTP_URL_REASON = "not an http(s) URL"

_PLACEHOLDER_VALUES = frozenset({
    "your_schwab_app_key_here",
    "your_schwab_app_secret_here",
    "your_app_key",
    "your_app_secret",
    "your_client_id",
    "your_client_secret",
    "your_registered_schwab_redirect_uri",
    "changeme",
    "change_me",
    "todo",
})

# Anchored/word-bounded so real credentials are never flagged. In particular
# `test-app-key` and `test-client-id` must stay valid — fixtures depend on it.
_PLACEHOLDER_PATTERNS = (
    re.compile(r"^your[_\- ]"),      # your_app_key, your-client-id
    re.compile(r"[_\- ]here$"),      # ..._here
    re.compile(r"^<.+>$"),           # <paste your key>
    re.compile(r"placeholder"),
    re.compile(r"\bexample\b"),      # env.example leftovers, example.com
    re.compile(r"change[_\- ]?me"),
)


def is_placeholder(value: str | None) -> bool:
    """True when a value is an obvious stand-in rather than a real credential.

    Empty is *missing*, not placeholder — the two are reported separately so
    the operator knows whether to add a key or replace a template value.
    """
    candidate = (value or "").strip().lower()
    if not candidate:
        return False
    if candidate in _PLACEHOLDER_VALUES:
        return True
    return any(p.search(candidate) for p in _PLACEHOLDER_PATTERNS)


def callback_url_problem(value: str | None) -> str | None:
    """Secret-free reason the callback URL is unusable, or None if it is fine."""
    parsed = urlparse((value or "").strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return NOT_HTTP_URL_REASON
    return None


def missing_schwab_keys(
    app_key: str, app_secret: str, callback_url: str = "", *, require_callback: bool = False
) -> list[str]:
    """Env keys that are absent or blank, in SCHWAB_ENV_KEYS order.

    `callback_url` is only reported missing when `require_callback` is set,
    which the OAuth routes do. Reading an already-stored token does not need it
    -- see SCHWAB_REQUIRED_ENV_KEYS.
    """
    keys = SCHWAB_ENV_KEYS if require_callback else SCHWAB_REQUIRED_ENV_KEYS
    values = {
        "SCHWAB_APP_KEY": app_key,
        "SCHWAB_APP_SECRET": app_secret,
        "SCHWAB_CALLBACK_URL": callback_url,
    }
    return [k for k in keys if not (values[k] or "").strip()]


def invalid_schwab_keys(app_key: str, app_secret: str, callback_url: str) -> dict[str, str]:
    """Env keys that are set but unusable, mapped to a secret-free reason."""
    problems: dict[str, str] = {}
    for key, value in zip(SCHWAB_ENV_KEYS, (app_key, app_secret, callback_url), strict=True):
        if (value or "").strip() and is_placeholder(value):
            problems[key] = PLACEHOLDER_REASON
    if (callback_url or "").strip() and "SCHWAB_CALLBACK_URL" not in problems:
        problem = callback_url_problem(callback_url)
        if problem:
            problems["SCHWAB_CALLBACK_URL"] = problem
    return problems


def describe_config_problem(missing: list[str], invalid: dict[str, str]) -> str:
    """One sentence naming what is wrong with .env. Never includes a value."""
    parts = []
    if missing:
        parts.append("missing " + ", ".join(missing))
    if invalid:
        parts.append(
            "placeholder or invalid "
            + ", ".join(f"{k} ({r})" for k, r in sorted(invalid.items()))
        )
    detail = "; ".join(parts) if parts else "incomplete"
    return (
        f"Schwab app config in gradea-backend/.env is {detail}. Copy "
        "gradea-backend/env.example to gradea-backend/.env, replace the "
        "placeholder values with your Schwab app credentials, then restart GradeA."
    )


class Settings(BaseSettings):
    """Application settings — single source of truth."""

    model_config = SettingsConfigDict(
        # Absolute, not ".env": pydantic-settings resolves a relative env_file
        # against the *process working directory*, so launching uvicorn from the
        # repo root instead of gradea-backend silently loads a different file --
        # or none -- and every edit to gradea-backend/.env appears to do nothing.
        # Same failure shape as the relative SCHWAB_TOKEN_PATH fixed in
        # AUTH-TOKEN-PATH-HONESTY-1; that slice anchored the token path and left
        # the file naming it still CWD-relative.
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ───────────────────────────────────────────────────
    host: str = Field("127.0.0.1", validation_alias="GRADEA_HOST")
    # 8002 is GradeA's own origin and the origin registered on GradeA's own
    # Schwab developer app, so Schwab's redirect lands here. Sibling projects
    # run their own ports and their own registrations; nothing is shared, which
    # is what keeps their refresh chains independent (INV-F, INV-16).
    port: int = Field(8002, validation_alias="GRADEA_PORT")
    log_level: str = Field("INFO", validation_alias="GRADEA_LOG_LEVEL")
    allowed_origins_raw: str = Field(
        "http://127.0.0.1:5173",
        validation_alias="GRADEA_ALLOWED_ORIGINS",
    )

    @property
    def allowed_origins(self) -> list[str]:
        return _csv(self.allowed_origins_raw)

    # ── Storage ──────────────────────────────────────────────────
    data_dir_raw: str | None = Field(None, validation_alias="GRADEA_DATA_DIR")
    db_path_raw: str | None = Field(None, validation_alias="GRADEA_DB_PATH")

    @property
    def data_dir(self) -> Path:
        if self.data_dir_raw:
            return Path(self.data_dir_raw)
        # Default: directory containing this settings.py
        return Path(__file__).resolve().parent

    @property
    def db_path(self) -> Path:
        if self.db_path_raw:
            return Path(self.db_path_raw)
        return self.data_dir / "gradea.db"

    @property
    def tokens_path(self) -> Path:
        """GradeA's own token file. Same path as `schwab_token_path` unless
        SCHWAB_TOKEN_PATH overrides it; kept as a named property because
        several call sites read "the file GradeA writes" rather than "the file
        GradeA was configured to read"."""
        return self.data_dir / "tokens.json"

    # ── Schwab token custody (FOUNDATION-AUTH-INTERNAL-1) ────────
    # GradeA owns its own Schwab app registration and runs its own OAuth flow,
    # so it is the only writer of this file. There is no candidate search and
    # no external custodian: one project, one credential, one token path. The
    # Schwab Token Vault on 8182 is BullMoose's internal auth and is not part
    # of GradeA's data path.
    schwab_token_path_raw: str | None = Field(
        None, validation_alias="SCHWAB_TOKEN_PATH"
    )

    @staticmethod
    def _absolute(raw: str) -> Path:
        """Expand `~` and anchor a relative value to the backend directory.

        A relative SCHWAB_TOKEN_PATH is almost never what someone means, and it
        resolves against the *working directory* -- so the same config reads a
        different file depending on where the server was launched from. Worse,
        a value meant as an absolute path but written relative silently reads
        some other file entirely. Anchoring to BACKEND_DIR makes the value
        deterministic, and `token_path` in the status payload always reports the
        path actually used.
        """
        expanded = Path(raw).expanduser()
        return expanded if expanded.is_absolute() else BACKEND_DIR / expanded

    @property
    def schwab_token_path(self) -> Path:
        """The one token file GradeA reads and writes, always absolute.

        `SCHWAB_TOKEN_PATH` wins when set; otherwise GradeA's own
        `<data_dir>/tokens.json`. Deliberately a single value with no fallback
        search: a candidate list can only ever answer "we looked somewhere
        else too", which is the shape of the bug it used to hide. If the file
        is not there, the error names this exact path.
        """
        if self.schwab_token_path_raw:
            return self._absolute(self.schwab_token_path_raw)
        return self.tokens_path

    @property
    def schwab_token_path_candidates(self) -> list[Path]:
        """Kept for the status payload's diagnostics field. Exactly one entry
        now, because exactly one path is ever consulted."""
        return [self.schwab_token_path]

    # ── Schwab ───────────────────────────────────────────────────
    schwab_app_key: str = Field("", validation_alias="SCHWAB_APP_KEY")
    schwab_app_secret: str = Field("", validation_alias="SCHWAB_APP_SECRET")
    # GradeA's own OAuth routes are the only auth path (FOUNDATION-AUTH-INTERNAL-1).
    # This must match the callback registered on GradeA's OWN Schwab developer
    # app, byte for byte -- a separate registration from any sibling project's,
    # which is what keeps the refresh chains independent.
    schwab_callback_url: str = Field(
        "https://127.0.0.1:8002",
        validation_alias="SCHWAB_CALLBACK_URL",
    )

    def _schwab_values(self) -> tuple[str, str, str]:
        return (self.schwab_app_key, self.schwab_app_secret, self.schwab_callback_url)

    def schwab_missing_keys(self) -> list[str]:
        return missing_schwab_keys(*self._schwab_values())

    def schwab_invalid_keys(self) -> dict[str, str]:
        return invalid_schwab_keys(*self._schwab_values())

    @property
    def schwab_configured(self) -> bool:
        """True when the minimum Schwab app config is present and usable."""
        return not self.schwab_missing_keys() and not self.schwab_invalid_keys()

    # ── Supabase (research corpus) ───────────────────────────────
    # The VTS email archive and the LaDuc mirror both live in Supabase. The
    # service key bypasses RLS and must stay server-side: it is read here,
    # used by sources/supabase.py, and never serialised into any /api response.
    supabase_url: str = Field("", validation_alias="SUPABASE_URL")
    supabase_service_key: str = Field("", validation_alias="SUPABASE_SERVICE_KEY")
    research_cache_ttl: int = Field(
        900, validation_alias="GRADEA_RESEARCH_CACHE_TTL"
    )

    @property
    def supabase_configured(self) -> bool:
        """True when both the project URL and a service key are present."""
        return bool(self.supabase_url and self.supabase_service_key)

    # ── Telegram (signal pushes; SIGNAL-BACKWARDATION-2) ─────────
    # Direct Bot API credentials for backwardation-tracker pushes.
    # Server-side only, never serialised into any /api response; the wire
    # exposes only the boolean `telegram_configured`.
    telegram_bot_token: str = Field("", validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field("", validation_alias="TELEGRAM_CHAT_ID")

    # ── EDGAR ────────────────────────────────────────────────────
    edgar_user_agent: str = Field(
        "GradeA Trading research@example.com",
        validation_alias="EDGAR_USER_AGENT",
    )

    # ── Supabase (VTS archive mirror; RESEARCH-VTS-1) ────────────
    # Read-only anon key. Server-side only: the browser sees /api/research/*
    # endpoints, never these values. Empty means the VTS archive surface
    # answers 503 (unconfigured), not an empty archive.
    supabase_url: str = Field("", validation_alias="SUPABASE_URL")
    supabase_key: str = Field("", validation_alias="SUPABASE_KEY")
    vts_status_cache_ttl: int = Field(
        300, validation_alias="GRADEA_VTS_STATUS_CACHE_TTL"
    )

    # ── Streaming ────────────────────────────────────────────────
    stream_interval: int = Field(15, validation_alias="GRADEA_STREAM_INTERVAL")
    stream_symbols_raw: str = Field(
        DEFAULT_STREAM_SYMBOLS,
        validation_alias="GRADEA_STREAM_SYMBOLS",
    )
    tape_queue_max: int = Field(256, validation_alias="GRADEA_TAPE_QUEUE_MAX")

    @property
    def stream_symbols(self) -> list[str]:
        return [s.upper() for s in _csv(self.stream_symbols_raw)]

    # ── Earnings ─────────────────────────────────────────────────
    earnings_universe_raw: str = Field(
        DEFAULT_EARNINGS_UNIVERSE,
        validation_alias="GRADEA_EARNINGS_UNIVERSE",
    )
    earnings_cache_ttl: int = Field(3600, validation_alias="GRADEA_EARNINGS_CACHE_TTL")

    @property
    def earnings_universe(self) -> list[str]:
        return [s.upper() for s in _csv(self.earnings_universe_raw)]

    # ── Dark pool ────────────────────────────────────────────────
    darkpool_cache_ttl: int = Field(21600, validation_alias="GRADEA_DARKPOOL_CACHE_TTL")

    # ── Chain cache ──────────────────────────────────────────────
    schwab_chain_cache_ttl_sec: int = Field(
        30, validation_alias="SCHWAB_CHAIN_CACHE_TTL_SEC"
    )

    # ── Chain snapshots (FOUNDATION-CHAIN-SNAPSHOT-1) ────────────
    # Append-only option-chain snapshot store. Empty symbols list disables
    # the scheduler (manual CLI captures still work with explicit symbols).
    snapshot_symbols_raw: str = Field(
        DEFAULT_RECORDER_SYMBOLS,
        validation_alias="GRADEA_SNAPSHOT_SYMBOLS",
    )
    snapshot_interval_min: int = Field(30, validation_alias="GRADEA_SNAPSHOT_INTERVAL_MIN")
    snapshot_max_dte: int = Field(120, validation_alias="GRADEA_SNAPSHOT_MAX_DTE")

    # ── Chain snapshot mirror (FOUNDATION-CHAIN-SNAPSHOT-MIRROR-1) ─
    # Off-box copy of the snapshot store in Supabase. Off by default: the
    # mirror also needs SUPABASE_URL + SUPABASE_SERVICE_KEY, and it must
    # never start pushing a real account's chain captures by accident.
    snapshot_mirror_enabled: bool = Field(
        False, validation_alias="GRADEA_SNAPSHOT_MIRROR_ENABLED"
    )
    snapshot_mirror_interval_min: int = Field(
        5, validation_alias="GRADEA_SNAPSHOT_MIRROR_INTERVAL_MIN"
    )

    @property
    def snapshot_symbols(self) -> list[str]:
        return [s.upper() for s in _csv(self.snapshot_symbols_raw)]

    # ── Karsan composite recorder (KARSAN-CONFIG-1) ──────────────
    # Daily near-close composite rows. Empty symbols list disables the
    # scheduler (manual CLI records still work with explicit symbols).
    karsan_symbols_raw: str = Field(
        DEFAULT_RECORDER_SYMBOLS,
        validation_alias="GRADEA_KARSAN_SYMBOLS",
    )

    @property
    def karsan_symbols(self) -> list[str]:
        return [s.upper() for s in _csv(self.karsan_symbols_raw)]

    # ── Chain trace ──────────────────────────────────────────────
    trace_poll_seconds: int = Field(10, validation_alias="GRADEA_TRACE_POLL_SECONDS")
    trace_keep: int = Field(200, validation_alias="GRADEA_TRACE_KEEP")
    trace_default_symbols_raw: str = Field(
        DEFAULT_TRACE_SYMBOLS,
        validation_alias="GRADEA_TRACE_SYMBOLS",
    )

    @property
    def trace_default_symbols(self) -> list[str]:
        return [s.upper() for s in _csv(self.trace_default_symbols_raw)]

    # ── Strategy signals ─────────────────────────────────────────
    signals_interval: int = Field(300, validation_alias="GRADEA_SIGNALS_INTERVAL")
    signals_bars: int = Field(120, validation_alias="GRADEA_SIGNALS_BARS")

    # ── Schwab rate limit (FOUNDATION-RATE-LIMIT-1) ─────────────────
    # Schwab's documented REST budget is ~65-70 req/min (see cache.py). 55
    # sits conservatively below that ceiling, leaving headroom for calls
    # schwab-py itself issues (token refresh) outside this gate's counting.
    schwab_rate_limit_per_min: int = Field(
        55, validation_alias="GRADEA_SCHWAB_RATE_LIMIT_PER_MIN"
    )
    # A caller on an empty bucket blocks rather than fails fast (these calls
    # serve synchronous dashboard requests, and a short wait beats an error
    # tile for a page load that's simply eager). This caps how long any one
    # call will wait for a bucket slot before raising RateLimitExceeded.
    schwab_rate_limit_wait_cap_sec: float = Field(
        10.0, validation_alias="GRADEA_SCHWAB_RATE_LIMIT_WAIT_CAP_SEC"
    )
    # Bounded retry budget for a Schwab 429 response, independent of the
    # bucket wait above.
    schwab_429_max_retries: int = Field(
        3, validation_alias="GRADEA_SCHWAB_429_MAX_RETRIES"
    )
    # Ceiling on how long a single Retry-After wait is honored. Schwab's
    # own windows can run ~60s; capping at 30s means a caller fails loudly
    # rather than blocking a dashboard request for a full minute.
    schwab_429_max_wait_sec: float = Field(
        30.0, validation_alias="GRADEA_SCHWAB_429_MAX_WAIT_SEC"
    )


# Singleton — import this in modules. Call settings.refresh() in tests if needed.
settings = Settings()


_UNSET = object()


def reload_settings(env_file: object = _UNSET) -> Settings:
    """Force re-read from environment. Useful in tests.

    `env_file=None` reads the process environment ONLY, ignoring
    gradea-backend/.env. A test that asserts what a setting DEFAULTS to has to
    ask for this: `monkeypatch.delenv` unsets the process variable and cannot
    unset a line in the .env file, so on any machine whose .env sets that
    variable the test reads the developer's local configuration and fails
    against a value the code never chose. That failure looks exactly like a
    code regression and is not one -- see test_tc_t34 in
    tests/test_token_custody.py, which failed on Taylor's checkout for a
    SCHWAB_TOKEN_PATH his .env sets on purpose.
    """
    global settings
    settings = Settings() if env_file is _UNSET else Settings(_env_file=env_file)
    return settings
