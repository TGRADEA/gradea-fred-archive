"""Named symbol universes — the single place universes are declared.

FOUNDATION-UNIVERSE-1. Before this slice, seven modules carried their own
hardcoded symbol lists, two of which were dead code that had silently
diverged from the settings defaults that actually won at runtime. This
module ends that: every named universe is declared here in ``_SEEDS`` and
nowhere else (INV-UNI-1).

Two consumption tiers, deliberately separate:

* **Seed tier (pure, no I/O).** ``seed_symbols()`` / ``seed_csv()`` /
  ``seed_weights()`` read the in-code seed data. Modules bind their
  import-time constants to these, so importing a module never touches the
  database and existing test monkeypatch seams keep working.
* **Registry tier (SQLite).** ``get_universe()`` / ``list_universes()`` /
  ``resolve()`` read the ``universes`` / ``universe_members`` tables, which
  hold the seeds plus any future user-defined rows. Seeding is idempotent:
  rows with ``source='seed'`` are refreshed from code, rows with
  ``source='user'`` are never touched.

Symbols here are the user-facing spellings Taylor types (``SPX``, not
``$SPX.X``) — mapping to Schwab symbology stays at the fetcher boundary
per the schwab-first-data skill.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

# ── Sector tags (GICS, static seed — enrichment via Schwab instruments is
# a later slice; rows carry source='seed' so provenance is explicit) ──────

_SECTOR: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology",
    "MSFT": "Information Technology",
    "NVDA": "Information Technology",
    "AVGO": "Information Technology",
    "ORCL": "Information Technology",
    "CRM": "Information Technology",
    "ADBE": "Information Technology",
    "AMD": "Information Technology",
    "CSCO": "Information Technology",
    "QCOM": "Information Technology",
    "IBM": "Information Technology",
    "TXN": "Information Technology",
    "INTU": "Information Technology",
    "NOW": "Information Technology",
    "ACN": "Information Technology",
    # Communication Services
    "GOOGL": "Communication Services",
    "GOOG": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "VZ": "Communication Services",
    "T": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    # Consumer Staples
    "WMT": "Consumer Staples",
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "PM": "Consumer Staples",
    # Financials
    "BRK.B": "Financials",
    "JPM": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "GS": "Financials",
    "SPGI": "Financials",
    # Health Care
    "LLY": "Health Care",
    "UNH": "Health Care",
    "JNJ": "Health Care",
    "ABBV": "Health Care",
    "MRK": "Health Care",
    "TMO": "Health Care",
    "ABT": "Health Care",
    "DHR": "Health Care",
    "PFE": "Health Care",
    "AMGN": "Health Care",
    "ISRG": "Health Care",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    # Industrials
    "GE": "Industrials",
    "CAT": "Industrials",
    "UNP": "Industrials",
    "RTX": "Industrials",
    # Utilities / Materials
    "NEE": "Utilities",
    "LIN": "Materials",
}

_ETFS = {"SPY", "QQQ", "IWM", "DIA"}
_INDICES = {"SPX", "NDX", "RUT", "VIX"}


def _asset_type(symbol: str) -> str:
    if " " in symbol:  # OCC option symbol, e.g. "SPY   260619C00580000"
        return "option"
    if symbol in _ETFS:
        return "etf"
    if symbol in _INDICES:
        return "index"
    return "equity"


# ── Seed data — THE single declaration point for named universes ─────────


@dataclass(frozen=True)
class UniverseSeed:
    description: str
    kind: str  # 'symbols' | 'weighted' | 'option_contracts'
    members: tuple[str, ...]
    weights: tuple[float, ...] | None = None  # parallel to members


_SCAN_11 = (
    "SPY",
    "QQQ",
    "IWM",
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "META",
    "AMD",
    "AMZN",
    "GOOG",
)

_EARNINGS_60 = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "GOOG",
    "AMZN",
    "META",
    "TSLA",
    "BRK.B",
    "LLY",
    "AVGO",
    "JPM",
    "V",
    "MA",
    "UNH",
    "XOM",
    "WMT",
    "JNJ",
    "PG",
    "HD",
    "COST",
    "ORCL",
    "ABBV",
    "CRM",
    "BAC",
    "KO",
    "MRK",
    "CVX",
    "AMD",
    "ADBE",
    "PEP",
    "TMO",
    "ACN",
    "LIN",
    "ABT",
    "WFC",
    "DIS",
    "MCD",
    "CSCO",
    "QCOM",
    "IBM",
    "GE",
    "CAT",
    "VZ",
    "TXN",
    "DHR",
    "INTU",
    "PFE",
    "UNP",
    "RTX",
    "GS",
    "LOW",
    "PM",
    "T",
    "SPGI",
    "ISRG",
    "COP",
    "AMGN",
    "NOW",
    "BKNG",
    "NEE",
)

_DISPERSION_WEIGHTED = (
    ("AAPL", 7.0),
    ("MSFT", 6.5),
    ("NVDA", 6.0),
    ("AMZN", 3.6),
    ("META", 2.4),
    ("GOOGL", 2.0),
    ("GOOG", 1.7),
    ("AVGO", 1.7),
    ("TSLA", 1.7),
    ("BRK.B", 1.6),
    ("JPM", 1.3),
    ("LLY", 1.3),
    ("V", 1.0),
    ("XOM", 1.0),
    ("UNH", 1.0),
    ("MA", 0.9),
    ("COST", 0.8),
    ("HD", 0.8),
    ("PG", 0.8),
    ("JNJ", 0.7),
    ("WMT", 0.7),
    ("ABBV", 0.6),
    ("BAC", 0.6),
    ("NFLX", 0.6),
    ("CRM", 0.6),
    ("MRK", 0.6),
    ("ORCL", 0.6),
    ("AMD", 0.6),
    ("CVX", 0.6),
    ("KO", 0.6),
    ("PEP", 0.5),
    ("ADBE", 0.5),
    ("CSCO", 0.5),
    ("TMO", 0.5),
    ("ACN", 0.5),
    ("ABT", 0.5),
    ("LIN", 0.5),
    ("MCD", 0.5),
    ("WFC", 0.5),
    ("DIS", 0.5),
    ("DHR", 0.5),
    ("INTU", 0.4),
    ("NOW", 0.4),
    ("VZ", 0.4),
    ("TXN", 0.4),
    ("AMGN", 0.4),
    ("QCOM", 0.4),
    ("IBM", 0.4),
    ("GE", 0.4),
    ("PM", 0.4),
)

_SEEDS: dict[str, UniverseSeed] = {
    "core_etfs": UniverseSeed(
        description="SPY/QQQ/IWM breadth trio used by market-regime signals.",
        kind="symbols",
        members=("SPY", "QQQ", "IWM"),
    ),
    "recorder_default": UniverseSeed(
        description=(
            "Daily recorder surface shared by the karsan composite and "
            "chain-snapshot schedulers (UNIVERSE-CONFIG-1)."
        ),
        kind="symbols",
        members=("SPY", "QQQ", "IWM", "DIA", "SPX", "VIX", "GLD", "USO", "TLT", "HYG"),
    ),
    "screener_default": UniverseSeed(
        description="Default options-screener scan surface.",
        kind="symbols",
        members=_SCAN_11,
    ),
    "wheel_default": UniverseSeed(
        description=(
            "Wheel scan surface — intentionally identical to "
            "screener_default until a real watchlist exists."
        ),
        kind="symbols",
        members=_SCAN_11,
    ),
    "neg_fly_default": UniverseSeed(
        description="Negative-butterfly scan surface (cash indices + liquid names).",
        kind="symbols",
        members=(
            "SPX",
            "NDX",
            "RUT",
            "SPY",
            "QQQ",
            "IWM",
            "NVDA",
            "TSLA",
            "AAPL",
            "MSFT",
            "META",
        ),
    ),
    "box_spread_default": UniverseSeed(
        description="Box-spread scan surface — cash indices only (European-style).",
        kind="symbols",
        members=("SPX", "NDX", "RUT"),
    ),
    "neg_vertical_default": UniverseSeed(
        description="Negative-vertical scan surface (cash indices + liquid names).",
        kind="symbols",
        members=(
            "SPX",
            "NDX",
            "RUT",
            "SPY",
            "QQQ",
            "IWM",
            "NVDA",
            "TSLA",
            "AAPL",
            "MSFT",
            "META",
            "AMD",
            "AMZN",
            "GOOG",
        ),
    ),
    "earnings_default": UniverseSeed(
        description="S&P-100-style earnings watch list (settings default).",
        kind="symbols",
        members=_EARNINGS_60,
    ),
    "dispersion_spx50": UniverseSeed(
        description="Representative SPX-50 dispersion basket, index-weight seeds.",
        kind="weighted",
        members=tuple(s for s, _ in _DISPERSION_WEIGHTED),
        weights=tuple(w for _, w in _DISPERSION_WEIGHTED),
    ),
    "stream_default": UniverseSeed(
        description="Tape/streamer default subscription set (settings default).",
        kind="symbols",
        members=("SPY", "QQQ", "VIX"),
    ),
    "trace_default": UniverseSeed(
        description="Chain-trace default polling set (settings default).",
        kind="symbols",
        members=("SPY", "QQQ"),
    ),
    "flow_tape_seeds": UniverseSeed(
        description=(
            "OCC option contracts seeding the flow tape and classifier — "
            "the shared config that keeps both in lockstep."
        ),
        kind="option_contracts",
        members=(
            "SPY   260619C00580000",
            "SPY   260619P00580000",
            "QQQ   260619C00500000",
            "QQQ   260619P00500000",
            "NVDA  260619C00130000",
            "NVDA  260619P00130000",
        ),
    ),
}


# ── Seed tier: pure accessors, no I/O ────────────────────────────────────


def seed_symbols(name: str) -> list[str]:
    """Members of a seed universe, in declared order. Pure — no DB."""
    return list(_SEEDS[name].members)


def seed_csv(name: str) -> str:
    """Seed members as a CSV string (for pydantic-settings defaults)."""
    return ",".join(_SEEDS[name].members)


def seed_weights(name: str) -> dict[str, float]:
    """Symbol → weight for a weighted seed universe. Pure — no DB."""
    seed = _SEEDS[name]
    if seed.weights is None:
        raise ValueError(f"universe {name!r} is not weighted")
    return dict(zip(seed.members, seed.weights, strict=True))


# ── Registry tier: SQLite-backed, seeds ∪ user rows ──────────────────────


@dataclass(frozen=True)
class UniverseMember:
    symbol: str
    asset_type: str  # 'equity' | 'etf' | 'index' | 'option'
    sector: str | None
    weight: float | None
    source: str  # 'seed' | 'user'


def _connect() -> sqlite3.Connection:
    # Imported at call time, not module level: settings.py consumes this
    # module's seed tier for its defaults, and storage.py consumes settings —
    # a module-level `import storage` here would close that cycle. The seed
    # tier stays import-safe; only the registry tier touches storage.
    import storage

    conn = sqlite3.connect(storage.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_seeded() -> None:
    """Create tables and sync seed rows. Idempotent; never touches user rows.

    Seed rows are wiped and rewritten from ``_SEEDS`` on every call so code
    updates propagate; rows with ``source='user'`` are never touched. DDL
    lives in storage.py's schema per convention.
    """
    import storage

    storage.init_db()
    with _connect() as conn:
        for name, seed in _SEEDS.items():
            conn.execute(
                "INSERT INTO universes(name, description, kind) VALUES(?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET description=excluded.description, "
                "kind=excluded.kind",
                (name, seed.description, seed.kind),
            )
            conn.execute(
                "DELETE FROM universe_members " "WHERE universe_name=? AND source='seed'",
                (name,),
            )
            weights = seed.weights or (None,) * len(seed.members)
            conn.executemany(
                "INSERT INTO universe_members"
                "(universe_name, symbol, asset_type, sector, weight, source) "
                "VALUES(?,?,?,?,?, 'seed') "
                "ON CONFLICT(universe_name, symbol) DO NOTHING",
                [
                    (name, sym, _asset_type(sym), _SECTOR.get(sym), w)
                    for sym, w in zip(seed.members, weights, strict=True)
                ],
            )


def get_universe(name: str) -> list[UniverseMember]:
    """All members (seed ∪ user) of a named universe. Raises KeyError if unknown."""
    ensure_seeded()
    with _connect() as conn:
        known = conn.execute("SELECT 1 FROM universes WHERE name=?", (name,)).fetchone()
        if known is None:
            raise KeyError(f"unknown universe: {name!r}")
        rows = conn.execute(
            "SELECT symbol, asset_type, sector, weight, source "
            "FROM universe_members WHERE universe_name=? "
            "ORDER BY source, symbol",
            (name,),
        ).fetchall()
    return [UniverseMember(**dict(r)) for r in rows]


def universe_kind(name: str) -> str:
    """The declared kind of a named universe. Raises KeyError if unknown.

    UNIVERSE-MEMBERS-1: a deliberate second query rather than widening
    get_universe()'s return shape — the record_symbols()/snapshot_symbols()
    2-tuple → 3-tuple widening in UNIVERSE-CONFIG-1 touched every caller and
    three test pins; a members route that needs the kind can pay one extra
    indexed SELECT instead of repeating that churn.
    """
    ensure_seeded()
    with _connect() as conn:
        row = conn.execute("SELECT kind FROM universes WHERE name=?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"unknown universe: {name!r}")
    return row["kind"]


def list_universes() -> list[dict]:
    """Name/description/kind/member_count/origin for every universe.

    origin is computed from ``_SEEDS`` membership, not stored: reseeding
    rewrites seed universes from code, so code is the authority on which
    names are seeds (UNIVERSE-ORIGIN-1). create_universe refuses seed-name
    collisions, so no user universe can shadow a seed.
    """
    ensure_seeded()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT u.name, u.description, u.kind, COUNT(m.symbol) AS member_count "
            "FROM universes u LEFT JOIN universe_members m "
            "ON m.universe_name = u.name "
            "GROUP BY u.name ORDER BY u.name"
        ).fetchall()
    return [
        {**dict(r), "origin": "seed" if r["name"] in _SEEDS else "user"}
        for r in rows
    ]


# ── Write tier (UNIVERSE-CRUD-1) ──────────────────────────────────────────
#
# The durability rule that shapes every write: ensure_seeded() upserts seed
# universes and wipes/rewrites source='seed' member rows on every call. A
# write only survives if it touches (a) a universe whose name is NOT in
# _SEEDS, or (b) member rows with source='user'. Anything else would
# silently resurrect — so it is refused loudly here, never allowed.

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_MEMBER_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,11}$")


class UniverseNameTakenError(ValueError):
    """Create collided with an existing universe (seed or user)."""


class UniverseProtectedError(ValueError):
    """Write would touch seed-owned state that reseeding resurrects."""


class UniverseInUseError(ValueError):
    """Delete refused: a recorder config references this universe by name."""


def clean_member_symbols(raw: list[str]) -> list[str]:
    """Upcase, strip, drop leading $, dedupe (order-preserving), validate.

    Registry-level rule, deliberately NOT karsan.clean_symbols: dotted
    symbols (BRK.B) are valid registry members — earnings_default already
    stores one — and karsan's MAX_SYMBOLS cap is a recorder budget, not a
    registry rule. Raises ValueError naming the first bad symbol.
    """
    out: list[str] = []
    for item in raw:
        sym = (item or "").strip().upper().lstrip("$")
        if not _MEMBER_SYMBOL_RE.match(sym):
            raise ValueError(f"invalid symbol {item!r}: must match {_MEMBER_SYMBOL_RE.pattern}")
        if sym not in out:
            out.append(sym)
    return out


def create_universe(name: str, description: str = "") -> None:
    """Create an empty user universe (kind='symbols' — the only kind users
    can create this slice; weighted/option_contracts stay code-declared).

    Raises ValueError on a malformed name, UniverseNameTakenError on any
    collision — including seed names: recreating core_etfs would look like
    it worked until the next reseed rewrote its description.
    """
    if not _NAME_RE.match(name or ""):
        raise ValueError(f"invalid universe name {name!r}: must match {_NAME_RE.pattern}")
    ensure_seeded()
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM universes WHERE name=?", (name,)).fetchone()
        if row is not None:
            raise UniverseNameTakenError(f"universe name taken: {name!r}")
        conn.execute(
            "INSERT INTO universes(name, description, kind) VALUES(?,?, 'symbols')",
            (name, description or f"User universe {name!r}."),
        )


def delete_universe(name: str) -> int:
    """Delete a USER universe and all its member rows; returns rows removed.

    Seed universes are protected (reseeding would resurrect the shell and
    orphan the intent). A universe referenced by a recorder config is
    protected too — deleting it would KeyError the recorder's next tick;
    repoint the recorder first, then delete.
    """
    if name in _SEEDS:
        raise UniverseProtectedError(f"seed universe is protected: {name!r}")
    ensure_seeded()  # also runs storage.init_db() — app_config must exist
    # before the in-use guard reads it (fresh-DB order matters here)
    import storage

    for key in ("karsan.symbols", "snapshots.symbols"):
        row = storage.get_config(key)
        if row is not None and isinstance(row[0], dict) and row[0].get("universe") == name:
            raise UniverseInUseError(
                f"universe {name!r} is referenced by config {key!r}; "
                "repoint that recorder before deleting"
            )
    with _connect() as conn:
        known = conn.execute("SELECT 1 FROM universes WHERE name=?", (name,)).fetchone()
        if known is None:
            raise KeyError(f"unknown universe: {name!r}")
        removed = conn.execute(
            "DELETE FROM universe_members WHERE universe_name=?", (name,)
        ).rowcount
        conn.execute("DELETE FROM universes WHERE name=?", (name,))
    return removed


def add_members(name: str, symbols: list[str]) -> dict:
    """Add source='user' member rows; returns {'added': [...], 'already_present': [...]}.

    Works on seed universes too — user rows survive reseeding by design
    (ensure_seeded only wipes source='seed'). Adding an existing member is
    an idempotent no-op reported under 'already_present', not an error.
    Raises KeyError on unknown universe, ValueError on bad symbols.
    """
    cleaned = clean_member_symbols(symbols)
    if not cleaned:
        raise ValueError("no symbols to add")
    ensure_seeded()
    with _connect() as conn:
        known = conn.execute("SELECT 1 FROM universes WHERE name=?", (name,)).fetchone()
        if known is None:
            raise KeyError(f"unknown universe: {name!r}")
        existing = {
            r["symbol"]
            for r in conn.execute(
                "SELECT symbol FROM universe_members WHERE universe_name=?", (name,)
            )
        }
        added = [s for s in cleaned if s not in existing]
        conn.executemany(
            "INSERT INTO universe_members"
            "(universe_name, symbol, asset_type, sector, weight, source) "
            "VALUES(?,?,?,?,NULL,'user')",
            [(name, s, _asset_type(s), _SECTOR.get(s)) for s in added],
        )
    return {"added": added, "already_present": [s for s in cleaned if s in existing]}


def remove_member(name: str, symbol: str) -> None:
    """Remove one source='user' member row.

    Seed rows are protected: the DELETE would succeed and then the next
    ensure_seeded() would rewrite the row — a lie shaped like success.
    Raises KeyError on unknown universe or member, UniverseProtectedError
    on a seed row.
    """
    sym = (symbol or "").strip().upper().lstrip("$")
    ensure_seeded()
    with _connect() as conn:
        known = conn.execute("SELECT 1 FROM universes WHERE name=?", (name,)).fetchone()
        if known is None:
            raise KeyError(f"unknown universe: {name!r}")
        row = conn.execute(
            "SELECT source FROM universe_members WHERE universe_name=? AND symbol=?",
            (name, sym),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown member: {sym!r} not in universe {name!r}")
        if row["source"] != "user":
            raise UniverseProtectedError(
                f"seed member is protected: {sym!r} in {name!r} is declared in "
                "code and would be resurrected by the next reseed"
            )
        conn.execute(
            "DELETE FROM universe_members WHERE universe_name=? AND symbol=?",
            (name, sym),
        )


def resolve(symbols: list[str] | None, universe_name: str) -> list[str]:
    """Explicit symbols if any survive cleaning, else the named universe.

    This is the override-aware entry point scan modules use: caller-supplied
    symbols win; blank/whitespace input falls back to the registry. Fallback
    to the universe is the documented contract, not a silent substitution.
    """
    cleaned = [s.strip().upper() for s in (symbols or []) if s and s.strip()]
    if cleaned:
        return cleaned
    return [m.symbol for m in get_universe(universe_name)]
