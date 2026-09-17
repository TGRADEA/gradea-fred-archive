"""Supabase (PostgREST) adapter — the research corpus behind the dashboard.

WHAT LIVES BEHIND THIS
----------------------
Two bodies of research, both already captured, neither previously reachable
from the dashboard:

  public.*  the VTS (Volatility Trading Strategies) email archive — a daily
            newsletter parsed into structured rows: a composite "barometer",
            fourteen named volatility metrics with percentile ranks, VIX term
            structure ratios, ETP decay, and per-strategy position/action.
  laduc.*   the LaDuc trading-room archive, mirrored up from the canonical
            local SQLite file by scripts/laduc_supabase_mirror.py.

WHY POSTGREST AND NOT PSYCOPG
-----------------------------
INV-D says third-party HTTP/SDK calls live under `sources/`. PostgREST is an
ordinary HTTP API, which keeps this module the same shape as `finra.py` and
`edgar.py` — one httpx client, one error type, no connection pool to own and
no event-loop story to get wrong inside FastAPI. A direct Postgres driver
would be faster for bulk reads, but nothing here reads in bulk: every query is
a handful of rows off an indexed column.

AUTH AND EXPOSURE
-----------------
Reads use the service-role key, which bypasses RLS. That key must never reach
the browser — it stays in gradea-backend/.env and the frontend only ever sees
this backend's own /api routes. That separation is the entire reason these
tables are read server-side instead of from JavaScript with an anon key.

The `laduc` schema has RLS enabled with zero policies, so anon and
authenticated are denied outright there. The `public` VTS tables currently
have RLS OFF (migration `disable_rls_vts_tables`, 2026-08-06) — see
_map/data-sources.md for that decision and its consequence.

NO SILENT FALLBACKS
-------------------
Every failure path raises `SupabaseError` with the status and body. Callers
decide what the user sees. A research page that quietly renders an empty
table when its credentials are wrong is worse than one that says so.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import httpx

log = logging.getLogger(__name__)

# PostgREST caps a single response; ask for more than this and it silently
# truncates unless you page. Everything in this module stays well under it,
# but the constant is here so a future caller has a number to check against.
POSTGREST_MAX_ROWS = 1000

DEFAULT_TIMEOUT = 15.0


class SupabaseError(RuntimeError):
    """A Supabase/PostgREST request failed or was not configured."""


class SupabaseNotConfigured(SupabaseError):
    """No project URL or service key. Distinct so callers can say so precisely."""


def _base_url(project_url: str) -> str:
    return project_url.rstrip("/") + "/rest/v1"


def _headers(service_key: str, *, schema: str, count: bool) -> dict[str, str]:
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Accept": "application/json",
        # PostgREST only exposes non-public schemas when asked by name, and it
        # is a separate header for reads vs writes. Reads use Accept-Profile;
        # the one writer (insert_rows) sets Content-Profile itself.
        "Accept-Profile": schema,
    }
    if count:
        headers["Prefer"] = "count=exact"
    return headers


def select(
    table: str,
    *,
    project_url: str,
    service_key: str,
    schema: str = "public",
    columns: str = "*",
    filters: Mapping[str, str] | None = None,
    order: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Run one PostgREST select and return the rows.

    `filters` is passed through as PostgREST operator syntax, e.g.
    ``{"as_of": "gte.2025-01-01", "metric_name": "eq.VIX Index"}``. Keeping it
    as raw operator strings rather than inventing a query DSL means anything
    PostgREST supports is available without this module growing a translator.

    `order` uses PostgREST syntax too: ``"as_of.desc"``, or
    ``"as_of.desc,metric_name.asc"``.
    """
    if not project_url or not service_key:
        raise SupabaseNotConfigured(
            "Supabase is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY in gradea-backend/.env"
        )

    params: dict[str, str] = {"select": columns}
    if filters:
        params.update(filters)
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = str(limit)
    if offset is not None:
        params["offset"] = str(offset)

    url = f"{_base_url(project_url)}/{table}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(
                url,
                params=params,
                headers=_headers(service_key, schema=schema, count=False),
            )
    except httpx.HTTPError as exc:
        raise SupabaseError(f"Supabase request failed: {exc}") from exc

    if r.status_code >= 400:
        raise SupabaseError(
            f"Supabase HTTP {r.status_code} on {schema}.{table}: {r.text[:300]}"
        )

    payload = r.json()
    if not isinstance(payload, list):
        raise SupabaseError(
            f"Supabase returned {type(payload).__name__}, expected a list, "
            f"on {schema}.{table}"
        )

    if limit is None and len(payload) >= POSTGREST_MAX_ROWS:
        # Not an error, but the caller is probably reading a truncated set and
        # would otherwise never know.
        log.warning(
            "supabase.select(%s.%s) returned %d rows, at or above PostgREST's "
            "default ceiling — the result may be truncated. Pass limit/offset.",
            schema,
            table,
            len(payload),
        )

    return payload


def distinct_values(
    table: str,
    column: str,
    *,
    project_url: str,
    service_key: str,
    schema: str = "public",
    limit: int = POSTGREST_MAX_ROWS,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[str]:
    """Every distinct non-null value of one column, sorted.

    PostgREST has no DISTINCT, so this reads the column and dedupes here. Only
    use it on columns with small cardinality — metric names, strategy labels,
    symbols. It is not a substitute for a real aggregate.
    """
    rows = select(
        table,
        project_url=project_url,
        service_key=service_key,
        schema=schema,
        columns=column,
        order=f"{column}.asc",
        limit=limit,
        timeout=timeout,
    )
    seen: dict[str, None] = {}
    for row in rows:
        value = row.get(column)
        if value is not None:
            seen.setdefault(str(value), None)
    return list(seen)


def latest_as_of(
    table: str,
    *,
    project_url: str,
    service_key: str,
    schema: str = "public",
    column: str = "as_of",
    timeout: float = DEFAULT_TIMEOUT,
) -> str | None:
    """The most recent value of the as-of column, or None if the table is empty.

    Every research read starts here. The archives are point-in-time captures
    with gaps — asking for "today" returns nothing on a day the newsletter did
    not arrive, so callers resolve the real latest date first and label it.
    """
    rows = select(
        table,
        project_url=project_url,
        service_key=service_key,
        schema=schema,
        columns=column,
        order=f"{column}.desc",
        limit=1,
        timeout=timeout,
    )
    if not rows:
        return None
    value = rows[0].get(column)
    return str(value) if value is not None else None


def rows_for_date(
    table: str,
    as_of: str,
    *,
    project_url: str,
    service_key: str,
    schema: str = "public",
    columns: str = "*",
    order: str | None = None,
    date_column: str = "as_of",
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """Every row for one as-of date."""
    return select(
        table,
        project_url=project_url,
        service_key=service_key,
        schema=schema,
        columns=columns,
        filters={date_column: f"eq.{as_of}"},
        order=order,
        timeout=timeout,
    )


def series(
    table: str,
    *,
    project_url: str,
    service_key: str,
    schema: str = "public",
    columns: str = "*",
    since: str | None = None,
    date_column: str = "as_of",
    extra_filters: Mapping[str, str] | None = None,
    limit: int = 500,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict[str, Any]]:
    """A time-ordered history, oldest first, optionally from a start date."""
    filters: dict[str, str] = dict(extra_filters or {})
    if since:
        filters[date_column] = f"gte.{since}"
    return select(
        table,
        project_url=project_url,
        service_key=service_key,
        schema=schema,
        columns=columns,
        filters=filters,
        order=f"{date_column}.asc",
        limit=limit,
        timeout=timeout,
    )


def health(
    *,
    project_url: str,
    service_key: str,
    tables: Iterable[tuple[str, str]],
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Probe each (schema, table) and report reachability and coverage.

    Used by the research health endpoint so a page can distinguish "the
    archive is empty" from "the archive is unreachable" — two states that look
    identical in a table and mean opposite things.
    """
    if not project_url or not service_key:
        return {
            "configured": False,
            "reachable": False,
            "error": "SUPABASE_URL / SUPABASE_SERVICE_KEY not set",
            "tables": [],
        }

    results: list[dict[str, Any]] = []
    reachable = True
    for schema, table in tables:
        entry: dict[str, Any] = {"schema": schema, "table": table}
        try:
            newest = latest_as_of(
                table,
                project_url=project_url,
                service_key=service_key,
                schema=schema,
                timeout=timeout,
            )
            entry["ok"] = True
            entry["latest_as_of"] = newest
        except SupabaseError as exc:
            reachable = False
            entry["ok"] = False
            entry["error"] = str(exc)[:200]
        results.append(entry)

    return {
        "configured": True,
        "reachable": reachable,
        "tables": results,
    }


def insert_rows(
    table: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    project_url: str,
    service_key: str,
    schema: str = "public",
    ignore_duplicates: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> int:
    """Insert rows in one PostgREST request. Returns the number sent.

    The only write in this module (FOUNDATION-CHAIN-SNAPSHOT-MIRROR-1). With
    ``ignore_duplicates`` (the default) a primary-key collision is skipped
    server-side rather than failing the batch, which is what makes a mirror
    re-send idempotent. There is deliberately no update or delete here: every
    Supabase writer in this codebase is append-only, and a helper that could
    revise rows would be one grep away from being used.
    """
    if not project_url or not service_key:
        raise SupabaseNotConfigured(
            "Supabase is not configured. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY in gradea-backend/.env"
        )
    payload = [dict(r) for r in rows]
    if not payload:
        return 0

    prefer = "return=minimal"
    if ignore_duplicates:
        prefer += ",resolution=ignore-duplicates"
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Content-Type": "application/json",
        "Content-Profile": schema,
        "Prefer": prefer,
    }
    url = f"{_base_url(project_url)}/{table}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise SupabaseError(f"Supabase request failed: {exc}") from exc

    if r.status_code >= 400:
        raise SupabaseError(
            f"Supabase HTTP {r.status_code} on {schema}.{table}: {r.text[:300]}"
        )
    return len(payload)
