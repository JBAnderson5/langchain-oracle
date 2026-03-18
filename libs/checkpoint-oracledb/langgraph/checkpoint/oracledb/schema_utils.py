from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import oracledb


def apply_migrations(
    cursor: oracledb.Cursor,
    migrations: Sequence[str],
    migrations_table: str,
    *,
    format_params: dict[str, Any] | None = None,
) -> None:
    """Apply ordered migrations tracked by a migrations table.

    The migrations list must include the CREATE TABLE statement for the
    migrations tracking table as its first entry.
    """
    if not migrations:
        return
    cursor.execute(_format_sql(migrations[0], format_params))
    cursor.execute(
        f"SELECT v FROM {migrations_table} ORDER BY v DESC FETCH FIRST 1 ROWS ONLY"
    )
    row = cursor.fetchone()
    version = row[0] if row else -1
    for v, migration in enumerate(migrations[version + 1 :], start=version + 1):
        cursor.execute(_format_sql(migration, format_params))
        cursor.execute(
            f"INSERT INTO {migrations_table} (v) VALUES (:v)",
            {"v": v},
        )


def _format_sql(sql: str, format_params: dict[str, Any] | None) -> str:
    if not format_params:
        return sql
    return sql.format(**format_params)


__all__ = ["apply_migrations"]
