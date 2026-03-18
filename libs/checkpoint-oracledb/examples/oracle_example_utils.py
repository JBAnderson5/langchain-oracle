"""Utilities for OracleDB examples."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import oracledb

from db_conn_utils import connect_to_oracle


def connect_or_start_oracledb(
    *,
    user: Optional[str] = None,
    password: Optional[str] = None,
    dsn: Optional[str] = None,
    wallet_location: Optional[str] = None,
    wallet_password: Optional[str] = None,
) -> tuple[oracledb.Connection, bool]:
    """Connect to Oracle using env vars or start Docker fallback.

    Returns:
        (connection, docker_oracle_started)
    """
    user = user or os.getenv("DB_USER")
    password = password or os.getenv("DB_PASSWORD")
    dsn = dsn or os.getenv("DB_DSN")
    wallet_location = wallet_location or os.getenv("DB_WALLET_LOCATION")
    wallet_password = wallet_password or os.getenv("DB_WALLET_PASSWORD")

    if user and password and dsn:
        print("connecting using provided oracle db environment variables")
        conn = connect_to_oracle(
            user=user,
            password=password,
            dsn=dsn,
            wallet_location=wallet_location,
            wallet_password=wallet_password,
        )
        return conn, False

    print(
        "missing at least one oracle db environment variable: DB_USER, DB_PASSWORD, DB_DSN"
    )
    print("starting docker version of Oracle DB")
    lib_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subprocess.run(["make", "-C", lib_root, "start-oracle"], check=True)

    conn = connect_to_oracle(password="OraclePwd_2025")
    return conn, True


def stop_oracle_docker() -> None:
    """Stop the dockerized Oracle instance used for local examples."""
    lib_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    subprocess.run(["make", "-C", lib_root, "stop-oracle"], check=True)