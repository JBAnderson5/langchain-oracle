"""Shared helpers for Oracle DB connections."""

from __future__ import annotations

import getpass
import os
import time
from typing import Optional
import oracledb


DB_USER=os.getenv("DB_USER","SYSTEM")
DB_PASSWORD=os.getenv("DB_PASSWORD")
DB_DSN=os.getenv("DB_DSN","127.0.0.1:1521/FREEPDB1")
DB_WALLET_LOCATION=os.getenv("DB_WALLET_LOCATION")
DB_WALLET_PASSWORD=os.getenv("DB_WALLET_PASSWORD")
#TNS_ADMIN=os.getenv("TNS_ADMIN")
DEFAULT_PROGRAM = "langgraph_storage_agent"


def set_env_securely(var_name: str, prompt: str):
    if var_name not in os.environ:
        value = getpass.getpass(prompt)
        os.environ[var_name] = value


def connect_to_oracle(
    *,
    max_retries: int = 3,
    retry_delay: int = 5,
    user: str = DB_USER,
    password: Optional[str] = DB_PASSWORD,
    dsn: str = DB_DSN,
    program: str = DEFAULT_PROGRAM,
    wallet_location: Optional[str] = DB_WALLET_LOCATION,
    wallet_password: Optional[str] = DB_WALLET_PASSWORD,
    test_connection: bool = True,
):
    """
    Connect to Oracle database with retry logic and optional prompts.

    Args:
        max_retries: Maximum number of connection attempts.
        retry_delay: Seconds to wait between retries.
    """

    if password is None:
        set_env_securely("DB_PASSWORD", f"Password for the DB user {user}: ")
        password = os.environ["DB_PASSWORD"]

    if wallet_location and wallet_password is None:
        set_env_securely(
            "DB_WALLET_PASSWORD", f"Password for the DB wallet in {wallet_location}: "
        )
        wallet_password = os.environ["DB_WALLET_PASSWORD"]

    wallet_kwargs = (
        {"wallet_password": wallet_password, "wallet_location": wallet_location}
        if wallet_location
        else {}
    )

    for attempt in range(1, max_retries + 1):
        try:
            conn = oracledb.connect(
                user=user,
                password=password,
                dsn=dsn,
                program=program,
                **wallet_kwargs,
            )

            if test_connection:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT banner FROM v$version WHERE banner LIKE 'Oracle%';"
                    )
                    cur.fetchone()

            return conn

        except oracledb.OperationalError as exc:
            error_msg = str(exc)
            print(f"✗ Connection failed (attempt {attempt}/{max_retries})")

            if "DPY-4011" in error_msg or "Connection reset by peer" in error_msg:
                print("  → This usually means:")
                print("    1. Database is still starting up (wait 2-3 minutes)")
                print("    2. Listener configuration issue")
                print("    3. Container is not running")

                if attempt < max_retries:
                    print(f"\n  Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    print("\n  💡 Try running: setup_oracle_database()")
                    print("     This will fix the listener and verify the connection.")
                    raise
            else:
                raise
        except Exception as exc:
            print(f"✗ Unexpected error: {exc}")
            raise

    raise ConnectionError("Failed to connect after all retries")
