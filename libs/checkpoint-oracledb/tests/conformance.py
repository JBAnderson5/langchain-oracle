import asyncio
from typing import AsyncIterator
import os

from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

from langgraph.checkpoint.oracledb.base_async_adapter import (
    OracleDBSaverAsyncAdapter as AsyncAdapter,
)
from db_conn_utils import connect_to_oracle



@checkpointer_test(name="OracleDBSaver")
async def oracle_checkpointer() -> AsyncIterator[AsyncAdapter]:
    conn = connect_to_oracle(
        user="SYSTEM",
        password=os.environ.get("ORACLE_PWD", "OraclePwd_2025"),
        dsn="127.0.0.1:1521/FREEPDB1",
        wallet_location=None,
        wallet_password=None,
        test_connection=True,
    )
    saver = AsyncAdapter(conn)
    yield saver
    conn.close()

# TODO: add conformance test for AsyncOracleDBSaver after it's implemented

async def main() -> None:
    report = await validate(oracle_checkpointer, progress=ProgressCallbacks.verbose())
    report.print_report()


asyncio.run(main())
