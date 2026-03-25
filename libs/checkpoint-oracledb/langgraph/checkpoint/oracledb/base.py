from __future__ import annotations

import json
import logging
import random
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langchain_oracledb.vectorstores.utils import _get_connection
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

import oracledb
from .schema_utils import apply_migrations

logger = logging.getLogger(__name__)

MIGRATIONS: list[str] = [
    """CREATE TABLE IF NOT EXISTS checkpoint_migrations (
        v NUMBER PRIMARY KEY
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id VARCHAR2(255) NOT NULL,
        checkpoint_ns VARCHAR2(255) DEFAULT '',
        checkpoint_id VARCHAR2(255) NOT NULL,
        parent_checkpoint_id VARCHAR2(255),
        type VARCHAR2(64),
        checkpoint JSON NOT NULL,
        metadata JSON DEFAULT JSON_OBJECT(),
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoint_blobs (
        thread_id VARCHAR2(255) NOT NULL,
        checkpoint_ns VARCHAR2(255) DEFAULT '',
        channel VARCHAR2(255) NOT NULL,
        version VARCHAR2(255) NOT NULL,
        type VARCHAR2(64) NOT NULL,
        blob BLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
    )""",
    """CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id VARCHAR2(255) NOT NULL,
        checkpoint_ns VARCHAR2(255) DEFAULT '',
        checkpoint_id VARCHAR2(255) NOT NULL,
        task_id VARCHAR2(255) NOT NULL,
        idx NUMBER NOT NULL,
        channel VARCHAR2(255) NOT NULL,
        type VARCHAR2(64) NOT NULL,
        blob BLOB NOT NULL,
        task_path VARCHAR2(512) DEFAULT '',
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )""",
    """CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON checkpoints(thread_id)""",
    """CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx ON checkpoint_blobs(thread_id)""",
    """CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx ON checkpoint_writes(thread_id)""",
]

ROOT_NAMESPACE_SENTINEL = "__langgraph_root__"


class OracleDBSaver(BaseCheckpointSaver[str]):
    """Checkpointer that stores checkpoints in Oracle DB."""

    def __init__(
        self,
        conn: oracledb.Connection | oracledb.ConnectionPool,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the OracleDB checkpointer.

        Args:
            conn: Oracle connection or pool used for checkpoint operations.
            serde: Optional serializer override.

        Example:
            >>> import oracledb
            >>> from langgraph.checkpoint.oracledb import OracleDBSaver
            >>> conn = oracledb.connect(...)
            >>> saver = OracleDBSaver(conn)
        """
        super().__init__(serde=serde)
        self.conn = conn
        self._is_setup = False
        self._lock = threading.Lock()

    # Using this encoding/decoding to work around Oracle DB treating "" as NULL
    @staticmethod
    def _encode_checkpoint_ns(checkpoint_ns: str | None) -> str:
        if checkpoint_ns is None or checkpoint_ns == "":
            return ROOT_NAMESPACE_SENTINEL
        return checkpoint_ns

    @staticmethod
    def _decode_checkpoint_ns(checkpoint_ns: str | None) -> str:
        if checkpoint_ns is None or checkpoint_ns == ROOT_NAMESPACE_SENTINEL:
            return ""
        return checkpoint_ns

    @staticmethod
    def _coerce_blob(blob: Any) -> bytes:
        if blob is None:
            return b""
        read_method = getattr(blob, "read", None)
        data = read_method() if callable(read_method) else blob
        if isinstance(data, memoryview):
            return data.tobytes()
        if isinstance(data, bytearray):
            return bytes(data)
        if isinstance(data, bytes):
            return data
        return bytes(cast(memoryview, data))

    @staticmethod
    def _normalize_json(value: Any) -> Any:
        if isinstance(value, Decimal):
            return int(value) if value % 1 == 0 else float(value)
        if isinstance(value, dict):
            return {k: OracleDBSaver._normalize_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [OracleDBSaver._normalize_json(v) for v in value]
        return value

    @contextmanager
    def _cursor(self, *, commit: bool = False) -> Iterator[oracledb.Cursor]:
        class _LoggingCursor:
            def __init__(self, inner: oracledb.Cursor) -> None:
                self._inner = inner

            def execute(self, statement: str, params: Any | None = None) -> Any:
                logger.debug("SQL execute: %s", params)
                return self._inner.execute(statement, params)

            def executemany(self, statement: str, params: Any | None = None) -> Any:
                rows = len(params) if params is not None else 0
                logger.debug("SQL executemany (%s rows): %s", rows, params)
                return self._inner.executemany(statement, params)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        with self._lock, _get_connection(self.conn) as connection:
            with connection.cursor() as cursor:
                logging_cursor = cast(oracledb.Cursor, _LoggingCursor(cursor))
                yield logging_cursor
            if commit:
                connection.commit()

    def setup(self) -> None:
        """Ensure the checkpoint schema is created and up to date.

        This is called internally to lazy load the schema and is not required to
        be called explicitly by the user.
        """
        if self._is_setup:
            return
        with self._cursor(commit=True) as cursor:
            apply_migrations(cursor, MIGRATIONS, "checkpoint_migrations")
        self._is_setup = True

    def _load_blobs(
        self, blob_values: list[tuple[str, str, bytes | None]]
    ) -> dict[str, Any]:
        if not blob_values:
            return {}
        logger.debug("_load_blobs: %d rows", len(blob_values))
        return {
            channel: self.serde.loads_typed((type_, self._coerce_blob(blob)))
            for channel, type_, blob in blob_values
            if type_ != "empty"
        }

    def _dump_blobs(
        self,
        thread_id: str,
        checkpoint_ns: str,
        values: dict[str, Any],
        versions: ChannelVersions,
    ) -> list[tuple[str, str, str, str, str, bytes | None]]:
        checkpoint_ns = self._encode_checkpoint_ns(checkpoint_ns)
        dumped: list[tuple[str, str, str, str, str, bytes | None]] = []
        for channel, version in versions.items():
            if channel in values:
                type_, blob = self.serde.dumps_typed(values[channel])
            else:
                type_, blob = "empty", None
            logger.debug(
                "_dump_blobs: channel=%s version=%s type=%s blob_type=%s blob_len=%s",
                channel,
                version,
                type_,
                type(blob).__name__,
                len(blob) if isinstance(blob, (bytes, bytearray)) else None,
            )
            dumped.append(
                (thread_id, checkpoint_ns, channel, str(version), type_, blob)
            )
        return dumped

    def _load_writes(
        self, writes: list[tuple[str, str, str, bytes]]
    ) -> list[tuple[str, str, Any]]:
        logger.debug("_load_writes: %d rows", len(writes))
        return [
            (task_id, channel, self.serde.loads_typed((type_, self._coerce_blob(blob))))
            for task_id, channel, type_, blob in writes
        ]

    def _dump_writes(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        task_path: str,
        writes: Sequence[tuple[str, Any]],
    ) -> list[tuple[str, str, str, str, int, str, str, bytes, str]]:
        checkpoint_ns = self._encode_checkpoint_ns(checkpoint_ns)
        dumped: list[tuple[str, str, str, str, int, str, str, bytes, str]] = []
        for idx, (channel, value) in enumerate(writes):
            type_, blob = self.serde.dumps_typed(value)
            coerced_blob = self._coerce_blob(blob)
            logger.debug(
                "_dump_writes: task_id=%s channel=%s idx=%s type=%s blob_type=%s blob_len=%s",
                task_id,
                channel,
                idx,
                type_,
                type(coerced_blob).__name__,
                len(coerced_blob)
                if isinstance(coerced_blob, (bytes, bytearray))
                else None,
            )
            dumped.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    WRITES_IDX_MAP.get(channel, idx),
                    channel,
                    type_,
                    coerced_blob,
                    task_path,
                )
            )
        return dumped

    def _fetch_blobs_for_checkpoint(
        self,
        cursor: oracledb.Cursor,
        thread_id: str,
        checkpoint_ns: str,
        channel_versions: dict[str, Any],
    ) -> dict[str, Any]:
        if not channel_versions:
            return {}
        checkpoint_ns = self._encode_checkpoint_ns(checkpoint_ns)
        where_clause_list = []
        params: dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
        for idx, (channel, version) in enumerate(channel_versions.items()):
            params[f"channel_{idx}"] = channel
            params[f"version_{idx}"] = str(version)
            where_clause_list.append(
                f"(channel = :channel_{idx} AND version = :version_{idx})"
            )
        where_clause = " OR ".join(where_clause_list)
        cursor.execute(
            f"""
            SELECT channel, type, blob
            FROM checkpoint_blobs
            WHERE thread_id = :thread_id
              AND checkpoint_ns = :checkpoint_ns
              AND ({where_clause})
            """,
            params,
        )
        return self._load_blobs(cursor.fetchall())

    def _get_checkpoint_rows(
        self,
        cursor: oracledb.Cursor,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        where_clause_list = []
        params: dict[str, Any] = {}
        if config:
            configurable = config.get("configurable", {})
            where_clause_list.append("thread_id = :thread_id")
            params["thread_id"] = configurable.get("thread_id")
            checkpoint_ns = configurable.get("checkpoint_ns")
            if checkpoint_ns is not None:
                where_clause_list.append("checkpoint_ns = :checkpoint_ns")
                params["checkpoint_ns"] = self._encode_checkpoint_ns(checkpoint_ns)
            if checkpoint_id := get_checkpoint_id(config):
                where_clause_list.append("checkpoint_id = :checkpoint_id")
                params["checkpoint_id"] = checkpoint_id
        if before and (before_id := get_checkpoint_id(before)):
            where_clause_list.append("checkpoint_id < :before_id")
            params["before_id"] = before_id
        if filter:
            for idx, (key, value) in enumerate(filter.items()):
                where_clause_list.append(
                    f"JSON_VALUE(metadata, '$.{key}') = :filter_{idx}"
                )
                params[f"filter_{idx}"] = json.dumps(value).strip('"')
        where_clause = (
            " WHERE " + " AND ".join(where_clause_list) if where_clause_list else ""
        )
        limit_clause = "" if limit is None else f" FETCH FIRST {int(limit)} ROWS ONLY"
        query = f"""
            SELECT thread_id,
                   checkpoint_ns,
                   checkpoint_id,
                   parent_checkpoint_id,
                   checkpoint,
                   metadata
            FROM checkpoints
            {where_clause}
            ORDER BY checkpoint_id DESC{limit_clause}
        """
        cursor.execute(query, params)
        columns = [str(col[0]).lower() for col in (cursor.description or [])]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple for the given config.

        If no checkpoint_id is provided in the config, the latest checkpoint is
        returned for the thread/namespace.

        Args:
            config: RunnableConfig containing configurable thread_id and optional
                checkpoint_id.

        Returns:
            The matching CheckpointTuple or None if not found.

        Example:
            >>> config = {"configurable": {"thread_id": "thread-1"}}
            >>> saver.get_tuple(config)
        """
        self.setup()
        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        db_checkpoint_ns = self._encode_checkpoint_ns(checkpoint_ns)
        checkpoint_id = get_checkpoint_id(config)
        logger.debug(
            "get_tuple: thread_id=%s checkpoint_ns=%s checkpoint_id=%s",
            thread_id,
            checkpoint_ns,
            checkpoint_id,
        )
        with self._cursor() as cursor:
            if checkpoint_id:
                cursor.execute(
                    """
                    SELECT thread_id,
                           checkpoint_ns,
                           checkpoint_id,
                           parent_checkpoint_id,
                           checkpoint,
                           metadata
                    FROM checkpoints
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                      AND checkpoint_id = :checkpoint_id
                    """,
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": db_checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    },
                )
            else:
                cursor.execute(
                    """
                    SELECT thread_id,
                           checkpoint_ns,
                           checkpoint_id,
                           parent_checkpoint_id,
                           checkpoint,
                           metadata
                    FROM checkpoints
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                    ORDER BY checkpoint_id DESC
                    FETCH FIRST 1 ROWS ONLY
                    """,
                    {"thread_id": thread_id, "checkpoint_ns": db_checkpoint_ns},
                )
            row = cursor.fetchone()
            if not row:
                return None
            (
                thread_id,
                db_checkpoint_ns,
                checkpoint_id,
                parent_checkpoint_id,
                checkpoint_json,
                metadata_json,
            ) = row
            checkpoint_ns = self._decode_checkpoint_ns(db_checkpoint_ns)
            checkpoint_data = self._normalize_json(
                checkpoint_json
                if isinstance(checkpoint_json, dict)
                else json.loads(checkpoint_json)
            )
            metadata = self._normalize_json(
                metadata_json
                if isinstance(metadata_json, dict)
                else json.loads(metadata_json)
            )
            channel_versions = checkpoint_data.get("channel_versions", {})
            channel_values = checkpoint_data.get("channel_values") or {}
            channel_values.update(
                self._fetch_blobs_for_checkpoint(
                    cursor, thread_id, db_checkpoint_ns, channel_versions
                )
            )
            checkpoint_data["channel_values"] = channel_values
            cursor.execute(
                """
                SELECT task_id, channel, type, blob
                FROM checkpoint_writes
                WHERE thread_id = :thread_id
                  AND checkpoint_ns = :checkpoint_ns
                  AND checkpoint_id = :checkpoint_id
                ORDER BY task_id, idx
                """,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": db_checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                },
            )
            writes = self._load_writes(cursor.fetchall())
        logger.debug("get_tuple: pending_writes=%s", len(writes) if writes else 0)
        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=cast(Checkpoint, checkpoint_data),
            metadata=cast(CheckpointMetadata, metadata),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Iterate over checkpoint tuples matching the filters.

        Args:
            config: RunnableConfig with a thread_id to scope the search, or None.
            filter: Optional metadata key/value filter.
            before: Optional config whose checkpoint_id bounds the search.
            limit: Optional limit on the number of checkpoints returned.

        Yields:
            CheckpointTuple values ordered by checkpoint_id descending.

        Example:
            >>> config = {"configurable": {"thread_id": "thread-1"}}
            >>> list(saver.list(config, limit=5))
        """
        self.setup()
        logger.debug(
            "list: config=%s filter=%s before=%s limit=%s",
            config,
            filter,
            before,
            limit,
        )
        with self._cursor() as cursor:
            rows = self._get_checkpoint_rows(
                cursor, config, filter=filter, before=before, limit=limit
            )
            for row in rows:
                db_checkpoint_ns = row["checkpoint_ns"]
                checkpoint_ns = self._decode_checkpoint_ns(db_checkpoint_ns)
                checkpoint_data = self._normalize_json(
                    row["checkpoint"]
                    if isinstance(row["checkpoint"], dict)
                    else json.loads(row["checkpoint"])
                )
                metadata = self._normalize_json(
                    row["metadata"]
                    if isinstance(row["metadata"], dict)
                    else json.loads(row["metadata"])
                )
                channel_versions = checkpoint_data.get("channel_versions", {})
                channel_values = checkpoint_data.get("channel_values") or {}
                channel_values.update(
                    self._fetch_blobs_for_checkpoint(
                        cursor,
                        row["thread_id"],
                        db_checkpoint_ns,
                        channel_versions,
                    )
                )
                checkpoint_data["channel_values"] = channel_values
                cursor.execute(
                    """
                    SELECT task_id, channel, type, blob
                    FROM checkpoint_writes
                    WHERE thread_id = :thread_id
                      AND checkpoint_ns = :checkpoint_ns
                      AND checkpoint_id = :checkpoint_id
                    ORDER BY task_id, idx
                    """,
                    {
                        "thread_id": row["thread_id"],
                        "checkpoint_ns": db_checkpoint_ns,
                        "checkpoint_id": row["checkpoint_id"],
                    },
                )
                writes = self._load_writes(cursor.fetchall())
                logger.debug(
                    "list: thread_id=%s checkpoint_id=%s pending_writes=%s",
                    row["thread_id"],
                    row["checkpoint_id"],
                    len(writes) if writes else 0,
                )
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": row["thread_id"],
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": row["checkpoint_id"],
                        }
                    },
                    checkpoint=cast(Checkpoint, checkpoint_data),
                    metadata=cast(CheckpointMetadata, metadata),
                    parent_config=(
                        {
                            "configurable": {
                                "thread_id": row["thread_id"],
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": row["parent_checkpoint_id"],
                            }
                        }
                        if row["parent_checkpoint_id"]
                        else None
                    ),
                    pending_writes=writes,
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint and its metadata.

        Args:
            config: RunnableConfig for the thread and optional parent checkpoint.
            checkpoint: Checkpoint payload to store.
            metadata: Metadata associated with the checkpoint.
            new_versions: Channel versions to persist alongside blobs.

        Returns:
            RunnableConfig pointing at the newly stored checkpoint.

        Example:
            >>> config = {"configurable": {"thread_id": "thread-1"}}
            >>> checkpoint = {"id": "ckpt-1", "channel_values": {}, "channel_versions": {}}
            >>> metadata = {}
            >>> new_versions = {}
            >>> saver.put(config, checkpoint, metadata, new_versions)
        """
        self.setup()
        logger.debug(
            "put: thread_id=%s checkpoint_id=%s metadata_keys=%s",
            config.get("configurable", {}).get("thread_id"),
            checkpoint.get("id"),
            list(metadata.keys()),
        )
        configurable = config["configurable"].copy()
        thread_id = configurable.pop("thread_id")
        checkpoint_ns = configurable.pop("checkpoint_ns", "")
        db_checkpoint_ns = self._encode_checkpoint_ns(checkpoint_ns)
        parent_checkpoint_id = configurable.pop("checkpoint_id", None)
        copy = checkpoint.copy()
        copy["channel_values"] = copy["channel_values"].copy()
        next_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }
        blob_values: dict[str, Any] = {}
        for key, value in checkpoint["channel_values"].items():
            if value is None or isinstance(value, (str, int, float, bool)):
                continue
            blob_values[key] = copy["channel_values"].pop(key)
        metadata_payload = get_serializable_checkpoint_metadata(config, metadata)
        with self._cursor(commit=True) as cursor:
            if blob_versions := {
                k: v for k, v in new_versions.items() if k in blob_values
            }:
                cursor.setinputsizes(blob=oracledb.DB_TYPE_BLOB)
                logger.debug("put: blob_versions=%s", blob_versions)
                blob_rows = self._dump_blobs(
                    thread_id, db_checkpoint_ns, blob_values, blob_versions
                )
                cursor.executemany(
                    """
                    MERGE INTO checkpoint_blobs cb
                    USING (SELECT :thread_id AS thread_id,
                                  :checkpoint_ns AS checkpoint_ns,
                                  :channel AS channel,
                                  :version AS version,
                                  :type AS type,
                                  :blob AS blob
                           FROM dual) src
                    ON (cb.thread_id = src.thread_id
                        AND cb.checkpoint_ns = src.checkpoint_ns
                        AND cb.channel = src.channel
                        AND cb.version = src.version)
                    WHEN NOT MATCHED THEN
                        INSERT (thread_id, checkpoint_ns, channel, version, type, blob)
                        VALUES (src.thread_id, src.checkpoint_ns, src.channel, src.version, src.type, src.blob)
                    """,
                    [
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": db_checkpoint_ns,
                            "channel": channel,
                            "version": version,
                            "type": type_,
                            "blob": blob,
                        }
                        for (
                            _thread_id,
                            _checkpoint_ns,
                            channel,
                            version,
                            type_,
                            blob,
                        ) in blob_rows
                    ],
                )
            cursor.execute(
                """
                MERGE INTO checkpoints ck
                USING (SELECT :thread_id AS thread_id,
                              :checkpoint_ns AS checkpoint_ns,
                              :checkpoint_id AS checkpoint_id,
                              :parent_checkpoint_id AS parent_checkpoint_id,
                              :checkpoint AS checkpoint,
                              :metadata AS metadata
                       FROM dual) src
                ON (ck.thread_id = src.thread_id
                    AND ck.checkpoint_ns = src.checkpoint_ns
                    AND ck.checkpoint_id = src.checkpoint_id)
                WHEN MATCHED THEN UPDATE SET
                    ck.checkpoint = src.checkpoint,
                    ck.metadata = src.metadata
                WHEN NOT MATCHED THEN
                    INSERT (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
                    VALUES (src.thread_id, src.checkpoint_ns, src.checkpoint_id, src.parent_checkpoint_id, src.checkpoint, src.metadata)
                """,
                {
                    "thread_id": thread_id,
                    "checkpoint_ns": db_checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                    "parent_checkpoint_id": parent_checkpoint_id,
                    "checkpoint": json.dumps(copy),
                    "metadata": json.dumps(metadata_payload),
                },
            )
        return cast(RunnableConfig, next_config)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending writes for a task.

        Args:
            config: RunnableConfig that includes thread_id and checkpoint_id.
            writes: Sequence of (channel, value) writes.
            task_id: Task identifier for ordering writes.
            task_path: Optional task path for ordering writes.

        Example:
            >>> config = {"configurable": {"thread_id": "thread-1", "checkpoint_id": "ckpt-1"}}
            >>> saver.put_writes(config, [("channel", {"foo": "bar"})], task_id="task-1")
        """
        self.setup()
        configurable = config.get("configurable", {})
        thread_id = configurable["thread_id"]
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        db_checkpoint_ns = self._encode_checkpoint_ns(checkpoint_ns)
        checkpoint_id = configurable["checkpoint_id"]
        logger.debug(
            "put_writes: thread_id=%s checkpoint_ns=%s checkpoint_id=%s task_id=%s writes=%s",
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            task_id,
            len(writes),
        )
        params = self._dump_writes(
            thread_id, db_checkpoint_ns, checkpoint_id, task_id, task_path, writes
        )
        logger.debug("put_writes: dumped_rows=%s", len(params))
        with self._cursor(commit=True) as cursor:
            cursor.setinputsizes(blob=oracledb.DB_TYPE_BLOB)
            # WRITES_IDX_MAP represents special channels that are overwritable
            # Allows a batch of special writes to be ovewritten
            # If one write is normal, it will just insert new values
            # TODO: should we split writes into two lists and process on a case by case basis instead of batch?
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                cursor.executemany(
                    """
                    MERGE INTO checkpoint_writes cw
                    USING (SELECT :thread_id AS thread_id,
                                  :checkpoint_ns AS checkpoint_ns,
                                  :checkpoint_id AS checkpoint_id,
                                  :task_id AS task_id,
                                  :idx AS idx,
                                  :channel AS channel,
                                  :type AS type,
                                  CASE WHEN :blob IS NULL THEN EMPTY_BLOB() ELSE :blob END AS blob,
                                  :task_path AS task_path
                           FROM dual) src
                    ON (cw.thread_id = src.thread_id
                        AND cw.checkpoint_ns = src.checkpoint_ns
                        AND cw.checkpoint_id = src.checkpoint_id
                        AND cw.task_id = src.task_id
                        AND cw.idx = src.idx)
                    WHEN MATCHED THEN UPDATE SET
                        cw.channel = src.channel,
                        cw.type = src.type,
                        cw.blob = src.blob
                    WHEN NOT MATCHED THEN
                        INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path)
                        VALUES (src.thread_id, src.checkpoint_ns, src.checkpoint_id, src.task_id, src.idx, src.channel, src.type, src.blob, src.task_path)
                    """,
                    [
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                            "task_id": task_id,
                            "idx": idx,
                            "channel": channel,
                            "type": type_,
                            "blob": blob,
                            "task_path": task_path,
                        }
                        for (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            idx,
                            channel,
                            type_,
                            blob,
                            task_path,
                        ) in params
                    ],
                )
            else:
                cursor.executemany(
                    """
                    MERGE INTO checkpoint_writes cw
                    USING (SELECT :thread_id AS thread_id,
                                  :checkpoint_ns AS checkpoint_ns,
                                  :checkpoint_id AS checkpoint_id,
                                  :task_id AS task_id,
                                  :idx AS idx,
                                  :channel AS channel,
                                  :type AS type,
                                  CASE WHEN :blob IS NULL THEN EMPTY_BLOB() ELSE :blob END AS blob,
                                  :task_path AS task_path
                           FROM dual) src
                    ON (cw.thread_id = src.thread_id
                        AND cw.checkpoint_ns = src.checkpoint_ns
                        AND cw.checkpoint_id = src.checkpoint_id
                        AND cw.task_id = src.task_id
                        AND cw.idx = src.idx)
                    WHEN NOT MATCHED THEN
                        INSERT (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path)
                        VALUES (src.thread_id, src.checkpoint_ns, src.checkpoint_id, src.task_id, src.idx, src.channel, src.type, src.blob, src.task_path)
                    """,
                    [
                        {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id,
                            "task_id": task_id,
                            "idx": idx,
                            "channel": channel,
                            "type": type_,
                            "blob": blob,
                            "task_path": task_path,
                        }
                        for (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            idx,
                            channel,
                            type_,
                            blob,
                            task_path,
                        ) in params
                    ],
                )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints, blobs, and writes for a thread.

        Example:
            >>> saver.delete_thread("thread-1")
        """
        self.setup()
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = :thread_id",
                {"thread_id": thread_id},
            )
            cursor.execute(
                "DELETE FROM checkpoint_blobs WHERE thread_id = :thread_id",
                {"thread_id": thread_id},
            )
            cursor.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = :thread_id",
                {"thread_id": thread_id},
            )

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        """Delete checkpoints and writes associated with specific run IDs.

        Example:
            >>> saver.delete_for_runs(["run-1", "run-2"])
        """
        if not run_ids:
            return
        self.setup()
        run_ids_clause = ",".join(f":run_id_{i}" for i in range(len(run_ids)))
        params = {f"run_id_{i}": rid for i, rid in enumerate(run_ids)}
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                f"""
                DELETE FROM checkpoint_writes
                WHERE (thread_id, checkpoint_ns, checkpoint_id) IN (
                    SELECT thread_id, checkpoint_ns, checkpoint_id
                    FROM checkpoints
                    WHERE JSON_VALUE(metadata, '$.run_id') IN ({run_ids_clause})
                )
                """,
                params,
            )
            cursor.execute(
                f"""
                DELETE FROM checkpoint_blobs
                WHERE (thread_id, checkpoint_ns, channel, version) IN (
                    SELECT cb.thread_id, cb.checkpoint_ns, cb.channel, cb.version
                    FROM checkpoint_blobs cb
                    JOIN checkpoints ck
                      ON ck.thread_id = cb.thread_id
                     AND ck.checkpoint_ns = cb.checkpoint_ns
                    WHERE JSON_VALUE(ck.metadata, '$.run_id') IN ({run_ids_clause})
                )
                """,
                params,
            )
            cursor.execute(
                f"""
                DELETE FROM checkpoints
                WHERE JSON_VALUE(metadata, '$.run_id') IN ({run_ids_clause})
                """,
                params,
            )

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        """Copy all checkpoint data from one thread to another.

        Example:
            >>> saver.copy_thread("source-thread", "target-thread")
        """
        if source_thread_id == target_thread_id:
            return
        self.setup()
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                """
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
                SELECT :target_thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata
                FROM checkpoints
                WHERE thread_id = :source_thread_id
                """,
                {
                    "source_thread_id": source_thread_id,
                    "target_thread_id": target_thread_id,
                },
            )
            cursor.execute(
                """
                INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                SELECT :target_thread_id, checkpoint_ns, channel, version, type, blob
                FROM checkpoint_blobs
                WHERE thread_id = :source_thread_id
                """,
                {
                    "source_thread_id": source_thread_id,
                    "target_thread_id": target_thread_id,
                },
            )
            cursor.execute(
                """
                INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path)
                SELECT :target_thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path
                FROM checkpoint_writes
                WHERE thread_id = :source_thread_id
                """,
                {
                    "source_thread_id": source_thread_id,
                    "target_thread_id": target_thread_id,
                },
            )

    def prune(
        self, thread_ids: Sequence[str], *, strategy: str = "keep_latest"
    ) -> None:
        """Prune checkpoints for one or more threads.

        Args:
            thread_ids: Thread IDs whose checkpoints should be pruned.
            strategy: "keep_latest" to keep the newest checkpoint per namespace,
                or "delete" to remove all checkpoints for the threads.

        Example:
            >>> saver.prune(["thread-1"], strategy="keep_latest")
        """
        if not thread_ids:
            return
        self.setup()
        ids_clause = ",".join(f":thread_id_{i}" for i in range(len(thread_ids)))
        params = {f"thread_id_{i}": tid for i, tid in enumerate(thread_ids)}
        with self._cursor(commit=True) as cursor:
            if strategy == "delete":
                cursor.execute(
                    f"DELETE FROM checkpoints WHERE thread_id IN ({ids_clause})",
                    params,
                )
                cursor.execute(
                    f"DELETE FROM checkpoint_blobs WHERE thread_id IN ({ids_clause})",
                    params,
                )
                cursor.execute(
                    f"DELETE FROM checkpoint_writes WHERE thread_id IN ({ids_clause})",
                    params,
                )
                return
            elif strategy != "keep_latest":
                raise ValueError(f"Unknown prune strategy: {strategy}")
            cursor.execute(
                f"""
                SELECT thread_id, checkpoint_ns, MAX(checkpoint_id) AS checkpoint_id
                FROM checkpoints
                WHERE thread_id IN ({ids_clause})
                GROUP BY thread_id, checkpoint_ns
                """,
                params,
            )
            keep_rows = cursor.fetchall()
            if not keep_rows:
                return
            keep_params = {}
            keep_clause_list = []
            for idx, (thread_id, checkpoint_ns, checkpoint_id) in enumerate(keep_rows):
                keep_params[f"keep_thread_{idx}"] = thread_id
                keep_params[f"keep_ns_{idx}"] = checkpoint_ns
                keep_params[f"keep_id_{idx}"] = checkpoint_id
                keep_clause_list.append(
                    "(thread_id = :keep_thread_{idx} AND checkpoint_ns = :keep_ns_{idx} AND checkpoint_id = :keep_id_{idx})".format(
                        idx=idx
                    )
                )
            keep_clause = " OR ".join(keep_clause_list)
            cursor.execute(
                f"""
                DELETE FROM checkpoint_writes
                WHERE thread_id IN ({ids_clause})
                  AND (thread_id, checkpoint_ns, checkpoint_id) NOT IN (
                    SELECT thread_id, checkpoint_ns, checkpoint_id
                    FROM checkpoints
                    WHERE {keep_clause}
                  )
                """,
                {**params, **keep_params},
            )
            cursor.execute(
                f"""
                DELETE FROM checkpoints
                WHERE thread_id IN ({ids_clause})
                  AND NOT ({keep_clause})
                """,
                {**params, **keep_params},
            )

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate the next version string for a channel.

        Example:
            >>> saver.get_next_version(None, None)
        """
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"
