from __future__ import annotations

import concurrent.futures
import datetime
import json
import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import oracledb
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)
from langchain_oracledb.vectorstores.oraclevs import OracleVS, create_index, get_processed_ids
from langchain_oracledb.vectorstores.utils import _get_connection

from .schema_utils import apply_migrations

logger = logging.getLogger(__name__)



STORE_MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS {migrations_table} (
        v NUMBER PRIMARY KEY
    )""",
    """
CREATE TABLE IF NOT EXISTS {store_table} (
    namespace VARCHAR2(512) NOT NULL,
    key VARCHAR2(512) NOT NULL,
    value JSON NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    ttl_minutes NUMBER,
    CONSTRAINT {store_table}_pk PRIMARY KEY (namespace, key)
)
""",
    """
CREATE INDEX {store_table}_namespace_idx ON {store_table} (namespace)
""",
    """
CREATE INDEX {store_table}_expires_idx ON {store_table} (expires_at)
""",
]

VECTOR_MAP_MIGRATIONS = [
    """CREATE TABLE IF NOT EXISTS {vector_map_migrations_table} (
        v NUMBER PRIMARY KEY
    )""",
    """
CREATE TABLE IF NOT EXISTS {vector_map_table} (
    vector_id RAW(16) NOT NULL,
    vector_doc_id VARCHAR2(512) NOT NULL,
    namespace VARCHAR2(512) NOT NULL,
    key VARCHAR2(512) NOT NULL,
    field_name VARCHAR2(1024) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT {vector_map_table}_pk PRIMARY KEY (vector_id),
    CONSTRAINT {vector_map_table}_uniq UNIQUE (namespace, key, field_name),
    CONSTRAINT {vector_map_table}_store_fk FOREIGN KEY (namespace, key)
        REFERENCES {store_table}(namespace, key) ON DELETE CASCADE,
    CONSTRAINT {vector_map_table}_vector_fk FOREIGN KEY (vector_id)
        REFERENCES {vector_table}(id) ON DELETE CASCADE
)
""",
    """
CREATE INDEX {vector_map_table}_namespace_idx ON {vector_map_table} (namespace)
""",
    """
CREATE INDEX {vector_map_table}_key_idx ON {vector_map_table} (key)
""",
]


class OracleIndexConfig(IndexConfig, total=False):
    """Configuration for vector embeddings in Oracle store."""

    distance_type: Literal["cosine", "l2", "inner_product"]
    __tokenized_fields: list[tuple[str, Literal["$"] | list[str]]]
    __estimated_num_vectors: int


@dataclass(frozen=True)
class _VectorRequest:
    namespace: tuple[str, ...]
    key: str
    field_name: str
    text: str


class OracleStore(BaseStore):
    """Oracle-backed BaseStore that delegates vector operations to OracleVS."""

    supports_ttl: bool = True

    def __init__(
        self,
        conn: oracledb.Connection | oracledb.ConnectionPool,
        *,
        table: str = "store",
        deserializer: Callable[[str | bytes | oracledb.LOB], dict[str, Any]] | None = None,
        index: OracleIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        vector_store_params: dict[str, Any] | None = None,
        vector_index_params: dict[str, Any] | None = None, # {"idx_name": "toolbox_vs_hnsw", "idx_type": "HNSW"}
    ) -> None:
        super().__init__()
        self._deserializer = deserializer
        self.conn = conn
        self.lock = threading.Lock()
        self.is_setup = False
        self.table_name = table
        self.vector_table_name = f"{table}_vectors".upper()
        self.vector_map_table_name = f"{table}_vector_map"
        self.migrations_table_name = f"{table}_migrations"
        self.vector_map_migrations_table_name = f"{table}_vector_map_migrations"
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
            self.enable_vector = True
        else:
            self.embeddings = None
            self.enable_vector = False
        self.ttl_config = ttl
        self.vector_store_params = vector_store_params or {}
        self.vector_index_params = vector_index_params
        self.vector_store: OracleVS | None = None
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

    @contextmanager
    def _cursor(self, *, commit: bool = False) -> Iterator[oracledb.Cursor]:
        """Context manager for Oracle cursor with optional commit."""
        with self.lock, _get_connection(self.conn) as connection:
            cursor = connection.cursor()
            try:
                yield cursor
                if commit:
                    connection.commit()
            finally:
                cursor.close()

    def _ensure_vector_store(self) -> OracleVS:
        if self.vector_store is not None:
            return self.vector_store
        if not self.embeddings:
            raise ValueError("Embedding configuration is required for vector operations.")
        distance_type = (self.index_config or {}).get("distance_type") or "cosine"
        strategy = _distance_strategy(str(distance_type))
        self.vector_store = OracleVS(
            client=self.conn,
            embedding_function=self.embeddings,
            table_name=self.vector_table_name,
            distance_strategy=strategy,
            mutate_on_duplicate=True,
            **self.vector_store_params,
        )
        if self.enable_vector and not self._vector_index_exists():
            create_index(self.conn, self.vector_store, params=self.vector_index_params)
        return self.vector_store

    def _vector_index_exists(self) -> bool:
        table_name = self.vector_table_name
        query = """
            SELECT 1
            FROM all_ind_columns
            WHERE table_name = :table_name
        """
        with self._cursor() as cursor:
            cursor.execute(query, {"table_name": table_name})
            return cursor.fetchone() is not None

    def setup(self) -> None:
        """Create required tables and migration tracking tables."""
        if self.is_setup:
            return
        format_params = {
            "migrations_table": self.migrations_table_name,
            "store_table": self.table_name,
        }
        with self._cursor(commit=True) as cursor:
            apply_migrations(cursor, STORE_MIGRATIONS, self.migrations_table_name, format_params=format_params)

        if self.index_config:
            self._ensure_vector_store()
            vector_params = {
                "vector_map_migrations_table": self.vector_map_migrations_table_name,
                "vector_map_table": self.vector_map_table_name,
                "vector_table": self.vector_table_name,
                "store_table": self.table_name,
            }
            with self._cursor(commit=True) as cursor:
                apply_migrations(
                    cursor,
                    VECTOR_MAP_MIGRATIONS,
                    self.vector_map_migrations_table_name,
                    format_params=vector_params,
                )

        self.is_setup = True

    def sweep_ttl(self) -> int:
        """Delete expired items and return count."""
        with self._cursor(commit=True) as cursor:
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP"
            )
            return cursor.rowcount

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """Start background TTL sweeper thread."""
        if not self.ttl_config:
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_result(None)
            return future

        if self._ttl_sweeper_thread and self._ttl_sweeper_thread.is_alive():
            future = concurrent.futures.Future()
            future.add_done_callback(
                lambda f: self._ttl_stop_event.set() if f.cancelled() else None
            )
            return future

        self._ttl_stop_event.clear()
        interval = float(
            sweep_interval_minutes or self.ttl_config.get("sweep_interval_minutes") or 5
        )
        future = concurrent.futures.Future()

        def _sweep_loop() -> None:
            try:
                while not self._ttl_stop_event.is_set():
                    if self._ttl_stop_event.wait(interval * 60):
                        break
                    try:
                        expired_items = self.sweep_ttl()
                        if expired_items > 0:
                            logger.info("Store swept %s expired items", expired_items)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.exception("Store TTL sweep failed", exc_info=exc)
                future.set_result(None)
            except Exception as exc:  # pragma: no cover - defensive
                future.set_exception(exc)

        thread = threading.Thread(target=_sweep_loop, daemon=True, name="ttl-sweeper")
        self._ttl_sweeper_thread = thread
        thread.start()
        future.add_done_callback(
            lambda f: self._ttl_stop_event.set() if f.cancelled() else None
        )
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper thread."""
        if not self._ttl_sweeper_thread or not self._ttl_sweeper_thread.is_alive():
            return True
        self._ttl_stop_event.set()
        self._ttl_sweeper_thread.join(timeout)
        success = not self._ttl_sweeper_thread.is_alive()
        if success:
            self._ttl_sweeper_thread = None
        return success

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        if not self.is_setup:
            self.setup()
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops
        with self._cursor(commit=True) as cursor:
            if GetOp in grouped_ops:
                self._batch_get_ops(
                    cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                    results,
                    cursor,
                )
            if SearchOp in grouped_ops:
                self._batch_search_ops(
                    cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                    results,
                    cursor,
                )
            if ListNamespacesOp in grouped_ops:
                self._batch_list_namespaces_ops(
                    cast(
                        Sequence[tuple[int, ListNamespacesOp]],
                        grouped_ops[ListNamespacesOp],
                    ),
                    results,
                    cursor,
                )
            if PutOp in grouped_ops:
                self._batch_put_ops(
                    cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
                    cursor,
                )
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        import asyncio

        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
        cursor: oracledb.Cursor,
    ) -> None:
        namespace_groups = defaultdict(list)
        refresh_ttls = defaultdict(list)
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key))
            refresh_ttls[op.namespace].append(getattr(op, "refresh_ttl", False))

        for namespace, items in namespace_groups.items():
            _, keys = zip(*items, strict=False)
            placeholders = ",".join([f":k{i}" for i in range(len(keys))])
            params = {f"k{i}": key for i, key in enumerate(keys)}
            params["namespace"] = _namespace_to_text(namespace)
            cursor.execute(
                f"""
                SELECT key, value, created_at, updated_at, expires_at, ttl_minutes
                FROM {self.table_name}
                WHERE namespace = :namespace AND key IN ({placeholders})
                """,
                params,
            )
            rows = cursor.fetchall()
            key_to_row = {
                row[0]: {
                    "key": row[0],
                    "value": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "expires_at": row[4],
                    "ttl_minutes": row[5],
                }
                for row in rows
            }

            if (
                any(refresh_ttls[namespace])
                and self.ttl_config
                and self.ttl_config.get("refresh_on_read", False)
            ):
                cursor.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET expires_at = CURRENT_TIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                    WHERE namespace = :namespace AND key IN ({placeholders}) AND ttl_minutes IS NOT NULL
                    """,
                    params,
                )

            for idx, key in items:
                row = key_to_row.get(key)
                if row:
                    results[idx] = _row_to_item(
                        namespace, row, loader=self._deserializer
                    )
                else:
                    results[idx] = None

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
        cursor: oracledb.Cursor,
    ) -> None:
        dedupped_ops: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped_ops[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped_ops.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        if deletes:
            namespace_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for op in deletes:
                namespace_groups[op.namespace].append(op.key)
            for namespace, keys in namespace_groups.items():
                placeholders = ",".join([f":k{i}" for i in range(len(keys))])
                params = {f"k{i}": key for i, key in enumerate(keys)}
                params["namespace"] = _namespace_to_text(namespace)
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE namespace = :namespace AND key IN ({placeholders})",
                    params,
                )
                if self.index_config:
                    self._delete_vectors_for_key(namespace, keys, cursor)

        if inserts:
            now = datetime.datetime.now(datetime.timezone.utc)
            for op in inserts:
                params: dict[str, Any] = {
                    "namespace": _namespace_to_text(op.namespace),
                    "key": op.key,
                    "value": json.dumps(op.value),
                    "created_at": now,
                    "updated_at": now,
                }
                if op.ttl is None:
                    params["expires_at"] = None
                    params["ttl_minutes"] = None
                else:
                    params["expires_at"] = now + datetime.timedelta(minutes=op.ttl)
                    params["ttl_minutes"] = float(op.ttl)

                cursor.execute(
                    f"""
                    MERGE INTO {self.table_name} s
                    USING (SELECT :namespace AS namespace,
                                  :key AS key,
                                  :value AS value,
                                  :created_at AS created_at,
                                  :updated_at AS updated_at,
                                  :expires_at AS expires_at,
                                  :ttl_minutes AS ttl_minutes
                           FROM dual) d
                    ON (s.namespace = d.namespace AND s.key = d.key)
                    WHEN MATCHED THEN UPDATE SET
                        s.value = d.value,
                        s.updated_at = d.updated_at,
                        s.expires_at = d.expires_at,
                        s.ttl_minutes = d.ttl_minutes
                    WHEN NOT MATCHED THEN INSERT
                        (namespace, key, value, created_at, updated_at, expires_at, ttl_minutes)
                    VALUES (d.namespace, d.key, d.value, d.created_at, d.updated_at, d.expires_at, d.ttl_minutes)
                    """,
                    params,
                )

            if self.index_config:
                for op in inserts:
                    if op.index is False:
                        self._delete_vectors_for_key(op.namespace, [op.key], cursor)
                    else:
                        self._upsert_vectors_for_op(op, cursor)

    def _delete_vectors_for_key(
        self, namespace: tuple[str, ...], keys: Sequence[str], cursor: oracledb.Cursor
    ) -> None:
        if not self.vector_store:
            return
        namespace_text = _namespace_to_text(namespace)
        placeholders = ",".join([f":k{i}" for i in range(len(keys))])
        params = {f"k{i}": key for i, key in enumerate(keys)}
        params["namespace"] = namespace_text
        cursor.execute(
            f"""
            SELECT vector_doc_id
            FROM {self.vector_map_table_name}
            WHERE namespace = :namespace AND key IN ({placeholders})
            """,
            params,
        )
        doc_ids = [row[0] for row in cursor.fetchall()]
        if doc_ids:
            self.vector_store.delete(ids=doc_ids)
        cursor.execute(
            f"DELETE FROM {self.vector_map_table_name} WHERE namespace = :namespace AND key IN ({placeholders})",
            params,
        )

    def _upsert_vectors_for_op(self, op: PutOp, cursor: oracledb.Cursor) -> None:
        if not self.vector_store:
            return
        vector_requests = self._build_vector_requests(op)
        if not vector_requests:
            return

        # Clear existing vector rows for this namespace/key since OracleVS will
        # generate new ids when we omit ids during insertion.
        self._delete_vectors_for_key(op.namespace, [op.key], cursor)

        now = datetime.datetime.now(datetime.timezone.utc)
        mapping_rows = []
        for req in vector_requests:
            metadata = self._vector_metadata(req)
            inserted_ids = self.vector_store.add_texts(
                texts=[req.text],
                metadatas=[metadata],
            )
            if not inserted_ids:
                continue
            doc_id = inserted_ids[0]
            processed_ids, _ = get_processed_ids(ids=[doc_id])
            vector_id = processed_ids[0]
            mapping_rows.append(
                {
                    "vector_id": vector_id,
                    "vector_doc_id": doc_id,
                    "namespace": _namespace_to_text(req.namespace),
                    "key": req.key,
                    "field_name": req.field_name,
                    "created_at": now,
                    "updated_at": now,
                }
            )

        if not mapping_rows:
            return

        cursor.executemany(
            f"""
            MERGE INTO {self.vector_map_table_name} vm
            USING (SELECT :vector_id AS vector_id,
                          :vector_doc_id AS vector_doc_id,
                          :namespace AS namespace,
                          :key AS key,
                          :field_name AS field_name,
                          :created_at AS created_at,
                          :updated_at AS updated_at
                   FROM dual) src
            ON (vm.vector_id = src.vector_id)
            WHEN MATCHED THEN UPDATE SET
                vm.vector_doc_id = src.vector_doc_id,
                vm.namespace = src.namespace,
                vm.key = src.key,
                vm.field_name = src.field_name,
                vm.updated_at = src.updated_at
            WHEN NOT MATCHED THEN INSERT
                (vector_id, vector_doc_id, namespace, key, field_name, created_at, updated_at)
            VALUES (src.vector_id, src.vector_doc_id, src.namespace, src.key, src.field_name, src.created_at, src.updated_at)
            """,
            mapping_rows,
        )

    def _build_vector_requests(self, op: PutOp) -> list[_VectorRequest]:
        if not self.index_config or not op.value:
            return []
        if op.index is None or op.index is True:
            paths = cast(dict, self.index_config)["__tokenized_fields"]
        else:
            if isinstance(op.index, bool):
                return []
            paths = [(ix, tokenize_path(ix)) for ix in op.index]
        requests: list[_VectorRequest] = []
        for path, tokenized_path in paths:
            texts = get_text_at_path(op.value, tokenized_path)
            for i, text in enumerate(texts):
                field_name = f"{path}.{i}" if len(texts) > 1 else path
                requests.append(
                    _VectorRequest(
                        namespace=op.namespace,
                        key=op.key,
                        field_name=field_name,
                        text=text,
                    )
                )
        return requests

    def _vector_metadata(self, req: _VectorRequest) -> dict[str, Any]:
        return {
            "namespace": list(req.namespace),
            "namespace_text": _namespace_to_text(req.namespace),
            "key": req.key,
            "field": req.field_name,
        }

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
        cursor: oracledb.Cursor,
    ) -> None:
        for (original_idx, op) in search_ops:
            if op.query and self.index_config:
                results[original_idx] = self._vector_search(op, cursor)
            else:
                results[original_idx] = self._sql_search(op, cursor)

    def _vector_search(self, op: SearchOp, cursor: oracledb.Cursor) -> list[SearchItem]:
        if not self.vector_store:
            return []

        ns_text = _namespace_to_text(op.namespace_prefix)
        filter_obj: dict[str, Any] | None = None
        if ns_text:
            filter_obj = {"namespace_text": {"$startsWith": ns_text}}

        vectors_per_doc = cast(dict, self.index_config)["__estimated_num_vectors"]
        desired = op.limit + op.offset
        k = max(1, int(desired * max(vectors_per_doc, 1) * 2))
        raw_results = self.vector_store.similarity_search_with_score(
            query=cast(str, op.query),
            k=k,
            filter=filter_obj,
        )

        scored: dict[tuple[tuple[str, ...], str], dict[str, Any]] = {}
        for doc, distance in raw_results:
            metadata = doc.metadata or {}
            namespace_list = metadata.get("namespace") or []
            namespace = tuple(cast(list[str], namespace_list))
            key = cast(str | None, metadata.get("key"))
            if not namespace or not key:
                continue
            score = _distance_to_score(distance, self.index_config)
            existing = scored.get((namespace, key))
            if existing is None or score > existing["score"]:
                scored[(namespace, key)] = {"score": score}

        ordered = sorted(
            ((ns, key, payload["score"]) for (ns, key), payload in scored.items()),
            key=lambda item: item[2],
            reverse=True,
        )
        sliced = ordered[op.offset : op.offset + op.limit]
        if not sliced:
            return []

        filter_clauses: list[str] = []
        filter_params: dict[str, Any] = {}
        if op.filter:
            for f_idx, (key, value) in enumerate(op.filter.items()):
                clause, params = _build_json_filter(key, value, f"f{f_idx}")
                filter_clauses.append(clause)
                filter_params.update(params)
        filter_sql = " AND " + " AND ".join(filter_clauses) if filter_clauses else ""

        key_placeholders = ",".join(
            [f"(:ns{i}, :k{i})" for i in range(len(sliced))]
        )
        params = {"ns" + str(i): _namespace_to_text(ns) for i, (ns, _, _) in enumerate(sliced)}
        params.update({"k" + str(i): key for i, (_, key, _) in enumerate(sliced)})
        params.update(filter_params)

        cursor.execute(
            f"""
            SELECT namespace, key, value, created_at, updated_at, expires_at, ttl_minutes
            FROM {self.table_name}
            WHERE (namespace, key) IN ({key_placeholders}){filter_sql}
            """,
            params,
        )
        rows = cursor.fetchall()
        row_lookup = {
            (cast(str, row[0]), cast(str, row[1])): {
                "key": cast(str, row[1]),
                "value": row[2],
                "created_at": row[3],
                "updated_at": row[4],
                "expires_at": row[5],
                "ttl_minutes": row[6],
            }
            for row in rows
        }

        items: list[SearchItem] = []
        for ns, key, score in sliced:
            row = row_lookup.get((_namespace_to_text(ns), key))
            if not row:
                continue
            item = _row_to_search_item(
                ns,
                {
                    **row,
                    "score": score,
                },
                loader=self._deserializer,
            )
            items.append(item)

        if (
            op.refresh_ttl
            and items
            and self.ttl_config
            and self.ttl_config.get("refresh_on_read", False)
        ):
            params_refresh = {}
            placeholders = []
            for i, item in enumerate(items):
                params_refresh[f"ns{i}"] = _namespace_to_text(item.namespace)
                params_refresh[f"k{i}"] = item.key
                placeholders.append(f"(:ns{i}, :k{i})")
            cursor.execute(
                f"""
                UPDATE {self.table_name}
                SET expires_at = CURRENT_TIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                WHERE (namespace, key) IN ({','.join(placeholders)}) AND ttl_minutes IS NOT NULL
                """,
                params_refresh,
            )

        return items

    def _sql_search(self, op: SearchOp, cursor: oracledb.Cursor) -> list[SearchItem]:
        filter_params: dict[str, Any] = {}
        filter_clauses: list[str] = []
        if op.filter:
            for f_idx, (key, value) in enumerate(op.filter.items()):
                clause, params = _build_json_filter(key, value, f"f{f_idx}")
                filter_clauses.append(clause)
                filter_params.update(params)

        ns_condition = "namespace LIKE :ns"
        params: dict[str, Any] = {
            "ns": f"{_namespace_to_text(op.namespace_prefix)}%",
            "offset": op.offset,
            "limit": op.limit,
        }
        params.update(filter_params)

        filter_sql = " AND " + " AND ".join(filter_clauses) if filter_clauses else ""
        query = f"""
            SELECT namespace, key, value, created_at, updated_at, expires_at, ttl_minutes, NULL AS score
            FROM {self.table_name}
            WHERE {ns_condition}{filter_sql}
            ORDER BY updated_at DESC
            OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY
        """

        cursor.execute(query, params)
        rows = cursor.fetchall()
        items = [
            _row_to_search_item(
                _decode_ns_text(row[0]),
                {
                    "key": row[1],
                    "value": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "expires_at": row[5],
                    "ttl_minutes": row[6],
                    "score": None,
                },
                loader=self._deserializer,
            )
            for row in rows
        ]

        if (
            op.refresh_ttl
            and rows
            and self.ttl_config
            and self.ttl_config.get("refresh_on_read", False)
        ):
            keys = [(row[0], row[1]) for row in rows]
            key_placeholders = ",".join(
                [f"(:ns{i}, :k{i})" for i in range(len(keys))]
            )
            params_refresh = {}
            for i, (namespace, key) in enumerate(keys):
                params_refresh[f"ns{i}"] = namespace
                params_refresh[f"k{i}"] = key
            cursor.execute(
                f"""
                UPDATE {self.table_name}
                SET expires_at = CURRENT_TIMESTAMP + NUMTODSINTERVAL(ttl_minutes, 'MINUTE')
                WHERE (namespace, key) IN ({key_placeholders}) AND ttl_minutes IS NOT NULL
                """,
                params_refresh,
            )

        return items

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
        cursor: oracledb.Cursor,
    ) -> None:
        for (idx, op) in list_ops:
            conditions = []
            params: dict[str, Any] = {}
            if op.match_conditions:
                for i, condition in enumerate(op.match_conditions):
                    if condition.match_type == "prefix":
                        conditions.append(f"namespace LIKE :prefix{i}")
                        params[f"prefix{i}"] = (
                            f"{_namespace_to_text(condition.path, handle_wildcards=True)}%"
                        )
                    elif condition.match_type == "suffix":
                        conditions.append(f"namespace LIKE :suffix{i}")
                        params[f"suffix{i}"] = (
                            f"%{_namespace_to_text(condition.path, handle_wildcards=True)}"
                        )
            where_sql = "WHERE " + " AND ".join(conditions) if conditions else ""

            if op.max_depth is not None:
                query = f"""
                SELECT DISTINCT
                    REGEXP_SUBSTR(namespace, '[^.]+', 1, LEVEL) AS part,
                    namespace
                FROM {self.table_name}
                {where_sql}
                CONNECT BY LEVEL <= :max_depth
                """
                params["max_depth"] = op.max_depth
                cursor.execute(query, params)
                namespaces = {
                    ".".join(row[1].split(".")[: op.max_depth])
                    for row in cursor.fetchall()
                }
                namespace_list = sorted(namespaces)
            else:
                cursor.execute(
                    f"SELECT DISTINCT namespace FROM {self.table_name} {where_sql} ORDER BY namespace",
                    params,
                )
                namespace_list = [row[0] for row in cursor.fetchall()]

            namespace_list = namespace_list[op.offset : op.offset + op.limit]
            results[idx] = [_decode_ns_text(ns) for ns in namespace_list]

def _distance_strategy(distance_type: str) -> DistanceStrategy:
    if distance_type == "l2":
        return DistanceStrategy.EUCLIDEAN_DISTANCE
    if distance_type == "inner_product":
        return DistanceStrategy.DOT_PRODUCT
    return DistanceStrategy.COSINE


def _distance_to_score(distance: float, index_config: OracleIndexConfig | None) -> float:
    if not index_config:
        return float(distance)
    distance_type = index_config.get("distance_type") or "cosine"
    if distance_type == "l2":
        return -float(distance)
    if distance_type == "inner_product":
        return float(distance)
    return 1.0 - float(distance)


def _build_json_filter(key: str, value: Any, param_prefix: str) -> tuple[str, dict[str, Any]]:
    if isinstance(value, dict) and any(k.startswith("$") for k in value):
        clauses = []
        params: dict[str, Any] = {}
        for idx, (op, op_val) in enumerate(value.items()):
            clause, clause_params = _json_op_clause(key, op, op_val, f"{param_prefix}_{idx}")
            clauses.append(clause)
            params.update(clause_params)
        return " AND ".join(clauses), params

    if value is None:
        return "json_value(value, '$.%s') IS NULL" % key, {}
    param_name = f"{param_prefix}_val"
    return "json_value(value, '$.%s') = :%s" % (key, param_name), {param_name: value}


def _json_op_clause(
    key: str, op: str, value: Any, param_name: str
) -> tuple[str, dict[str, Any]]:
    op_map = {
        "$eq": "=",
        "$ne": "!=",
        "$gt": ">",
        "$gte": ">=",
        "$lt": "<",
        "$lte": "<=",
    }
    if op not in op_map:
        raise ValueError(f"Unsupported operator: {op}")
    return f"json_value(value, '$.{key}') {op_map[op]} :{param_name}", {param_name: value}


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return ".".join(namespace)


def _decode_ns_text(namespace: str) -> tuple[str, ...]:
    return tuple(namespace.split("."))


def _row_to_item(
    namespace: tuple[str, ...],
    row: dict[str, Any],
    *,
    loader: Callable[[str | bytes | oracledb.LOB], dict[str, Any]] | None = None,
) -> Item:
    val = row["value"]
    if not isinstance(val, dict):
        val = (loader or _json_loads)(val)
    return Item(
        value=val,
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_search_item(
    namespace: tuple[str, ...],
    row: dict[str, Any],
    *,
    loader: Callable[[str | bytes | oracledb.LOB], dict[str, Any]] | None = None,
) -> SearchItem:
    val = row["value"]
    if not isinstance(val, dict):
        val = (loader or _json_loads)(val)
    score = row.get("score")
    if score is not None:
        try:
            score = float(score)
        except ValueError:
            score = None
    return SearchItem(
        value=val,
        key=row["key"],
        namespace=namespace,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        score=score,
    )


def _json_loads(content: str | bytes | oracledb.LOB) -> Any:
    if isinstance(content, oracledb.LOB):
        return json.loads(content.read())
    if isinstance(content, bytes):
        return json.loads(content.decode())
    return json.loads(content)


def _group_ops(ops: Iterable[Op]) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


def _ensure_index_config(
    index_config: OracleIndexConfig,
) -> tuple[Embeddings | None, OracleIndexConfig]:
    index_config = index_config.copy()
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
    tot = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {fields}")
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized
    index_config["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(
        index_config.get("embed"),
    )
    return embeddings, index_config


__all__ = [
    "OracleStore",
    "OracleIndexConfig",
]