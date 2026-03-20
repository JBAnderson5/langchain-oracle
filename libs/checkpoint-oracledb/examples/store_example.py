from __future__ import annotations

import os
import sys
import uuid
import json
from typing import Any

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)


from langgraph.store.oracledb import OracleIndexConfig, OracleStore

from langchain_oci import OCIGenAIEmbeddings
from oracle_example_utils import connect_or_start_oracledb, stop_oracle_docker, enable_oracle_vectors


def _print_items(label: str, items: list[Any]) -> None:
    print(label)
    for item in items:
        try:
            print(json.dumps(item,indent = 4))
        except:
            print(item)
    print("\n")


def main() -> None:
    print("Langgraph docs on persistence: https://docs.langchain.com/oss/python/langgraph/persistence")
    print("\n")
    conn, docker_oracle = connect_or_start_oracledb()
    print("\n")

    print("creating oracle store object")
    store = OracleStore(conn)
    print("Schema will be created if missing on any public function call by internally calling setup() function")
    print("\n")

    namespace = ("examples", "stores")
    key = f"doc-{uuid.uuid4().hex[:8]}"

    print("putting item into store")
    store.put(namespace, key, {"title": "OracleStore", "content": "Hello from Oracle", "tags": ["demo", "oracle"]})
    print("\n")

    print("getting item from store")
    item = store.get(namespace, key)
    print(item)
    print("\n")

    print("adding a second item for searches")
    second_key = f"doc-{uuid.uuid4().hex[:8]}"
    store.put(namespace, second_key, {"title": "Vector search", "content": "Search me", "tags": ["demo"]})
    print("\n")

    print("searching items by namespace")
    results = store.search(("examples",), limit=10)
    _print_items("Search results:", results)

    print("searching items with JSON filter")
    filtered = store.search(("examples",), filter={"title": "OracleStore"}, limit=10)
    _print_items("Filtered results:", filtered)

    print("listing namespaces")
    namespaces = store.list_namespaces()
    _print_items("Namespaces:", namespaces)

    print("deleting item")
    store.delete(namespace, key)
    print("\n")

    print("Vector search example using embeddings from oci gen ai service")
    print("\n")

    if docker_oracle:
        conn = enable_oracle_vectors()

    embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-v4.0",
    service_endpoint=f"https://inference.generativeai.{os.getenv('OCI_REGION')}.oci.oraclecloud.com",
    compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
    auth_profile=os.getenv("OCI_AUTH_PROFILE"),
    )

    vector_dim = len(embeddings.embed_query("dimension probe"))
    vector_store = OracleStore(
        conn=conn,
        index=OracleIndexConfig(
            embed= embeddings,
            dims= vector_dim,
            fields= ["content"], # TODO: how configurable?
            distance_type="cosine", # TODO: how configurable?
        ),
    )


    vector_store.put(namespace, "vector-doc-1", {"content": "Hello vector world"})
    vector_store.put(namespace, "vector-doc-2", {"content": "Another piece of info"})
    vector_store.put(namespace, "vector-doc-3", {"content": "A whole new world"})
    query = "vector world"
    vector_results = vector_store.search(
        ("examples",),
        query=query,
        limit=5,
    )
    _print_items(f"Vector search results for query: {query}", vector_results)


    # TODO: do an example of an onnx or similar in db embedding model

    conn.close()

    if docker_oracle:
        stop_oracle_docker()


if __name__ == "__main__":
    main()