from __future__ import annotations

import inspect
import json
import os
import sys
import uuid
from typing import Annotated, Any, cast

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedStore

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from langgraph.store.oracledb import OracleIndexConfig, OracleVSStore

from langchain_oci import ChatOCIOpenAI, OCIGenAIEmbeddings
from oci_openai import OciUserPrincipalAuth

from oracle_example_utils import (
    connect_or_start_oracledb,
    enable_oracle_vectors,
    stop_oracle_docker,
)


KNOWLEDGE_NAMESPACE = ("knowledge", "base")


def _print_items(label: str, items: list[Any]) -> None:
    print(label)
    for item in items:
        try:
            print(json.dumps(item, indent=4))
        except Exception:
            print(item)
    print("\n")


@tool
def search_knowledge(query: str, store: Annotated[Any, InjectedStore()],limit: int = 5) -> str:
    """Search the knowledge base namespace for relevant documentation."""
    results = store.search(KNOWLEDGE_NAMESPACE, query=query, limit=limit)
    if not results:
        print("no KB results")
        return "No relevant knowledge found."
    lines = []
    print("KB Results")
    for item in results:
        if isinstance(item.value, dict):
            title = item.value.get("title", "untitled")
            content = item.value.get("content", "")
        else:
            title = "untitled"
            content = str(item.value)
        snippet = content[:300].replace("\n", " ")
        lines.append(f"- {title}: {snippet}")
        print(title)
    return "Knowledge base matches:\n" + "\n".join(lines)

def _seed_knowledge_base(store: OracleVSStore) -> None:
    print("Seeding knowledge base with docstrings...")
    modules = {
        "langchain_oci.agents.react": "langchain_oci.agents.react",
        "langchain_oci.chat_models.oci_generative_ai": "langchain_oci.chat_models.oci_generative_ai",
        "langgraph.store.base": "langgraph.store.base",
        "langgraph.checkpoint.base": "langgraph.checkpoint.base",
    }
    for title, module_path in modules.items():
        module = __import__(module_path, fromlist=["*"])
        doc = inspect.getdoc(module) or ""
        if not doc:
            print(f"No docstring found for {module_path}")
            continue
        key = module_path
        store.put(
            KNOWLEDGE_NAMESPACE,
            key,
            {"title": title, "content": doc, "source": module_path},
            index=["title", "content", "source"],
        )

    results = store.search(KNOWLEDGE_NAMESPACE, limit=50)
    print(f"Knowledge base item count: {len(results)}")


def _create_llm() -> ChatOCIOpenAI:
    auth_profile = os.getenv("OCI_AUTH_PROFILE", "DEFAULT")
    auth = OciUserPrincipalAuth(profile_name=auth_profile)
    region = os.getenv("OCI_REGION")
    compartment_id = os.getenv("OCI_COMPARTMENT_ID")
    model = os.getenv("OCI_MODEL_ID")

    if not region or not compartment_id or not model:
        raise ValueError(
            "Missing OCI_REGION, OCI_COMPARTMENT_ID, or OCI_MODEL_ID env vars."
        )

    service_endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"

    return ChatOCIOpenAI(
        auth=auth,
        compartment_id=compartment_id,
        region=region,
        model=model,
        service_endpoint=service_endpoint,
        store=False,
    )


def _build_graph(store: OracleVSStore, llm: ChatOCIOpenAI) -> StateGraph:
    def retrieve_knowledge(state: MessagesState) -> dict[str, list[Any]]:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        last = messages[-1]
        query = getattr(last, "content", None)
        if query is None and isinstance(last, dict):
            query = last.get("content")
        query = str(query or "")
        tool_result = search_knowledge.invoke({"query": query, "store": store})
        return {
            "messages": [SystemMessage(content=f"Knowledge lookup:\n{tool_result}")]
        }

    def call_llm(state: MessagesState) -> dict[str, list[Any]]:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("retrieve", retrieve_knowledge)
    builder.add_node("llm", call_llm)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "llm")
    builder.add_edge("llm", END)
    return builder


def main() -> None:
    print("Langgraph docs on persistence: https://docs.langchain.com/oss/python/langgraph/persistence")
    conn, docker_oracle = connect_or_start_oracledb()
    if docker_oracle:
        conn = enable_oracle_vectors()

    embeddings = OCIGenAIEmbeddings(
        model_id="cohere.embed-v4.0",
        service_endpoint=f"https://inference.generativeai.{os.getenv('OCI_REGION')}.oci.oraclecloud.com",
        compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        auth_profile=os.getenv("OCI_AUTH_PROFILE"),
    )
    vector_dim = len(embeddings.embed_query("dimension probe"))
    store = OracleVSStore(
        conn=conn,
        index=OracleIndexConfig(
            embed=embeddings,
            dims=vector_dim,
            fields=["content"],
            distance_type="cosine",
        ),
    )

    _seed_knowledge_base(store)

    llm = _create_llm()
    graph = _build_graph(store, llm).compile(store=store)

    config = cast(RunnableConfig, {"configurable": {"thread_id": "kb-thread-1"}})
    user_question = "How do I instantiate ChatOCIOpenAI?"
    print(f"\nUser question: {user_question}\n")

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": user_question},
            ]
        },
        config,
    )


    for m in result["messages"]:
        print(type(m))
    last_response = result["messages"][-1]
    print(last_response.content[-1]["text"])

    conn.close()
    if docker_oracle:
        stop_oracle_docker()


if __name__ == "__main__":
    main()