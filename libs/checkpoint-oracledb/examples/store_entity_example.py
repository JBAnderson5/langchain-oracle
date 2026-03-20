from __future__ import annotations
import os
import sys
import uuid
import json
from typing import Annotated, Any, cast

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from langgraph.store.oracledb import OracleIndexConfig, OracleStore

from langchain_oci import OCIGenAIEmbeddings, create_oci_agent
from oracle_example_utils import connect_or_start_oracledb, stop_oracle_docker, enable_oracle_vectors

namespace = ("myagent", "myuser")

def _print_items(label: str, items: list[Any]) -> None:
    print(label)
    for item in items:
        try:
            print(json.dumps(item,indent = 4))
        except:
            print(item)
    print("\n")

@tool
def create_entity(key,value,store: Annotated[Any,InjectedStore()]):
    """create new entities to remember later.
    Provide the name/type for the entity in key.
    Store the contents of the entity in value"""
    print(f"saving memory key: {key}, Value: {value}")
    store.put(namespace,key,{"content": value})


@tool
def search_entity(query: str, store: Annotated[Any,InjectedStore()]) -> str:
    """Search stored user memories for relevant preferences/context.
    Use query parameter for natural language search"""
    print(f"searching for entity: {query}")
    results = store.search(namespace, query=query, limit=3)
    _print_items("search results",results)
    if not results:
        return "No relevant Entity found."
    lines = []
    for item in results:
        lines.append(f"- {item.value}")
    return "Relevant Entities:\n" + "\n".join(lines)


def main() -> None:
    print("Langgraph docs on persistence: https://docs.langchain.com/oss/python/langgraph/persistence")
    conn, docker_oracle = connect_or_start_oracledb()
    if docker_oracle:
        conn = enable_oracle_vectors()
    print("\n")


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

    system_prompt = (
        "You are a helpful assistant."
        "When you learn something new about the user, save them in create_entity tool."
        "When asked about the user's preferences, past conversations, or stored context, call the search_entity tool to retrieve relevant memories before answering."
    )

    agent = create_oci_agent(
        model_id=os.getenv("OCI_MODEL_ID"),
        service_endpoint=f"https://inference.generativeai.{os.getenv('OCI_REGION')}.oci.oraclecloud.com",
        compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        tools=[create_entity,search_entity],
        auth_profile=os.getenv("OCI_AUTH_PROFILE"),
        auth_type="API_KEY",
        system_prompt=system_prompt,
        store = vector_store
    )

    print("creating entities")
    seed_messages = [
    "I love olive oil. My favorite is putting olive oil on the pizzas I make. ",
    "My favorite cuisine is Italian.",
    "I have a dog named Nova.",
    ]
    for msg in seed_messages:
        response = agent.invoke({"messages": [msg]}, config=RunnableConfig(
            {"configurable": {"thread_id": "create-entity-thread-1"}}
        ))
        for message in response["messages"]:
            print(message.content)
            try:
                print(message.tool_calls)
            except:
                continue

    _print_items("Current Entities",vector_store.search(namespace))


    print("asking about things from previous thread")
    chat_messages = [
        "What foods do I like?",
        "Do I have any pets?",
    ]

    for msg in chat_messages:
        response = agent.invoke({"messages": [msg]}, config=RunnableConfig(
            {"configurable": {"thread_id": "entity-chat-thread-2"}}
        ))
        for message in response["messages"]:
            print(message.content)
            try:
                print(message.tool_calls)
            except:
                continue



    conn.close()

    if docker_oracle:
        stop_oracle_docker()


if __name__ == "__main__":
    main()