from __future__ import annotations

import json
import os
import sys
import operator
import uuid
from typing import Annotated, TypedDict, cast

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from langchain_oci import ChatOCIGenAI, create_oci_agent

from langgraph.checkpoint.oracledb import OracleDBSaver

from oracle_example_utils import connect_or_start_oracledb, stop_oracle_docker


class DemoState(TypedDict):
    messages: Annotated[list[str], operator.add]


def _format_messages(messages: object) -> object:
    if not isinstance(messages, list):
        return messages
    formatted = []
    for item in messages:
        role = getattr(item, "type", None)
        content = getattr(item, "content", None)
        if role is not None or content is not None:
            formatted.append({"role": role, "content": content})
            continue
        if isinstance(item, dict):
            formatted.append(
                {
                    "role": item.get("role") or item.get("type"),
                    "content": item.get("content") or item.get("text"),
                }
            )
            continue
        formatted.append(str(item))
    return formatted


def print_checkpoint_tuple(label: str, checkpoint_tuple) -> None:
    if checkpoint_tuple is None:
        print(f"{label}: no checkpoint found")
        return
    configurable = checkpoint_tuple.config.get("configurable", {})
    checkpoint_id = configurable.get("checkpoint_id")
    channel_values = checkpoint_tuple.checkpoint.get("channel_values")
    if isinstance(channel_values, dict) and "messages" in channel_values:
        channel_values = {**channel_values}
        channel_values["messages"] = _format_messages(channel_values["messages"])
    print(f"{label} checkpoint_id: {checkpoint_id}")
    print(
        f"{label} channel_values:\n"
        + json.dumps(channel_values, indent=2, sort_keys=True, default=str)
    )


def append_message(state: DemoState) -> DemoState:
    count = len(state.get("messages", []))
    return {"messages": [f"auto-msg-{count + 1}"]}


def main() -> None:
    conn, started_oracle = connect_or_start_oracledb()

    checkpointer = OracleDBSaver(conn)


    print("\nBuilding a minimal StateGraph with checkpointer")
    graph = StateGraph(DemoState)
    graph.add_node("append", append_message)
    graph.set_entry_point("append")
    graph.add_edge("append", END)
    app = graph.compile(checkpointer=checkpointer)

    config = cast(RunnableConfig, {"configurable": {"thread_id": "langgraph-demo-thread-1"}})
    result1 = app.invoke({"messages": ["hello"]}, config)
    result2 = app.invoke({"messages": ["second run"]}, config)


    print("\nCreating an OCI agent with checkpointer")
    print("ChatOCIGenAI only supports checkpointers when you build it with create_oci_agent() function")

    system_prompt = "You are a helpful assistant."

    agent = create_oci_agent(
        model_id=os.getenv("OCI_MODEL_ID"),
        service_endpoint=f"https://inference.generativeai.{os.getenv('OCI_REGION')}.oci.oraclecloud.com",
        compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
        checkpointer=checkpointer,
        tools=[],
        auth_profile=os.getenv("OCI_AUTH_PROFILE"),
        auth_type="API_KEY",
        system_prompt=system_prompt,
    )

    agent_config = cast(
        RunnableConfig, {"configurable": {"thread_id": f"oci-agent-demo-thread-1"}}
    )
    response = agent.invoke({"messages": ["Hello World."]}, config=agent_config)

    print("\nLatest checkpoint for LangGraph thread:")
    print_checkpoint_tuple("LangGraph", checkpointer.get_tuple(config))

    print("\nLatest checkpoint for OCI agent thread:")
    print_checkpoint_tuple("OCI Agent", checkpointer.get_tuple(agent_config))

    conn.close()
    if started_oracle:
        stop_oracle_docker()


if __name__ == "__main__":
    main()

