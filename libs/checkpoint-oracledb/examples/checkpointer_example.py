from __future__ import annotations

import os
import sys
import uuid
from typing import cast

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import oracledb

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.oracledb import OracleDBSaver
from langgraph.checkpoint.oracledb.db_conn_utils import connect_to_oracle

from oracle_example_utils import connect_or_start_oracledb, stop_oracle_docker

def main() -> None:

    print("Langgraph docs on persistence: https://docs.langchain.com/oss/python/langgraph/persistence")
    print("\n")
    conn, docker_oracle = connect_or_start_oracledb()
    print("\n")

    print("creating oracle checkpointer object")
    saver = OracleDBSaver(conn)
    print("Schema will be created if missing on any public function call by internally calling setup() function")
    print("\n")

    print("When invoking a graph with a checkpointer, you must specify a thread_id as part of the configurable portion of the config")
    print("Thread ID is the primary key of a checkpoint. It can be anything you want and should be set at the application level")
    thread_id = f"thread-myuser-{uuid.uuid4().hex[:8]}"
    config = cast(RunnableConfig, {"configurable": {"thread_id": thread_id}})
    print("example config:")
    print(config)
    print("\n")

    print("You generally don't have to worry about creating checkpoints with put() or put_writes() function.")
    print("Just pass the checkpointer as a parameter in your graph.compile() function and the graph will automatically save the state of each thread")
    version = saver.get_next_version(None, None)
    checkpoint = cast(
        Checkpoint,
        {
        "id": "ckpt-1",
        "channel_versions": {"foo": version},
        "channel_values": {"foo": {"value": 123}},
        },
    )
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    metadata = cast(CheckpointMetadata, {"run_id": run_id})
    new_versions = cast(ChannelVersions, {"foo": version})
    next_config = saver.put(config, checkpoint, metadata, new_versions)

    saver.put_writes(
        next_config,
        [("foo", {"event": "started"})],
        task_id="task-1",
        task_path="workflow.step",
    )
    print("\n")

    print("You can get a checkpoint by calling get_tuple() and passing the thread id and optionally checkpoint id in a config")
    print("example config:\n",next_config)
    latest = saver.get_tuple(next_config)
    print("Latest checkpoint:\n", latest)
    print("\n")

    print("Creating a second checkpoint in the same thread")
    next_version = saver.get_next_version(version, None)
    second_checkpoint = cast(
        Checkpoint,
        {
            "id": "ckpt-2",
            "channel_versions": {"foo": next_version},
            "channel_values": {"foo": {"value": 456}},
        },
    )
    second_metadata = cast(CheckpointMetadata, {"run_id": run_id, "step": "second"})
    second_versions = cast(ChannelVersions, {"foo": next_version})
    second_config = saver.put(next_config, second_checkpoint, second_metadata, second_versions)
    print("Second checkpoint config:\n", second_config)
    print("\n")

    print("Creating a second thread with its own checkpoint")
    second_thread_id = f"thread-my2user-{uuid.uuid4().hex[:8]}"
    second_thread_config = cast(
        RunnableConfig, {"configurable": {"thread_id": second_thread_id}}
    )
    second_thread_checkpoint = cast(
        Checkpoint,
        {
            "id": "ckpt-1",
            "channel_versions": {"foo": saver.get_next_version(None, None)},
            "channel_values": {"foo": {"value": 999}},
        },
    )
    second_thread_metadata = cast(
        CheckpointMetadata, {"run_id": f"run-{uuid.uuid4().hex[:8]}", "thread": "secondary"}
    )
    second_thread_versions = cast(
        ChannelVersions, {"foo": second_thread_checkpoint["channel_versions"]["foo"]}
    )
    saver.put(second_thread_config, second_thread_checkpoint, second_thread_metadata, second_thread_versions)
    print("\n")

    print("You can also list checkpoints and filter by thread_id, time, and/or metadata")
    filtered = list(
        saver.list(
            config=RunnableConfig(),
            #filter={"run_id": run_id},
            before=config,
            limit=10,
        )
    )
    print("List checkpoints:")
    for item in filtered:
        configurable = item.config.get("configurable", {})
        print(
            f" thread id:{configurable.get('thread_id')}, checkpoint id: {item.checkpoint['id']}"
        )
    print("\n")

    print("you can also copy threads")
    target_thread_id = f"{thread_id}-copy"
    saver.copy_thread(thread_id, target_thread_id)
    copy_config = cast(RunnableConfig, {"configurable": {"thread_id": target_thread_id}})
    copy = saver.get_tuple(copy_config)
    print("copied checkpoint:\n", copy)
    print("\n")

    print("You have three options for deleting checkpoint data.")
    print("You can prune checkpoints for multiple threads")
    saver.prune([thread_id, target_thread_id], strategy="keep_latest")

    print("You can delete a thread")
    saver.delete_thread(target_thread_id)

    print("Or you can delete a specific run of a thread")
    saver.delete_for_runs([run_id])

    conn.close()

    if docker_oracle:
        stop_oracle_docker()





if __name__ == "__main__":
    main()