
# setup python environment

recommended to use uv

`brew install uv`

this will create/update .venv to match pyproject.toml and uv.lock
`cd langchain-oracle/libs/checkpoint-oracledb`
`uv sync`

this installs additional dependencies when doing things like running tests as declared in pyproject.toml
`uv sync --group test`

## adding dependencies

`uv add <package names>`

or to add to a specific group like for testing
`uv add --group test <package names>`

# Running Example Code

## Setup Environment Variables

You can source environment variables from .env file:

```bash
set -a
source .env
set +a
```

or set them via command line:
`export DB_USER=SYSTEM`

DeSource environment variables:
`unset DB_DSN DB_USER DB_WALLET_LOCATION DB_WALLET_PASSWORD`

### Oracle DB
If ommitted, the script will try to use a local docker version of Oracle DB. Make sure a docker daemon like Docker or Rancher Desktop is running.

Required environment variables for a non-docker connection:
- DB_USER
- DB_PASSWORD
- DB_DSN (Easy Connect string, e.g. "dbhost/orclpdb1")
optional variables that may be required to connect to your oracle DB:
- DB_WALLET_LOCATION
- DB_WALLET_PASSWORD

### OCI GenAI variables:

- OCI_COMPARTMENT_ID
- OCI_MODEL_ID
- OCI_REGION
- OCI_AUTH_PROFILE

## Available example scripts

### Checkpointer
pure checkpointer example that shows all the public functions and details of the checkpointer data structures
`uv run python examples/checkpointer_example.py`

Shows how to use checkpointers when compiling a langgraph graph or a langchain-oracle create_oci_agent graph.
`uv run python examples/checkpointer_agents_example.py`

### (Memory) Store

Store example that shows how to create a non-vector store and a vector store using OCI gen ai service. Uses put() to create memories and search() to retrieve them
`uv run python examples/store_example.py`

Store example that uses a vector store in combination with a create_oci_agent() to show how agents can create and retrieve entity information about a user
`uv run python examples/store_entity_example.py`

Store example using ChatOCIOpenAI with Langgraph. Loads a knowledge base from python module docstrings and dynamically fills llm's context window with knowledge base results from the user query
`uv run python examples/store_knowledge_base_example.py`

# TODO:
- figure out who to talk to to merge code into upstream langchain-oracle - Federico Kamelhar
- check latest cohere/gemini embedding models
- review postgres tests/make ours more comprehensive
- build real async version (down the line)
- how to add checkpointer/thread history to memory store (or extend checkpointers to include vectors)?
- study summarization and anthropic compression
- copy memory/migrate memories from one AI harness/system to another
- add wayflow example to checkpointer_agents_example.py once wayflow supports checkpoints

- __add `backend=` support to `create_deep_research_agent(...)`__
- Patch the Langfuse callback to treat missing usage as 0:

```python
# in langfuse/langchain/CallbackHandler.py
if usage_model["output"] is None:
    usage_model["output"] = 0
```





# testing

make sure docker daemon is running via rancher desktop/docker
`open -a Docker`
or
`open -a "Rancher Desktop"`

- TODO: should we switch to podman?

Testing requires additional dependencies:
`uv sync --group test`

## Running Tests

`make conformance`
- this runs langgraphs conformance tests for checkpointers

`make test`
- this runs our unit tests (we don't have any yet)


## linting and formatting

```bash
make format    # full formatting

make format_diff    # diff-only formatting
make type           # type checking only


make lint      # full lint

make lint_package   # only langgraph package
make lint_tests     # only tests
make lint_diff      # only files changed vs main

```

# uv python environment troubleshooting

"""
deactivate
unset VIRTUAL_ENV
hash -r
"""
