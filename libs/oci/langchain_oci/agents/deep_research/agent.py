# Copyright (c) 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""OCI Deep Research Agent - deepagents-based research agent with OCI GenAI."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from langchain_core.tools import BaseTool

from langchain_oci.agents.common import OCIConfig, filter_none, merge_model_kwargs
from langchain_oci.agents.datastores import VectorDataStore, create_datastore_tools
from langchain_oci.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_oci.common.auth import OCIAuthType

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

create_agent: Any = None
try:
    import langchain.agents as _langchain_agents
except (ImportError, AttributeError):
    pass
else:
    create_agent = getattr(_langchain_agents, "create_agent", None)

create_react_agent: Any = None
try:
    import langgraph.prebuilt as _langgraph_prebuilt
except ImportError:
    pass
else:
    create_react_agent = getattr(_langgraph_prebuilt, "create_react_agent", None)


def _get_lightweight_agent_factory() -> tuple[Callable[..., Any], bool]:
    """Get the lightweight agent factory across supported langchain versions."""
    if callable(create_agent):
        return create_agent, False

    if callable(create_react_agent):
        return create_react_agent, True

    raise ImportError(
        "Could not import lightweight agent factory. "
        "Please install langchain>=1.0.0 or langgraph."
    )


def _should_use_lightweight_agent(
    *,
    middleware: Optional[Sequence[Any]],
    datastores: Optional[Dict[str, VectorDataStore]],
    subagents: Optional[List[Any]],
    skills: Optional[List[str]],
    memory: Optional[List[str]],
    backend: Any = None,
    cache: Any = None,
    interrupt_on: Optional[Dict[str, Any]] = None,
    response_format: Any = None,
    context_schema: Optional[type] = None,
) -> bool:
    # Deep agent features require create_deep_agent
    if subagents or skills or memory:
        return False
    if backend or cache or interrupt_on or response_format or context_schema:
        return False
    if datastores:
        return True
    return middleware is not None and len(middleware) == 0


def _check_deep_research_prerequisites() -> None:
    """Validate runtime prerequisites for deep research.

    Raises clear errors when the environment cannot support deep agents,
    rather than letting users hit cryptic schema conversion or hash errors.
    """
    import sys

    min_version = (3, 11)
    if sys.version_info < min_version:
        msg = (
            "Deep Research requires Python 3.11 or later. "
            f"Current version: {sys.version.split()[0]}. "
            "The deepagents package and its middleware are not compatible "
            "with earlier Python versions."
        )
        raise RuntimeError(msg)

    try:
        import deepagents  # noqa: F401
    except ImportError:
        raise ImportError(
            "Deep Research requires the 'deepagents' package. "
            "Install it with: pip install 'langchain-oci[deep-research]' "
            "or: pip install deepagents"
        ) from None


def create_deep_research_agent(
    tools: Optional[Sequence[Union[BaseTool, Callable[..., Any]]]] = None,
    *,
    # Datastores - if provided, auto-routing search is enabled
    datastores: Optional[Dict[str, VectorDataStore]] = None,
    default_datastore: Optional[str] = None,
    default_store: Optional[str] = None,  # Alias for default_datastore
    embedding_model: Any = None,
    top_k: int = 5,
    # OCI-specific options
    model_id: str = "google.gemini-2.5-pro",
    compartment_id: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    auth_type: str | OCIAuthType = OCIAuthType.API_KEY,
    auth_profile: str = "DEFAULT",
    auth_file_location: str = "~/.oci/config",
    # Deep agent options
    system_prompt: Optional[str] = None,
    subagents: Optional[List[Any]] = None,
    skills: Optional[List[str]] = None,
    memory: Optional[List[str]] = None,
    middleware: Optional[Sequence[Any]] = None,
    response_format: Any = None,
    context_schema: Optional[type] = None,
    # LangGraph options
    checkpointer: Any = None,
    store: Any = None,
    backend: Any = None,
    cache: Any = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    interrupt_on: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    name: Optional[str] = None,
    # Model kwargs
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_input_tokens: Optional[int] = None,  # noqa: ARG001 - Intentionally ignored
    **model_kwargs: Any,
) -> "CompiledStateGraph":
    """Create a Deep Research Agent using OCI GenAI and deepagents.

    This agent is designed for multi-step research tasks that require:
    - Searching multiple data sources (OpenSearch, ADB)
    - Planning and reflection
    - Synthesizing information into reports

    Args:
        tools: Custom tools for the agent.
        datastores: Dict of vector datastores for auto-routing search.
        default_datastore: Fallback datastore if routing is inconclusive.
        default_store: Alias for default_datastore.
        embedding_model: Custom embedding model for datastores.
        top_k: Number of search results to return.
        model_id: OCI model identifier (Gemini models recommended).
        compartment_id: OCI compartment OCID.
        service_endpoint: OCI GenAI service endpoint.
        auth_type: OCI authentication type.
        auth_profile: OCI config profile name.
        auth_file_location: Path to OCI config file.
        system_prompt: Custom system prompt for the agent.
        subagents: List of subagents for delegation.
        skills: List of skill names to enable.
        memory: List of memory namespaces.
        middleware: Custom middleware. Pass empty list to disable defaults.
        response_format: Structured output response format for the agent.
        context_schema: Schema for typed context passed into the agent graph.
        checkpointer: LangGraph checkpointer for persistence/memory.
        store: LangGraph store for long-term memory.
        backend: State backend for the deep agent (e.g., StoreBackend).
            When provided, the agent uses this backend instead of the default
            ephemeral StateBackend, enabling persistent skills and memory.
        cache: LangGraph cache for caching LLM calls.
        interrupt_before: Node names to interrupt before (lightweight path).
        interrupt_after: Node names to interrupt after (lightweight path).
        interrupt_on: Mapping of tool names to interrupt configs for
            human-in-the-loop approval (e.g., ``{"edit_file": True}``).
            Used by the deep agent path via HumanInTheLoopMiddleware.
        debug: Enable debug mode.
        name: Name for the agent.
        temperature: Model temperature.
        max_tokens: Maximum output tokens (e.g., 65536 for Gemini 2.5 Pro).
        max_input_tokens: Ignored. Input limits are model-determined.
        **model_kwargs: Additional model kwargs.

    Returns:
        CompiledStateGraph: A compiled deep research agent.

    Example:
        >>> from langchain_oci.agents.deep_research import OpenSearch, ADB
        >>>
        >>> agent = create_deep_research_agent(
        ...     datastores={
        ...         "docs": OpenSearch(
        ...             endpoint="https://opensearch:9200",
        ...             index_name="company-docs",
        ...             datastore_description="internal documentation, policies",
        ...         ),
        ...         "sales": ADB(
        ...             dsn="mydb_low",
        ...             user="ADMIN",
        ...             password="...",
        ...             datastore_description="sales data, revenue, customers",
        ...         ),
        ...     },
        ...     compartment_id="ocid1.compartment...",
        ... )
    """
    _check_deep_research_prerequisites()

    # Resolve OCI configuration
    oci_config = OCIConfig.resolve(
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type=auth_type,
        auth_profile=auth_profile,
        auth_file_location=auth_file_location,
    )

    # Build tools list
    all_tools: list[BaseTool | Callable[..., Any]] = []

    if datastores:
        datastore_tools = create_datastore_tools(
            stores=datastores,
            default_store=default_store or default_datastore,
            embedding_model=embedding_model,
            compartment_id=oci_config.compartment_id,
            service_endpoint=oci_config.service_endpoint,
            auth_type=oci_config.auth_type,
            auth_profile=oci_config.auth_profile,
            top_k=top_k,
        )
        all_tools.extend(datastore_tools)

    if tools:
        all_tools.extend(tools)

    # Create OCI chat model
    llm = ChatOCIGenAI(
        model_id=model_id,
        compartment_id=oci_config.compartment_id,
        service_endpoint=oci_config.service_endpoint,
        auth_type=oci_config.auth_type,
        auth_profile=oci_config.auth_profile,
        auth_file_location=oci_config.auth_file_location,
        model_kwargs=merge_model_kwargs(
            model_kwargs,
            temperature,
            max_tokens,
            model_id=model_id,
        ),
    )

    if _should_use_lightweight_agent(
        middleware=middleware,
        datastores=datastores,
        subagents=subagents,
        skills=skills,
        memory=memory,
        backend=backend,
        cache=cache,
        interrupt_on=interrupt_on,
        response_format=response_format,
        context_schema=context_schema,
    ):
        create_agent_func, use_legacy_api = _get_lightweight_agent_factory()
        prompt_key = "prompt" if use_legacy_api else "system_prompt"
        agent_kwargs = {
            "model": llm,
            "tools": all_tools,
            **filter_none(
                middleware=None if use_legacy_api else tuple(middleware or ()),
                checkpointer=checkpointer,
                store=store,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                name=name,
                **{prompt_key: system_prompt},
            ),
            "debug": debug,
        }
        compiled = create_agent_func(
            **{key: value for key, value in agent_kwargs.items() if value is not None}
        )
    else:
        try:
            from deepagents import create_deep_agent
        except ImportError as ex:
            raise ImportError(
                "deepagents required. Install with: pip install deepagents"
            ) from ex

        # Build agent kwargs - only include non-None values
        agent_kwargs = {
            "model": llm,
            "tools": all_tools,
            **filter_none(
                system_prompt=system_prompt,
                subagents=subagents,
                skills=skills,
                memory=memory,
                middleware=middleware,
                response_format=response_format,
                context_schema=context_schema,
                checkpointer=checkpointer,
                store=store,
                backend=backend,
                cache=cache,
                interrupt_on=interrupt_on,
                name=name,
            ),
        }

        # debug=False is meaningful, so handle separately
        if debug:
            agent_kwargs["debug"] = True

        compiled = create_deep_agent(**agent_kwargs)

    # Expose the underlying OCI chat model for explicit cleanup in long-lived
    # processes (and in our integration tests). This avoids aiohttp
    # "Unclosed client session" warnings when async pooling is used.
    setattr(compiled, "_oci_llm", llm)
    return compiled
