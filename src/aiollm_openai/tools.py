"""OpenAI's built-in tools, as the shared request takes them.

``WebSearch`` and ``CodeExecution`` of the shared layer send the options
both providers have. These functions are for the rest: OpenAI's own
options (``search_context_size``, ``external_web_access``, the code
interpreter's container), and the file search tool, which only OpenAI
has. Each returns a ``ProviderTool`` of this provider, built from the
generated request parameters, so a field the tool does not have, or a
value it refuses, raises ``ValueError`` here and not at the request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import MISSING

from aiollm.types.requests.builtin import ProviderTool

from aiollm_openai.api import PROVIDER
from aiollm_openai.methods.params.tools import (
    AutoCodeInterpreterToolParam,
    CodeInterpreterTool,
    FileSearchTool,
    WebSearchTool,
    WebSearchToolFilters,
)
from aiollm_openai.methods.params.tools import (
    WebSearchApproximateLocationWebSearchApproximateLocation as Location,
)

type WebSearchVersion = Literal["web_search", "web_search_2025_08_26"]
"""The ``type`` of the web search tool: the current one, or the dated one."""

type MemoryLimit = Literal["1g", "4g", "16g", "64g"]
"""The memory of a code interpreter's container."""


def web_search(
    *,
    version: WebSearchVersion = "web_search",
    allowed_domains: list[str] | MISSING = MISSING,
    search_context_size: Literal["low", "medium", "high"] | MISSING = MISSING,
    external_web_access: bool | MISSING = MISSING,
    user_location: Location | MISSING = MISSING,
) -> ProviderTool:
    """Build OpenAI's web search tool, with its own options.

    Args:
        version: The ``type`` of the tool; the shared ``WebSearch``
            sends ``web_search``.
        allowed_domains: Search only these domains.
        search_context_size: How much of the context window the search
            may use; OpenAI's default is ``medium``.
        external_web_access: ``False`` for an offline, cache-only search.
        user_location: Where the user is, to localize the results; the
            shared ``UserLocation`` is for the shared ``WebSearch``.

    Returns:
        The tool, for ``tools=`` of a request.

    Raises:
        ValueError: An option has a value the API refuses.

    Example:
        >>> web_search(search_context_size="high")
    """
    tool = WebSearchTool(
        type=version,
        filters=(
            MISSING
            if allowed_domains is MISSING
            else WebSearchToolFilters(allowed_domains=allowed_domains)
        ),
        search_context_size=search_context_size,
        external_web_access=external_web_access,
        user_location=user_location,
    )
    return _tool(tool)


def file_search(
    vector_store_ids: list[str],
    *,
    max_num_results: int | MISSING = MISSING,
) -> ProviderTool:
    """Build OpenAI's file search tool: the model reads your vector stores.

    Only OpenAI has it, so it has no shared counterpart.

    Args:
        vector_store_ids: The vector stores to search.
        max_num_results: The most results to return, 1 to 50.

    Returns:
        The tool, for ``tools=`` of a request.

    Raises:
        ValueError: An option has a value the API refuses.

    Example:
        >>> file_search(["vs_123"], max_num_results=5)
    """
    return _tool(
        FileSearchTool(
            vector_store_ids=vector_store_ids, max_num_results=max_num_results
        )
    )


def code_interpreter(
    *,
    container: str | MISSING = MISSING,
    file_ids: list[str] | MISSING = MISSING,
    memory_limit: MemoryLimit | MISSING = MISSING,
) -> ProviderTool:
    """Build OpenAI's code interpreter with a container of your choice.

    The shared ``CodeExecution`` runs it in an ``auto`` container, which
    OpenAI creates (and bills) for the request.

    Args:
        container: The id of a container to reuse; then ``file_ids``
            and ``memory_limit`` belong to it, not to this call.
        file_ids: Uploaded files to make available to the code, in a
            new container.
        memory_limit: The memory of a new container.

    Returns:
        The tool, for ``tools=`` of a request.

    Raises:
        ValueError: ``container`` is given with ``file_ids`` or
            ``memory_limit``, or an option has a value the API refuses.

    Example:
        >>> code_interpreter(memory_limit="4g")
    """
    if container is not MISSING:
        if file_ids is not MISSING or memory_limit is not MISSING:
            message = "file_ids and memory_limit are for a new container"
            raise ValueError(message)
        return _tool(CodeInterpreterTool(container=container))
    auto = AutoCodeInterpreterToolParam(
        file_ids=file_ids, memory_limit=memory_limit
    )
    return _tool(CodeInterpreterTool(container=auto))


def _tool(
    parameters: WebSearchTool | FileSearchTool | CodeInterpreterTool,
) -> ProviderTool:
    """Wrap validated tool parameters as a tool of this provider.

    Args:
        parameters: The generated request parameters of the tool.

    Returns:
        The tool, its definition the parameters as OpenAI takes them.
    """
    return ProviderTool(
        provider=PROVIDER, definition=parameters.model_dump(mode="json")
    )


__all__ = ["code_interpreter", "file_search", "web_search"]
