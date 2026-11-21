"""Adapter of shared generation to OpenAI's Responses API."""

from __future__ import annotations

import copy
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Self

from pydantic import MISSING

from aiollm.adapters.base import BaseCompositeAdapter
from aiollm.adapters.validation import (
    check_builtin_tools,
    check_conversation,
    check_files,
    checked_options,
    merge_options,
    request_errors,
)
from aiollm.enums import ServerToolPhase, ServerToolStatus, ToolChoice
from aiollm.http.exceptions.response import APIResponseError
from aiollm.types.requests.builtin import (
    BuiltinTool,
    CodeExecution,
    WebSearch,
)
from aiollm.types.requests.generation import (
    ForcedTool,
    GenerationRequest,
    InputItem,
    Tool,
    ToolCall,
)
from aiollm.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    ReasoningDeltaEvent,
    ServerToolCallEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallDoneEvent,
    ToolCallStartEvent,
)
from aiollm.types.responses.generation import GenerationResponse

from aiollm_openai.adapters.conversions import (
    INPUT_CONVERSION,
    STOP_CONVERSION,
    USAGE_CONVERSION,
    check_status,
    native_output,
    output_text,
    reasoning_summary,
    response_citations,
    response_server_calls,
    server_tool_call,
    tool_calls,
)
from aiollm_openai.api import PROVIDER
from aiollm_openai.methods.responses import (
    CreateResponse,
    StreamResponse,
)
from aiollm_openai.types import (
    FunctionToolCall,
    Response,
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from aiollm_openai.types.options import OpenAIGenerationOptions

if TYPE_CHECKING:
    from aiollm.adapters.protocols import IConversion
    from aiollm.adapters.stop import Stop
    from aiollm.types.responses.generation import TokenUsage

    from aiollm_openai.types import ResponseUsage


SHARED_FIELDS: Final[Mapping[str, str]] = {
    "safety_identifier": "end_user_id",
}
"""Shared request field of each method field with another name."""

DEFAULT_SCHEMA_NAME: Final = "output"
"""``name`` of a JSON Schema format when the adapter is given none."""


class OpenAIGenerationAdapter(
    BaseCompositeAdapter[
        GenerationRequest,
        Response,
        GenerationResponse[Response],
        ResponseStreamEvent,
        GenerationEvent[ResponseStreamEvent],
    ],
):
    """Generates text with ``POST /responses``.

    Request fields map one to one; a list of turns becomes a list of
    ``EasyInputMessage`` items. OpenAI's own limits are checked when the
    method is built, so a request that OpenAI would reject, such as
    ``max_output_tokens`` below 16, more than 16 metadata pairs or a
    conversation that does not end with a ``user`` turn, raises
    ``RequestValidationError`` before anything is sent.

    The conversions of the input, the token counts and the stop are
    keyword arguments, so any of them can be swapped without a subclass;
    a stream's terminal event goes through the same ones as a response.

    Attributes:
        input_conversion: Turns the shared ``input`` into OpenAI's.
        usage_conversion: Turns OpenAI's ``ResponseUsage`` into shared
            counts.
        stop_conversion: Tells how a finished response stopped.

    Example:
        >>> generation = GenerationResource(session, OpenAIGenerationAdapter())
        >>> response = await generation.create(
        ...     GenerationRequest(model="gpt-5", input="Hi")
        ... )
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        input_conversion: IConversion[
            str | Sequence[InputItem], str | list[dict[str, Any]]
        ] = INPUT_CONVERSION,
        usage_conversion: IConversion[
            ResponseUsage, TokenUsage
        ] = USAGE_CONVERSION,
        stop_conversion: IConversion[Response, Stop] = STOP_CONVERSION,
        schema_name: str = DEFAULT_SCHEMA_NAME,
    ) -> None:
        """Store the conversions and the name of an output schema.

        Args:
            input_conversion: Turns the shared ``input`` into OpenAI's.
            usage_conversion: Turns OpenAI's ``ResponseUsage`` into
                shared counts.
            stop_conversion: Tells how a finished response stopped.
            schema_name: The ``name`` OpenAI requires with a JSON Schema
                format; the shared ``output_schema`` has none.
        """
        self.input_conversion = input_conversion
        self.usage_conversion = usage_conversion
        self.stop_conversion = stop_conversion
        self.schema_name = schema_name
        self.provider_options: dict[str, Any] = {}

    def with_options(self, options: OpenAIGenerationOptions, /) -> Self:
        """Return a copy that sends the options with every method.

        Args:
            options: Fields of ``CreateResponse`` that the shared request
                does not hold, such as ``{"store": False}``.

        Returns:
            A copy of the adapter; this one is left as it is.

        Raises:
            RequestValidationError: A key is not one of the options.
        """
        adapter = copy.copy(self)
        adapter.provider_options = checked_options(
            options, OpenAIGenerationOptions, self.provider
        )
        return adapter

    def to_method(self, request: GenerationRequest) -> CreateResponse:
        """Build the method that creates a response at once.

        Args:
            request: The shared request.

        Returns:
            ``CreateResponse`` with the request's fields.

        Raises:
            RequestValidationError: OpenAI cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider, SHARED_FIELDS):
            return CreateResponse(**fields)

    def to_response(
        self,
        raw: Response,
        request: GenerationRequest,
    ) -> GenerationResponse[Response]:
        """Convert a finished response.

        Args:
            raw: OpenAI's response.
            request: The request it answers.

        Returns:
            The generated text with its token counts, the calls of the
            built-in tools OpenAI ran (``server_tool_calls``) and the
            sources its text cites (``citations``); an incomplete
            response keeps its text and gets ``incomplete_reason``, and
            a refusal gets ``IncompleteReason.REFUSAL`` and ``refusal``.

        Raises:
            APIResponseError: The response did not finish. A failed one
                is raised by ``CreateResponse`` before it gets here.
        """
        check_status(raw)
        stop = self.stop_conversion(raw)
        return GenerationResponse[Response](
            id=raw.id,
            provider=self.provider,
            model=raw.model,
            text=output_text(raw),
            usage=self._usage(raw.usage),
            incomplete_reason=stop.incomplete_reason,
            raw_incomplete_reason=stop.raw_incomplete_reason,
            refusal=stop.refusal,
            reasoning=reasoning_summary(raw),
            tool_calls=tool_calls(raw),
            server_tool_calls=response_server_calls(raw),
            citations=response_citations(raw),
            native=native_output(raw),
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> StreamResponse:
        """Build the method that streams a response.

        Args:
            request: The shared request.

        Returns:
            ``StreamResponse`` with the request's fields.

        Raises:
            RequestValidationError: OpenAI cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider, SHARED_FIELDS):
            return StreamResponse(**fields)

    async def to_events(
        self,
        items: AsyncIterator[ResponseStreamEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[ResponseStreamEvent]]:
        """Convert the events of a Responses stream.

        Args:
            items: The events OpenAI sent.
            request: The request the stream answers.

        Yields:
            One shared event per OpenAI event; an item of a built-in
            tool call is a ``ServerToolCallEvent`` when it is added and
            when it is done. The stream's end is
            checked by ``StreamResponse`` and by the use case.

        Raises:
            APIResponseError: Arguments came for an item that no
                ``response.output_item.added`` opened as a function call.
        """
        calls: dict[str, str] = {}
        async for item in items:
            yield self._to_event(item, calls)

    def _to_event(
        self,
        event: ResponseStreamEvent,
        calls: dict[str, str],
    ) -> GenerationEvent[ResponseStreamEvent]:
        """Convert one event of a Responses stream.

        Args:
            event: The event OpenAI sent.
            calls: The ``call_id`` of each function call item opened so
                far, by the item's ``id``; the deltas name the item.

        Returns:
            A text or reasoning summary delta, the start, a piece or the
            end of a function call, the completion (also of an
            incomplete response), or ``OtherEvent`` for every other
            event.
            ``response.failed`` never gets here: ``StreamResponse``
            raises it as an ``APIStatusError``.

        Raises:
            APIResponseError: A delta names an item that is no open
                function call.
        """
        match event:
            case ResponseOutputItemAddedEvent(item=FunctionToolCall() as call):
                calls[call.id or call.call_id] = call.call_id
                return ToolCallStartEvent[ResponseStreamEvent](
                    call_id=call.call_id, name=call.name, raw=event
                )
            case ResponseFunctionCallArgumentsDeltaEvent():
                if event.item_id not in calls:
                    message = (
                        "OpenAI stream sent arguments for item "
                        f"{event.item_id} without its function_call"
                    )
                    raise APIResponseError(message)
                return ToolCallDeltaEvent[ResponseStreamEvent](
                    call_id=calls[event.item_id],
                    arguments=event.delta,
                    raw=event,
                )
            case ResponseOutputItemDoneEvent(item=FunctionToolCall() as call):
                done = ToolCall(
                    id=call.call_id, name=call.name, arguments=call.arguments
                )
                return ToolCallDoneEvent[ResponseStreamEvent](
                    call=done, raw=event
                )
            case (
                ResponseOutputItemAddedEvent() | ResponseOutputItemDoneEvent()
            ):
                return _item_event(event)
            case ResponseTextDeltaEvent():
                return TextDeltaEvent[ResponseStreamEvent](
                    text=event.delta, raw=event
                )
            case ResponseReasoningSummaryTextDeltaEvent():
                return ReasoningDeltaEvent[ResponseStreamEvent](
                    text=event.delta, raw=event
                )
            case ResponseCompletedEvent() | ResponseIncompleteEvent():
                response = event.response
                stop = self.stop_conversion(response)
                return CompletedEvent[ResponseStreamEvent](
                    id=response.id,
                    provider=self.provider,
                    model=response.model,
                    text=output_text(response),
                    usage=self._usage(response.usage),
                    incomplete_reason=stop.incomplete_reason,
                    raw_incomplete_reason=stop.raw_incomplete_reason,
                    refusal=stop.refusal,
                    reasoning=reasoning_summary(response),
                    tool_calls=tool_calls(response),
                    server_tool_calls=response_server_calls(response),
                    citations=response_citations(response),
                    native=native_output(response),
                    raw=event,
                )
            case _:
                return OtherEvent[ResponseStreamEvent](raw=event)

    def _usage(self, usage: ResponseUsage | None) -> TokenUsage | None:
        """Convert the token counts, if OpenAI reported them.

        Args:
            usage: Token counts of a response, if sent.

        Returns:
            The shared counts, or ``None`` without counts.
        """
        return None if usage is None else self.usage_conversion(usage)

    def _fields(self, request: GenerationRequest) -> dict[str, Any]:
        """Take the method fields of a request.

        Args:
            request: The shared request.

        Returns:
            The fields that both methods take, the provider options of
            ``with_options`` included.

        Raises:
            RequestValidationError: The conversation does not end with a
                ``user`` turn, a part names another provider's file, or a
                built-in tool is one OpenAI does not run (see
                ``check_builtin_tools``).
        """
        check_conversation(request, self.provider)
        check_files(request, self.provider)
        check_builtin_tools(request, self.provider, (WebSearch, CodeExecution))
        fields = {
            "model": request.model,
            "input": self.input_conversion(request.input),
            "instructions": request.instructions,
            "max_output_tokens": request.max_output_tokens,
            "safety_identifier": request.end_user_id,
            "reasoning": (
                MISSING
                if request.effort is MISSING
                else {"effort": request.effort.value}
            ),
            "text": (
                MISSING
                if request.output_schema is MISSING
                else {
                    "format": {
                        "type": "json_schema",
                        "name": self.schema_name,
                        "schema": request.output_schema,
                        "strict": True,
                    }
                }
            ),
            "tools": _tools(request),
            "tool_choice": _tool_choice(request.tool_choice),
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        return merge_options(fields, self.provider_options, self.provider)


TOOL_CHOICES: Final[Mapping[ToolChoice, str]] = {
    ToolChoice.AUTO: "auto",
    ToolChoice.NONE: "none",
    ToolChoice.REQUIRED: "required",
}
"""OpenAI's ``tool_choice`` of each shared mode."""


def _item_event(
    event: ResponseOutputItemAddedEvent | ResponseOutputItemDoneEvent,
) -> GenerationEvent[ResponseStreamEvent]:
    """Convert the event of an output item that is not a function call.

    Args:
        event: ``response.output_item.added`` or ``.done``, whose item is
            no ``function_call``.

    Returns:
        The ``STARTED`` or ``DONE`` event of a built-in tool call for the
        call items OpenAI runs itself, the call ``IN_PROGRESS`` when it
        starts; ``OtherEvent`` for any other item (a message, reasoning).
    """
    item = event.item.model_dump(
        mode="json", by_alias=True, exclude_unset=True
    )
    call = server_tool_call(item)
    if call is None:
        return OtherEvent[ResponseStreamEvent](raw=event)
    if isinstance(event, ResponseOutputItemAddedEvent):
        return ServerToolCallEvent[ResponseStreamEvent](
            phase=ServerToolPhase.STARTED,
            call=call.model_copy(
                update={"status": ServerToolStatus.IN_PROGRESS, "error": None}
            ),
            raw=event,
        )
    return ServerToolCallEvent[ResponseStreamEvent](
        phase=ServerToolPhase.DONE, call=call, raw=event
    )


def _tools(request: GenerationRequest) -> list[dict[str, Any]] | MISSING:
    """Build ``tools`` from the function and the built-in tools.

    Args:
        request: The shared request.

    Returns:
        The function tools, then the built-in ones, or ``MISSING``
        without any.
    """
    functions = [] if request.tools is MISSING else request.tools
    tools = [_tool(tool) for tool in functions]
    tools += [_builtin_tool(tool) for tool in request.builtin_tools]
    return tools or MISSING


def _builtin_tool(tool: BuiltinTool) -> dict[str, Any]:
    """Build OpenAI's tool from a shared built-in tool.

    Args:
        tool: The shared tool; ``check_builtin_tools`` has accepted it.

    Returns:
        ``WebSearch`` as a ``web_search`` tool with its domains as
        ``filters.allowed_domains`` and its location as an
        ``approximate`` one, ``CodeExecution`` as a ``code_interpreter``
        tool in an ``auto`` container, and a ``ProviderTool`` (the only
        other kind) as its own definition.
    """
    match tool:
        case WebSearch():
            search: dict[str, Any] = {"type": "web_search"}
            if tool.allowed_domains is not MISSING:
                search["filters"] = {
                    "allowed_domains": list(tool.allowed_domains)
                }
            location = tool.user_location
            if location is not MISSING:
                search["user_location"] = {
                    "type": "approximate",
                    **{
                        name: getattr(location, name)
                        for name in ("country", "region", "city", "timezone")
                        if getattr(location, name) is not MISSING
                    },
                }
            return search
        case CodeExecution():
            return {"type": "code_interpreter", "container": {"type": "auto"}}
        case _:
            return copy.deepcopy(dict(tool.definition))


def _tool(tool: Tool) -> dict[str, Any]:
    """Build OpenAI's function tool from a shared tool.

    Args:
        tool: The shared tool.

    Returns:
        A ``function`` tool. Its ``strict`` is required, so a tool that
        leaves it out sends ``null``, which OpenAI takes as not strict
        (seen in a live check, 2026-11-06), as Anthropic's default.
    """
    function: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": None if tool.strict is MISSING else tool.strict,
    }
    if tool.description is not MISSING:
        function["description"] = tool.description
    return function


def _tool_choice(
    choice: ToolChoice | ForcedTool | MISSING,
) -> str | dict[str, Any] | MISSING:
    """Build OpenAI's ``tool_choice``.

    Args:
        choice: The shared choice, if given.

    Returns:
        The mode, ``{"type": "function", "name": ...}`` for a forced
        tool, or ``MISSING`` when no choice is given.
    """
    if choice is MISSING:
        return MISSING
    if isinstance(choice, ForcedTool):
        return {"type": "function", "name": choice.name}
    return TOOL_CHOICES[choice]
