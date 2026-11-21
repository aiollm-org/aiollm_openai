"""Conversions of OpenAI's Responses types into shared types."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any, Final, TypeIs

from pydantic import MISSING

from aiollm.adapters.base import BaseConversion
from aiollm.adapters.stop import Stop
from aiollm.enums import (
    CitationKind,
    IncompleteReason,
    Role,
    ServerToolKind,
    ServerToolStatus,
)
from aiollm.http.exceptions.response import APIResponseError
from aiollm.types.requests.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    ProviderFile,
)
from aiollm.types.requests.generation import (
    InputItem,
    NativeOutput,
    ToolCall,
    ToolCallTurn,
    Turn,
)
from aiollm.types.responses.builtin import (
    Citation,
    ProducedFile,
    ServerToolCall,
)
from aiollm.types.responses.generation import TokenUsage

from aiollm_openai.api import PROVIDER
from aiollm_openai.types import (
    FunctionToolCall,
    OutputMessage,
    OutputTextContent,
    ReasoningItem,
    RefusalContent,
    Response,
    ResponseUsage,
)

INCOMPLETE_REASONS: Final[Mapping[str, IncompleteReason]] = {
    "max_output_tokens": IncompleteReason.MAX_TOKENS,
}
"""Shared reason of each ``incomplete_details.reason`` that has one.

Every other reason (``content_filter``, ``max_messages``, ``steered``,
none at all, a reason OpenAI adds later) is ``IncompleteReason.OTHER``.
"""


class InputConversion(
    BaseConversion[str | Sequence[InputItem], str | list[dict[str, Any]]],
):
    """Converts the shared ``input`` into OpenAI's ``input``.

    A text is sent as it is. In a list, a text turn becomes an
    ``EasyInputMessage`` (``role``, ``content``), an ``assistant`` turn
    with OpenAI's own output (``native``: a response that used built-in
    tools, whose search calls and reasoning items OpenAI needs back)
    those output items, a turn of calls its
    OpenAI output items when it has them (``native``, reasoning items
    with it), else an ``assistant`` message with its text and a
    ``function_call`` per call, and a tool result a
    ``function_call_output``.
    """

    def __call__(
        self, source: str | Sequence[InputItem], /
    ) -> str | list[dict[str, Any]]:
        """Convert the input.

        Args:
            source: The text or the conversation of the shared request.

        Returns:
            The text, or the input items.
        """
        if isinstance(source, str):
            return source
        items: list[dict[str, Any]] = []
        for item in source:
            match item:
                case Turn(native=NativeOutput() as native) if (
                    item.role is Role.ASSISTANT and native.provider == PROVIDER
                ):
                    items += [dict(output) for output in native.items]
                case Turn():
                    items.append(
                        {
                            "role": item.role.value,
                            "content": item.content
                            if isinstance(item.content, str)
                            else [_part(part) for part in item.content],
                        }
                    )
                case ToolCallTurn() if _own(item.native):
                    items += [dict(native) for native in item.native.items]
                case ToolCallTurn():
                    if item.text is not MISSING:
                        items.append(
                            {
                                "role": Role.ASSISTANT.value,
                                "content": item.text,
                            }
                        )
                    items += [
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                        for call in item.tool_calls
                    ]
                case _:
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": item.output,
                        }
                    )
        return items


DEFAULT_FILENAME: Final = "document.pdf"
"""The name OpenAI is told for a document given without one."""


def _data_url(data: bytes, media_type: str) -> str:
    """Write bytes as a ``data:`` URL.

    Args:
        data: The content.
        media_type: Its MIME type.

    Returns:
        The URL, the content in base64.
    """
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


def _part(part: ContentPart) -> dict[str, Any]:
    """Convert a part of a turn into an item of a message's content.

    Text becomes ``input_text``; an image ``input_image`` with its
    ``data:`` URL, URL or ``file_id`` and ``detail: "auto"``, OpenAI's
    default, which the schema requires; a document ``input_file`` with
    ``file_data`` (a ``data:`` URL) and ``filename``, ``file_url`` or
    ``file_id``. The adapter has checked that a file is OpenAI's.

    Args:
        part: The part.

    Returns:
        The content item.
    """
    match part:
        case ImagePart(data=bytes() as data, media_type=str() as kind):
            return {
                "type": "input_image",
                "image_url": _data_url(data, kind),
                "detail": "auto",
            }
        case ImagePart(file=ProviderFile() as file):
            return {
                "type": "input_image",
                "file_id": file.id,
                "detail": "auto",
            }
        case ImagePart():
            return {
                "type": "input_image",
                "image_url": part.url,
                "detail": "auto",
            }
        case DocumentPart(data=bytes() as data):
            filename = part.filename
            return {
                "type": "input_file",
                "file_data": _data_url(data, part.media_type),
                "filename": DEFAULT_FILENAME
                if filename is MISSING
                else filename,
            }
        case DocumentPart(file=ProviderFile() as file):
            return {"type": "input_file", "file_id": file.id}
        case DocumentPart():
            return {"type": "input_file", "file_url": part.url}
        case _:
            return {"type": "input_text", "text": part.text}


class UsageConversion(BaseConversion[ResponseUsage, TokenUsage]):
    """Converts OpenAI's token counts; they map one to one."""

    def __call__(self, source: ResponseUsage, /) -> TokenUsage:
        """Convert the counts of a response.

        Args:
            source: Token counts of a response.

        Returns:
            The shared token counts.
        """
        inputs = source.input_tokens_details
        outputs = source.output_tokens_details
        return TokenUsage(
            input_tokens=source.input_tokens,
            output_tokens=source.output_tokens,
            total_tokens=source.total_tokens,
            cache_read_tokens=inputs.cached_tokens,
            cache_write_tokens=inputs.cache_write_tokens,
            reasoning_tokens=outputs.reasoning_tokens,
        )


class StopConversion(BaseConversion[Response, Stop]):
    """Tells how a finished response stopped.

    A refusal part makes it a refusal, whatever the status, as OpenAI
    marks a refusal by a content part, not by a status.
    """

    def __call__(self, source: Response, /) -> Stop:
        """Convert the stop of a finished response.

        Args:
            source: A ``completed`` or ``incomplete`` response, also the
                one inside a stream's terminal event.

        Returns:
            ``REFUSAL`` and the refusal text if the output has a refusal
            part; else, for an incomplete response, ``MAX_TOKENS`` or
            ``OTHER`` with OpenAI's reason; nothing for a response that
            finished.
        """
        return Stop(
            _incomplete_reason(source),
            _raw_incomplete_reason(source),
            _refusal(source),
        )


def _refusal(response: Response) -> str | None:
    """Collect the refusal of a response.

    Args:
        response: A finished response.

    Returns:
        The text of all ``refusal`` parts of the output messages joined
        together, or ``None`` if there are none.
    """
    parts = [
        part.refusal
        for item in response.output
        if isinstance(item, OutputMessage)
        for part in item.content
        if isinstance(part, RefusalContent)
    ]
    return "".join(parts) if parts else None


def _raw_incomplete_reason(response: Response) -> str | None:
    """Take OpenAI's own reason of an early stop.

    Args:
        response: A finished response.

    Returns:
        ``incomplete_details.reason`` of an incomplete response, such as
        ``"max_output_tokens"``, or ``None`` if the response is not
        incomplete or OpenAI gives no reason.
    """
    if response.status != "incomplete":
        return None
    details = response.incomplete_details
    return None if details is None else details.reason


def _incomplete_reason(response: Response) -> IncompleteReason | None:
    """Tell why a response stopped before the end.

    A refusal part makes it a refusal, whatever the status, as OpenAI
    marks a refusal by a content part, not by a status.

    Args:
        response: A finished response.

    Returns:
        ``REFUSAL`` if the output has a refusal part; else, for an
        incomplete response, ``MAX_TOKENS`` or ``OTHER`` from its
        details; ``None`` for a response that finished.
    """
    if _refusal(response) is not None:
        return IncompleteReason.REFUSAL
    if response.status != "incomplete":
        return None
    reason = _raw_incomplete_reason(response)
    return INCOMPLETE_REASONS.get(reason or "", IncompleteReason.OTHER)


def output_text(response: Response) -> str | None:
    """Join the text of all output messages.

    The Responses API sends no such field: its schema names
    ``output_text`` as a property the SDKs compute, so ``Response``
    reads it as ``None``. It is collected here the way the SDKs do,
    from the ``output_text`` parts of the ``message`` items.

    Args:
        response: A response.

    Returns:
        The joined text, or ``None`` if the output has no text part.
    """
    parts = [
        part.text
        for item in response.output
        if isinstance(item, OutputMessage)
        for part in item.content
        if isinstance(part, OutputTextContent)
    ]
    return "".join(parts) if parts else None


def tool_calls(response: Response) -> list[ToolCall]:
    """Collect the function calls of a response.

    Args:
        response: A response.

    Returns:
        A shared call per ``function_call`` item, in output order, with
        its ``call_id``; empty without calls.
    """
    return [
        ToolCall(id=item.call_id, name=item.name, arguments=item.arguments)
        for item in response.output
        if isinstance(item, FunctionToolCall)
    ]


def _own(native: NativeOutput | MISSING) -> TypeIs[NativeOutput]:
    """Tell OpenAI's own output from another provider's or none.

    Args:
        native: The ``native`` of a turn of calls.

    Returns:
        Whether it is OpenAI's output, to be sent in place of the turn.
    """
    return native is not MISSING and native.provider == PROVIDER


def output_items(response: Response) -> list[dict[str, Any]]:
    """Dump a response's output items as OpenAI sent them.

    Args:
        response: A response.

    Returns:
        Each output item with only the fields OpenAI sent.
    """
    return [
        item.model_dump(mode="json", by_alias=True, exclude_unset=True)
        for item in response.output
    ]


def native_output(response: Response) -> NativeOutput | None:
    """Keep a response's output as OpenAI's next request takes it.

    Args:
        response: A response.

    Returns:
        Its output items with only the fields OpenAI sent, which are
        valid input items (reasoning items with their ``id`` or
        ``encrypted_content``, function calls, the calls of built-in
        tools, messages with their annotations); ``None`` without
        output.
    """
    items = output_items(response)
    return NativeOutput(provider=PROVIDER, items=items) if items else None


SERVER_TOOL_KINDS: Final[Mapping[str, ServerToolKind]] = {
    "web_search_call": ServerToolKind.WEB_SEARCH,
    "file_search_call": ServerToolKind.FILE_SEARCH,
    "code_interpreter_call": ServerToolKind.CODE_EXECUTION,
    "image_generation_call": ServerToolKind.IMAGE_GENERATION,
    "mcp_call": ServerToolKind.MCP,
}
"""Shared kind of each output item that is a call OpenAI ran itself.

The other tool calls of the output (computer, shell, patch, custom, local
shell) are for the client to run.
"""

SERVER_TOOL_STATUSES: Final[Mapping[str, ServerToolStatus]] = {
    "completed": ServerToolStatus.COMPLETED,
    "failed": ServerToolStatus.FAILED,
    "incomplete": ServerToolStatus.INCOMPLETE,
}
"""Shared status of each final ``status`` of a call item.

A call that is still ``in_progress``, ``searching``, ``interpreting`` or
``generating`` in a finished response never got its result, so it is
``INCOMPLETE``.
"""

FAILED: Final = "failed"
"""What a failed call says when OpenAI tells no more."""


def server_tool_call(item: Mapping[str, Any]) -> ServerToolCall | None:
    """Turn an output item of a built-in tool call into a shared call.

    Args:
        item: An output item, as OpenAI sent it.

    Returns:
        The call: a search's queries, a code interpreter's code, the
        status (``COMPLETED``, ``FAILED`` with the item's error, or
        ``INCOMPLETE`` for any other), the item in ``raw``; ``None`` for
        an item that is no call OpenAI ran.
    """
    kind = SERVER_TOOL_KINDS.get(str(item.get("type")))
    if kind is None:
        return None
    status = SERVER_TOOL_STATUSES.get(
        str(item.get("status")), ServerToolStatus.INCOMPLETE
    )
    code = item.get("code")
    return ServerToolCall(
        id=item["id"],
        kind=kind,
        status=status,
        queries=_queries(item),
        code=code if isinstance(code, str) else None,
        error=_error(item) if status is ServerToolStatus.FAILED else None,
        raw=[dict(item)],
    )


def _queries(item: Mapping[str, Any]) -> list[str]:
    """Collect what a search call looked for.

    Args:
        item: A ``web_search_call`` or ``file_search_call`` item.

    Returns:
        The ``query`` and ``queries`` of a web search's ``search``
        action, or the ``queries`` of a file search, without repeats;
        empty for any other item.
    """
    action = item.get("action")
    found = [
        *(item.get("queries") or []),
        *(
            [action.get("query"), *(action.get("queries") or [])]
            if isinstance(action, Mapping)
            else []
        ),
    ]
    return list(dict.fromkeys(q for q in found if isinstance(q, str)))


def _error(item: Mapping[str, Any]) -> str:
    """Tell what went wrong in a failed call item.

    Args:
        item: A failed call item.

    Returns:
        The item's ``error`` (a text, or the ``message``, ``code`` or
        ``type`` of an object, whichever it has first), else
        ``"failed"``.
    """
    error = item.get("error")
    if isinstance(error, Mapping):
        detail = error.get("message") or error.get("code") or error.get("type")
        error = None if detail is None else str(detail)
    return error if isinstance(error, str) and error else FAILED


def server_tool_calls(
    output: Sequence[Mapping[str, Any]],
) -> list[ServerToolCall]:
    """Collect the built-in tool calls of output items.

    Args:
        output: The output items of a response, as OpenAI sent them.

    Returns:
        A call per item that is one, in output order.
    """
    calls = (server_tool_call(item) for item in output)
    return [call for call in calls if call is not None]


def output_citations(output: Sequence[Mapping[str, Any]]) -> list[Citation]:
    """Collect the citations of the output text.

    OpenAI gives each annotation's position in the ``output_text`` part
    it sits in; here it is moved to the joined text of the response.
    A ``url_citation`` is a ``URL``, a ``file_citation`` a ``FILE`` (OpenAI
    gives no span for it, only the file's place in a list) and a
    ``container_file_citation`` a ``CONTAINER_FILE``; a ``file_path``
    stays in ``raw`` and ``native``.

    Args:
        output: The output items of a response, as OpenAI sent them.

    Returns:
        The citations, in order.
    """
    citations: list[Citation] = []
    offset = 0
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") != "output_text":
                continue
            for annotation in part.get("annotations") or []:
                citation = _citation(annotation, offset)
                if citation is not None:
                    citations.append(citation)
            offset += len(str(part.get("text", "")))
    return citations


def output_files(output: Sequence[Mapping[str, Any]]) -> list[ProducedFile]:
    """Collect the files of a container that the output text names.

    The code interpreter keeps the files its code writes in its
    container; OpenAI names one when the text refers to it, with a
    ``container_file_citation``. Each file comes once, however often the
    text cites it. OpenAI does not say which call wrote it.

    Args:
        output: The output items of a response, as OpenAI sent them.

    Returns:
        The files, in the order the text first cites them.
    """
    files: dict[tuple[str, str], ProducedFile] = {}
    for citation in output_citations(output):
        if (
            citation.kind is not CitationKind.CONTAINER_FILE
            or citation.container_id is None
            or citation.file_id is None
        ):
            continue
        key = (citation.container_id, citation.file_id)
        files.setdefault(
            key,
            ProducedFile(
                provider=PROVIDER,
                id=citation.file_id,
                filename=citation.filename,
                container_id=citation.container_id,
            ),
        )
    return list(files.values())


def response_files(response: Response) -> list[ProducedFile]:
    """Collect the files of a container that a response names.

    Args:
        response: A response.

    Returns:
        Its files, as ``output_files`` gives them; empty without.
    """
    return output_files(output_items(response))


def response_server_calls(response: Response) -> list[ServerToolCall]:
    """Collect the built-in tool calls of a response.

    Args:
        response: A response.

    Returns:
        Its calls, as ``server_tool_calls`` gives them; empty without.
    """
    return server_tool_calls(output_items(response))


def response_citations(response: Response) -> list[Citation]:
    """Collect the citations of a response.

    Args:
        response: A response.

    Returns:
        Its citations, as ``output_citations`` gives them; empty without.
    """
    return output_citations(output_items(response))


def _citation(annotation: Mapping[str, Any], offset: int) -> Citation | None:
    """Turn an annotation of an output text into a shared citation.

    Args:
        annotation: The annotation as OpenAI sent it.
        offset: Where its text part starts in the joined text.

    Returns:
        The citation, or ``None`` for an annotation that is no source.
    """
    kind = annotation.get("type")
    if kind == "url_citation":
        return Citation(
            kind=CitationKind.URL,
            url=annotation.get("url"),
            title=annotation.get("title"),
            start_index=annotation["start_index"] + offset,
            end_index=annotation["end_index"] + offset,
        )
    if kind == "file_citation":
        return Citation(
            kind=CitationKind.FILE,
            file_id=annotation.get("file_id"),
            filename=annotation.get("filename"),
        )
    if kind == "container_file_citation":
        return Citation(
            kind=CitationKind.CONTAINER_FILE,
            container_id=annotation.get("container_id"),
            file_id=annotation.get("file_id"),
            filename=annotation.get("filename"),
            start_index=annotation["start_index"] + offset,
            end_index=annotation["end_index"] + offset,
        )
    return None


def reasoning_summary(response: Response) -> str | None:
    """Join the summaries of all reasoning items.

    Args:
        response: A response.

    Returns:
        The joined summaries, or ``None`` if there are none (the
        request did not ask for them).
    """
    summary = "".join(
        part.text
        for item in response.output
        if isinstance(item, ReasoningItem)
        for part in item.summary
    )
    return summary or None


def check_status(response: Response) -> None:
    """Reject a response that did not finish.

    A response is created without ``background``, so OpenAI answers
    only when it has finished: ``completed`` or ``incomplete`` (stopped
    early, with the text generated so far). A ``failed`` one never gets
    here: ``CreateResponse`` raises it as an ``APIStatusError``. Any
    other status leaves no generated text to return.

    Args:
        response: The parsed response.

    Raises:
        APIResponseError: The status is neither ``completed`` nor
            ``incomplete``.
    """
    if response.status in {"completed", "incomplete"}:
        return
    message = (
        f"OpenAI response {response.id} has status {response.status!r}, "
        "expected 'completed' or 'incomplete'"
    )
    raise APIResponseError(message)


INPUT_CONVERSION: Final = InputConversion()
"""The default conversion of the shared ``input``."""

USAGE_CONVERSION: Final = UsageConversion()
"""The default conversion of token counts."""

STOP_CONVERSION: Final = StopConversion()
"""The default conversion of a response's stop."""
