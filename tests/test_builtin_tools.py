"""Built-in tools at OpenAI: the tools, their calls and their sources.

The shared scenarios of the test kit cover a web search end to end; these
cover what is OpenAI's own: the other call items, the annotations, the
tools of ``aiollm_openai.tools`` and the replay of a response's output.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import MISSING
from pydantic_core import to_jsonable_python

from aiollm.enums import (
    CitationKind,
    Provider,
    Role,
    ServerToolKind,
    ServerToolPhase,
    ServerToolStatus,
)
from aiollm.exceptions import RequestValidationError
from aiollm.types.requests import (
    CodeExecution,
    GenerationRequest,
    ProviderTool,
    Turn,
    UserLocation,
    WebSearch,
)
from aiollm.types.requests.generation import NativeOutput
from aiollm.types.responses import (
    CompletedEvent,
    OtherEvent,
    ServerToolCallEvent,
    ToolCallStartEvent,
)

from aiollm_openai import tools
from aiollm_openai.api import PROVIDER
from aiollm_openai.methods import CreateResponse
from aiollm_openai.methods.params.tools import (
    WebSearchApproximateLocationWebSearchApproximateLocation as Location,
)
from tests.test_generation import REQUEST, adapter, convert, response
from tests.test_generation_contract import (
    CITED_MESSAGE,
    SEARCH_ITEM,
    openai_searched_stream,
)
from tests.test_responses import MESSAGE, state


def request(*tool_list: object, **fields: Any) -> GenerationRequest:
    """The reference request with tools."""
    data = REQUEST.model_dump(by_alias=True) | {"tools": list(tool_list)}
    return GenerationRequest.model_validate(data | fields)


def body_tools(*tool_list: object) -> list[dict[str, Any]]:
    """The ``tools`` of the method the adapter builds."""
    method = adapter.to_method(request(*tool_list))
    assert isinstance(method, CreateResponse)
    return method.get_body_map()["tools"]


def convert_output(*items: dict[str, Any]) -> Any:
    """The shared response of the given output items."""
    return adapter.to_response(response(output=list(items)), REQUEST)


def text_message(text: str, *annotations: dict[str, Any]) -> dict[str, Any]:
    """An output message of one text part with annotations."""
    part = MESSAGE["content"][0] | {"text": text, "annotations": annotations}
    return MESSAGE | {"content": [part]}


class TestRequest:
    def test_a_web_search_sends_what_it_has(self) -> None:
        search = WebSearch(
            allowed_domains=["example.com"],
            user_location=UserLocation(
                country="FR", region="IDF", city="Paris", timezone="CET"
            ),
        )

        assert body_tools(search) == [
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["example.com"]},
                "user_location": {
                    "type": "approximate",
                    "country": "FR",
                    "region": "IDF",
                    "city": "Paris",
                    "timezone": "CET",
                },
            }
        ]

    def test_a_code_execution_runs_in_an_auto_container(self) -> None:
        assert body_tools(CodeExecution()) == [
            {"type": "code_interpreter", "container": {"type": "auto"}}
        ]

    def test_a_provider_tool_goes_as_its_definition(self) -> None:
        search = tools.file_search(["vs_1"], max_num_results=4)

        assert body_tools(search) == [
            {
                "type": "file_search",
                "vector_store_ids": ["vs_1"],
                "max_num_results": 4,
            }
        ]

    def test_a_request_of_built_in_tools_only_has_no_tool_choice(
        self,
    ) -> None:
        method = adapter.to_method(request(WebSearch()))

        body = method.get_body_map()
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body

    def test_a_tool_of_another_provider_is_refused(self) -> None:
        other = ProviderTool(
            provider=Provider("someone-else"), definition={"type": "x"}
        )

        with pytest.raises(RequestValidationError) as caught:
            adapter.to_method(request(other))

        assert caught.value.field == "tools"

    def test_the_definition_is_copied(self) -> None:
        tool = tools.web_search(allowed_domains=["a.com"])

        [sent] = body_tools(tool)
        sent["filters"]["allowed_domains"].append("b.com")

        assert tool.definition["filters"] == {"allowed_domains": ["a.com"]}


class TestTools:
    def test_web_search_takes_openais_options(self) -> None:
        tool = tools.web_search(
            version="web_search_2025_08_26",
            allowed_domains=["example.com"],
            search_context_size="high",
            external_web_access=False,
            user_location=Location(type="approximate", city="Paris"),
        )

        assert tool.provider == PROVIDER
        assert tool.definition == {
            "type": "web_search_2025_08_26",
            "external_web_access": False,
            "filters": {"allowed_domains": ["example.com"]},
            "search_context_size": "high",
            "user_location": {"type": "approximate", "city": "Paris"},
        }

    def test_a_bare_web_search_is_just_its_type(self) -> None:
        assert tools.web_search().definition == {"type": "web_search"}

    def test_a_code_interpreter_may_reuse_a_container(self) -> None:
        assert tools.code_interpreter(container="cntr_1").definition == {
            "type": "code_interpreter",
            "container": "cntr_1",
        }

    def test_a_new_container_has_its_files_and_memory(self) -> None:
        assert tools.code_interpreter(
            file_ids=["file_1"], memory_limit="4g"
        ).definition == {
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": ["file_1"],
                "memory_limit": "4g",
            },
        }

    def test_a_bare_code_interpreter_is_an_auto_container(self) -> None:
        assert tools.code_interpreter().definition == {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        }

    @pytest.mark.parametrize(
        "options", [{"file_ids": ["f"]}, {"memory_limit": "1g"}]
    )
    def test_a_reused_container_takes_no_new_container_options(
        self, options: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="new container"):
            tools.code_interpreter(container="cntr_1", **options)

    def test_a_value_the_tool_refuses_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_num_results"):
            tools.file_search(["vs_1"], max_num_results="many")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="search_context_size"):
            tools.web_search(search_context_size="huge")  # type: ignore[arg-type]


class TestCalls:
    def test_a_search_has_its_queries_without_repeats(self) -> None:
        item = SEARCH_ITEM | {
            "action": {
                "type": "search",
                "query": "a",
                "queries": ["a", "b"],
                "sources": [],
            }
        }

        [call] = convert_output(item).server_tool_calls

        assert call.queries == ["a", "b"]
        assert call.kind is ServerToolKind.WEB_SEARCH
        assert call.raw == [item]

    def test_opening_a_page_is_a_search_call_without_queries(self) -> None:
        item = SEARCH_ITEM | {
            "action": {"type": "open_page", "url": "https://example.com/"}
        }

        [call] = convert_output(item).server_tool_calls

        assert call.queries == []
        assert call.status is ServerToolStatus.COMPLETED

    def test_a_search_without_an_action_has_no_queries(self) -> None:
        item = {
            key: value for key, value in SEARCH_ITEM.items() if key != "action"
        }

        [call] = convert_output(item).server_tool_calls

        assert call.queries == []

    def test_a_file_search_has_its_queries(self) -> None:
        item = {
            "type": "file_search_call",
            "id": "fs_1",
            "status": "completed",
            "queries": ["policy", "refunds"],
            "results": None,
        }

        [call] = convert_output(item).server_tool_calls

        assert call.kind is ServerToolKind.FILE_SEARCH
        assert call.queries == ["policy", "refunds"]

    def test_a_code_interpreter_call_has_its_code(self) -> None:
        item = {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "status": "completed",
            "container_id": "cntr_1",
            "code": "print(1)",
            "outputs": [{"type": "logs", "logs": "1\n"}],
        }

        [call] = convert_output(item).server_tool_calls

        assert call.kind is ServerToolKind.CODE_EXECUTION
        assert call.code == "print(1)"
        assert call.status is ServerToolStatus.COMPLETED
        assert call.raw[0]["container_id"] == "cntr_1"

    def test_a_code_interpreter_call_without_code_has_none(self) -> None:
        item = {
            "type": "code_interpreter_call",
            "id": "ci_1",
            "status": "completed",
            "container_id": "cntr_1",
            "code": None,
            "outputs": None,
        }

        [call] = convert_output(item).server_tool_calls

        assert call.code is None

    def test_an_image_generation_failure_gives_its_message(self) -> None:
        item = {
            "type": "image_generation_call",
            "id": "ig_1",
            "status": "failed",
            "result": None,
            "error": {"type": "moderation", "message": "Blocked"},
        }

        [call] = convert_output(item).server_tool_calls

        assert call.kind is ServerToolKind.IMAGE_GENERATION
        assert call.status is ServerToolStatus.FAILED
        assert call.error == "Blocked"

    def test_an_error_object_without_a_message_gives_its_type(self) -> None:
        item = {
            "type": "mcp_call",
            "id": "mcp_1",
            "status": "failed",
            "name": "lookup",
            "arguments": "{}",
            "server_label": "docs",
            "error": {"type": "mcp_tool_execution_error", "content": []},
        }

        [call] = convert_output(item).server_tool_calls

        assert call.error == "mcp_tool_execution_error"

    def test_an_error_code_without_a_message_is_text(self) -> None:
        item = {
            "type": "mcp_call",
            "id": "mcp_1",
            "status": "failed",
            "name": "lookup",
            "arguments": "{}",
            "server_label": "docs",
            "error": {
                "type": "mcp_protocol_error",
                "code": -32000,
                "message": "",
            },
        }

        [call] = convert_output(item).server_tool_calls

        assert call.error == "-32000"

    def test_an_mcp_failure_gives_its_error(self) -> None:
        item = {
            "type": "mcp_call",
            "id": "mcp_1",
            "status": "failed",
            "name": "lookup",
            "arguments": "{}",
            "server_label": "docs",
            "error": {
                "type": "mcp_protocol_error",
                "code": -32000,
                "message": "timed out",
            },
        }

        [call] = convert_output(item).server_tool_calls

        assert call.kind is ServerToolKind.MCP
        assert call.error == "timed out"

    def test_a_failure_without_a_reason_says_failed(self) -> None:
        [call] = convert_output(
            SEARCH_ITEM | {"status": "failed"}
        ).server_tool_calls

        assert call.status is ServerToolStatus.FAILED
        assert call.error == "failed"

    def test_a_call_that_never_finished_is_incomplete(self) -> None:
        [call] = convert_output(
            SEARCH_ITEM | {"status": "searching"}
        ).server_tool_calls

        assert call.status is ServerToolStatus.INCOMPLETE
        assert call.error is None

    def test_functions_and_messages_are_no_server_calls(self) -> None:
        call = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_time",
            "arguments": "{}",
            "status": "completed",
        }

        result = convert_output(call, SEARCH_ITEM, MESSAGE)

        assert [c.id for c in result.server_tool_calls] == ["search_1"]
        assert [c.id for c in result.tool_calls] == ["call_1"]


class TestCitations:
    def test_a_url_citation_is_a_url(self) -> None:
        [citation] = convert_output(CITED_MESSAGE).citations

        assert citation.kind is CitationKind.URL
        assert citation.url == "https://example.com/paris"
        assert citation.title == "Paris weather"
        assert (citation.start_index, citation.end_index) == (0, 9)

    def test_the_span_moves_to_the_joined_text(self) -> None:
        first = text_message("One. ")
        second = text_message(
            "Two.",
            {
                "type": "url_citation",
                "url": "https://example.com/two",
                "title": "Two",
                "start_index": 0,
                "end_index": 3,
            },
        )

        result = convert_output(first, SEARCH_ITEM, second)

        [citation] = result.citations
        assert result.text == "One. Two."
        assert (citation.start_index, citation.end_index) == (5, 8)

    def test_a_file_citation_is_a_file_without_a_span(self) -> None:
        message = text_message(
            "From the file.",
            {
                "type": "file_citation",
                "file_id": "file_1",
                "filename": "report.pdf",
                "index": 3,
            },
        )

        [citation] = convert_output(message).citations

        assert citation.kind is CitationKind.FILE
        assert citation.file_id == "file_1"
        assert citation.filename == "report.pdf"
        assert citation.start_index is None
        assert citation.end_index is None

    def test_a_container_file_citation_is_a_container_file(self) -> None:
        message = text_message(
            "See the chart.",
            {
                "type": "container_file_citation",
                "container_id": "cntr_1",
                "file_id": "cfile_1",
                "filename": "chart.png",
                "start_index": 4,
                "end_index": 13,
            },
        )

        [citation] = convert_output(message).citations

        assert citation.kind is CitationKind.CONTAINER_FILE
        assert citation.container_id == "cntr_1"
        assert citation.file_id == "cfile_1"
        assert citation.filename == "chart.png"
        assert (citation.start_index, citation.end_index) == (4, 13)

    def test_a_file_path_is_no_citation(self) -> None:
        message = text_message(
            "Saved.", {"type": "file_path", "file_id": "f", "index": 0}
        )

        result = convert_output(message)

        assert result.citations == []
        assert result.native is not None
        assert result.native.items[0]["content"][0]["annotations"] == [
            {"type": "file_path", "file_id": "f", "index": 0}
        ]

    def test_a_refusal_part_has_no_citations_and_no_span(self) -> None:
        refusal = MESSAGE | {
            "content": [{"type": "refusal", "refusal": "No."}]
        }

        result = convert_output(refusal, CITED_MESSAGE)

        [citation] = result.citations
        assert (citation.start_index, citation.end_index) == (0, 9)


class TestStream:
    async def test_a_search_streams_as_started_then_done(self) -> None:
        received = await convert(openai_searched_stream())

        tool_events = [
            e for e in received if isinstance(e, ServerToolCallEvent)
        ]
        assert [(e.phase, e.call.status) for e in tool_events] == [
            (ServerToolPhase.STARTED, ServerToolStatus.IN_PROGRESS),
            (ServerToolPhase.DONE, ServerToolStatus.COMPLETED),
        ]
        started, done = tool_events
        assert started.call.queries == []
        assert done.call.queries == ["weather in Paris"]
        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert [c.id for c in completed.server_tool_calls] == ["search_1"]
        assert [c.url for c in completed.citations] == [
            "https://example.com/paris"
        ]

    async def test_a_failed_call_starts_without_its_error(self) -> None:
        item = {
            "type": "mcp_call",
            "id": "mcp_1",
            "status": "failed",
            "name": "lookup",
            "arguments": "{}",
            "server_label": "docs",
            "error": {
                "type": "mcp_protocol_error",
                "code": -32000,
                "message": "timed out",
            },
        }
        added = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
            "sequence_number": 1,
        }

        [event] = await convert([added])

        assert isinstance(event, ServerToolCallEvent)
        assert event.call.status is ServerToolStatus.IN_PROGRESS
        assert event.call.error is None

    async def test_a_message_and_reasoning_are_other_events(self) -> None:
        reasoning = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
        }
        payloads = [
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": item,
                "sequence_number": number,
            }
            for number, item in enumerate(
                [reasoning, MESSAGE | {"content": []}]
            )
        ] + [
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": reasoning,
                "sequence_number": 2,
            }
        ]

        received = await convert(payloads)

        assert all(isinstance(e, OtherEvent) for e in received)

    async def test_a_function_call_is_still_a_tool_call(self) -> None:
        item = {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_time",
            "arguments": "",
            "status": "in_progress",
        }

        [event] = await convert(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": item,
                    "sequence_number": 1,
                }
            ]
        )

        assert isinstance(event, ToolCallStartEvent)

    async def test_the_end_of_a_stream_has_the_calls_and_citations(
        self,
    ) -> None:
        completed = state(
            "response.completed", 1, output=[SEARCH_ITEM, CITED_MESSAGE]
        )

        [event] = await convert([completed])

        assert isinstance(event, CompletedEvent)
        assert [c.kind for c in event.server_tool_calls] == [
            ServerToolKind.WEB_SEARCH
        ]
        assert len(event.citations) == 1


class TestInput:
    def test_a_turn_with_openais_output_goes_back_as_it_is(self) -> None:
        searched = adapter.to_response(
            response(output=[SEARCH_ITEM, CITED_MESSAGE]), REQUEST
        )
        turn = searched.to_turn()

        method = adapter.to_method(
            GenerationRequest(
                model="m", input=[Turn.user("Hi"), turn, Turn.user("More")]
            )
        )

        assert isinstance(turn, Turn)
        assert searched.native is not None
        assert to_jsonable_python(method.get_body_map()["input"]) == [
            {"role": "user", "content": "Hi"},
            *searched.native.items,
            {"role": "user", "content": "More"},
        ]

    def test_output_of_another_provider_is_not_read(self) -> None:
        turn = Turn(
            role=Role.ASSISTANT,
            content="Hi there.",
            native=NativeOutput(
                provider=Provider("someone-else"), items=[{"x": 1}]
            ),
        )

        method = adapter.to_method(
            GenerationRequest(
                model="m", input=[Turn.user("Hi"), turn, Turn.user("More")]
            )
        )

        assert method.get_body_map()["input"][1] == {
            "role": "assistant",
            "content": "Hi there.",
        }

    def test_a_plain_answer_goes_back_as_text(self) -> None:
        plain = adapter.to_response(response(), REQUEST).to_turn()

        assert isinstance(plain, Turn)
        assert plain.native is MISSING
        method = adapter.to_method(
            GenerationRequest(
                model="m", input=[Turn.user("Hi"), plain, Turn.user("More")]
            )
        )
        assert method.get_body_map()["input"][1]["role"] == "assistant"
