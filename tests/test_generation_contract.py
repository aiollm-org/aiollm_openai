"""Shared generation behaves for OpenAI as for every provider.

The scenarios of ``aiollm.testing.contract.generation`` run against
``OpenAIClient.generation`` on the local server, with the payloads of
OpenAI's reference examples. Only the wire format is the provider's
own; the shared results and errors must be the same.
"""

from __future__ import annotations

import json
from typing import Any

from aiollm.testing.contract.generation import (
    CALLS,
    SCHEMA,
    GenerationCase,
    GenerationContract,
)
from aiollm.types.responses import TokenUsage

from aiollm_openai import OpenAIClient
from aiollm_openai.api import PROVIDER as OPENAI_PROVIDER
from aiollm_openai.client import session_factory as openai_factory
from aiollm_openai.methods import StreamResponse
from tests import test_responses as openai


def openai_called() -> dict[str, Any]:
    """An OpenAI response with text and ``CALLS``."""
    calls = [
        {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(arguments),
            "status": "completed",
        }
        for n, (call_id, name, arguments) in enumerate(CALLS)
    ]
    return openai.RESPONSE | {"output": [openai.MESSAGE, *calls]}


def openai_called_stream() -> list[dict[str, Any]]:
    """An OpenAI stream of ``CALLS``, the arguments in two pieces."""
    events: list[dict[str, Any]] = [openai.state("response.created", 0)]
    for n, (call_id, name, arguments) in enumerate(CALLS):
        item = {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": call_id,
            "name": name,
            "arguments": "",
            "status": "in_progress",
        }
        text = json.dumps(arguments)
        events += [
            {
                "type": "response.output_item.added",
                "output_index": n,
                "item": item,
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{n}",
                "output_index": n,
                "delta": text[:3],
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{n}",
                "output_index": n,
                "delta": text[3:],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": f"fc_{n}",
                "output_index": n,
                "arguments": text,
            },
            {
                "type": "response.output_item.done",
                "output_index": n,
                "item": item | {"arguments": text, "status": "completed"},
            },
        ]
    events.append(
        openai.state("response.completed", 0, output=openai_called()["output"])
    )
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return events


OPENAI = GenerationCase(
    provider=OPENAI_PROVIDER,
    client=OpenAIClient,
    factory=openai_factory,
    path="/v1/responses",
    response=openai.RESPONSE,
    text=openai.MESSAGE["content"][0]["text"],
    usage=TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    ),
    truncated=openai.RESPONSE
    | {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    refused=openai.RESPONSE
    | {
        "output": [
            openai.MESSAGE
            | {"content": [{"type": "refusal", "refusal": "I can't."}]}
        ]
    },
    refusal="I can't.",
    stream=openai.STREAM,
    deltas=["Hi"],
    stream_usage=TokenUsage(
        input_tokens=328,
        output_tokens=52,
        total_tokens=380,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
    ),
    overloaded={
        "type": "error",
        "code": "server_is_overloaded",
        "message": "Overloaded",
        "sequence_number": 1,
    },
    quota={
        "type": "error",
        "code": "insufficient_quota",
        "message": "You exceeded your current quota",
        "sequence_number": 1,
    },
    bad_request={
        "error": {
            "code": "invalid_value",
            "message": "Bad",
            "type": "invalid_request_error",
        }
    },
    raw_stream=lambda: StreamResponse(model="model-1", input="Hi"),
    options={"store": False, "temperature": 0.2},
    other_options={"store": True, "top_p": 0.5},
    bad_option={"temperature": 5},
    effort={"reasoning": {"effort": "high"}},
    output_schema={
        "reasoning": {"effort": "high"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "output",
                "schema": {"type": "object", "required": ["n"]},
                "strict": True,
            }
        },
    },
    tools={
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather",
                "parameters": SCHEMA,
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_time",
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
            },
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    },
    forced={"tool_choice": {"type": "function", "name": "get_time"}},
    only_parallel={"parallel_tool_calls": True},
    called=openai_called(),
    called_stream=openai_called_stream(),
    cut_off={
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    },
    input_field="input",
    continued_streamed=[
        *openai_called()["output"],
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    continued=[
        *openai_called()["output"],
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    continued_shared=[
        {"role": "assistant", "content": "Checking."},
        *(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": json.dumps(arguments),
            }
            for call_id, name, arguments in CALLS
        ),
        *(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": f"result of {name}",
            }
            for call_id, name, _ in CALLS
        ),
    ],
    parts={
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What is in these?"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBORw0K",
                "detail": "auto",
            },
            {
                "type": "input_image",
                "image_url": "https://example.com/cat.jpg",
                "detail": "auto",
            },
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                "filename": "report.pdf",
            },
            {
                "type": "input_file",
                "file_data": "data:application/pdf;base64,JVBERi0xLjc=",
                "filename": "document.pdf",
            },
            {
                "type": "input_file",
                "file_url": "https://example.com/report.pdf",
            },
        ],
    },
    after_results=[
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "noon",
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "What is in these?"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,iVBORw0K",
                    "detail": "auto",
                },
            ],
        },
    ],
)


class TestOpenAIGeneration(GenerationContract):
    case = OPENAI
