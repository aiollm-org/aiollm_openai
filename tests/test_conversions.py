"""Conversions of OpenAI's generation adapter can be swapped.

The adapter takes its input, usage and stop conversions as keyword
arguments; a swapped one is used for the request, the response and the
stream alike, and gets the provider's own types.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aiollm.adapters import BaseConversion, IConversion, Stop
from aiollm.enums import IncompleteReason
from aiollm.types.requests import InputItem
from aiollm.types.responses import CompletedEvent, TokenUsage

from aiollm_openai.adapters import OpenAIGenerationAdapter
from aiollm_openai.adapters.conversions import (
    INPUT_CONVERSION as INPUT,
)
from aiollm_openai.adapters.conversions import (
    STOP_CONVERSION,
    USAGE_CONVERSION,
)
from aiollm_openai.types import Response, ResponseUsage
from tests import test_generation as openai

USAGE = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)

STOP = Stop(IncompleteReason.OTHER, "swapped", "because")


class Recording:
    """Swapped conversions that record what they were given."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        self.seen: list[object] = []

    def usage(self, source: object, /) -> TokenUsage:
        """Record the source and give fixed counts."""
        self.seen.append(source)
        return USAGE

    def stop(self, source: object, /) -> Stop:
        """Record the source and give a fixed stop."""
        self.seen.append(source)
        return STOP


class OpenAIInput(
    BaseConversion[str | Sequence[InputItem], str | list[dict[str, Any]]],
):
    """Sends a fixed text instead of the request's input."""

    def __call__(
        self, source: str | Sequence[InputItem], /
    ) -> str | list[dict[str, Any]]:
        """Ignore the input."""
        return "swapped"


def test_default_conversions_are_conversions() -> None:
    defaults: list[IConversion[Any, Any]] = [
        INPUT,
        USAGE_CONVERSION,
        STOP_CONVERSION,
    ]
    adapter = OpenAIGenerationAdapter()

    assert all(isinstance(each, BaseConversion) for each in defaults)
    assert adapter.input_conversion is INPUT
    assert adapter.usage_conversion is USAGE_CONVERSION
    assert adapter.stop_conversion is STOP_CONVERSION


def check_completed(events: list[Any]) -> None:
    """Check that the completion went through the swapped conversions."""
    completed = events[-1]
    assert isinstance(completed, CompletedEvent)
    assert completed.usage == USAGE
    assert completed.incomplete_reason is STOP.incomplete_reason
    assert completed.raw_incomplete_reason == STOP.raw_incomplete_reason
    assert completed.refusal == STOP.refusal


async def test_swapped_conversions_are_used() -> None:
    recording = Recording()
    adapter = OpenAIGenerationAdapter(
        input_conversion=OpenAIInput(),
        usage_conversion=recording.usage,
        stop_conversion=recording.stop,
    )

    method = adapter.to_method(openai.REQUEST)
    response = adapter.to_response(openai.response(), openai.REQUEST)
    events = [
        event
        async for event in adapter.to_events(
            openai.events(openai.STREAM), openai.REQUEST
        )
    ]

    assert method.input == "swapped"
    assert response.usage == USAGE
    assert response.incomplete_reason is STOP.incomplete_reason
    assert response.raw_incomplete_reason == STOP.raw_incomplete_reason
    assert response.refusal == STOP.refusal
    check_completed(events)
    # Buffered, then streamed: the stop, then the usage of each.
    assert [type(source) for source in recording.seen] == [
        Response,
        ResponseUsage,
        Response,
        ResponseUsage,
    ]
