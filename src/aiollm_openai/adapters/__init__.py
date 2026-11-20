"""OpenAI's adapters between shared types and its methods."""

from __future__ import annotations

from aiollm_openai.adapters.conversions import (
    output_text,
    reasoning_summary,
    tool_calls,
)
from aiollm_openai.adapters.files import (
    OpenAIFilesAdapter,
)
from aiollm_openai.adapters.generation import (
    OpenAIGenerationAdapter,
)

__all__ = [
    "OpenAIFilesAdapter",
    "OpenAIGenerationAdapter",
    "output_text",
    "reasoning_summary",
    "tool_calls",
]
