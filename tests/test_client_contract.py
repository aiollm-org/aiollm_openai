"""What every provider client does the same way, run for OpenAI.

The scenarios of ``aiollm.testing.contract.clients`` run against
``OpenAIClient``; what is OpenAI's own (its error resolver) stays in
``test_client.py``.
"""

from __future__ import annotations

from aiollm.providers.openai.api import OPENAI_API, PROVIDER
from aiollm.providers.openai.client import OpenAIClient
from aiollm.providers.openai.resolver import OpenAIErrorResolver
from aiollm.testing.contract.clients import ClientCase, ClientContract

OPENAI = ClientCase(
    client=OpenAIClient,
    provider=PROVIDER,
    api=OPENAI_API,
    url=("/models", "https://api.openai.com/v1/models"),
    resolver=OpenAIErrorResolver,
    env="OPENAI_API_KEY",
)


class TestOpenAIClient(ClientContract):
    case = OPENAI
