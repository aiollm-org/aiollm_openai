"""What every provider client does the same way, run for OpenAI.

The scenarios of ``aiollm.testing.contract.clients`` run against
``OpenAIClient``; what is OpenAI's own (its error resolver) stays in
``test_client.py``.
"""

from __future__ import annotations

from aiollm.testing.contract.clients import ClientCase, ClientContract

from aiollm_openai.api import OPENAI_API, PROVIDER
from aiollm_openai.client import OpenAIClient
from aiollm_openai.resolver import OpenAIErrorResolver

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
