"""Shared fixtures: the ``aiollm.testing`` plugin, set up for this suite."""

from __future__ import annotations

import pytest

pytest_plugins = ["aiollm.testing.plugin"]


@pytest.fixture
def provider_key_envs() -> tuple[str, ...]:
    """The key the OpenAI client reads, cleared in every test."""
    return ("OPENAI_API_KEY",)
