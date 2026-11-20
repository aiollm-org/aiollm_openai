"""The example of README.md runs as it is written."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from aiollm.testing import TEST_TOKEN, LocalAPI, Reply
from aiollm.testing.aiohttp import Recorder

from aiollm_openai.client import session_factory
from tests.test_responses import MESSAGE, RESPONSE

README = Path(__file__).resolve().parents[1] / "README.md"


def example(title: str) -> str:
    """The Python block of README.md with this title."""
    blocks = re.findall(
        r'^```python title="([^"]+)"\n(.*?)^```$',
        README.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    found = [code for name, code in blocks if name == title]
    assert len(found) == 1, title
    return found[0]


def test_every_example_is_tested() -> None:
    titles = re.findall(
        r'^```python title="([^"]+)"', README.read_text("utf-8"), re.M
    )

    assert titles == ["quick_start.py"]


async def test_quick_start(
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_factory, "api", api)
    monkeypatch.setenv("OPENAI_API_KEY", TEST_TOKEN)
    recorder.reply(Reply(json=RESPONSE))
    namespace: dict[str, Any] = {}
    exec(
        compile(example("quick_start.py"), "quick_start.py", "exec"), namespace
    )  # noqa: S102

    text = await namespace["ask"]()

    assert text == MESSAGE["content"][0]["text"]
    (request,) = recorder.requests
    assert request.raw_path == "/v1/responses"
    assert request.headers["Authorization"] == f"Bearer {TEST_TOKEN}"
