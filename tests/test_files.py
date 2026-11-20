"""OpenAI's Files methods on the wire.

The client runs against the local recorder: what goes out (path,
query, form parts, headers) and how results and errors come back.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from aiollm.http.exceptions.status import ServiceUnavailableError
from aiollm.testing import TEST_TOKEN, LocalAPI, Reply
from aiollm.testing.aiohttp import Recorder
from aiollm.types.files import BytesFile

from aiollm_openai import OpenAIClient
from aiollm_openai import methods as openai
from aiollm_openai.client import session_factory
from aiollm_openai.methods.params import FileExpirationAfter

CONTRACTS = Path(__file__).parent / "contracts"

PDF = b"%PDF-1.4\n%%EOF\n"
"""Bytes the upload sends; the server never looks into them."""


def documented(operation: str) -> Any:
    """Read a documented result of the Files snapshot.

    Args:
        operation: Such as ``createFile``.

    Returns:
        Its body.
    """
    path = CONTRACTS / "files" / "responses.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    (body,) = [item["body"] for item in items if item["id"] == operation]
    return body


OPENAI_FILE = documented("createFile")


@pytest.fixture
def local(api: LocalAPI, monkeypatch: pytest.MonkeyPatch) -> LocalAPI:
    """Point the client at the recorder."""
    monkeypatch.setattr(session_factory, "api", api)
    return api


@pytest.mark.usefixtures("local")
async def test_upload_sends_expires_after_as_bracketed_fields(
    recorder: Recorder,
) -> None:
    # OpenAI refuses expires_after as one JSON-encoded field (400, live
    # 2026-11-19); the form spells the object out key by key.
    recorder.reply(Reply(json=OPENAI_FILE))

    async with OpenAIClient(TEST_TOKEN) as client:
        file = await client.send_request(
            openai.CreateFile(
                file=BytesFile(PDF, "report.pdf"),
                purpose="user_data",
                expires_after=FileExpirationAfter(
                    anchor="created_at", seconds=3600
                ),
            )
        )

    assert file.id == OPENAI_FILE["id"]
    assert recorder.last.method == "POST"
    assert recorder.last.raw_path == "/v1/files"
    parts = {part.name: part for part in recorder.last.parts}
    assert list(parts) == [
        "file",
        "purpose",
        "expires_after[anchor]",
        "expires_after[seconds]",
    ]
    assert parts["file"].filename == "report.pdf"
    assert parts["file"].data == PDF
    assert parts["purpose"].data == b"user_data"
    assert parts["expires_after[anchor]"].data == b"created_at"
    assert parts["expires_after[seconds]"].data == b"3600"


@pytest.mark.usefixtures("local")
async def test_an_upload_is_not_retried(recorder: Recorder) -> None:
    # A 503 is transient, yet a second POST could store the file twice;
    # the recorder fails the test if a retry comes.
    recorder.reply(Reply(status=503, json={"error": {"message": "Try again"}}))

    async with OpenAIClient(TEST_TOKEN) as client:
        with pytest.raises(ServiceUnavailableError) as caught:
            await client.send_request(
                openai.CreateFile(
                    file=BytesFile(PDF, "a.pdf"), purpose="user_data"
                )
            )

    assert caught.value.transient is True
    assert len(recorder.requests) == 1


@pytest.mark.usefixtures("local")
async def test_reading_is_retried(recorder: Recorder) -> None:
    recorder.reply(
        Reply(status=503, json={"error": {"message": "Try again"}}),
        Reply(json=OPENAI_FILE),
    )

    async with OpenAIClient(TEST_TOKEN) as client:
        file = await client.send_request(
            openai.RetrieveFile(file_id="file-abc123")
        )

    assert file.id == "file-abc123"
    assert [r.raw_path for r in recorder.requests] == [
        "/v1/files/file-abc123"
    ] * 2


@pytest.mark.usefixtures("local")
async def test_list_sends_its_query(recorder: Recorder) -> None:
    recorder.reply(Reply(json=documented("listFiles")))

    async with OpenAIClient(TEST_TOKEN) as client:
        page = await client.send_request(
            openai.ListFiles(limit=2, order="asc", after="file-1")
        )

    assert recorder.last.method == "GET"
    assert recorder.last.raw_path == "/v1/files"
    assert dict(recorder.last.query) == {
        "limit": "2",
        "order": "asc",
        "after": "file-1",
    }
    assert recorder.last.body == b""
    assert [file.id for file in page.data] == ["file-abc123", "file-abc456"]


@pytest.mark.usefixtures("local")
async def test_a_file_id_is_escaped_in_the_path(recorder: Recorder) -> None:
    # An id is a path segment, never a path: a slash in it cannot reach
    # another endpoint.
    recorder.reply(Reply(json={"id": "x", "object": "file", "deleted": True}))

    async with OpenAIClient(TEST_TOKEN) as client:
        await client.send_request(openai.DeleteFile(file_id="../batches/x"))

    assert recorder.last.method == "DELETE"
    assert recorder.last.raw_path == "/v1/files/..%2Fbatches%2Fx"


@pytest.mark.usefixtures("local")
async def test_download_streams_the_content_to_a_file(
    recorder: Recorder,
    tmp_path: Path,
) -> None:
    content = bytes(range(256)) * 64
    recorder.reply(
        Reply(
            chunks=[content[:1000], content[1000:]],
            content_type="application/octet-stream",
        )
    )
    target = tmp_path / "out.bin"

    async with (
        OpenAIClient(TEST_TOKEN) as client,
        client.stream(openai.DownloadFile(file_id="file-abc123")) as stream,
    ):
        written = await stream.save(target)

    assert written == len(content)
    assert target.read_bytes() == content
    assert recorder.last.raw_path == "/v1/files/file-abc123/content"


@pytest.mark.usefixtures("local")
async def test_file_names_and_ids_never_reach_a_log(
    recorder: Recorder,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A file name or id may say whose data the file holds.
    caplog.set_level(logging.DEBUG, logger="aiollm")
    recorder.reply(
        Reply(json=OPENAI_FILE | {"id": "file-of-a-user"}),
        Reply(chunks=[b"x"], content_type="application/octet-stream"),
    )

    async with OpenAIClient(TEST_TOKEN) as client:
        await client.send_request(
            openai.CreateFile(
                file=BytesFile(PDF, "diagnosis-of-a-user.pdf"),
                purpose="user_data",
            )
        )
        method = openai.DownloadFile(file_id="file-of-a-user")
        async with client.stream(method) as stream:
            assert [chunk async for chunk in stream] == [b"x"]

    assert "Sending GET /files/{file_id}/content" in caplog.text
    assert "file-of-a-user" not in caplog.text
    assert "diagnosis-of-a-user" not in caplog.text
