"""Shared files behave for OpenAI as for every provider.

The scenarios of ``aiollm.testing.contract.files`` run against
``OpenAIClient.files`` on the local server, with the documented results
of OpenAI's Files API; only the wire format is the provider's own, the
shared results and errors must be the same.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aiollm.providers.openai import OpenAIClient
from aiollm.providers.openai.api import PROVIDER as OPENAI_PROVIDER
from aiollm.providers.openai.client import session_factory as openai_factory
from aiollm.providers.openai.types import OpenAIFile
from aiollm.testing.contract.files import FilesCase, FilesContract
from tests.providers.openai import test_responses as openai
from tests.providers.openai.test_files import documented

OPENAI_FILE = documented("createFile")

OPENAI = FilesCase(
    provider=OPENAI_PROVIDER,
    client=OpenAIClient,
    factory=openai_factory,
    raw_type=OpenAIFile,
    file=OPENAI_FILE,
    size=120000,
    created_at=datetime(2023, 2, 28, 18, 56, 42, tzinfo=UTC),
    expires_at=datetime(2023, 3, 30, 18, 56, 42, tzinfo=UTC),
    deleted=documented("deleteFile"),
    page={
        "object": "list",
        "data": [OPENAI_FILE],
        "first_id": OPENAI_FILE["id"],
        "last_id": OPENAI_FILE["id"],
        "has_more": True,
    },
    cursor=("after", OPENAI_FILE["id"]),
    last_page={
        "object": "list",
        "data": [OPENAI_FILE | {"id": "file-older"}],
        "first_id": "file-older",
        "last_id": "file-older",
        "has_more": False,
    },
    upload_parts={
        "purpose": b"user_data",
        "expires_after[anchor]": b"created_at",
        "expires_after[seconds]": b"3600",
    },
    bare_upload_parts=["file", "purpose"],
    options=(
        {"purpose": "vision"},
        {
            "purpose": b"vision",
            "expires_after[anchor]": b"created_at",
            "expires_after[seconds]": b"3600",
        },
    ),
    message_path="/v1/responses",
    message=openai.RESPONSE,
    input_field="input",
    stored_parts=[
        {"type": "input_file", "file_id": OPENAI_FILE["id"]},
        {
            "type": "input_image",
            "file_id": OPENAI_FILE["id"],
            "detail": "auto",
        },
    ],
)


class TestOpenAIFiles(FilesContract):
    case = OPENAI
