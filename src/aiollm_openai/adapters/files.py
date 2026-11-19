"""Adapter of shared files to OpenAI's Files API."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final, Self

from pydantic import MISSING

from aiollm.adapters.base import BaseFilesAdapter
from aiollm.adapters.validation import (
    checked_options,
    merge_options,
    request_errors,
)
from aiollm.providers.openai.api import PROVIDER
from aiollm.providers.openai.methods.files import (
    CreateFile,
    DeleteFile,
    DownloadFile,
    ListFiles,
    RetrieveFile,
)
from aiollm.providers.openai.types.files import ListFilesResponse, OpenAIFile
from aiollm.providers.openai.types.options import OpenAIFileOptions
from aiollm.types.requests.content import ProviderFile
from aiollm.types.requests.upload import FileUpload
from aiollm.types.responses.files import StoredFile

DEFAULT_PURPOSE: Final = "user_data"
"""``purpose`` of an upload that does not set it: a model's input."""

DEFAULT_PAGE_SIZE: Final = 100
"""Files asked for in one page of ``client.files.list()``."""

SHARED_FIELDS: Final = {"expires_after": "expires_in_seconds"}
"""Shared upload field of each method field with another name."""


def _utc(timestamp: int | None) -> datetime | None:
    """Turn a Unix timestamp into a UTC datetime.

    Args:
        timestamp: Seconds since the epoch, if OpenAI sent one.

    Returns:
        The time in UTC, or ``None`` without one.
    """
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC)


class OpenAIFilesAdapter(BaseFilesAdapter[OpenAIFile, ListFilesResponse]):
    """Stores files with OpenAI's ``/files``.

    An upload is a ``user_data`` file unless ``provider_options`` sets
    another ``purpose``; ``expires_in_seconds`` becomes
    ``expires_after`` from the file's creation. The list goes newest
    first, ``page_size`` files a request, by ``after`` the last id.

    Attributes:
        page_size: Files asked for in one page of the list.

    Example:
        >>> files = FilesResource(session, OpenAIFilesAdapter())
        >>> stored = await files.upload(Path("report.pdf"))
    """

    provider = PROVIDER

    def __init__(self, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        """Store the page size.

        Args:
            page_size: Files asked for in one page of the list.
        """
        self.page_size = page_size
        self.provider_options: dict[str, Any] = {}

    def with_options(self, options: OpenAIFileOptions, /) -> Self:
        """Return a copy that sends the options with every upload.

        Args:
            options: Fields of ``CreateFile`` that the shared upload does
                not hold, such as ``{"purpose": "vision"}``.

        Returns:
            A copy of the adapter; this one is left as it is.

        Raises:
            RequestValidationError: A key is not one of the options.
        """
        adapter = copy.copy(self)
        adapter.provider_options = checked_options(
            options, OpenAIFileOptions, self.provider
        )
        return adapter

    def to_upload_method(self, request: FileUpload) -> CreateFile:
        """Build the method that stores a file.

        Args:
            request: The shared upload.

        Returns:
            ``CreateFile`` with the file, its purpose and its expiry.

        Raises:
            RequestValidationError: OpenAI cannot take the upload, such
                as an expiry outside 3600 to 2 592 000 seconds or an
                unknown purpose.
        """
        seconds = request.expires_in_seconds
        fields = {
            "file": request.file,
            "purpose": DEFAULT_PURPOSE,
            "expires_after": (
                MISSING
                if seconds is MISSING
                else {"anchor": "created_at", "seconds": seconds}
            ),
        }
        fields = merge_options(fields, self.provider_options, self.provider)
        with request_errors(self.provider, SHARED_FIELDS):
            return CreateFile(**fields)

    def to_retrieve_method(self, file_id: str) -> RetrieveFile:
        """Build the method that reads a file's metadata.

        Args:
            file_id: OpenAI's id of the file.

        Returns:
            ``RetrieveFile``.
        """
        return RetrieveFile(file_id=file_id)

    def to_list_method(self, cursor: str | None) -> ListFiles:
        """Build the method that reads one page of the stored files.

        Args:
            cursor: The id of the last file of the page before, or
                ``None`` for the first page.

        Returns:
            ``ListFiles`` of ``page_size`` files, newest first.
        """
        return ListFiles(
            limit=self.page_size,
            order="desc",
            after=MISSING if cursor is None else cursor,
        )

    def to_page(
        self, raw: ListFilesResponse
    ) -> tuple[Sequence[OpenAIFile], str | None]:
        """Take the files of a page and where the next one starts.

        Args:
            raw: The page.

        Returns:
            Its files, and ``last_id`` while ``has_more`` is true.
        """
        more = raw.has_more and bool(raw.data)
        return raw.data, raw.last_id if more else None

    def to_delete_method(self, file_id: str) -> DeleteFile:
        """Build the method that deletes a file.

        Args:
            file_id: OpenAI's id of the file.

        Returns:
            ``DeleteFile``.
        """
        return DeleteFile(file_id=file_id)

    def to_download_method(self, file_id: str) -> DownloadFile:
        """Build the method that streams a file's content.

        Args:
            file_id: OpenAI's id of the file.

        Returns:
            ``DownloadFile``; OpenAI refuses a ``user_data`` file.
        """
        return DownloadFile(file_id=file_id)

    def to_stored_file(self, raw: OpenAIFile) -> StoredFile[OpenAIFile]:
        """Convert OpenAI's file into the shared metadata.

        Args:
            raw: OpenAI's file.

        Returns:
            Its id, name, ``bytes`` as ``size`` and its Unix times in
            UTC.
        """
        created_at = datetime.fromtimestamp(raw.created_at, UTC)
        return StoredFile[OpenAIFile](
            ref=ProviderFile(provider=self.provider, id=raw.id),
            filename=raw.filename,
            size=raw.bytes,
            created_at=created_at,
            expires_at=_utc(raw.expires_at),
            raw=raw,
        )
