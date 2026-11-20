"""Methods of OpenAI's Files API."""

from __future__ import annotations

from aiollm.enums import BodyFormat, ResponseFormat
from aiollm.methods.base import BaseAPIMethod, BaseStreamMethod

from aiollm_openai.methods.params.files import (
    CreateFileParams,
    DeleteFileParams,
    DownloadFileParams,
    ListFilesParams,
    RetrieveFileParams,
)
from aiollm_openai.types.files import (
    DeleteFileResponse,
    ListFilesResponse,
    OpenAIFile,
)


class CreateFile(CreateFileParams, BaseAPIMethod[OpenAIFile]):
    """Upload a file, to refer to it by its ``id`` in later requests.

    The upload is never retried: ``POST /files`` is not idempotent, and a
    retry after a lost answer would store the file twice. A timeout or a
    lost connection may therefore leave a file at OpenAI whose ``id`` you
    never got; give the file an ``expires_after`` where that matters.
    ``expires_after`` is sent as ``expires_after[anchor]`` and
    ``expires_after[seconds]``, the form fields OpenAI reads.

    A file ``id`` refers to data in the project: treat it as a server-side
    reference and never take one from an end user.

    Example:
        >>> file = await client.send_request(
        ...     CreateFile(file=Path("report.pdf"), purpose="user_data")
        ... )
        >>> file.id
    """

    __request_path__ = "/files"
    __request_method__ = "POST"
    __request_body_format__ = BodyFormat.MULTIPART
    __retryable__ = False


class ListFiles(ListFilesParams, BaseAPIMethod[ListFilesResponse]):
    """List the files of the project, one page at a time.

    Pass the page's ``last_id`` as ``after`` while ``has_more`` is true.

    Example:
        >>> page = await client.send_request(ListFiles(limit=100))
        >>> while page.has_more:
        ...     page = await client.send_request(
        ...         ListFiles(limit=100, after=page.last_id)
        ...     )
    """

    __request_path__ = "/files"
    __request_method__ = "GET"
    __request_query__ = frozenset({"after", "limit", "order", "purpose"})
    __retryable__ = True


class RetrieveFile(RetrieveFileParams, BaseAPIMethod[OpenAIFile]):
    """Get the metadata of a file.

    Example:
        >>> file = await client.send_request(RetrieveFile(file_id="file-1"))
    """

    __request_path__ = "/files/{file_id}"
    __request_method__ = "GET"
    __retryable__ = True


class DeleteFile(DeleteFileParams, BaseAPIMethod[DeleteFileResponse]):
    """Delete a file.

    It is retried on a transient failure, as the official SDKs do; a
    retry after a deletion whose answer was lost raises
    ``NotFoundError``, since the file is already gone.

    Example:
        >>> await client.send_request(DeleteFile(file_id="file-1"))
    """

    __request_path__ = "/files/{file_id}"
    __request_method__ = "DELETE"
    __retryable__ = True


class DownloadFile(DownloadFileParams, BaseStreamMethod[bytes]):
    """Stream the content of a file, never holding it whole in memory.

    Only opening the download is retried. ``Stream.save(path)`` writes it
    to a file.

    Example:
        >>> method = DownloadFile(file_id="file-1")
        >>> async with client.stream(method) as content:
        ...     await content.save(Path("output.jsonl"))
    """

    __request_path__ = "/files/{file_id}/content"
    __request_method__ = "GET"
    __response_format__ = ResponseFormat.BYTES
    __retryable__ = True
