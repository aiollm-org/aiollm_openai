"""OpenAI's Files methods against their upstream snapshot.

``operations.json`` of the ``files`` snapshot says how an
operation is sent (method, path, path and query parameters, body) and
what it returns; each method class must send it that way and parse the
documented results without losing a field. The fields themselves are
checked with every other request model in ``test_wire_contracts.py``.
"""

from __future__ import annotations

import json
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import pytest
from jsonschema import Draft202012Validator

from aiollm.enums import BodyFormat, ResponseFormat
from aiollm.methods import BaseAPIMethod, BaseStreamMethod
from aiollm.methods.base import BaseMethod
from aiollm.types.base import BaseAPIType

from aiollm_openai import methods

CONTRACTS: Final = Path(__file__).parent / "contracts"


@dataclass(frozen=True)
class FilesCase:
    """A group of Files endpoints.

    Attributes:
        name: Snapshot folder.
        methods: The module that exports the method classes.
    """

    name: str
    methods: ModuleType


CASES: Final = [pytest.param(FilesCase("files", methods), id="files")]

BODY_FORMATS: Final = {
    (): BodyFormat.NONE,
    ("multipart/form-data",): BodyFormat.MULTIPART,
}
"""The body format of each documented set of request media types."""


def load(case: FilesCase, file: str) -> Any:
    """Read one file of a snapshot.

    Args:
        case: The provider.
        file: File name in its snapshot folder.

    Returns:
        The decoded JSON.
    """
    path = CONTRACTS / case.name / file
    return json.loads(path.read_text(encoding="utf-8"))


def method_of(case: FilesCase, operation: str) -> type[BaseMethod[Any]]:
    """Find the method class of an operation.

    Args:
        case: The provider.
        operation: Such as ``listFiles``.

    Returns:
        The class named after it, such as ``ListFiles``.
    """
    method: type[BaseMethod[Any]] = getattr(
        case.methods, operation[:1].upper() + operation[1:]
    )
    return method


def response_model(method: BaseMethod[Any]) -> type[BaseAPIType]:
    """Give the model a method parses its JSON result into.

    Args:
        method: The method.

    Returns:
        Its response type, a provider model.
    """
    model = method.response_type
    assert isinstance(model, type)
    assert issubclass(model, BaseAPIType)
    return model


def wire_names(
    method: type[BaseMethod[Any]], names: AbstractSet[str]
) -> list[str]:
    """Give fields' names on the wire.

    Args:
        method: The method class.
        names: Field names.

    Returns:
        Their serialization aliases, else the names, sorted.
    """
    fields = method.model_fields
    return sorted(fields[name].serialization_alias or name for name in names)


@pytest.mark.parametrize("case", CASES)
def test_every_operation_has_its_method(case: FilesCase) -> None:
    for operation in load(case, "operations.json"):
        method = method_of(case, operation)

        assert issubclass(method, (BaseAPIMethod, BaseStreamMethod))
        assert method.__name__ in case.methods.__all__


@pytest.mark.parametrize("case", CASES)
def test_methods_send_what_upstream_documents(case: FilesCase) -> None:
    for operation, shape in load(case, "operations.json").items():
        method = method_of(case, operation)
        request = method.model_construct()
        params = method.__request_path_params__
        query = method.__request_query__

        assert request.request_method == shape["method"], operation
        assert request.request_path_template == shape["path"], operation
        assert wire_names(method, params) == sorted(shape["path_parameters"])
        assert wire_names(method, query) == shape["query"], operation
        assert (
            request.request_body_format is BODY_FORMATS[tuple(shape["body"])]
        ), operation


@pytest.mark.parametrize("case", CASES)
def test_methods_read_what_upstream_returns(case: FilesCase) -> None:
    for operation, shape in load(case, "operations.json").items():
        method = method_of(case, operation)
        request = method.model_construct()

        if shape["response"] == ["application/octet-stream"]:
            assert issubclass(method, BaseStreamMethod), operation
            assert request.response_format is ResponseFormat.BYTES
            continue

        assert shape["response"] == ["application/json"], operation
        assert request.response_format is ResponseFormat.JSON
        upstream = response_model(request).__upstream_schema__
        assert upstream == shape["schema"], operation


@pytest.mark.parametrize("case", CASES)
def test_only_the_upload_is_never_retried(case: FilesCase) -> None:
    # POST /files is not idempotent: a retry after a lost answer would
    # store the file twice. Reading and deleting are safe to repeat.
    for operation in load(case, "operations.json"):
        request = method_of(case, operation).model_construct()

        assert request.retryable is (operation != "createFile"), operation


@pytest.mark.parametrize("case", CASES)
def test_documented_results_parse_without_loss(case: FilesCase) -> None:
    # Validation alone would pass a field read into the wrong place or
    # dropped: what the model gives back must be the JSON it came from.
    schema = Draft202012Validator(load(case, "response.schema.json"))
    documented = load(case, "responses.json")
    assert documented

    for example in documented:
        body = example["body"]
        method = method_of(case, example["id"]).model_construct()
        model = response_model(method).model_validate(body)

        assert list(schema.iter_errors(body)) == [], example["id"]
        assert model.model_dump(mode="json", exclude_unset=True) == body
