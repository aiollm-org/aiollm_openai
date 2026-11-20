"""OpenAI's methods and types follow its own schemas.

The schemas and examples in ``contracts/`` are a snapshot of
OpenAI's official OpenAPI file
(``manifest.json`` says which; ``update.py`` rebuilds them). The
request schemas are closed here: an object that lists its properties
takes no others, so a misspelled or removed field fails
instead of passing as an unknown extra.

A snapshot does not follow the provider by itself: these tests catch
the library drifting from the snapshot, and the provider drifting from
the library only once the snapshot is refreshed.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    Final,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

import pytest
from jsonschema import Draft202012Validator
from pydantic import MISSING, BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from aiollm.adapters import ICompositeAdapter
from aiollm.enums import Effort, ToolChoice
from aiollm.methods import BaseAPIMethod, BaseStreamMethod
from aiollm.response.parse.stream import StreamResponseParser
from aiollm.session.aiohttp import AiohttpSession
from aiollm.testing import Reply
from aiollm.testing.aiohttp import Recorder
from aiollm.testing.wire import (
    Snapshot,
    allows_null,
    errors,
    field_problems,
    naming_problems,
    provider_models,
    resolve,
)
from aiollm.types.matchers import AnyOf, EventTypes
from aiollm.types.requests import (
    DocumentPart,
    GenerationRequest,
    ImagePart,
    TextPart,
    Tool,
    ToolCallTurn,
    ToolResult,
    Turn,
)

from aiollm_openai.adapters import OpenAIGenerationAdapter
from aiollm_openai.adapters.conversions import output_text
from aiollm_openai.methods import CreateResponse, StreamResponse
from aiollm_openai.types import Response, ResponseStreamEvent
from aiollm_openai.types.options import OpenAIGenerationOptions
from tests import test_responses as openai
from tests.contracts.update import event_tags, type_tags

CONTRACTS = Path(__file__).parent / "contracts"


def snapshot(name: str) -> Snapshot:
    """The snapshot of an endpoint.

    Args:
        name: Snapshot folder, the endpoint's name.

    Returns:
        The snapshot.
    """
    return Snapshot(CONTRACTS, name)


def load(name: str, file: str) -> Any:
    """Read one snapshot file.

    Args:
        name: Snapshot folder, the endpoint's name.
        file: File name in it.

    Returns:
        The decoded JSON.
    """
    return snapshot(name).load(file)


def validator(case: Case, file: str) -> Draft202012Validator:
    """Build a validator of a snapshot schema; the request one is closed.

    Args:
        case: The endpoint.
        file: Schema file name.

    Returns:
        The validator.
    """
    return snapshot(case.name).validator(file)


def examples(case: Case, file: str) -> list[Any]:
    """Return the documentation examples of a snapshot.

    Args:
        case: The endpoint.
        file: ``responses.json`` or ``events.json``.

    Returns:
        The example bodies.
    """
    return snapshot(case.name).examples(file)


def modules(
    package: str, *, skip: frozenset[str] = frozenset()
) -> tuple[ModuleType, ...]:
    """Import every module of a package.

    Args:
        package: The package, such as ``aiollm_openai.types``.
        skip: Module names left out, such as the hand-written options.

    Returns:
        The modules, in name order.
    """
    root = importlib.import_module(package)
    return tuple(
        importlib.import_module(f"{package}.{info.name}")
        for info in sorted(pkgutil.iter_modules(root.__path__))
        if info.name not in skip
    )


@dataclass(frozen=True)
class Case:
    """One endpoint and the library's side of it.

    Attributes:
        name: Snapshot folder, the endpoint's name.
        create: The buffered method.
        stream: The streamed method.
        request: Every request field, set to a valid value.
        response: The response model.
        events: The stream event union.
        documented: Event types the provider documents outside the SDK
            schema, with where.
        examples: The hand-written payloads of the other tests:
            responses and stream events.
        adapter: The shared generation adapter of the endpoint.
        options: The adapter's provider options.
        models: The modules of the endpoint's response and event models.
        params: The modules of the endpoint's request parameter models.
    """

    name: str
    create: type[BaseAPIMethod[Any]]
    stream: type[BaseStreamMethod[Any]]
    request: dict[str, Any]
    response: type[BaseModel]
    events: TypeAdapter[Any]
    documented: dict[str, str]
    examples: tuple[list[dict[str, Any]], list[dict[str, Any]]]
    adapter: ICompositeAdapter[GenerationRequest, Any, Any, Any, Any]
    options: type
    models: tuple[ModuleType, ...]
    params: tuple[ModuleType, ...]


OPENAI = Case(
    name="responses",
    create=CreateResponse,
    stream=StreamResponse,
    request={
        "model": "gpt-5",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
        ],
        "instructions": "Be brief",
        "max_output_tokens": 64,
        "metadata": {"user": "u-1"},
        "safety_identifier": "u-1",
        "reasoning": {
            "effort": "high",
            "summary": "auto",
            "context": "auto",
            "mode": "standard",
        },
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {"type": "object"},
                "description": "The answer",
                "strict": True,
            },
            "verbosity": "low",
        },
        "previous_response_id": "resp_1",
        "store": False,
        "temperature": 0.5,
        "top_p": 0.9,
        "include": ["reasoning.encrypted_content"],
        "max_tool_calls": 3,
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Current weather",
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
            }
        ],
        "tool_choice": {"type": "function", "name": "get_weather"},
        "parallel_tool_calls": False,
    },
    response=Response,
    events=TypeAdapter(ResponseStreamEvent),
    documented={},
    examples=([openai.RESPONSE], openai.STREAM),
    adapter=OpenAIGenerationAdapter(),
    options=OpenAIGenerationOptions,
    models=modules("aiollm_openai.types", skip=frozenset({"options"})),
    params=modules("aiollm_openai.methods.params"),
)

CASES = [pytest.param(case, id=case.name) for case in (OPENAI,)]


@dataclass(frozen=True)
class ProviderTypes:
    """Every generated model of a provider and the snapshots they mirror.

    Attributes:
        snapshots: The snapshots of every
            endpoint the provider's types serve.
        models: The modules of the response and event models.
        params: The modules of the request parameter models.
        covered: The methods whose fields
            ``test_method_fields_match_the_request_schema`` checks; their
            parameter mixins are left to that test. Every other parameter
            model, such as ``ListFilesParams``, is checked here against
            the ``$defs`` entry of its name.
    """

    snapshots: tuple[Snapshot, ...]
    models: tuple[ModuleType, ...]
    params: tuple[ModuleType, ...]
    covered: tuple[type[BaseModel], ...]


TYPES = ProviderTypes(
    snapshots=(snapshot("responses"), snapshot("files")),
    models=OPENAI.models,
    params=OPENAI.params,
    covered=(OPENAI.create, OPENAI.stream),
)
"""The provider's types, checked against all of its snapshots."""


def frame(data: dict[str, Any]) -> bytes:
    """Encode one SSE frame.

    Args:
        data: The event.

    Returns:
        ``event:`` and ``data:`` lines of the event.
    """
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n".encode()


def test_the_snapshot_has_examples() -> None:
    manifest = json.loads((CONTRACTS / "manifest.json").read_text("utf-8"))

    for case in (OPENAI,):
        entry = manifest["endpoints"][case.name]
        assert examples(case, "responses.json")
        assert examples(case, "events.json")
        assert entry["responses"] == len(examples(case, "responses.json"))
        assert entry["events"] == len(examples(case, "events.json"))
        # A refresh keeps an example its new schema refuses and lists it
        # here, so it is fixed or removed by hand, never dropped.
        assert entry["rejected"] == []


FULL_REQUEST = GenerationRequest(
    model="m",
    input="Hi",
    instructions="Be brief",
    max_output_tokens=64,
    effort=Effort.HIGH,
    output_schema={"type": "object"},
    end_user_id="u-1",
    tools=[Tool(name="f", parameters={"type": "object"})],
    tool_choice=ToolChoice.REQUIRED,
    parallel_tool_calls=False,
)
"""A shared request with every field set."""


@pytest.mark.parametrize("case", CASES)
def test_the_full_request_sets_every_field_the_library_sends(
    case: Case,
) -> None:
    # The methods take every field of the provider's schema (checked
    # below); the full request covers the ones the shared adapter and the
    # provider options fill, so the body checks see each of them.
    adapter: Any = case.adapter
    sent = {
        name
        for name, value in adapter._fields(FULL_REQUEST).items()
        if value is not MISSING
    }
    method = case.create(**case.request)

    assert sent | get_type_hints(case.options).keys() <= (
        method.model_fields_set
    )


@pytest.mark.parametrize("case", CASES)
async def test_requests_pass_the_request_schema(
    case: Case,
    recorder: Recorder,
    session: AiohttpSession,
) -> None:
    schema = validator(case, "request.schema.json")
    recorder.reply(
        Reply(json=examples(case, "responses.json")[0]),
        Reply(
            chunks=[frame(event) for event in examples(case, "events.json")],
            content_type="text/event-stream",
        ),
    )

    await session.send_request(case.create(**case.request))
    buffered = json.loads(recorder.last.body)
    # Only the request matters: the examples are not one stream.
    async with session.stream(case.stream(**case.request)):
        pass
    streamed = json.loads(recorder.last.body)

    assert errors(schema, buffered) == []
    assert errors(schema, streamed) == []
    assert streamed == buffered | {"stream": True}


CONVERSATION = GenerationRequest(
    model="model-1",
    input=[
        Turn.user("Hi"),
        Turn.assistant("Hello! How can I help?"),
        Turn.user(
            [
                TextPart(text="What is in these?"),
                ImagePart(data=b"\x89PNG", media_type="image/png"),
                ImagePart(url="https://example.com/cat.jpg"),
                DocumentPart(data=b"%PDF-1.7", filename="report.pdf"),
                DocumentPart(data=b"%PDF-1.7"),
                DocumentPart(url="https://example.com/report.pdf"),
            ]
        ),
        Turn.assistant("A cat and a report."),
        Turn.user("Tell me a joke"),
    ],
    instructions="Be brief",
    max_output_tokens=64,
    end_user_id="u-1",
)
"""A shared request with every field set and several turns, one of
them of every kind of part and source."""


@pytest.mark.parametrize("case", CASES)
def test_shared_conversation_passes_the_request_schema(case: Case) -> None:
    schema = validator(case, "request.schema.json")

    for method in (
        case.adapter.to_method(CONVERSATION),
        case.adapter.to_stream_method(CONVERSATION),
    ):
        body = to_jsonable_python(method.get_body_map())
        assert errors(schema, body) == []


@pytest.mark.parametrize("native", [True, False], ids=["native", "shared"])
@pytest.mark.parametrize("case", CASES)
def test_a_tool_loop_passes_the_request_schema(
    case: Case, native: bool
) -> None:
    # The response's own output (or, from another provider, the shared
    # calls) and the results that follow it make a valid body, so the
    # second request of a loop is what the provider takes.
    adapter: Any = case.adapter
    called = adapter.to_response(
        case.response.model_validate(LOOP[case.name]), CONVERSATION
    )
    turn = called.to_turn()
    assert isinstance(turn, ToolCallTurn)
    if not native:
        turn = turn.model_copy(update={"native": MISSING})
    request = GenerationRequest(
        model="model-1",
        input=[
            Turn.user("Weather in Paris?"),
            turn,
            ToolResult(call_id=turn.tool_calls[0].id, output="15 C"),
            Turn.user("And in short?"),
        ],
        tools=[Tool(name="get_weather", parameters={"type": "object"})],
    )
    schema = validator(case, "request.schema.json")

    body = json.loads(json.dumps(adapter.to_method(request).get_body_map()))

    assert errors(schema, body) == []


LOOP: Final = {
    "responses": openai.RESPONSE
    | {
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "encrypted_content": "gAAAA",
            },
            openai.MESSAGE,
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
                "status": "completed",
            },
        ]
    },
}
"""A response of each endpoint that calls a tool, with reasoning."""


def request_definition(
    case: Case,
    *,
    streamed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the request schema of a buffered or a streamed call.

    OpenAI's schema is a union of a streaming variant (``stream``
    required) and a buffered one.

    Args:
        case: The endpoint.
        streamed: Whether the call is streamed.

    Returns:
        The object schema of the call and the schema's definitions.
    """
    schema = load(case.name, "request.schema.json")
    definitions: dict[str, Any] = schema.get("$defs", {})
    variants = [
        resolve(variant, definitions)
        for variant in schema.get("anyOf", [schema])
    ]

    if len(variants) == 1:
        return variants[0], definitions

    (variant,) = [
        variant
        for variant in variants
        if ("stream" in variant.get("required", ())) is streamed
    ]
    return variant, definitions


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("streamed", [False, True], ids=["create", "stream"])
def test_method_fields_match_the_request_schema(
    case: Case,
    streamed: bool,
) -> None:
    # A field takes None exactly where upstream takes null, and leaves
    # itself out by default only where upstream lets it be absent.
    method = case.stream if streamed else case.create
    definition, definitions = request_definition(case, streamed=streamed)
    properties = definition.get("properties", {})
    required = set(definition.get("required", []))
    problems: list[str] = []

    for name, field in method.model_fields.items():
        wire = field.serialization_alias or name
        where = f"{method.__name__}.{name}"

        if get_origin(field.annotation) is Literal:
            # A constant of the method itself, such as StreamResponse's
            # stream: True; upstream's one schema serves both methods.
            continue
        if wire not in properties:
            problems.append(f"{where}: not in the request schema")
            continue

        nullable = allows_null(properties[wire], definitions)
        takes_none = field.annotation is Any or type(None) in get_args(
            field.annotation
        )

        if nullable and not takes_none:
            problems.append(f"{where}: upstream takes null, the field not")
        if takes_none and not nullable:
            problems.append(f"{where}: the field takes None, upstream not")
        if wire in required and field.default is MISSING:
            problems.append(f"{where}: required upstream, left out here")

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("case", CASES)
def test_a_bare_shared_request_sends_only_what_it_must(case: Case) -> None:
    # Nothing that was not given is sent, not even as null.
    request = GenerationRequest(model="m", input="Hi")
    expected = {
        "responses": {"model", "input"},
    }[case.name]

    for method in (
        case.adapter.to_method(request),
        case.adapter.to_stream_method(request),
    ):
        assert set(method.get_body_map()) - {"stream"} == expected


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize(
    "field",
    [
        pytest.param({"modle": "x"}, id="misspelled"),
        pytest.param({"top_k": 5}, id="unknown"),
    ],
)
def test_the_closed_request_schema_rejects_unknown_fields(
    case: Case,
    field: dict[str, Any],
) -> None:
    body = json.loads(case.create(**case.request).model_dump_json())

    assert errors(validator(case, "request.schema.json"), body | field)


def documented_text(case: Case, body: dict[str, Any]) -> str:
    """Join the text of a documented response straight from its JSON.

    Args:
        case: The endpoint.
        body: The documented response.

    Returns:
        Its text parts, joined, as the provider's SDK reads them.
    """
    assert case is OPENAI
    return "".join(
        part["text"]
        for item in body["output"]
        if item["type"] == "message"
        for part in item["content"]
        if part["type"] == "output_text"
    )


USAGE_FIELDS: Final = {
    "responses": ("input_tokens", "output_tokens", "total_tokens"),
}
"""The token counts of each endpoint's ``usage``, compared as they are."""


@pytest.mark.parametrize("case", CASES)
def test_documented_responses_parse(case: Case) -> None:
    # Validation alone would pass a field read into the wrong place or
    # dropped by extra="ignore": compare what the model gives with the
    # JSON it came from.
    for body in examples(case, "responses.json"):
        model = case.response.model_validate(body)
        assert isinstance(model, Response)
        text = output_text(model)

        assert model.id == body["id"]
        assert model.model == body["model"]
        assert (text or "") == documented_text(case, body)

        if body.get("usage") is None:
            assert model.usage is None
            continue

        for name in USAGE_FIELDS[case.name]:
            assert getattr(model.usage, name) == body["usage"][name], name


@pytest.mark.parametrize("case", CASES)
def test_documented_events_parse(case: Case) -> None:
    for body in examples(case, "events.json"):
        event = case.events.validate_python(body)

        assert event.type == body["type"]


@pytest.mark.parametrize("case", CASES)
def test_known_types_exist_upstream(case: Case) -> None:
    stream = load(case.name, "stream.schema.json")
    upstream = event_tags(stream) | case.documented.keys()
    nested = type_tags(load(case.name, "response.schema.json"))
    nested |= type_tags(stream)

    ours = event_tags(case.events.json_schema())

    assert ours - upstream == set()
    assert type_tags(case.response.model_json_schema()) - nested == set()


def matched_types(matcher: object) -> set[str]:
    """Collect the event types a matcher names.

    Args:
        matcher: A matcher, possibly combined with ``|``.

    Returns:
        The ``types`` of every ``EventTypes`` in it.
    """
    if isinstance(matcher, AnyOf):
        return set().union(*map(matched_types, matcher.matchers))
    if isinstance(matcher, EventTypes):
        return set(matcher.types)
    return set()


@pytest.mark.parametrize("case", CASES)
def test_declared_event_types_exist_upstream(case: Case) -> None:
    # A misspelled terminal would never match, and every complete stream
    # would end in APIStreamInterruptedError.
    upstream = event_tags(load(case.name, "stream.schema.json"))
    upstream |= case.documented.keys()
    method = case.stream(**case.request)
    parser = method.stream_parser
    assert isinstance(parser, StreamResponseParser)

    terminal = matched_types(method.stream_terminal)
    errors = matched_types(parser.error_events)

    assert terminal
    assert terminal - upstream == set()
    assert errors - upstream == set()


@pytest.mark.parametrize("case", CASES)
def test_hand_written_examples_pass_the_schemas(case: Case) -> None:
    responses, events = case.examples
    response_schema = validator(case, "response.schema.json")
    event_schema = validator(case, "stream.schema.json")

    for body in responses:
        assert errors(response_schema, body) == []
    for body in events:
        if body["type"] in case.documented:
            continue
        assert errors(event_schema, body) == [], body["type"]


SIDES = [pytest.param(False, id="responses"), pytest.param(True, id="params")]
"""The response and event models, and the request parameter models."""


@pytest.mark.parametrize("params", SIDES)
def test_every_model_names_its_upstream_schema(params: bool) -> None:
    types = TYPES
    models = list(
        provider_models(
            types.params if params else types.models, types.covered
        )
    )

    assert models
    assert naming_problems(types.snapshots, models, params=params) == []


@pytest.mark.parametrize("params", SIDES)
def test_models_accept_what_upstream_allows(params: bool) -> None:
    types = TYPES
    models = provider_models(
        types.params if params else types.models, types.covered
    )
    problems = field_problems(types.snapshots, models, params=params)

    assert not problems, "\n".join(problems)
