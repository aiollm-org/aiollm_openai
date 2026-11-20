# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static checks of OpenAI's public API.

Never run: mypy (through ``pytest --mypy``) and pyright check it. Every
``# type: ignore[...]`` marks a line that must stay a type error; mypy's
``warn_unused_ignores`` and pyright's ``reportUnnecessaryTypeIgnoreComment``
fail the check if that error disappears.
"""

from __future__ import annotations

from typing import assert_type

from aiollm import (
    CompletedEvent,
    GenerationStream,
    IGenerationClient,
    IncompleteReason,
    Provider,
    Stream,
    bind_resource,
)
from aiollm.adapters import IConversion
from aiollm.adapters import Stop as GenerationStop
from aiollm.client import BaseClient
from aiollm.http import BearerAuth
from aiollm.resources import (
    GenerationResource,
    IFilesClient,
    IFilesResource,
    IGenerationResource,
)
from aiollm.session import ISession, ISessionFactory
from aiollm.session.aiohttp import AiohttpSession
from aiollm.types.requests import GenerationRequest, ToolCallTurn, Turn
from aiollm.types.responses import GenerationResponse, StoredFile, TokenUsage

from aiollm_openai import OpenAIClient
from aiollm_openai.adapters import OpenAIGenerationAdapter
from aiollm_openai.client import session_factory
from aiollm_openai.methods import CreateResponse, StreamResponse
from aiollm_openai.types import (
    OpenAIFile,
    Response,
    ResponseStreamEvent,
    ResponseUsage,
)
from aiollm_openai.types.options import OpenAIGenerationOptions


async def responses_infer_their_types(session: AiohttpSession) -> None:
    create = CreateResponse(model="m", input="x", instructions="Be brief")
    assert_type(await session.send_request(create), Response)
    async with session.stream(StreamResponse(model="m", input="x")) as events:
        assert_type(events, Stream[ResponseStreamEvent])
    CreateResponse(model="m", input="x", max_tokens=16)  # type: ignore[call-arg]


async def generation_infers_its_types(session: AiohttpSession) -> None:
    generation = GenerationResource(session, OpenAIGenerationAdapter())
    request = GenerationRequest(model="m", input="x")
    response = await generation.create(request)
    assert_type(response, GenerationResponse[Response])
    assert_type(response.incomplete_reason, IncompleteReason | None)
    assert_type(response.raw_incomplete_reason, str | None)
    assert_type(response.refusal, str | None)
    assert_type(response.to_turn(), Turn | ToolCallTurn)
    async with generation.stream(request) as events:
        assert_type(events, GenerationStream[ResponseStreamEvent])


def openai_usage(source: ResponseUsage, /) -> TokenUsage:
    """A conversion of OpenAI's counts."""
    return TokenUsage(input_tokens=source.input_tokens)


def text_usage(source: str, /) -> TokenUsage:
    """A conversion of something else than OpenAI's counts."""
    return TokenUsage(input_tokens=len(source))


def conversions_are_typed(
    stop: IConversion[Response, GenerationStop],
    response: Response,
) -> None:
    assert_type(stop(response), GenerationStop)
    OpenAIGenerationAdapter(
        stop_conversion=stop, usage_conversion=openai_usage
    )
    OpenAIGenerationAdapter(usage_conversion=text_usage)  # type: ignore[arg-type]


async def client_generation_is_typed(client: OpenAIClient) -> None:
    response = await client.generation.create(model="m", input="x")
    assert_type(response, GenerationResponse[Response])
    async with client.generation.stream(model="m", input="x") as events:
        completed = await events.final()
        assert_type(completed, CompletedEvent[ResponseStreamEvent])


def clients_take_session_arguments() -> None:
    OpenAIClient("sk", max_body_size=1_000, trust_env=True)
    OpenAIClient("sk", max_body_sizee=1)  # type: ignore[call-arg]
    assert_type(session_factory("sk", max_body_size=1), AiohttpSession)
    session_factory("sk", max_body_sizee=1)  # type: ignore[call-arg]


def any_generation_client(client: IGenerationClient) -> IGenerationClient:
    return client


def clients_are_generation_clients(openai: OpenAIClient) -> None:
    typed: IGenerationClient[Response, ResponseStreamEvent] = openai
    resource: IGenerationResource[Response, ResponseStreamEvent]
    resource = typed.generation
    assert resource.session is openai.session
    any_generation_client(openai)


def any_files_client(client: IFilesClient) -> IFilesClient:
    return client


async def clients_are_files_clients(openai: OpenAIClient) -> None:
    typed: IFilesClient[OpenAIFile] = openai
    resource: IFilesResource[OpenAIFile] = typed.files
    assert resource.session is openai.session
    any_files_client(openai)
    any_files_client(BaseClient(openai.session))  # type: ignore[arg-type]

    stored = await openai.files.upload(
        b"%PDF", provider_options={"purpose": "vision"}
    )
    assert_type(stored, StoredFile[OpenAIFile])
    await openai.files.upload(
        b"%PDF",
        provider_options={"purpose": "photos"},  # type: ignore[arg-type]
    )
    async for item in openai.files.list():
        assert_type(item, StoredFile[OpenAIFile])


def session_factories_are_typed() -> None:
    factory: ISessionFactory[AiohttpSession, BearerAuth, ...] = session_factory
    assert_type(factory("sk"), AiohttpSession)
    session: ISession = OpenAIClient.session_factory("sk")
    assert session


def clients_bind_their_resources() -> None:
    assert_type(
        OpenAIClient("sk").generation,
        GenerationResource[
            Response, ResponseStreamEvent, OpenAIGenerationOptions
        ],
    )
    # An adapter of options the resource does not take.
    no_options = GenerationResource[Response, ResponseStreamEvent]
    bind_resource(no_options, OpenAIGenerationAdapter)  # type: ignore[arg-type]
    assert_type(OpenAIClient.provider, Provider)


async def provider_options_are_typed(request: GenerationRequest) -> None:
    generation = OpenAIClient("sk").generation
    response = await generation.create(
        request, provider_options={"store": False, "temperature": 0.2}
    )
    assert_type(response, GenerationResponse[Response])
    await generation.create(
        model="m", input="x", provider_options={"store": None}, timeout=5
    )
    async with generation.stream(
        request, provider_options={"top_p": 0.5}
    ) as events:
        assert_type(events, GenerationStream[ResponseStreamEvent])
    await generation.create(request, provider_options={"stor": False})  # type: ignore[call-overload]
    await generation.create(request, provider_options={"temperature": "hot"})  # type: ignore[call-overload]
