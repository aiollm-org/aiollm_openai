# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Up to 0.14.0 the package was `aiollm.providers.openai`, part of aiollm: its
earlier changes are in [aiollm's changelog](https://github.com/aiollm-org/aiollm/blob/master/CHANGELOG.md).

## [Unreleased]

<!-- towncrier release notes start -->

## [0.16.0] - 2026-11-21

### Added

- Built-in tools of the shared request: `WebSearch` (with `allowed_domains` and
  `user_location`) and `CodeExecution` go in `tools=` beside the functions and
  reach OpenAI as `web_search` and `code_interpreter` (in an `auto` container);
  a `ProviderTool` of this provider is sent as its own definition, and a tool
  OpenAI does not run raises `RequestValidationError`. The new
  `aiollm_openai.tools` module builds those provider tools for the rest:
  `web_search` (`search_context_size`, `external_web_access`, the dated type),
  `file_search` and `code_interpreter` (a container to reuse, or the files and
  memory of a new one). The calls OpenAI ran come back as `server_tool_calls`
  (`web_search_call`, `file_search_call`, `code_interpreter_call`,
  `image_generation_call` and `mcp_call` items; `COMPLETED`, `FAILED` with the
  item's error, `INCOMPLETE`) and the annotations of the text as `citations`
  (`url_citation` a `URL`, `file_citation` a `FILE`, `container_file_citation`
  a `CONTAINER_FILE`, their spans moved to the joined text), in a response and
  in the `CompletedEvent` of a stream, where each call is also a
  `ServerToolCallEvent` when its item is added and when it is done. An
  `assistant` turn of `to_turn()` with OpenAI's own output goes back as those
  output items, so the search calls and reasoning items reach the next request.

### Changed

- Requires `aiollm-core` 0.16 (`>=0.16,<0.17`), the release with the shared
  built-in tool types.

## [0.15.0] - 2026-11-20

### Changed

- **Breaking:** the OpenAI provider is its own package, `aiollm-openai`,
  imported as `aiollm_openai`: `aiollm.providers.openai` becomes
  `aiollm_openai`, and each of its modules keeps its name
  (`aiollm.providers.openai.types` becomes `aiollm_openai.types`, and so on).
  `aiollm_openai.__version__` is its own version.
- The core this package depends on is now the `aiollm-core` distribution
  (it was `aiollm`), from PyPI: `aiollm-core>=0.15,<0.16`. The module is
  still `aiollm`.

[Unreleased]: https://github.com/aiollm-org/aiollm_openai/compare/v0.16.0...HEAD
[0.16.0]: https://github.com/aiollm-org/aiollm_openai/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/aiollm-org/aiollm_openai/releases/tag/v0.15.0
