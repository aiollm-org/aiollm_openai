# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`aiollm-openai` (import `aiollm_openai`) is aiollm's OpenAI provider: `OpenAIClient`, OpenAI's Responses and Files APIs, and the adapters that give it aiollm's provider-neutral `client.generation` and `client.files`. It was `aiollm.providers.openai` up to aiollm 0.14.0; its history came along (`git log --follow`). Pre-alpha; the public surface is unstable.

The core is [aiollm-org/aiollm](https://github.com/aiollm-org/aiollm), installed from PyPI as `aiollm-core`, one minor version of it (`>=0.17,<0.18` in `pyproject.toml`); between two releases of the core, a change that needs its unreleased code pins a commit of it (Conventions), and a release goes back to the range. **The core's `CLAUDE.md` holds the rules this package follows too**: docstrings, logging, naming, `MISSING` and `None` in requests, the request and stream flows, errors, the shared layer. This file adds only what is this package's own. The code says what the package does; an agreed decision says what it should do; a text that lags behind the code is fixed. `CONTRIBUTING.md` is the guide for contributors; keep it in step with this file. The user documentation is the core's site (<https://aiollm-org.github.io/aiollm/>); a docstring here is user documentation too.

## Commands

The project is managed with `uv` (see `uv.lock`) and requires Python 3.13 or newer (`.python-version` pins 3.13). The version is read by hatchling from `src/aiollm_openai/__meta__.py`.

```bash
uv sync --all-groups                 # install runtime + dev + test + codegen deps
uv run ruff check src tests          # lint (line length 79)
uv run ruff format src tests         # format
uv run mypy src                      # strict type checking (pydantic.mypy plugin)
uv run pyright                       # strict pyright over src and tests
uv run pytest                        # tests (asyncio_mode = auto)
uv run pytest --cov                  # with coverage (branch, fail_under = 98; kept at 100 %)
uv run pytest --mypy -m mypy tests/typing   # static checks of the public API
uv run rumdl check README.md CHANGELOG.md CONTRIBUTING.md CLAUDE.md .claude
uv run python scripts/generate_types.py     # the types and request params (pinned generator and OpenAPI file)
uv run python scripts/refresh_upstream.py   # move the OpenAPI pin to upstream, rebuild snapshot and types
OPENAI_OPENAPI=<openapi.yaml> uv run python -m tests.contracts.update   # rebuild the contract snapshot
uv run towncrier create +slug.fixed.md      # a changelog fragment in CHANGES/
uv pip install -e ../aiollm          # try a change of the core from a sibling checkout (until the next uv sync)
```

CI (`.github/workflows/tests.yml`) runs the checks, the Markdown lint and a regeneration of the types that must change no file (job `lint and types`), pytest on Ubuntu and Windows × every supported CPython (3.13, 3.14, 3.15; the `tests` job sums up the matrix for the branch ruleset), three clean environments, every package from a wheel (`package`): this wheel with the core it brings from PyPI, imported and run through `check_extension_api`; the core alone at that version, which must import without aiohttp or a provider; and both providers on that core, the other one from PyPI without its dependencies, `uv audit` (`audit`) and, on pull requests, `towncrier check` (`changelog`; the `no changelog` label skips it). `core-master.yml` runs the tests every Monday (and by hand) against the core's `master` instead of the pin, so a change of the core that breaks this package shows up before a core release. `upstream.yml` runs `scripts/refresh_upstream.py` every Monday at 06:00 UTC (and by hand): an OpenAI file ahead of the pin becomes one pull request on `upstream/refresh`, opened with the organization secret `UPSTREAM_PR_TOKEN` so CI runs on it; once a commit was pushed to that open pull request by hand, the branch is no longer rebuilt and newer sources only get a comment there. `publish.yml` publishes a tag `v<version>` to PyPI by trusted publishing; a release needs the core from PyPI, never a git pin, which PyPI refuses.

## Layout

- `src/aiollm_openai/`: `api.py` (`OPENAI_API`, `PROVIDER: Final = Provider("openai")`), `resolver.py` (`OpenAIErrorResolver`), `client.py` (`OpenAIClient`, `session_factory`), `tools.py` (`web_search`, `file_search`, `code_interpreter`: OpenAI's built-in tools with their own options, as `ProviderTool`s built from the generated parameters), `methods/` (hand-written `responses.py` and `files.py` over the generated `methods/params/`), `types/` (generated but `options.py`), `adapters/` (hand-written: `conversions.py`, `generation.py`, `files.py`). The package imports from the core only the names of its extension API (`aiollm/testing/extension_api.json`); `tests/test_extension_api.py` runs `check_extension_api("aiollm_openai")`. A name missing there is added in the core first (a core pull request), never imported around it.
- `tests/`: the provider's tests, `contracts/` (the wire-contract snapshot: `responses/` and `files/`, `manifest.json`, `update.py`), `typing/check_public_api.py` (never run: every `# type: ignore[...]` there marks a line that must stay a type error) and `public_api.json` (every package's `__all__`; a public name added, renamed or removed is a decision, updated in the same change with a fragment). `conftest.py` loads the core's test kit (`aiollm.testing.plugin`) and names `OPENAI_API_KEY` as the key to clear in every test. `test_readme.py` runs the README's example.
- `scripts/`: `generate_types.py` and `refresh_upstream.py`.

## Tests

- The shared scenarios are the core's (`aiollm.testing.contract`): `test_generation_contract.py`, `test_files_contract.py` and `test_client_contract.py` subclass `GenerationContract`, `FilesContract` and `ClientContract` with this provider's cases. What is OpenAI's own is in `test_client.py`, `test_generation.py`, `test_responses.py`, `test_files.py` and the rest.
- The local server (`aiollm.testing.aiohttp.Recorder`) answers only what a test scripted with `recorder.reply(...)`; any other request gets a 500 and fails the test at teardown.
- `test_wire_contracts.py` checks the endpoints against the snapshot (request schemas closed, so unknown fields fail), with the core's `aiollm.testing.wire`. Every model sets `__upstream_schema__`, the upstream `$def` it mirrors, and each field is checked against it, request parameters both ways. The examples (`responses.json`, `events.json`) are added by hand from OpenAI's documentation or a live run; a refresh never changes them and lists the ones the new schemas refuse under `rejected` in `manifest.json`.

## Schema source and generated code

- One pinned source: OpenAI's official OpenAPI file (`openapi.yaml` of `openai/openai-openapi`), never a third-party dump. `scripts/generate_types.py` pins it by `OPENAI_OPENAPI_URL` (a commit) and `OPENAI_OPENAPI_SHA256`, the hash of the snapshot's file (`test_codegen_pin.py` checks); a new file changes the snapshot and the pin together.
- The response and event types (`types/<family>.py`) and the request parameters (`methods/params/`) are generated by aiollm-codegen, the `codegen` dependency group pinned to a commit of `aiollm-org/aiollm_codegen` (`--codegen DIR` runs a local checkout), and never edited by hand; a defect is fixed in aiollm-codegen and the pin moves. `types/options.py`, the method classes and the adapters stay hand-written. `methods/params/` is deeper than the core's depth rule allows, decided with the maintainer (2026-11-06).
- Ruff gives the core (`aiollm`) an import section of its own between third-party and first-party, so the generated modules keep the layout aiollm-codegen writes.
- `scripts/refresh_upstream.py` compares the pin with the newest commit of OpenAI's file; behind, it moves the pin, rebuilds the snapshot and the types, and writes the pull request's description with the public names they add or remove against `tests/public_api.json` (updated by hand).

## Conventions

- The core's conventions apply (its `CLAUDE.md`, "Conventions"); a conflict is settled with the maintainer.
- A change of the core and of this package at once: the core's pull request first; this package's pull request pins the commit of that branch, then the commit of its merge, until the core's next release puts the range back.
- A change pull request adds a fragment to `CHANGES/` and never bumps the version or edits `CHANGELOG.md`; a release is a separate pull request (`ai-api-release`). Tags are set by the maintainer only.
- A provider type uses `Literal` for a schema constant that names the object (`type`, `object`, `role: "assistant"`) and `str` for an enumeration that grows (`status`, error codes).
- Provider-only request fields reach `client.generation` through `provider_options=`: `OpenAIGenerationOptions` (`types/options.py`), whose keys are method fields of the same type and never a field the shared request fills (`test_options.py`).

## Skills

`.claude/skills/`: `ai-api-endpoint` (a new endpoint or field, from the upstream schema to the wire contracts) and `ai-api-release` (fragments and releases). They follow this file; when a skill and this file disagree, this file wins and the skill is fixed.
