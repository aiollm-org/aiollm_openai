# Contributing

Thank you for helping with aiollm-openai, aiollm's OpenAI provider. It is pre-alpha: the public API can still change. For anything larger than a fix, open an issue first and describe the change, so that we agree on the design before you spend time on code.

The rules are written down in [`CLAUDE.md`](https://github.com/aiollm-org/aiollm_openai/blob/master/CLAUDE.md), this package's profile, and in the core's [`CLAUDE.md`](https://github.com/aiollm-org/aiollm/blob/master/CLAUDE.md) and [`CONTRIBUTING.md`](https://github.com/aiollm-org/aiollm/blob/master/CONTRIBUTING.md), whose rules for code, tests and docstrings hold here too.

## Setting up

You need Python 3.13 or newer, git and [uv](https://docs.astral.sh/uv/), whose `uv.lock` pins every version. Fork the repository, clone your fork, create a branch and install everything:

```bash
git clone https://github.com/<you>/aiollm_openai.git
cd aiollm_openai
git switch -c my-change
uv sync --all-groups
uv run pre-commit install
```

The core, [aiollm](https://github.com/aiollm-org/aiollm), comes from a commit pinned in `pyproject.toml`, with its test kit (`aiollm[testing]`).

## A change in the core and here

When a change needs the core too, the core's pull request comes first. Then this package's pull request pins the commit of that branch in `pyproject.toml` (both lines that name `aiollm`) and runs `uv lock`; once the core's pull request is merged, the pin moves to the commit of its merge. Locally, with both repositories side by side:

```bash
uv sync --all-groups
uv pip install -e ../aiollm          # the core from your checkout, until the next uv sync
```

## Making changes

- **Generated code is regenerated, not edited.** `src/aiollm_openai/types/<family>.py` and `methods/params/` come from [aiollm-codegen](https://github.com/aiollm-org/aiollm_codegen) through `uv run python scripts/generate_types.py`. The generator (the `codegen` dependency group) and OpenAI's OpenAPI file (a URL and a SHA-256 in the script) are pinned, and CI fails when a regeneration changes any file; fix a defect in aiollm-codegen and move the pin.
- **The schema is OpenAI's.** Methods and types follow OpenAI's official OpenAPI file field by field, and `tests/test_wire_contracts.py` checks them against the snapshot in `tests/contracts/`.
- **Only the extension API of the core.** The package imports from `aiollm` only the names of `aiollm/testing/extension_api.json`; `tests/test_extension_api.py` checks it. A missing name is added to the core first.
- **Public names are exported on purpose.** A sorted `__all__` in each package, and `tests/public_api.json` updated in the same change.

## Checks

```bash
uv run ruff format src tests
uv run ruff check --fix src tests
uv run mypy src
uv run pyright
uv run pytest --cov                  # coverage stays at 100 % in practice (floor 98 %)
uv run pytest --mypy -m mypy tests/typing
uv run rumdl check README.md CHANGELOG.md CONTRIBUTING.md CLAUDE.md .claude
uv run python scripts/generate_types.py && git status --short   # changes nothing
```

The tests never reach the real API: they run against the core's local `Recorder` server, which answers only what a test scripted and fails the test on any other request. `OPENAI_API_KEY` is cleared for every test.

## Describe changes

Add a changelog fragment to `CHANGES/` (`uv run towncrier create +my-change.fixed.md --content "..."`): one paragraph about what a user sees; a breaking change starts with `**Breaking:**` and says how to migrate. Do not bump the version or edit `CHANGELOG.md`. A change no user would notice asks for the `no changelog` label. See [`CHANGES/README.md`](CHANGES/README.md).

## Complete

Open a pull request against `master`. It is merged when these pass:

| Check | What it runs |
|---|---|
| `lint and types` | ruff, mypy, pyright, the typing tests, the Markdown lint and a regeneration of the types |
| `tests` | pytest with coverage on Linux and Windows, CPython 3.13, 3.14 and 3.15 |
| `package` | the wheel in a clean environment, with the core it pins |
| `audit` | known vulnerabilities in the locked dependencies (`uv audit`) |
| `changelog` | a changelog fragment, unless the PR has the `no changelog` label |

Releases, version numbers and tags are made by the maintainer.

## Contributing with Claude

The repository is set up for [Claude Code](https://claude.com/claude-code): `CLAUDE.md` is loaded into every session, and the skills in `.claude/skills/` carry the workflows (`ai-api-endpoint` for a new endpoint or field, `ai-api-release` for fragments and releases). You are the author of the pull request and answer for every line of it.

### Directives

1. **Discuss before code.** For anything beyond a small fix, ask Claude for a proposal first (what changes, the options, the trade-offs, the tests) and read it before you let it write code. Open the issue with that proposal.
2. **Facts come from sources, not from memory.** What a provider accepts or returns is checked against its schema or documentation, and the source is named. Claude must not guess a field, a status or a default.
3. **Test first for every bug.** Ask Claude to write the failing test before the fix and to show that it fails on the unfixed code.
4. **Results are run, not predicted.** Ask for the actual output of the full set of checks above. "It should pass" is not a result; neither is a check that was skipped without saying so.
5. **Nothing is weakened to get green.** No new `skip`, `xfail`, `type: ignore`, `noqa` or relaxed setting without a reason in the code and in the pull request; no lowered coverage.
6. **Small, focused diffs.** One concern per pull request, no drive-by refactoring or formatting of unrelated files, and no new abstraction without a second use for it.
7. **You decide what leaves your machine.** Review every diff yourself. Claude does not push, open or merge pull requests, publish or delete anything unless you tell it to, one action at a time.
8. **No secrets in the conversation.** Never paste a real API key into the session; the tests use fakes and a local server, and the library must never log a key, a header, a body, a query string or a file name.
9. **Your name on the commit.** Commit messages and pull requests carry no model names or AI co-author trailers. If you want to say that a pull request was made with Claude, say it in its description.
10. **One task per session.** Start a fresh session for a new task and give it the issue; a long session that drifted across topics makes worse decisions.

If you use a tool that condenses command output for the agent, exclude `pytest` from it, so that the coverage report stays visible.

## License

By contributing, you agree that your contributions are licensed under the MIT License of the project.
