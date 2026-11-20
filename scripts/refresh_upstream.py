"""Follow OpenAI's pinned OpenAPI file and refresh what it makes.

Run from the aiollm checkout (``.github/workflows/upstream.yml`` runs it
every week):

    uv run python scripts/openai/refresh_upstream.py --out DIR

It compares the commit of ``OPENAI_OPENAPI_URL`` in
``scripts/openai/generate_types.py`` with the newest commit of
``openai/openai-openapi`` that changed ``openapi.yaml``. A pin that is
behind is moved (the URL and the SHA-256), then the contract snapshot is
rebuilt and the types regenerated, so the working tree holds the whole
change for a pull request; ``DIR/pull-request.md`` describes it, with the
public names the new types add or remove against
``tests/providers/openai/public_api.json``, which a person updates.
Under GitHub Actions the output ``pins`` says whether the file was
written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "scripts" / "openai" / "generate_types.py"
PUBLIC_API: Final = ROOT / "tests" / "providers" / "openai" / "public_api.json"

EXPORTS: Final = (
    "import importlib, json, sys; "
    "print(json.dumps({n: importlib.import_module(n).__all__ "
    "for n in sys.argv[1:]}))"
)
"""Prints the ``__all__`` of the packages named on the command line."""

OPENAI_REPOSITORY: Final = "openai/openai-openapi"
"""Repository of OpenAI's OpenAPI file."""

OPENAI_COMMIT: Final = re.compile(r'"([0-9a-f]{40})/openapi\.yaml"')
"""The commit in ``OPENAI_OPENAPI_URL``."""

OPENAI_DATE: Final = re.compile(r"at the commit of (\d{4}-\d{2}-\d{2})")
"""The date in the docstring of ``OPENAI_OPENAPI_URL``."""

OPENAI_SHA256: Final = re.compile(
    r'(OPENAI_OPENAPI_SHA256: Final = \(\n\s*")([0-9a-f]{64})"'
)
"""The hash in ``OPENAI_OPENAPI_SHA256``."""


@dataclass(frozen=True)
class Change:
    """A source that moved.

    Attributes:
        source: What moved, as the report names it.
        old: The pinned value.
        new: The upstream value.
        link: Where to read the difference.
    """

    source: str
    old: str
    new: str
    link: str


def fetch(url: str) -> bytes:
    """Download a URL.

    ``GH_TOKEN`` is sent to the GitHub API when set, for its rate limit.

    Args:
        url: The URL.

    Returns:
        The body.
    """
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "aiollm-upstream-check"}
    )
    token = os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        body: bytes = response.read()
    return body


def fetch_json(url: str) -> Any:
    """Download and decode a JSON document.

    Args:
        url: The URL.

    Returns:
        The decoded value.
    """
    return json.loads(fetch(url))


def run(*command: str, **env: str) -> str:
    """Run a command in the checkout.

    Args:
        *command: The command and its arguments.
        **env: Variables added to the environment.

    Returns:
        Its standard output.
    """
    result = subprocess.run(  # noqa: S603
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    return result.stdout


def follow_openai(openapi: Path) -> Change | None:
    """Move OpenAI's pin to the newest commit of its file.

    Args:
        openapi: Where to write the pinned file, new or not.

    Returns:
        The change, or ``None`` when the pin is current.
    """
    script = SCRIPT.read_text(encoding="utf-8")
    match = OPENAI_COMMIT.search(script)
    assert match is not None
    old = match.group(1)
    commits = fetch_json(
        f"https://api.github.com/repos/{OPENAI_REPOSITORY}/commits"
        "?path=openapi.yaml&per_page=1"
    )
    new: str = commits[0]["sha"]
    raw = f"https://raw.githubusercontent.com/{OPENAI_REPOSITORY}"
    openapi.write_bytes(fetch(f"{raw}/{new}/openapi.yaml"))
    if new == old:
        return None
    date = commits[0]["commit"]["committer"]["date"][:10]
    digest = hashlib.sha256(openapi.read_bytes()).hexdigest()
    script = OPENAI_COMMIT.sub(f'"{new}/openapi.yaml"', script)
    script = OPENAI_DATE.sub(f"at the commit of {date}", script)
    script = OPENAI_SHA256.sub(rf'\g<1>{digest}"', script)
    SCRIPT.write_text(script, encoding="utf-8", newline="\n")
    return Change(
        source=f"OpenAI `openapi.yaml` ({OPENAI_REPOSITORY})",
        old=old[:7],
        new=f"{new[:7]} ({date})",
        link=f"https://github.com/{OPENAI_REPOSITORY}/compare/{old}...{new}",
    )


@dataclass(frozen=True)
class Names:
    """The public names of a package that differ from the snapshot.

    Attributes:
        package: The package.
        added: Names it exports that the snapshot lacks.
        removed: Names of the snapshot it no longer exports.
    """

    package: str
    added: list[str]
    removed: list[str]


def public_names() -> list[Names]:
    """Compare the exported names with ``public_api.json``.

    The packages are imported in a new interpreter, so the names read
    are those of the types just regenerated.

    Returns:
        One entry per package whose names differ.
    """
    snapshot: dict[str, list[str]] = json.loads(
        PUBLIC_API.read_text(encoding="utf-8")
    )
    current: dict[str, list[str]] = json.loads(
        run("uv", "run", "python", "-c", EXPORTS, *snapshot)
    )
    return [
        Names(
            package=package,
            added=sorted(set(current[package]) - set(names)),
            removed=sorted(set(names) - set(current[package])),
        )
        for package, names in snapshot.items()
        if set(current[package]) != set(names)
    ]


def names_section(names: list[Names]) -> str:
    """Describe the changed public names for the pull request.

    Args:
        names: The packages whose names differ.

    Returns:
        A Markdown section.
    """
    if not names:
        return "No public name is added or removed.\n"
    lines = [
        "The public names differ from",
        "`tests/providers/openai/public_api.json`, so",
        "`test_the_public_api_matches_its_snapshot` fails until it is",
        "updated; a removed or renamed name is a breaking change:",
        "",
    ]
    for entry in names:
        lines.append(f"- `{entry.package}`")
        lines.extend(f"  - added `{name}`" for name in entry.added)
        lines.extend(f"  - removed `{name}`" for name in entry.removed)
    return "\n".join(lines) + "\n"


def table(changes: list[Change]) -> str:
    """Write changes as a Markdown table.

    Args:
        changes: The changes.

    Returns:
        The table.
    """
    rows = [
        "| Source | Pinned | Upstream | Difference |",
        "|---|---|---|---|",
        *(
            f"| {c.source} | {c.old} | {c.new} | [link]({c.link}) |"
            for c in changes
        ),
    ]
    return "\n".join(rows) + "\n"


PULL_REQUEST: Final = """\
The weekly check found a new version of OpenAI's OpenAPI file. This
pull request moves the pin, rebuilds the contract snapshot
(`tests/providers/openai/contracts/`) and regenerates OpenAI's types.

{table}
{names}
Before merging, read the diff of the snapshot and of
`src/aiollm/providers/openai`:

- a field, type or event that was removed or narrowed is a breaking
  change: add a `**Breaking:**` fragment to `CHANGES/`;
- a new field or event is an `added` fragment, if users can reach it;
- an example listed under `rejected` in `manifest.json` fails the tests
  until it is fixed or removed by hand.

Opened by `.github/workflows/upstream.yml`
(`scripts/openai/refresh_upstream.py`).
"""


def output(name: str, value: str) -> None:
    """Set an output of the GitHub Actions step, when there is one.

    Args:
        name: The output's name.
        value: Its value.
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(f"{name}={value}\n")


def main() -> None:
    """Check the source and refresh what moved."""
    parser = argparse.ArgumentParser(
        description="Follow OpenAI's pinned OpenAPI file."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "aiollm-upstream-openai",
        help="folder for pull-request.md",
    )
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary:
        openapi = Path(temporary) / "openapi.yaml"
        change = follow_openai(openapi)
        if change is not None:
            env = {"OPENAI_OPENAPI": str(openapi)}
            run(
                "uv",
                "run",
                "python",
                "-m",
                "tests.providers.openai.contracts.update",
                **env,
            )
            run("uv", "run", "python", str(SCRIPT), **env)
            body = PULL_REQUEST.format(
                table=table([change]), names=names_section(public_names())
            )
            (out / "pull-request.md").write_text(body, encoding="utf-8")

    output("pins", "true" if change else "false")
    if change is not None:
        sys.stdout.write(f"{change.source}: {change.old} -> {change.new}\n")


if __name__ == "__main__":
    main()
