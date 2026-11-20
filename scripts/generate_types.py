"""Regenerate OpenAI's types and request parameters with aiollm-codegen.

Run from the aiollm checkout:

    uv run python scripts/openai/generate_types.py [--codegen DIR]

Both inputs are pinned, so the same commit always gives the same files
(the CI job ``lint and types`` regenerates them and fails on any
difference). The generator is the ``codegen`` dependency group, a commit
of aiollm-codegen named in ``pyproject.toml`` and ``uv.lock``;
``--codegen DIR`` runs a local checkout instead, to try a change of the
generator. The types come from OpenAI's official OpenAPI file at the
commit of ``OPENAI_OPENAPI_URL``, downloaded and checked against
``OPENAI_OPENAPI_SHA256`` (the file of the contract snapshot,
``tests/providers/openai/contracts/manifest.json``, a test checks), or
from the file ``OPENAI_OPENAPI`` names, which must be the same;
restricted to ``OPENAI_OPERATIONS``. To follow a new file, refresh the
contract snapshot from it and change the URL and the hash here together.

The script replaces ``types/<family>.py`` and ``methods/params/`` with
the generated modules, keeps the hand-written ``types/options.py`` and
the method classes in ``methods/``, and writes ``types/__init__.py``
from the generated one with the options added. Nothing else is touched.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
PACKAGE: Final = ROOT / "src" / "aiollm" / "providers" / "openai"
MODULE: Final = "aiollm.providers.openai"

OPENAI_OPENAPI_URL: Final = (
    "https://raw.githubusercontent.com/openai/openai-openapi/"
    "c898c67d98dfa5b3fb258825ae82183472b8736e/openapi.yaml"
)
"""OpenAI's OpenAPI file, at the commit of 2026-11-20."""

OPENAI_OPENAPI_SHA256: Final = (
    "b52a3625713c364c027d589d8b102a174d7f492e79babac73d459daf0105ea9c"
)
"""SHA-256 of that file."""

OPENAI_OPERATIONS: Final = (
    "createResponse",
    "createFile",
    "listFiles",
    "retrieveFile",
    "deleteFile",
    "downloadFile",
)
"""The operations of OpenAI's file the library implements."""

HAND_WRITTEN_TYPES = {"options.py"}
"""Modules of ``types/`` the script never replaces."""


def codegen(codegen_dir: Path | None, *args: str, output: Path) -> None:
    """Run aiollm-codegen: the pinned one, or a local checkout."""
    if codegen_dir is None:
        command = ["uv", "run", "--group", "codegen", "aiollm-codegen"]
        cwd = ROOT
    else:
        command = ["uv", "run", "aiollm-codegen"]
        cwd = codegen_dir
    command += [*args, "-o", str(output)]
    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603


def openai_openapi(directory: Path) -> Path:
    """The pinned OpenAPI file of OpenAI: ``OPENAI_OPENAPI``, or a download.

    Raises:
        SystemExit: The file is not the pinned one.
    """
    given = os.environ.get("OPENAI_OPENAPI")
    if given:
        path = Path(given).resolve()
    else:
        path = directory / "openai.openapi.yaml"
        with urllib.request.urlopen(OPENAI_OPENAPI_URL) as response:  # noqa: S310
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != OPENAI_OPENAPI_SHA256:
        message = (
            f"{path} has SHA-256 {digest}, not the pinned "
            f"{OPENAI_OPENAPI_SHA256}; to follow a new file, refresh the "
            "contract snapshot from it and change OPENAI_OPENAPI_URL and "
            "OPENAI_OPENAPI_SHA256 in this script"
        )
        raise SystemExit(message)
    return path


def exported(module: Path) -> tuple[list[str], list[str]]:
    """Read the import lines and ``__all__`` of a generated ``__init__``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    lines = module.read_text(encoding="utf-8").splitlines()
    imports: list[str] = []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imports.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            names = list(ast.literal_eval(node.value))
    return imports, names


def options_names() -> list[str]:
    """Public names of the hand-written ``types/options.py``."""
    tree = ast.parse((PACKAGE / "types" / "options.py").read_text("utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def install(generated: Path) -> None:
    """Replace the generated modules of the package."""
    types_dir = PACKAGE / "types"
    params_dir = PACKAGE / "methods" / "params"

    for module in types_dir.glob("*.py"):
        if module.name not in HAND_WRITTEN_TYPES:
            module.unlink()
    shutil.rmtree(params_dir, ignore_errors=True)
    params_dir.mkdir(parents=True)

    for module in (generated / "types").glob("*.py"):
        if module.name != "__init__.py":
            shutil.copy2(module, types_dir / module.name)
    for module in (generated / "methods" / "params").glob("*.py"):
        shutil.copy2(module, params_dir / module.name)

    imports, names = exported(generated / "types" / "__init__.py")
    options = options_names()
    body = [
        '"""OpenAI\'s own types and the options of a generation call.',
        "",
        "Every module but ``options`` is written by aiollm-codegen through",
        "``scripts/openai/generate_types.py``; do not edit them by hand.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        *imports,
        f"from {MODULE}.types.options import (",
        *[f"    {option}," for option in sorted(options)],
        ")",
        "",
        "__all__ = [",
        *[f'    "{item}",' for item in sorted({*names, *options})],
        "]",
        "",
    ]
    (types_dir / "__init__.py").write_text(
        "\n".join(body), encoding="utf-8", newline="\n"
    )


def main() -> None:
    """Generate the types and install them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--codegen",
        type=Path,
        help="run this checkout of aiollm-codegen, not the pinned one",
    )
    args = parser.parse_args()
    codegen_dir: Path | None = args.codegen and args.codegen.resolve()

    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        openapi = openai_openapi(out)
        codegen(
            codegen_dir,
            str(openapi),
            "--package",
            *(
                argument
                for operation in OPENAI_OPERATIONS
                for argument in ("--operation", operation)
            ),
            output=out / "openai",
        )
        install(out / "openai")

    ruff = ["uv", "run", "ruff"]
    for command in (
        [*ruff, "check", "--select", "I", "--fix", "-q"]
        + [str(PACKAGE / "types" / "__init__.py")],
        [*ruff, "format", "-q", str(PACKAGE)],
    ):
        subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603
    print("OpenAI types regenerated", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
