"""The names re-exported by OpenAI's packages."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

SNAPSHOT = Path(__file__).with_name("public_api.json")
"""The public names of every package, as the last release left them."""

PACKAGES = list(json.loads(SNAPSHOT.read_text(encoding="utf-8")))


def test_every_package_is_listed() -> None:
    root = importlib.import_module("aiollm_openai")
    folder = Path(root.__file__ or "").parent
    found = {
        ".".join(["aiollm_openai", *init.parent.relative_to(folder).parts])
        for init in folder.rglob("__init__.py")
    }

    assert found == set(PACKAGES)


@pytest.mark.parametrize("name", PACKAGES)
def test_all_names_exist_and_are_sorted(name: str) -> None:
    package = importlib.import_module(name)
    exported: list[str] = package.__all__

    assert exported == sorted(exported)
    assert all(hasattr(package, item) for item in exported)


def test_the_public_api_matches_its_snapshot() -> None:
    # A public name added, renamed or removed must be a decision: update
    # public_api.json in the same change, with a changelog fragment.
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = {
        name: list(importlib.import_module(name).__all__) for name in PACKAGES
    }

    assert current == snapshot
