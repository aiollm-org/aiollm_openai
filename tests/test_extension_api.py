"""The package takes only the core's extension API."""

from __future__ import annotations

from aiollm.testing.extension import check_extension_api


def test_the_package_takes_only_the_extension_api() -> None:
    # A name of the core outside aiollm/testing/extension_api.json is
    # not a contract: it can change in any release of the core.
    assert check_extension_api("aiollm_openai") == []
