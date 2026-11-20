# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Up to 0.14.0 the package was `aiollm.providers.openai`, part of aiollm: its
earlier changes are in [aiollm's changelog](https://github.com/aiollm-org/aiollm/blob/master/CHANGELOG.md).

## [Unreleased]

<!-- towncrier release notes start -->

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

[Unreleased]: https://github.com/aiollm-org/aiollm_openai/compare/v0.15.0...HEAD
[0.15.0]: https://github.com/aiollm-org/aiollm_openai/releases/tag/v0.15.0
