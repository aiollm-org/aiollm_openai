# aiollm-openai

OpenAI's Responses and Files APIs for [aiollm](https://github.com/aiollm-org/aiollm), the async-first toolkit for typed HTTP API clients.

`OpenAIClient` sends OpenAI's own methods (`CreateResponse`, `StreamResponse` and the five Files methods) with types generated from OpenAI's official OpenAPI file, and has aiollm's provider-neutral `client.generation` and `client.files`, so code written for any aiollm provider runs on it.

> **Status: pre-alpha.** The package was part of aiollm up to 0.14.0 as `aiollm.providers.openai`; it is now `aiollm_openai`. Its public API may still change: before 1.0 a breaking change bumps the minor version.

## Installation

The package requires Python 3.13 or newer and installs the core it needs, [aiollm-core](https://pypi.org/project/aiollm-core/):

```bash
pip install aiollm-openai
```

Or with uv: `uv add aiollm-openai`. To work on the package itself, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

`OpenAIClient()` reads the key from `OPENAI_API_KEY`; pass a token to give it yourself.

```python title="quick_start.py"
from aiollm_openai import OpenAIClient


async def ask() -> str:
    async with OpenAIClient() as client:
        response = await client.generation.create(
            model="gpt-6-astra",
            input="Hi",
            instructions="Be brief",
        )
    return response.text
```

Run it with `asyncio.run(ask())`.

## Documentation

The guide and the API reference are on the aiollm site, **[aiollm-org.github.io/aiollm](https://aiollm-org.github.io/aiollm/)**: [provider clients](https://aiollm-org.github.io/aiollm/guide/providers/), [shared generation](https://aiollm-org.github.io/aiollm/guide/generation/) and [files](https://aiollm-org.github.io/aiollm/guide/files/). The changes up to 0.14.0 are in [aiollm's changelog](https://github.com/aiollm-org/aiollm/blob/master/CHANGELOG.md); the later ones in [CHANGELOG.md](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).
